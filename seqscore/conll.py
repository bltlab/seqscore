import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, TextIO

from seqscore.encoding import Encoding, EncodingError, get_encoding
from seqscore.model import AnnotatedSequence, LabeledSequence, SequenceProvenance
from seqscore.output import report_scores
from seqscore.util import PathType
from seqscore.validation import (
    InvalidLabelError,
    SequenceValidationResult,
    ValidationResult,
    validate_labels,
)

DOCSTART = "-DOCSTART-"
EMPTY_OTHER_FIELD = "-X-"


ALL_TYPES = "ALL"


class CoNLLFormatError(Exception):
    pass


@dataclass(frozen=True)
class LineSpec:
    """Defines the fields and delimiters for a CoNLL-format line"""

    token_index: int
    ner_label_index: int

    def __post_init__(self) -> None:
        # This will only catch cases where the indices are identical, not
        # when they refer to the same position (such as 1 and -1 in a
        # sequence of length two).
        if self.token_index == self.ner_label_index:
            raise ValueError(
                f"Token index ({self.token_index}) and "
                f"label index ({self.ner_label_index}) cannot be the same"
            )


@dataclass(frozen=True, slots=True)
class _CoNLLToken:
    text: str
    label: str
    is_docstart: bool
    line_num: int
    orig_fields: tuple[str, ...]

    @classmethod
    def from_line(
        cls, line: str, line_num: int, source_name: str, line_spec: LineSpec
    ) -> "_CoNLLToken":
        # Note: The caller must strip the line of any trailing whitespace
        # TODO: Sense the file rather than the line so we get consistency across lines
        # Try tab first since it's safer, then space
        splits = line.split("\t")
        if len(splits) == 1:
            splits = line.split(" ")

        if len(splits) < 2:
            if line.startswith("#"):
                raise CoNLLFormatError(
                    f"Line {line_num} of {source_name} does not appear to be delimited "
                    "and begins with #. Perhaps you want to use the --allow-comment-lines "
                    f"flag? Line contents: {repr(line)}"
                )
            else:
                raise CoNLLFormatError(
                    f"Line {line_num} of {source_name} is not delimited by space or tab: {repr(line)}"
                )

        text = splits[line_spec.token_index]
        label = splits[line_spec.ner_label_index]
        orig_fields = tuple(splits)
        is_docstart = text == DOCSTART
        return cls(text, label, is_docstart, line_num, orig_fields)


@dataclass(frozen=True)
class CoNLLIngester:
    encoding: Encoding
    line_spec: LineSpec
    allow_comment_lines: bool = field(default=False, kw_only=True)
    ignore_document_boundaries: bool = field(default=False, kw_only=True)

    def ingest(
        self,
        source: TextIO,
        source_name: str,
        repair: Optional[str],
        *,
        quiet: bool = False,
    ) -> list[list[AnnotatedSequence]]:
        all_documents: list[list[AnnotatedSequence]] = []
        document: list[AnnotatedSequence] = []

        for source_sequence, comment in self._parse_file(
            source, source_name, parse_comments=self.allow_comment_lines
        ):
            if source_sequence[0].is_docstart:
                # We can ony receive DOCSTART in a sequence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sequence.
                assert len(source_sequence) == 1

                # TODO: Preserve document-level comments
                # _parse_file attaches a comment preceding a DOCSTART to the DOCSTART
                # sequence, but we drop it here because a document is just a list of
                # sequences with nowhere to store it, so it does not round-trip on output.

                # End current document and start a new one if we're attending to boundaries.
                # We skip this if the builder is empty, which will happen for the very
                # first document in the corpus (as there is no previous document to end).
                if not self.ignore_document_boundaries and document:
                    all_documents.append(document)
                    document = []
                continue

            # Create mentions from tokens in sequence
            tokens, labels, line_nums, orig_fields = self._decompose_sequence(
                source_sequence
            )

            # Validate before decoding
            try:
                validation = validate_labels(
                    labels,
                    self.encoding,
                    repair=repair,
                    tokens=tokens,
                    line_nums=line_nums,
                    source_name=source_name,
                )
            except InvalidLabelError as err:
                # Try to catch lines that start with # in case they are comments
                if tokens and tokens[0].startswith("#"):
                    raise InvalidLabelError(
                        err.label,
                        str(err)
                        + f" The first token {repr(tokens[0])} of this sentence starts with '#'."
                        + " If it's a comment, consider enabling --allow-comment-lines.",
                    ) from err
                else:
                    raise err

            if not validation.is_valid():
                # Exit immediately if there are state errors
                state_errors = validation.invalid_state_errors()
                if state_errors:
                    raise EncodingError(
                        "Stopping due to invalid label(s) in sequence "
                        + f"at line {line_nums[0]} of {source_name}:\n"
                        + "\n".join(err.msg for err in state_errors)
                        + f"\nThe above labels are not valid for the chunk encoding {self.encoding.name}."
                        + "\nCorrect your data or specify the correct encoding using --labels."
                    )

                if repair:
                    if not quiet:
                        msg = (
                            [
                                f"Validation errors in sequence beginning at line {line_nums[0]} of {source_name}:"
                            ]
                            + [error.msg for error in validation.errors]
                            + [
                                f"Used method {repair} to repair:",
                                f"Old: {labels}",
                                f"New: {validation.repaired_labels}",
                            ]
                        )
                        print("\n".join(msg), file=sys.stderr)
                    labels = validation.repaired_labels
                else:
                    raise EncodingError(
                        f"Stopping due to validation errors in {source_name}:\n"
                        + "\n".join(err.msg for err in validation.errors)
                    )

            try:
                mentions = self.encoding.decode_labels(labels)
            except AssertionError as e:  # pragma: no cover
                # Unreachable unless there is a bug in the decoder or validation
                raise ValueError(
                    "Encountered an error decoding this sequence despite passing validation: "
                    + " ".join(labels),
                ) from e

            try:
                sequences = AnnotatedSequence(
                    tokens=tokens,
                    labels=labels,
                    mentions=tuple(mentions),
                    token_fields=orig_fields,
                    provenance=SequenceProvenance(line_nums[0], source_name),
                    comment=comment,
                )
            except ValueError as e:  # pragma: no cover
                # Unreachable unless there is a bug in validation
                raise ValueError(
                    f"Invalid sequence error in sequence beginning at line {line_nums[0]} of {source_name}"
                ) from e
            document.append(sequences)

        # There is always a final document since empty input is rejected by _parse_file.
        assert document
        all_documents.append(document)

        return all_documents

    def validate(
        self,
        source: TextIO,
        source_name: str,
    ) -> list[list[SequenceValidationResult]]:
        all_results: list[list[SequenceValidationResult]] = []
        document_results: list[SequenceValidationResult] = []

        for source_sequence, _ in self._parse_file(
            source,
            source_name,
            parse_comments=self.allow_comment_lines,
        ):
            if source_sequence[0].is_docstart:
                # We can ony receive DOCSTART in a sequence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sequence.
                assert len(source_sequence) == 1

                # If we care about document boundaries and have results for this document,
                # add it and move on.
                if not self.ignore_document_boundaries and document_results:
                    all_results.append(document_results)
                    document_results = []

                # Go to the next sequence
                continue

            # Create mentions from tokens in sequence
            tokens, labels, line_nums, _ = self._decompose_sequence(source_sequence)

            # Validate
            document_results.append(
                validate_labels(labels, self.encoding, tokens=tokens, line_nums=line_nums)
            )

        # There is always a final document since empty input is rejected by _parse_file.
        assert document_results
        all_results.append(document_results)

        return all_results

    @staticmethod
    def _decompose_sequence(
        source_sequence: Sequence[_CoNLLToken],
    ) -> tuple[
        tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[tuple[str, ...], ...]
    ]:
        tokens = tuple(tok.text for tok in source_sequence)
        labels = tuple(tok.label for tok in source_sequence)
        line_nums = tuple(tok.line_num for tok in source_sequence)
        orig_fields = tuple(tok.orig_fields for tok in source_sequence)
        return tokens, labels, line_nums, orig_fields

    def _parse_file(
        self,
        input_file: TextIO,
        source_name: str,
        *,
        parse_comments: bool = False,
    ) -> Iterable[tuple[tuple[_CoNLLToken, ...], Optional[str]]]:
        sequence: list = []
        comment: Optional[str] = None
        # Line of the most recent DOCSTART that has not yet been followed by a
        # sequence, or None if there is none. Used to reject empty documents (a
        # DOCSTART with no sequences before the next DOCSTART or end of file).
        last_docstart_line: Optional[int] = None
        # Whether anything at all (a sequence or a DOCSTART) has been yielded,
        # used to reject completely empty input.
        have_yielded = False
        line_num = 0
        for line in input_file:
            line_num += 1
            # We only remove trailing space and newline. If there's other weird whitespace at the
            # end of a line, it could very well be something else (e.g. actual data columns)
            line = line.rstrip(" \n")

            # Parse comments at the start of sequences if specified
            if line.startswith("#") and not sequence:
                if parse_comments:
                    if comment:
                        # Add a second comment line if there was one already
                        comment += f"\n{line}"
                    else:
                        comment = line
                    continue

            # Handle whitespace-only lines
            if not line.strip():
                # Clear out sequence if there's anything in it
                if sequence:
                    self._check_sequence(sequence)
                    yield tuple(sequence), comment
                    have_yielded = True
                    # Reset state
                    sequence = []
                    comment = None
                    last_docstart_line = None
                # Always skip empty lines
                continue

            token = _CoNLLToken.from_line(line, line_num, source_name, self.line_spec)
            # Skip document starts, but ensure sequence is empty when we reach them
            if token.is_docstart:
                if sequence:
                    raise CoNLLFormatError(
                        f"Encountered {DOCSTART} at line {line_num} of {source_name} in the middle of a sequence"
                    )
                # If we're in the middle of another document, raise an error
                if last_docstart_line is not None:
                    raise CoNLLFormatError(
                        f"Encountered {DOCSTART} at line {last_docstart_line} of "
                        f"{source_name} with no sequences before the {DOCSTART} at "
                        f"line {line_num}"
                    )
                # Record this DOCSTART
                last_docstart_line = line_num
                # Yield the DOCSTART as its own single-token sequence. The sequence
                # variable is empty here (otherwise we would have raised an error earlier),
                # so we leave it unchanged for the next sentence.
                docstart_sequence = (token,)
                self._check_sequence(docstart_sequence)
                # A comment before a DOCSTART belongs to the document boundary, not
                # the first sentence of the document, so yield it with the DOCSTART
                # and clear it rather than carrying it forward to the next sequence.
                yield docstart_sequence, comment
                comment = None
                have_yielded = True
            else:
                sequence.append(token)

        # Finish the last sequence if needed
        if sequence:
            self._check_sequence(sequence)
            yield tuple(sequence), comment
            have_yielded = True
            last_docstart_line = None

        # A trailing DOCSTART with no sequences after it is an empty document
        if last_docstart_line is not None:
            raise CoNLLFormatError(
                f"Encountered {DOCSTART} at line {last_docstart_line} of "
                f"{source_name} with no sequences following it"
            )

        # A file with no sequences at all is empty and invalid
        if not have_yielded:
            raise CoNLLFormatError(f"{source_name} contains no sequences")

    @staticmethod
    def _check_sequence(sequence: Sequence[_CoNLLToken]) -> None:
        # We should only return DOCSTART in a sequence by itself. This isn't a constraint
        # on the layout of the input document, but rather one we are enforcing so that consumers
        # get document boundaries as their own sequences.
        if sequence[0].is_docstart and len(sequence) > 1:
            raise ValueError(
                f"Returned {DOCSTART} as part of a sequence at line {sequence[0].line_num}"
            )


def ingest_conll_file(
    input_path: PathType,
    mention_encoding_name: str,
    file_encoding: str,
    line_spec: LineSpec,
    *,
    repair: Optional[str] = None,
    ignore_document_boundaries: bool,
    allow_comment_lines: bool,
    quiet: bool = False,
) -> list[list[AnnotatedSequence]]:
    mention_encoding = get_encoding(mention_encoding_name)

    if repair and repair not in mention_encoding.supported_repair_methods():
        raise ValueError(
            f"Cannot repair mention encoding {mention_encoding_name} using method {repair}.\n"
            + 'Set --repair-method to "none" for this encoding.'
        )

    ingester = CoNLLIngester(
        mention_encoding,
        line_spec,
        allow_comment_lines=allow_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding=file_encoding) as input_file:
        docs = ingester.ingest(input_file, str(input_path), repair, quiet=quiet)
    return docs


def validate_conll_file(
    input_path: str,
    mention_encoding_name: str,
    file_encoding: str,
    line_spec: LineSpec,
    *,
    ignore_document_boundaries: bool,
    allow_comment_lines: bool,
) -> ValidationResult:
    encoding = get_encoding(mention_encoding_name)
    ingester = CoNLLIngester(
        encoding,
        line_spec,
        allow_comment_lines=allow_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding=file_encoding) as input_file:
        results = ingester.validate(input_file, input_path)

    n_docs = len(results)
    n_sequences = sum(len(doc_results) for doc_results in results)
    n_tokens = sum(sent.n_tokens for doc_results in results for sent in doc_results)

    errors = tuple(
        chain.from_iterable(
            result.errors for doc_results in results for result in doc_results
        )
    )
    return ValidationResult(errors, n_tokens, n_sequences, n_docs)


def repair_conll_file(
    input_file: PathType,
    mention_encoding_name: str,
    repair: Optional[str],
    file_encoding: str,
    line_spec: LineSpec,
    *,
    ignore_document_boundaries: bool,
    allow_comment_lines: bool,
    quiet: bool,
) -> list[list[AnnotatedSequence]]:
    return ingest_conll_file(
        input_file,
        mention_encoding_name,
        file_encoding,
        line_spec,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        allow_comment_lines=allow_comment_lines,
        quiet=quiet,
    )


def write_docs_using_encoding(
    docs: Sequence[Sequence[AnnotatedSequence]],
    mention_encoding_name: str,
    file_encoding: str,
    delim: str,
    line_spec: LineSpec,
    output_path: PathType,
    *,
    discard_extra_fields: bool = False,
    always_write_docstart: bool = False,
) -> None:
    mention_encoding = get_encoding(mention_encoding_name)
    output_docstart = len(docs) > 1 or always_write_docstart

    with open(output_path, "w", encoding=file_encoding) as file:
        for doc in docs:
            write_doc_using_encoding(
                doc,
                mention_encoding,
                delim,
                file,
                line_spec,
                output_docstart=output_docstart,
                discard_extra_fields=discard_extra_fields,
            )


def write_docs_raw(
    docs: Sequence[Sequence[LabeledSequence]],
    file_encoding: str,
    delim: str,
    line_spec: LineSpec,
    output_path: PathType,
    *,
    outside_label: str = "O",
    discard_extra_fields: bool = False,
    always_write_docstart: bool = False,
) -> None:
    output_docstart = len(docs) > 1 or always_write_docstart

    with open(output_path, "w", encoding=file_encoding) as file:
        for doc in docs:
            write_doc_raw(
                doc,
                delim,
                file,
                line_spec,
                output_docstart=output_docstart,
                outside_label=outside_label,
                discard_extra_fields=discard_extra_fields,
            )


def write_doc_raw(
    doc: Sequence[LabeledSequence],
    delim: str,
    file: TextIO,
    line_spec: LineSpec,
    *,
    output_docstart: bool,
    outside_label: str = "O",
    discard_extra_fields: bool = False,
) -> None:
    if output_docstart:
        # Get the fields of the first token of the first sentence
        if doc[0].token_fields and not discard_extra_fields:
            # Figure out how many fields there are
            sequence_orig_fields = doc[0].token_fields[0]
            # Create the right number of fields
            fields = [EMPTY_OTHER_FIELD] * len(sequence_orig_fields)
            # Fill in the token and label
            fields[line_spec.token_index] = DOCSTART
            fields[line_spec.ner_label_index] = outside_label
        else:
            fields = [DOCSTART, outside_label]
        # Write output
        print(delim.join(fields), file=file)
        print(file=file)

    for sequence in doc:
        # Write any comment lines before the sequence they belong to. The comment
        # already includes the leading "#" (and any embedded newlines for multi-line
        # comments), so it is written verbatim.
        if sequence.comment is not None:
            print(sequence.comment, file=file)

        for (token, orig_fields), label in zip(
            sequence.tokens_with_fields(), sequence.labels
        ):
            if orig_fields and not discard_extra_fields:
                fields = list(orig_fields)
                fields[line_spec.token_index] = token
                fields[line_spec.ner_label_index] = label
            else:
                fields = [token, label]
            # Write output
            print(delim.join(fields), file=file)

        # Print an empty line after each sequence
        print(file=file)


def write_doc_using_encoding(
    doc: Sequence[AnnotatedSequence],
    encoding: Encoding,
    delim: str,
    file: TextIO,
    line_spec: LineSpec,
    *,
    output_docstart: bool,
    discard_extra_fields: bool = False,
) -> None:
    # Re-encode mentions -> labels for each sequence, then defer to write_doc_raw
    # for the actual DOCSTART/token/blank-line writing so the two writers share
    # one implementation.
    raw_doc = [
        LabeledSequence(
            tokens=sequence.tokens,
            labels=tuple(encoding.encode_sequence(sequence)),
            token_fields=sequence.token_fields,
            provenance=sequence.provenance,
            comment=sequence.comment,
        )
        for sequence in doc
    ]
    write_doc_raw(
        raw_doc,
        delim,
        file,
        line_spec,
        output_docstart=output_docstart,
        outside_label=encoding.dialect.outside,
        discard_extra_fields=discard_extra_fields,
    )


def score_conll_files(
    pred_files: Sequence[PathType],
    reference_file: PathType,
    mention_encoding_name: str,
    repair: Optional[str],
    file_encoding: str,
    line_spec: LineSpec,
    *,
    ignore_document_boundaries: bool,
    allow_comment_lines: bool,
    output_format: str,
    delim: str,
    error_counts: bool = False,
    full_precision: bool = False,
    quiet: bool = False,
    table_format: str = "github",
    file: Optional[TextIO] = None,
) -> None:
    """Load the reference and prediction files, then delegate presentation to report_scores."""
    assert len(pred_files) > 0, "List of files to score cannot be empty"

    ref_docs = ingest_conll_file(
        reference_file,
        mention_encoding_name,
        file_encoding,
        line_spec,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        allow_comment_lines=allow_comment_lines,
        quiet=quiet,
    )

    pred_docs_by_file: list[tuple[str, Sequence[Sequence[AnnotatedSequence]]]] = []
    for pred_file in pred_files:
        pred_docs = ingest_conll_file(
            pred_file,
            mention_encoding_name,
            file_encoding,
            line_spec,
            repair=repair,
            ignore_document_boundaries=ignore_document_boundaries,
            allow_comment_lines=allow_comment_lines,
            quiet=quiet,
        )
        pred_docs_by_file.append((str(pred_file), pred_docs))

    report_scores(
        pred_docs_by_file=pred_docs_by_file,
        ref_docs=ref_docs,
        output_format=output_format,
        delim=delim,
        error_counts=error_counts,
        full_precision=full_precision,
        table_format=table_format,
        file=file,
    )
