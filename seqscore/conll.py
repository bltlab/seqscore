import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from math import sqrt
from statistics import mean, stdev
from typing import (
    Any,
    DefaultDict,
    Optional,
    TextIO,
)

from tabulate import tabulate

from seqscore.encoding import Encoding, EncodingError, get_encoding
from seqscore.model import AnnotatedSequence, LabeledSequence, SequenceProvenance
from seqscore.output import (
    FORMAT_CONLLEVAL,
    FORMAT_DELIM,
    FORMAT_PRETTY,
)
from seqscore.scoring import (
    AccuracyScore,
    ClassificationScore,
    compute_scores,
    convert_score,
)
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
                    encoding_name = self.encoding.__class__.__name__
                    raise EncodingError(
                        "Stopping due to invalid label(s) in sequence "
                        + f"at line {line_nums[0]} of {source_name}:\n"
                        + "\n".join(err.msg for err in state_errors)
                        + f"\nThe above labels are not valid for the chunk encoding {encoding_name}."
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

        # Add final document if non-empty
        if document:
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

        if document_results:
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
                    sequence = []
                    comment = None
                # Always skip empty lines
                continue

            token = _CoNLLToken.from_line(line, line_num, source_name, self.line_spec)
            # Skip document starts, but ensure sequence is empty when we reach them
            if token.is_docstart:
                if sequence:
                    raise CoNLLFormatError(
                        f"Encountered {DOCSTART} at line {line_num} of {source_name} in the middle of a sequence"
                    )
                else:
                    # Yield it by itself. Since the sequence variable is empty, leave it unchanged.
                    tmp_sent = (token,)
                    self._check_sequence(tmp_sent)
                    # Don't return the comment yet, it will be returned with the sequence
                    yield tmp_sent, None
            else:
                sequence.append(token)

        # Finish the last sequence if needed
        if sequence:
            self._check_sequence(sequence)
            yield tuple(sequence), comment

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


# TODO: Refactor to remove CoNLL-specific file loading so that this can move to the scoring module
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
) -> None:
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

    # Flag for whether we're scoring multiple files
    multi_files = len(pred_files) > 1

    # Data to accumulate across files
    score_summaries = []
    all_class_scores = []
    all_acc_scores = []

    # Used to track whether this is the first summary for including the header for delim
    first_summary = True
    # Used to track how many fields are in the header
    header_len = -1

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

        class_scores, acc_scores = compute_scores(
            pred_docs, ref_docs, count_fp_fn_examples=error_counts
        )
        all_class_scores.append(class_scores)
        all_acc_scores.append(acc_scores)

        if error_counts:
            if multi_files:
                raise ValueError(
                    "Outputting error counts is only available for a single prediction file"
                )

            if output_format == FORMAT_CONLLEVAL:
                raise ValueError(
                    f"Format {repr(output_format)} is not supported with error counts"
                )
            elif output_format in (FORMAT_PRETTY, FORMAT_DELIM):
                header = ["Count", "Error", "Type", "Tokens"]

                # Combine counts across the two counters
                combined_counts: Counter[tuple[str, str, str]] = Counter()
                for counter, error_type in zip(
                    (class_scores.false_pos_examples, class_scores.false_neg_examples),
                    ("FP", "FN"),
                ):
                    for item, count in counter.items():
                        combined_counts[
                            (error_type, item.type, " ".join(item.tokens))
                        ] = count

                # Sort by count descending (the negative reverses the default
                # ascending sort), breaking ties on the token string
                # (item[0] is the (error_type, mention_type, token_str) key,
                # so item[0][2] is the token string; item[1] is the count).
                rows = [
                    [count, error_type, mention_type, token_str]
                    for (
                        error_type,
                        mention_type,
                        token_str,
                    ), count in sorted(
                        combined_counts.items(), key=lambda item: (-item[1], item[0][2])
                    )
                ]

                if output_format == FORMAT_PRETTY:
                    print(tabulate(rows, header, tablefmt="github"))
                else:
                    # Delimited output
                    score_summaries.append(delim.join(header))
                    score_summaries.extend(_join_delim(row, delim) for row in rows)
                    print("\n".join(score_summaries))

            # Exit early since all the following logic is for printing scores
            return

        if output_format == FORMAT_CONLLEVAL:
            score_summaries.append(format_output_conlleval(class_scores, acc_scores))
        elif output_format in (FORMAT_PRETTY, FORMAT_DELIM):
            header, rows = format_output_table(class_scores, full_precision)
            if output_format == FORMAT_PRETTY:
                if full_precision:
                    raise ValueError("Cannot use full_precision with pretty formatting")
                # We don't allow full_precision in this case so we can use the usual float format
                score_summaries.append(
                    tabulate(rows, header, tablefmt="github", floatfmt="6.2f")
                )
            else:
                # Delimited output
                # Write the header if needed
                if first_summary:
                    # Add filename to header if needed
                    if multi_files:
                        header = ["File"] + header
                    score_summaries.append(delim.join(header))
                    header_len = len(header)
                    first_summary = False

                # Add filename to row if needed
                if multi_files:
                    rows = [[pred_file] + row for row in rows]

                # Double check that we have the same number of columns as the header.
                for row in rows:
                    assert len(row) == header_len, (
                        "Row column count does not match header"
                    )
                score_summaries.extend(_join_delim(row, delim) for row in rows)
        else:
            raise ValueError(f"Unrecognized output format: {output_format}")

    # Compute summary statistics across files when multiple files are scored
    if multi_files:
        type_scores: DefaultDict[str, list] = defaultdict(list)
        for class_score in all_class_scores:
            for entity_type, entity_score in class_score.type_scores.items():
                type_scores[entity_type].append(entity_score.f1)

        entity_type_means = {
            entity_type: mean(scores) for entity_type, scores in type_scores.items()
        }
        entity_type_means[ALL_TYPES] = mean(score.f1 for score in all_class_scores)

        entity_type_stderrs = {
            entity_type: stdev(scores) / sqrt(len(scores))
            for entity_type, scores in type_scores.items()
        }
        all_f1s = [score.f1 for score in all_class_scores]
        entity_type_stderrs[ALL_TYPES] = stdev(all_f1s) / sqrt(len(all_f1s))

    # For delimited, just join all the rows
    if output_format == FORMAT_DELIM:
        if multi_files:
            for entity_type, num in entity_type_stderrs.items():
                score_summaries.append(
                    _join_delim(
                        [
                            "SE",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(num, full_precision),
                            "NA",
                            "NA",
                            "NA",
                        ],
                        delim,
                    )
                )
            for entity_type, num in entity_type_means.items():
                score_summaries.append(
                    _join_delim(
                        [
                            "Mean",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(num, full_precision),
                            "NA",
                            "NA",
                            "NA",
                        ],
                        delim,
                    )
                )
        print("\n".join(score_summaries))
    else:
        if not multi_files:
            print(score_summaries[0])
        else:
            # Use the index because we care whether we're at the last entry
            for idx, (filename, summary) in enumerate(zip(pred_files, score_summaries)):
                print(filename)
                print(summary)
                # Don't print an extra trailing newline
                if idx != len(pred_files) - 1:
                    print()

            # Print mean ± SE summary table
            ref_scores = all_class_scores[0]
            summary_header = ["Type", "Mean F1", "SE", "Reference"]
            summary_rows = [
                [
                    ALL_TYPES,
                    entity_type_means[ALL_TYPES] * 100,
                    entity_type_stderrs[ALL_TYPES] * 100,
                    ref_scores.total_ref,
                ]
            ]
            for entity_type in sorted(entity_type_means):
                if entity_type == ALL_TYPES:
                    continue
                summary_rows.append(
                    [
                        entity_type,
                        entity_type_means[entity_type] * 100,
                        entity_type_stderrs[entity_type] * 100,
                        ref_scores.type_scores[entity_type].total_ref,
                    ]
                )
            print()
            print("Summary")
            print(
                tabulate(
                    summary_rows,
                    summary_header,
                    tablefmt="github",
                    floatfmt="6.2f",
                )
            )


def format_output_conlleval(
    class_scores: ClassificationScore,
    acc_scores: AccuracyScore,
) -> str:
    """Format output like conlleval.pl.

    Example:
    processed 15 tokens with 3 phrases; found: 4 phrases; correct: 2.
    accuracy:  93.33%; precision:  50.00%; recall:  66.67%; FB1:  57.14
                  LOC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3
                  ORG: precision: 100.00%; recall: 100.00%; FB1: 100.00  1
    """
    n_phrases = class_scores.true_pos + class_scores.false_neg
    lines = [
        f"processed {acc_scores.total} tokens with {n_phrases} phrases; "
        + f"found: {class_scores.total_pos} phrases; correct: {class_scores.true_pos}.",
        f"accuracy: {100 * acc_scores.accuracy:6.2f}%; "
        + f"precision: {100 * class_scores.precision:6.2f}%; "
        + f"recall: {100 * class_scores.recall:6.2f}%; "
        + f"FB1: {100 * class_scores.f1:6.2f}",
    ]

    # Add lines for each type
    for type_name, score in sorted(class_scores.type_scores.items()):
        lines.append(
            f"{type_name.rjust(17)}: "  # This is the width that conlleval uses
            + f"precision: {100 * score.precision:6.2f}%; "
            + f"recall: {100 * score.recall:6.2f}%; "
            + f"FB1: {100 * score.f1:6.2f}  {score.total_pos}"
        )

    return "\n".join(lines)


def format_output_table(
    class_scores: ClassificationScore,
    full_precision: bool,
) -> tuple[list[str], list[list[Any]]]:
    header = [
        "Type",
        "Precision",
        "Recall",
        "F1",
        "Reference",
        "Predicted",
        "Correct",
    ]
    rows = [
        [
            ALL_TYPES,
            convert_score(class_scores.precision, full_precision),
            convert_score(class_scores.recall, full_precision),
            convert_score(class_scores.f1, full_precision),
            class_scores.total_ref,
            class_scores.total_pos,
            class_scores.true_pos,
        ]
    ]

    # Add lines for each type
    for type_name, score in sorted(class_scores.type_scores.items()):
        rows.append(
            [
                type_name,
                convert_score(score.precision, full_precision),
                convert_score(score.recall, full_precision),
                convert_score(score.f1, full_precision),
                score.total_ref,
                score.total_pos,
                score.true_pos,
            ]
        )

    return header, rows


def _join_delim(items: Iterable[Any], delim: str) -> str:
    return delim.join(str(item) for item in items)
