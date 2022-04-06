import sys
from collections import defaultdict
from itertools import chain
from statistics import mean, stdev
from typing import Any, DefaultDict, Iterable, List, Optional, Sequence, TextIO, Tuple

from attr import attrib, attrs
from tabulate import tabulate

from seqscore.encoding import Encoding, EncodingError, get_encoding
from seqscore.model import LabeledSequence, SequenceProvenance
from seqscore.scoring import (
    AccuracyScore,
    ClassificationScore,
    compute_scores,
    convert_score,
)
from seqscore.util import PathType
from seqscore.validation import (
    SequenceValidationResult,
    ValidationResult,
    validate_labels,
)

DOCSTART = "-DOCSTART-"


FORMAT_PRETTY = "pretty"
FORMAT_CONLL = "conlleval"
FORMAT_DELIM = "delim"
SUPPORTED_SCORE_FORMATS = (FORMAT_PRETTY, FORMAT_CONLL, FORMAT_DELIM)


@attrs(frozen=True)
class _CoNLLToken:
    text: str = attrib()
    label: str = attrib()
    is_docstart: bool = attrib()
    line_num: int = attrib()

    @classmethod
    def from_line(cls, line: str, line_num: int, source_name: str) -> "_CoNLLToken":
        # Note: The caller must strip the line of any trailing whitespace
        # TODO: Sense the file rather than the line so we get consistency across lines
        # Try tab first since it's safer, then space
        splits = line.split("\t")
        if len(splits) == 1:
            splits = line.split(" ")

        if len(splits) < 2:
            raise ValueError(
                f"Line {line_num} of {source_name} is not delimited by space or tab: {repr(line)}"
            )

        text = splits[0]
        label = splits[-1]
        is_docstart = text == DOCSTART
        return cls(text, label, is_docstart, line_num)


@attrs(frozen=True)
class CoNLLIngester:
    encoding: Encoding = attrib()
    ignore_comment_lines: bool = attrib(default=False, kw_only=True)
    ignore_document_boundaries: bool = attrib(default=True, kw_only=True)

    def ingest(
        self,
        source: TextIO,
        source_name: str,
        repair: Optional[str],
        *,
        quiet: bool = False,
    ) -> Iterable[List[LabeledSequence]]:
        document_counter = 0
        document: List[LabeledSequence] = []

        for source_sequence in self._parse_file(
            source, source_name, ignore_comments=self.ignore_comment_lines
        ):
            if source_sequence[0].is_docstart:
                # We can ony receive DOCSTART in a sequence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sequence.
                assert len(source_sequence) == 1
                # End current document and start a new one if we're attending to boundaries.
                # We skip this if the builder is empty, which will happen for the very
                # first document in the corpus (as there is no previous document to end).
                if not self.ignore_document_boundaries and document:
                    document_counter += 1
                    yield document
                    document = []
                continue

            # Create mentions from tokens in sequence
            tokens, labels, line_nums = self._decompose_sequence(source_sequence)

            # Validate before decoding
            validation = validate_labels(
                labels, self.encoding, repair=repair, tokens=tokens, line_nums=line_nums
            )
            if not validation.is_valid():
                # Exit immediately if there are state errors
                state_errors = validation.invalid_state_errors()
                if state_errors:
                    raise EncodingError(
                        "Stopping due to invalid label(s) in sequence "
                        + f"at line {line_nums[0]} of {source_name}:\n"
                        + "\n".join(err.msg for err in state_errors)
                        + "\nInvalid label errors cannot be repaired; correct your data instead."
                    )

                if repair:
                    if not quiet:
                        msg = (
                            [
                                f"Validation errors in sequence at line {line_nums[0]} of {source_name}:"
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
            except EncodingError as e:
                raise ValueError(
                    "Encountered an error decoding this sequence despite passing validation: "
                    + " ".join(labels),
                ) from e

            sequences = LabeledSequence(
                tokens,
                labels,
                mentions,
                provenance=SequenceProvenance(line_nums[0], source_name),
            )
            document.append(sequences)

        # Yield final document if non-empty
        if document:
            document_counter += 1
            yield document

    def validate(
        self, source: TextIO, source_name: str
    ) -> List[List[SequenceValidationResult]]:
        all_results: List[List[SequenceValidationResult]] = []
        document_results: List[SequenceValidationResult] = []

        for source_sequence in self._parse_file(
            source, source_name, ignore_comments=self.ignore_comment_lines
        ):
            if source_sequence[0].is_docstart:
                # We can ony receive DOCSTART in a sequence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sequence.
                assert len(source_sequence) == 1

                # If we care about document boundaries and we have results for this documents,
                # add it and move on.
                if not self.ignore_document_boundaries and document_results:
                    all_results.append(document_results)
                    document_results = []

                # Go to the next sequence
                continue

            # Create mentions from tokens in sequence
            tokens, labels, line_nums = self._decompose_sequence(source_sequence)

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
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[int, ...]]:
        tokens = tuple(tok.text for tok in source_sequence)
        labels = tuple(tok.label for tok in source_sequence)
        line_nums = tuple(tok.line_num for tok in source_sequence)
        return tokens, labels, line_nums

    @classmethod
    def _parse_file(
        cls, input_file: TextIO, source_name: str, *, ignore_comments: bool = False
    ) -> Iterable[Tuple[_CoNLLToken, ...]]:
        sequence: list = []
        line_num = 0
        for line in input_file:
            line_num += 1
            line = line.strip()

            if ignore_comments and line.startswith("#"):
                continue

            if not line:
                # Clear out sequence if there's anything in it
                if sequence:
                    cls._check_sequence(sequence)
                    yield tuple(sequence)
                    sequence = []
                # Always skip empty lines
                continue

            token = _CoNLLToken.from_line(line, line_num, source_name)
            # Skip document starts, but ensure sequence is empty when we reach them
            if token.is_docstart:
                if sequence:
                    raise ValueError(
                        f"Encountered DOCSTART at line {line_num} while still in sequence"
                    )
                else:
                    # Yield it by itself. Since the sequence variable is empty, leave it unchanged.
                    tmp_sent = (token,)
                    cls._check_sequence(tmp_sent)
                    yield tmp_sent
            else:
                sequence.append(token)

        # Finish the last sequence if needed
        if sequence:
            cls._check_sequence(sequence)
            yield tuple(sequence)

    @staticmethod
    def _check_sequence(sequence: Sequence[_CoNLLToken]):
        # We should only return DOCSTART in a sequence by itself. This isn't a constraint
        # on the layout of the input document, but rather one we are enforcing so that consumers
        # get document boundaries as their own sequences.
        if sequence[0].is_docstart and len(sequence) > 1:
            raise ValueError(
                f"Returned -DOCSTART- as part of a sequence at line {sequence[0].line_num}"
            )


def ingest_conll_file(
    input_path: PathType,
    mention_encoding_name: str,
    file_encoding: str,
    *,
    repair: Optional[str] = None,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    quiet: bool = False,
) -> List[List[LabeledSequence]]:
    mention_encoding = get_encoding(mention_encoding_name)

    if repair and repair not in mention_encoding.supported_repair_methods():
        raise ValueError(
            f"Cannot repair mention encoding {mention_encoding_name} using method {repair}.\n"
            + 'Set --repair-method to "none" for this encoding.'
        )

    ingester = CoNLLIngester(
        mention_encoding,
        ignore_comment_lines=ignore_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding=file_encoding) as input_file:
        docs = list(ingester.ingest(input_file, str(input_path), repair, quiet=quiet))
    return docs


def validate_conll_file(
    input_path: str,
    mention_encoding_name: str,
    file_encoding: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
) -> ValidationResult:
    encoding = get_encoding(mention_encoding_name)
    ingester = CoNLLIngester(
        encoding,
        ignore_comment_lines=ignore_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding=file_encoding) as input_file:
        results = ingester.validate(input_file, input_path)

    n_docs = len(results)
    n_sequences = sum(len(doc_results) for doc_results in results)
    n_tokens = sum(sent.n_tokens for doc_results in results for sent in doc_results)

    errors = list(
        chain.from_iterable(
            result.errors for doc_results in results for result in doc_results
        )
    )
    return ValidationResult(errors, n_tokens, n_sequences, n_docs)


def repair_conll_file(
    input_file: PathType,
    output_file: PathType,
    mention_encoding_name: str,
    repair: Optional[str],
    file_encoding: str,
    output_delim: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    quiet: bool,
) -> None:
    docs = ingest_conll_file(
        input_file,
        mention_encoding_name,
        file_encoding,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        quiet=quiet,
    )

    output_docstart = len(docs) > 1

    with open(output_file, "w", encoding=file_encoding) as file:
        for doc in docs:
            _write_doc_labels(doc, output_delim, file, output_docstart=output_docstart)


def _write_doc_labels(
    doc: Sequence[LabeledSequence], delim: str, file: TextIO, *, output_docstart: bool
) -> None:
    if output_docstart:
        print(f"{DOCSTART}{delim}O", file=file)
        print(file=file)

    for sequence in doc:
        for token, label in sequence.tokens_with_labels():
            print(f"{token}{delim}{label}", file=file)
        print(file=file)


def write_docs_using_encoding(
    docs: Sequence[Sequence[LabeledSequence]],
    mention_encoding_name: str,
    file_encoding: str,
    delim: str,
    output_path: PathType,
) -> None:
    mention_encoding = get_encoding(mention_encoding_name)
    output_docstart = len(docs) > 1

    with open(output_path, "w", encoding=file_encoding) as file:
        for doc in docs:
            write_doc_using_encoding(
                doc, mention_encoding, delim, file, output_docstart=output_docstart
            )


def write_doc_using_encoding(
    doc: Sequence[LabeledSequence],
    encoding: Encoding,
    delim: str,
    file: TextIO,
    *,
    output_docstart: bool,
) -> None:
    if output_docstart:
        print(f"{DOCSTART}{delim}O", file=file)
        print(file=file)

    for sequence in doc:
        labels = encoding.encode_sequence(sequence)
        for token, label in zip(sequence.tokens, labels):
            print(f"{token}{delim}{label}", file=file)
        print(file=file)


# TODO: Refactor to remove CoNLL-specific file loading so that this can move to the scoring module
def score_conll_files(
    pred_files: Sequence[PathType],
    reference_file: PathType,
    mention_encoding_name: str,
    repair: Optional[str],
    file_encoding: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    output_format: str,
    delim: str,
    quiet: bool = False,
) -> None:
    assert len(pred_files) > 0, "List of files to score cannot be empty"

    ref_docs = ingest_conll_file(
        reference_file,
        mention_encoding_name,
        file_encoding,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
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
            repair=repair,
            ignore_document_boundaries=ignore_document_boundaries,
            ignore_comment_lines=ignore_comment_lines,
            quiet=quiet,
        )

        class_scores, acc_scores = compute_scores(pred_docs, ref_docs)
        all_class_scores.append(class_scores)
        all_acc_scores.append(class_scores)

        if output_format == FORMAT_CONLL:
            score_summaries.append(format_output_conlleval(class_scores, acc_scores))
        elif output_format in (FORMAT_PRETTY, FORMAT_DELIM):
            header, rows = format_output_table(class_scores)
            if output_format == FORMAT_PRETTY:
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

                # Double check that we have the same number of columns as the header. This
                # should be the case as long as the system doesn't produce a type that doesn't
                # exist in the reference.
                # TODO: Figure out how to handle a system producing a type not in the reference
                for row in rows:
                    assert (
                        len(row) == header_len
                    ), "Row column count does not match header"
                score_summaries.extend(_join_delim(row, delim) for row in rows)
        else:
            raise ValueError(f"Unrecognized output format: {output_format}")

    # For delimited, just join all the rows
    if output_format == FORMAT_DELIM:
        if multi_files:
            # Compute summary statistics
            type_scores: DefaultDict[str, List] = defaultdict(list)
            for class_score in all_class_scores:
                for entity_type, entity_score in class_score.type_scores.items():
                    type_scores[entity_type].append(entity_score.f1)

            entity_type_means = {
                entity_type: mean(scores) for entity_type, scores in type_scores.items()
            }
            entity_type_means["ALL"] = mean(score.f1 for score in all_class_scores)
            entity_type_sds = {
                entity_type: stdev(scores) for entity_type, scores in type_scores.items()
            }
            entity_type_sds["ALL"] = stdev(score.f1 for score in all_class_scores)

            for entity_type, num in entity_type_sds.items():
                score_summaries.append(
                    # TODO: Change SD precision
                    _join_delim(
                        [
                            "SD",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(num),
                            "NA",
                            "NA",
                            "NA",
                        ],
                        delim,
                    )
                )
            # Add aggregates
            for entity_type, num in entity_type_means.items():
                score_summaries.append(
                    _join_delim(
                        [
                            "Mean",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(num),
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
            # TODO: Sort out aggregates here?
            # Index because we care about when we're at the last entry
            for idx, (filename, summary) in enumerate(zip(pred_files, score_summaries)):
                print(filename)
                print(summary)
                # Don't print an extra trailing newline
                if idx != len(pred_files) - 1:
                    print()


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
) -> Tuple[List[str], List[List[Any]]]:
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
            "ALL",
            convert_score(class_scores.precision),
            convert_score(class_scores.recall),
            convert_score(class_scores.f1),
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
                convert_score(score.precision),
                convert_score(score.recall),
                convert_score(score.f1),
                score.total_ref,
                score.total_pos,
                score.true_pos,
            ]
        )

    return header, rows


def _join_delim(items: Iterable[Any], delim: str) -> str:
    return delim.join(str(item) for item in items)
