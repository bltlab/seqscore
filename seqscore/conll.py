import decimal
import sys
from itertools import chain
from typing import Any, Iterable, List, Optional, Sequence, TextIO, Tuple

from attr import attrib, attrs
from tabulate import tabulate

from seqscore.encoding import (
    REPAIR_NONE,
    Encoding,
    EncodingError,
    LabeledSentence,
    ValidationResult,
    get_encoding,
    validate_sentence,
)
from seqscore.scoring import AccuracyScore, ClassificationScore, compute_scores
from seqscore.util import PathType

DOCSTART = "-DOCSTART-"


DISPLAY_PRETTY = "pretty"
DISPLAY_CONLL = "conlleval"
DISPLAY_DELIM = "delim"
SUPPORTED_DISPLAY_FORMATS = (DISPLAY_PRETTY, DISPLAY_CONLL, DISPLAY_DELIM)


@attrs(frozen=True)
class _CoNLLToken:
    text: str = attrib()
    label: str = attrib()
    is_docstart: bool = attrib()
    line_num: int = attrib()

    # TODO: Make delimiter configurable. Currently splits on any whitespace.
    @classmethod
    def from_line(cls, line: str, line_num: int) -> "_CoNLLToken":
        splits = line.split()
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
        self, source: TextIO, source_name: str, repair: Optional[str]
    ) -> Iterable[List[LabeledSentence]]:
        document_counter = 0
        document: List[LabeledSentence] = []

        for source_sentence in self._parse_file(
            source, ignore_comments=self.ignore_comment_lines
        ):
            if source_sentence[0].is_docstart:
                # We can ony receive DOCSTART in a sentence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sentence.
                assert len(source_sentence) == 1
                # End current document and start a new one if we're attending to boundaries.
                # We skip this if the builder is empty, which will happen for the very
                # first document in the corpus (as there is no previous document to end).
                if not self.ignore_document_boundaries and document:
                    document_counter += 1
                    yield document
                    document = []
                continue

            # Create mentions from tokens in sentence
            tokens, labels, line_nums = self._decompose_sentence(source_sentence)

            # Validate before decoding
            validation = validate_sentence(
                tokens, labels, line_nums, self.encoding, repair=repair
            )
            if not validation.is_valid():
                if repair:
                    msg = (
                        [
                            f"Validation errors in sentence at line {line_nums[0]} of {source_name}:"
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

            orig_sentence = LabeledSentence(tokens, labels)
            try:
                mentions = self.encoding.decode_mentions(orig_sentence)
            except EncodingError as e:
                raise ValueError(
                    "Encountered an error decoding this sequence despite passing validation: "
                    + " ".join(labels),
                ) from e

            final_sentence = LabeledSentence(
                orig_sentence.tokens, orig_sentence.labels, mentions
            )
            document.append(final_sentence)

        # Yield final document if non-empty
        if document:
            document_counter += 1
            yield document

    def validate(self, source: TextIO) -> List[List[ValidationResult]]:
        all_results: List[List[ValidationResult]] = []
        document_results: List[ValidationResult] = []

        for source_sentence in self._parse_file(
            source, ignore_comments=self.ignore_comment_lines
        ):
            if source_sentence[0].is_docstart:
                # We can ony receive DOCSTART in a sentence by itself, see _parse_file.
                # But we check anyway to be absolutely sure we aren't throwing away a sentence.
                assert len(source_sentence) == 1

                # If we care about document boundaries and we have results for this documents,
                # add it and move on.
                if not self.ignore_document_boundaries and document_results:
                    all_results.append(document_results)
                    document_results = []

                # Go to the next sentence
                continue

            # Create mentions from tokens in sentence
            tokens, labels, line_nums = self._decompose_sentence(source_sentence)

            # Validate
            document_results.append(
                validate_sentence(tokens, labels, line_nums, self.encoding)
            )

        if document_results:
            all_results.append(document_results)

        return all_results

    @staticmethod
    def _decompose_sentence(
        source_sentence: Sequence[_CoNLLToken],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[int, ...]]:
        tokens = tuple(tok.text for tok in source_sentence)
        labels = tuple(tok.label for tok in source_sentence)
        line_nums = tuple(tok.line_num for tok in source_sentence)
        return tokens, labels, line_nums

    @classmethod
    def _parse_file(
        cls, input_file: TextIO, *, ignore_comments: bool = False
    ) -> Iterable[Tuple[_CoNLLToken, ...]]:
        sentence: list = []
        line_num = 0
        for line in input_file:
            line_num += 1
            line = line.strip()

            if ignore_comments and line.startswith("#"):
                continue

            if not line:
                # Clear out sentence if there's anything in it
                if sentence:
                    cls._check_sentence(sentence)
                    yield tuple(sentence)
                    sentence = []
                # Always skip empty lines
                continue

            token = _CoNLLToken.from_line(line, line_num)
            # Skip document starts, but ensure sentence is empty when we reach them
            if token.is_docstart:
                if sentence:
                    raise ValueError(
                        f"Encountered DOCSTART at line {line_num} while still in sentence"
                    )
                else:
                    # Yield it by itself. Since the sentence variable is empty, leave it unchanged.
                    tmp_sent = (token,)
                    cls._check_sentence(tmp_sent)
                    yield tmp_sent
            else:
                sentence.append(token)

        # Finish the last sentence if needed
        if sentence:
            cls._check_sentence(sentence)
            yield tuple(sentence)

    @staticmethod
    def _check_sentence(sentence: Sequence[_CoNLLToken]):
        # We should only return DOCSTART in a sentence by itself. This isn't a constraint
        # on the layout of the input document, but rather one we are enforcing so that consumers
        # get document boundaries as their own sentences.
        if sentence[0].is_docstart and len(sentence) > 1:
            raise ValueError(
                f"Returned -DOCSTART- as part of a sentence at line {sentence[0].line_num}"
            )


def ingest_conll_file(
    input_path: PathType,
    encoding_name: str,
    *,
    repair: Optional[str] = None,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
) -> List[List[LabeledSentence]]:
    if repair == REPAIR_NONE:
        repair = None
    encoding = get_encoding(encoding_name)
    ingester = CoNLLIngester(
        encoding,
        ignore_comment_lines=ignore_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding="utf8") as input_file:
        docs = list(ingester.ingest(input_file, str(input_path), repair))
    return docs


def validate_conll_file(
    input_path: str,
    encoding_name: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
) -> None:
    encoding = get_encoding(encoding_name)
    ingester = CoNLLIngester(
        encoding,
        ignore_comment_lines=ignore_comment_lines,
        ignore_document_boundaries=ignore_document_boundaries,
    )
    with open(input_path, encoding="utf8") as input_file:
        results = ingester.validate(input_file)
        n_docs = len(results)
        n_sentences = sum(len(doc_results) for doc_results in results)
        n_tokens = sum(len(sent) for doc_results in results for sent in doc_results)

        errors = list(
            chain.from_iterable(
                result.errors for doc_results in results for result in doc_results
            )
        )
        if errors:
            print(
                f"Encountered {len(errors)} errors in {n_tokens} tokens, {n_sentences} sentences, "
                + f"and {n_docs} documents in {input_path}"
            )
            print("\n".join(err.msg for err in errors))
            sys.exit(1)
        else:
            print(
                f"No errors found in {n_tokens} tokens, {n_sentences} sentences, "
                + f"and {n_docs} documents in {input_path}"
            )


def score_conll_files(
    pred_file: PathType,
    reference_file: PathType,
    repair: Optional[str],
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    output_format: str,
    delim: str,
) -> None:
    if repair == REPAIR_NONE:
        repair = None

    # We only support scoring BIO
    mention_encoding_name = "BIO"
    pred_docs = ingest_conll_file(
        pred_file,
        mention_encoding_name,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )

    ref_docs = ingest_conll_file(
        reference_file,
        mention_encoding_name,
        repair=repair,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )

    class_scores, acc_scores = compute_scores(pred_docs, ref_docs)

    if output_format == DISPLAY_CONLL:
        summary = format_output_conlleval(class_scores, acc_scores)
    elif output_format in (DISPLAY_PRETTY, DISPLAY_DELIM):
        header, rows = format_output_table(class_scores)
        if output_format == DISPLAY_PRETTY:
            summary = tabulate(rows, header, tablefmt="github", floatfmt="6.2f")
        else:
            # Since rows are a mix of types, we need to convert to string
            summary = "\n".join(
                [delim.join(header)]
                + [delim.join(str(item) for item in row) for row in rows]
            )
    else:
        raise ValueError(f"Unrecognized output format: {output_format}")

    print(summary)


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
            _pretty_format_num(class_scores.precision),
            _pretty_format_num(class_scores.recall),
            _pretty_format_num(class_scores.f1),
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
                _pretty_format_num(score.precision),
                _pretty_format_num(score.recall),
                _pretty_format_num(score.f1),
                score.total_ref,
                score.total_pos,
                score.true_pos,
            ]
        )

    return header, rows


def _pretty_format_num(num: float) -> decimal.Decimal:
    with decimal.localcontext() as ctx:
        ctx.rounding = decimal.ROUND_HALF_UP
        ctx.prec = 4
        dec = decimal.Decimal(num) * decimal.Decimal(100)

    return dec
