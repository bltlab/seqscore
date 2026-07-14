from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import DefaultDict, Optional, Union

from seqscore.encoding import Encoding, EncodingError, get_encoding
from seqscore.model import AnnotatedSequence, Mention
from seqscore.validation import validate_labels


def _defaultdict_classification_score() -> DefaultDict[str, "ClassificationScore"]:
    return defaultdict(ClassificationScore)


@dataclass(frozen=True, slots=True)
class TokensWithType:
    tokens: tuple[str, ...]
    type: str


class TokenCountError(ValueError):
    def __init__(
        self,
        ref_token_count: int,
        pred_token_count: int,
        ref_last_token: str,
        pred_last_token: str,
        line_num: int,
        source: Optional[str],
    ):
        self.ref_token_count: int = ref_token_count
        self.pred_token_count: int = pred_token_count
        self.ref_last_token: str = ref_last_token
        self.pred_last_token: str = pred_last_token
        self.line_num: int = line_num
        self.source: Optional[str] = source

        src = f" of {source}" if source else ""
        msg = "\n".join(
            [
                f"Token count mismatch at line {line_num}{src}",
                f"Reference sequence contains {ref_token_count} tokens; "
                + f"predicted sequence contains {pred_token_count}.",
                f"Last token of reference sequence: {ref_last_token!r}",
                f"Last token of predicted sequence: {pred_last_token!r}",
                "Correct the predictions to have the same number of tokens as the reference.",
                "If the predicted sequence was truncated, pad it with the outside label (O) "
                + "to match the length of the reference sequence.",
            ]
        )
        super().__init__(msg)

    @classmethod
    def from_sequences(
        cls, ref_sequence: AnnotatedSequence, pred_sequence: AnnotatedSequence
    ) -> "TokenCountError":
        if pred_sequence.provenance is None:
            raise ValueError(
                f"Cannot create {cls.__name__} from sequence without provenance"
            )
        # AnnotatedSequence enforces non-empty tokens, so tokens[-1] is always safe.
        return cls(
            len(ref_sequence),
            len(pred_sequence),
            ref_sequence.tokens[-1],
            pred_sequence.tokens[-1],
            pred_sequence.provenance.starting_line,
            pred_sequence.provenance.source,
        )


class TokenMismatchError(ValueError):
    """Raised when tokens have different content but the same length."""

    def __init__(
        self,
        ref_token: str,
        pred_token: str,
        differing_index: int,
        line_num: int,
        source: Optional[str],
    ):
        self.ref_token: str = ref_token
        self.pred_token: str = pred_token
        self.differing_index: int = differing_index
        self.line_num: int = line_num
        self.source: Optional[str] = source

        src = f" of {source}" if source else ""
        msg = "\n".join(
            [
                f"Tokens do not match at line {line_num}{src}",
                f"First differing index: {differing_index}",
                f"Reference token at that index: {ref_token}",
                f"Prediction token at that index: {pred_token}",
                "Correct the predictions to have the same tokens as the reference.",
            ]
        )
        super().__init__(msg)

    @classmethod
    def from_sequences(
        cls, ref_sequence: AnnotatedSequence, pred_sequence: AnnotatedSequence
    ) -> "TokenMismatchError":
        if ref_sequence.provenance is None or pred_sequence.provenance is None:
            raise ValueError(
                f"Cannot create {cls.__name__} from sequence without provenance"
            )
        # Precondition: ref_sequence.tokens and pred_sequence.tokens are equal
        # length and differ at least once, so the first differing index always
        # exists. Use default=None and raise early to avoid a bare StopIteration.
        differing_index = next(
            (
                i
                for i, (ref_token, pred_token) in enumerate(
                    zip(ref_sequence.tokens, pred_sequence.tokens)
                )
                if ref_token != pred_token
            ),
            None,
        )
        if differing_index is None:  # pragma: no cover
            raise ValueError(
                "Tokens are identical — this should not happen; "
                "TokenMismatchError is only raised when tokens differ"
            )
        return cls(
            ref_sequence.tokens[differing_index],
            pred_sequence.tokens[differing_index],
            differing_index,
            pred_sequence.provenance.starting_line,
            pred_sequence.provenance.source,
        )


@dataclass
class ClassificationScore:
    true_pos: int = field(default=0, kw_only=True)
    false_pos: int = field(default=0, kw_only=True)
    false_neg: int = field(default=0, kw_only=True)
    type_scores: DefaultDict[str, "ClassificationScore"] = field(
        default_factory=_defaultdict_classification_score, kw_only=True
    )
    false_pos_examples: Counter[TokensWithType] = field(default_factory=Counter)
    false_neg_examples: Counter[TokensWithType] = field(default_factory=Counter)

    def count_false_positive(self, tokens: Iterable[str], type_: str) -> None:
        self.false_pos_examples[TokensWithType(tuple(tokens), type_)] += 1

    def count_false_negative(self, tokens: Iterable[str], type_: str) -> None:
        self.false_neg_examples[TokensWithType(tuple(tokens), type_)] += 1

    def update(self, score: "ClassificationScore") -> None:
        self.true_pos += score.true_pos
        self.false_pos += score.false_pos
        self.false_neg += score.false_neg
        for entity_type, entity_score in score.type_scores.items():
            self.type_scores[entity_type].update(entity_score)

    @property
    def total_pos(self) -> int:
        return self.true_pos + self.false_pos

    @property
    def total_ref(self) -> int:
        return self.true_pos + self.false_neg

    @property
    def precision(self) -> float:
        total = self.total_pos
        if not total:
            return 0.0
        return self.true_pos / total

    @property
    def recall(self) -> float:
        total = self.total_ref
        if not total:
            return 0.0
        return self.true_pos / total

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if not precision or not recall:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


@dataclass
class AccuracyScore:
    hits: int = field(default=0, kw_only=True)
    total: int = field(default=0, kw_only=True)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total


def compute_scores(
    pred_docs: Sequence[Sequence[AnnotatedSequence]],
    ref_docs: Sequence[Sequence[AnnotatedSequence]],
    *,
    count_fp_fn_examples: bool = False,
) -> tuple[ClassificationScore, AccuracyScore]:
    accuracy = AccuracyScore()
    classification = ClassificationScore()

    if len(pred_docs) != len(ref_docs):
        raise ValueError(
            f"Prediction has {len(pred_docs)} documents, reference has {len(ref_docs)}. "
            "Consider setting --ignore-document-boundaries/ignore_document_boundaries."
        )

    for pred_doc, ref_doc in zip(pred_docs, ref_docs):
        if len(pred_doc) != len(ref_doc):
            raise ValueError(
                f"Prediction has {len(pred_doc)} sequences, reference has {len(ref_doc)}"
            )

        for pred_sequence, ref_sequence in zip(pred_doc, ref_doc):
            if len(pred_sequence) != len(ref_sequence):
                raise TokenCountError.from_sequences(ref_sequence, pred_sequence)

            if pred_sequence.tokens != ref_sequence.tokens:
                raise TokenMismatchError.from_sequences(ref_sequence, pred_sequence)

            score_sequence_label_accuracy(
                pred_sequence.labels, ref_sequence.labels, accuracy
            )
            score_sequence_mentions(
                pred_sequence.mentions,
                ref_sequence.mentions,
                classification,
                tokens=ref_sequence.tokens,
                count_fp_fn_examples=count_fp_fn_examples,
            )

    return classification, accuracy


def score_sequence_label_accuracy(
    pred_labels: Sequence[str],
    ref_labels: Sequence[str],
    score: AccuracyScore,
) -> None:
    """Update an AccuracyScore for a single sequence's labels."""

    if len(pred_labels) != len(ref_labels):
        raise ValueError(
            f"Prediction has {len(pred_labels)} labels, reference has {len(ref_labels)}"
        )

    # Compute label accuracy
    for pred_label, ref_label in zip(pred_labels, ref_labels):
        if pred_label == ref_label:
            score.hits += 1
        score.total += 1


def score_sequence_mentions(
    pred_mentions: Sequence[Mention],
    ref_mentions: Sequence[Mention],
    score: ClassificationScore,
    *,
    tokens: Optional[Sequence[str]] = (),
    count_fp_fn_examples: bool = False,
) -> None:
    """Update a ClassificationScore for a single sequence's mentions.

    Since mentions are defined per-sequence, the behavior is not defined
    if you provide mentions corresponding to multiple sequences. Tokens
    must be provided if you want false positives and negative examples
    to be counted.
    """
    if count_fp_fn_examples and not tokens:
        raise ValueError(
            "Tokens must be provided to count false positive/negative examples"
        )

    # Compute span accuracy
    pred_mentions_set = set(pred_mentions)
    ref_mentions_set = set(ref_mentions)

    # Positives
    for pred in pred_mentions_set:
        if pred in ref_mentions_set:
            # True positive
            score.true_pos += 1
            score.type_scores[pred.type].true_pos += 1
        else:
            # False positive
            score.false_pos += 1
            score.type_scores[pred.type].false_pos += 1
            if count_fp_fn_examples:
                error_tokens = tokens[pred.span.start : pred.span.end]
                score.count_false_positive(error_tokens, pred.type)

    # Negatives
    for ref in ref_mentions_set:
        if ref not in pred_mentions_set:
            score.false_neg += 1
            score.type_scores[ref.type].false_neg += 1
            if count_fp_fn_examples:
                error_tokens = tokens[ref.span.start : ref.span.end]
                score.count_false_negative(error_tokens, ref.type)


def score_label_sequences(
    pred_label_sequences: Sequence[Sequence[str]],
    ref_label_sequences: Sequence[Sequence[str]],
    encoding_name: str,
    *,
    repair: Optional[str],
) -> tuple[ClassificationScore, AccuracyScore]:
    """Return accuracy and classification scores for predicted and reference label sequences."""
    if len(pred_label_sequences) != len(ref_label_sequences):
        raise ValueError(
            f"Different number of sequences in predicted ({len(pred_label_sequences)}) and "
            + f"reference ({len(ref_label_sequences)})"
        )

    encoder = get_encoding(encoding_name)

    classification_score = ClassificationScore()
    accuracy_score = AccuracyScore()

    for pred_labels, ref_labels in zip(pred_label_sequences, ref_label_sequences):
        # This takes care of checking that the lengths of the labels match
        score_sequence_label_accuracy(pred_labels, ref_labels, accuracy_score)
        pred_mentions = _repair_label_sequence(pred_labels, encoder, repair)
        ref_mentions = _repair_label_sequence(ref_labels, encoder, repair)
        score_sequence_mentions(pred_mentions, ref_mentions, classification_score)

    return classification_score, accuracy_score


def _repair_label_sequence(
    labels: Sequence[str], encoder: Encoding, repair: Optional[str]
) -> Sequence[Mention]:
    validation = validate_labels(labels, encoder, repair=repair)
    if not validation.is_valid():
        if repair:
            labels = validation.repaired_labels
        else:
            raise EncodingError(
                "Cannot score sequence due to validation errors.\n"
                + f"Labels:\n{labels}\n"
                + "Errors:\n"
                + "\n".join(err.msg for err in validation.errors)
            )
    return encoder.decode_labels(labels)


def convert_score(num: float, full_precision: bool) -> Union[Decimal, float]:
    if full_precision:
        # Leave it unchanged
        return num
    else:
        # Convert a 0-1 score to the 0-100 range with two decimal places
        dec = Decimal(num) * 100
        return dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
