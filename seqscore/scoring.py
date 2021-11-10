from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import DefaultDict, Optional, Sequence, Tuple

from attr import Factory, attrib, attrs

from seqscore.encoding import Encoding, EncodingError, get_encoding
from seqscore.model import LabeledSequence, Mention
from seqscore.validation import validate_labels


def _defaultdict_classification_score() -> DefaultDict[str, "ClassificationScore"]:
    return defaultdict(ClassificationScore)


class TokenCountError(ValueError):
    def __init__(
        self,
        reference_token_count: int,
        pred_token_count: int,
        line_num: int,
        source: Optional[str],
    ):
        self.reference_token_count: int = reference_token_count
        self.other_token_count: int = pred_token_count
        self.line_num: int = line_num
        self.source: Optional[str] = source

        # Insertable string if source is specified
        src = f" of {source}" if source else ""
        msg = "\n".join(
            [
                f"Token count mismatch at line {line_num}{src}",
                f"Reference sequence contains {reference_token_count} tokens; "
                + f"predicted sequence contains {pred_token_count}.",
                "Correct the predictions to have the same number of tokens as the reference.",
            ]
        )
        super().__init__(msg)

    @classmethod
    def from_predicted_sequence(
        cls, reference_token_count: int, pred_sequence: LabeledSequence
    ):
        if pred_sequence.provenance is None:
            raise ValueError(
                f"Cannot create {cls.__name__} from sequence without provenance"
            )
        return cls(
            reference_token_count,
            len(pred_sequence),
            pred_sequence.provenance.starting_line,
            pred_sequence.provenance.source,
        )


@attrs
class ClassificationScore:
    true_pos: int = attrib(default=0, kw_only=True)
    false_pos: int = attrib(default=0, kw_only=True)
    false_neg: int = attrib(default=0, kw_only=True)
    type_scores: DefaultDict[str, "ClassificationScore"] = attrib(
        default=Factory(_defaultdict_classification_score), kw_only=True
    )

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


@attrs
class AccuracyScore:
    hits: int = attrib(default=0, kw_only=True)
    total: int = attrib(default=0, kw_only=True)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total


def compute_scores(
    pred_docs: Sequence[Sequence[LabeledSequence]],
    ref_docs: Sequence[Sequence[LabeledSequence]],
) -> Tuple[ClassificationScore, AccuracyScore]:
    accuracy = AccuracyScore()
    classification = ClassificationScore()

    # TODO: Recommend use of ignore_document_boundaries if this error is encountered
    if len(pred_docs) != len(ref_docs):
        raise ValueError(
            f"Prediction has {len(pred_docs)} documents, "
            f"reference has {len(ref_docs)}"
        )

    for pred_doc, ref_doc in zip(pred_docs, ref_docs):
        if len(pred_doc) != len(ref_doc):
            raise ValueError(
                f"Prediction has {len(pred_doc)} sequences, "
                f"reference has {len(ref_doc)}"
            )

        for pred_sequence, ref_sequence in zip(pred_doc, ref_doc):
            if len(pred_sequence) != len(ref_sequence):
                raise TokenCountError.from_predicted_sequence(
                    len(ref_sequence), pred_sequence
                )

            # Fail if tokens have been changed
            # TODO: Consider removing this check or providing a flag to disable it
            # TODO: Change to a more verbose error that uses the provenance
            if pred_sequence.tokens != ref_sequence.tokens:
                raise ValueError(
                    "Tokens do not match between predictions and reference.\n"
                    f"Prediction: {pred_sequence.tokens}\n"
                    f"Reference: {ref_sequence.tokens}"
                )

            score_sequence_label_accuracy(
                pred_sequence.labels, ref_sequence.labels, accuracy
            )
            score_sequence_mentions(
                pred_sequence.mentions, ref_sequence.mentions, classification
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
            f"Prediction has {len(pred_labels)} labels, "
            f"reference has {len(ref_labels)}"
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
) -> None:
    """Update a ClassificationScore for a single sequence's mentions.

    Since mentions are defined per-sequence, the behavior is not defined
    if you provide mentions corresponding to multiple sequences.
    """
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

    # Negatives
    for pred in ref_mentions_set:
        if pred not in pred_mentions_set:
            score.false_neg += 1
            score.type_scores[pred.type].false_neg += 1


# TODO: Consider taking an iterable and checking sequence lengths
def score_label_sequences(
    pred_label_sequences: Sequence[Sequence[str]],
    ref_label_sequences: Sequence[Sequence[str]],
    encoding_name: str,
    *,
    repair: Optional[str],
) -> Tuple[ClassificationScore, AccuracyScore]:
    encoder = get_encoding(encoding_name)

    classifcation_score = ClassificationScore()
    accuracy_score = AccuracyScore()

    for pred_labels, ref_labels in zip(pred_label_sequences, ref_label_sequences):
        # This takes care of checking that the lengths match
        score_sequence_label_accuracy(pred_labels, ref_labels, accuracy_score)
        pred_mentions = _repair_label_sequence(pred_labels, encoder, repair)
        ref_mentions = _repair_label_sequence(ref_labels, encoder, repair)
        score_sequence_mentions(pred_mentions, ref_mentions, classifcation_score)

    return classifcation_score, accuracy_score


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


def convert_score(num: float) -> Decimal:
    """Convert a 0-1 score to the 0-100 range with two decimal places."""
    dec = Decimal(num) * 100
    return dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
