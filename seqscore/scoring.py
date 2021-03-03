from collections import defaultdict
from typing import DefaultDict, Optional, Sequence, Tuple

from attr import Factory, attrib, attrs

from seqscore.encoding import LabeledSentence, Mention


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
                f"Reference sentence contains {reference_token_count} tokens; "
                + f"predicted sentence contains {pred_token_count}.",
                "Correct the predictions to have the same number of tokens as the reference.",
            ]
        )
        super().__init__(msg)

    @classmethod
    def from_predicted_sentence(
        cls, reference_token_count: int, pred_sentence: LabeledSentence
    ):
        if pred_sentence.provenance is None:
            raise ValueError(
                f"Cannot create {cls.__name__} from sentence without provenance"
            )
        return cls(
            reference_token_count,
            len(pred_sentence),
            pred_sentence.provenance.starting_line,
            pred_sentence.provenance.source,
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
    pred_docs: Sequence[Sequence[LabeledSentence]],
    ref_docs: Sequence[Sequence[LabeledSentence]],
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
                f"Prediction has {len(pred_doc)} sentences, "
                f"reference has {len(ref_doc)}"
            )

        for pred_sentence, ref_sentence in zip(pred_doc, ref_doc):
            if len(pred_sentence) != len(ref_sentence):
                raise TokenCountError.from_predicted_sentence(
                    len(ref_sentence), pred_sentence
                )

            # Fail if tokens have been changed
            # TODO: Consider removing this check or providing a flag to disable it
            # TODO: Change to a more verbose error that uses the provenance
            if pred_sentence.tokens != ref_sentence.tokens:
                raise ValueError(
                    "Tokens do not match between predictions and reference. "
                    f"Prediction: {pred_sentence.tokens}\n"
                    f"Reference: {ref_sentence.tokens}"
                )

            score_sentence_labels(pred_sentence.labels, ref_sentence.labels, accuracy)
            score_sentence_mentions(
                pred_sentence.mentions, ref_sentence.mentions, classification
            )

    return classification, accuracy


def score_sentence_labels(
    pred_labels: Sequence[str],
    ref_labels: Sequence[str],
    score: AccuracyScore,
) -> None:
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


def score_sentence_mentions(
    pred_mentions: Sequence[Mention],
    ref_mentions: Sequence[Mention],
    score: ClassificationScore,
) -> None:
    """Update a ClassificationScore for a sentence's mentions."""
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
