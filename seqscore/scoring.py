from collections import defaultdict
from typing import DefaultDict, Sequence, Tuple

from attr import Factory, attrib, attrs

from seqscore.encoding import LabeledSentence


def _defaultdict_classification_score() -> DefaultDict[str, "ClassificationScore"]:
    return defaultdict(ClassificationScore)


@attrs
class ClassificationScore:
    true_pos: int = attrib(default=0, kw_only=True)
    false_pos: int = attrib(default=0, kw_only=True)
    false_neg: int = attrib(default=0, kw_only=True)
    type_scores: DefaultDict[str, "ClassificationScore"] = attrib(
        default=Factory(_defaultdict_classification_score), kw_only=True
    )

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
                raise ValueError(
                    f"Prediction has {len(pred_sentence)} tokens, "
                    f"reference has {len(ref_sentence)}"
                )

            # Fail if tokens have been changed. Note that if this check is removed, the span
            # scoring logic needs to be changed to ignore the tokens, which are part of a mention.
            if pred_sentence.tokens != ref_sentence.tokens:
                raise ValueError(
                    "Tokens do not match between predictions and reference. "
                    f"Prediction: {pred_sentence.tokens}\n"
                    f"Reference: {ref_sentence.tokens}"
                )

            # Compute label accuracy
            for pred_label, ref_label in zip(pred_sentence.labels, ref_sentence.labels):
                if pred_label == ref_label:
                    accuracy.hits += 1
                accuracy.total += 1

            # Compute span accuracy
            pred_mentions = set(pred_sentence.mentions)
            ref_mentions = set(ref_sentence.mentions)

            # Positives
            for pred in pred_mentions:
                if pred in ref_mentions:
                    # True positive
                    classification.true_pos += 1
                    classification.type_scores[pred.mention.type].true_pos += 1
                else:
                    # False positive
                    classification.false_pos += 1
                    classification.type_scores[pred.mention.type].false_pos += 1

            # Negatives
            for pred in ref_mentions:
                if pred not in pred_mentions:
                    classification.false_neg += 1
                    classification.type_scores[pred.mention.type].false_neg += 1

    return classification, accuracy
