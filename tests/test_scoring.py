import pytest

from seqscore.encoding import Mention, Span
from seqscore.scoring import (
    AccuracyScore,
    ClassificationScore,
    score_sentence_labels,
    score_sentence_mentions,
)


def test_score_sentence_labels_correct() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    pred_labels = ref_labels[:]
    score = AccuracyScore()
    score_sentence_labels(pred_labels, ref_labels, score)
    assert score.total == 4
    assert score.hits == 4
    assert score.accuracy == 1.0


def test_score_sentence_labels_incorrect() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    pred_labels = ref_labels[:]
    pred_labels[2] = "B-LOC"
    score = AccuracyScore()
    score_sentence_labels(pred_labels, ref_labels, score)
    assert score.total == 4
    assert score.hits == 3
    assert score.accuracy == pytest.approx(3 / 4)


def test_score_sentence_labels_invalid() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    # Shorter predictions than reference
    pred_labels = ref_labels[:-1]
    with pytest.raises(ValueError):
        score_sentence_labels(pred_labels, ref_labels, AccuracyScore())


def test_score_sentence_mentions_correct() -> None:
    ref_mentions = [Mention(Span(0, 2), "PER"), Mention(Span(4, 5), "ORG")]
    pred_mentions = [Mention(Span(0, 2), "PER"), Mention(Span(4, 5), "ORG")]
    score = ClassificationScore()
    score_sentence_mentions(pred_mentions, ref_mentions, score)
    assert score.true_pos == 2
    assert score.false_pos == 0
    assert score.false_neg == 0
    assert score.type_scores == {
        "PER": ClassificationScore(true_pos=1),
        "ORG": ClassificationScore(true_pos=1),
    }
    assert score.total_ref == 2
    assert score.total_pos == 2
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_score_sentence_mentions_incorrect1() -> None:
    ref_mentions = [
        Mention(Span(0, 2), "LOC"),
        Mention(Span(4, 5), "PER"),
        Mention(Span(7, 8), "MISC"),
        Mention(Span(9, 11), "MISC"),
    ]
    pred_mentions = [
        Mention(Span(0, 2), "ORG"),
        Mention(Span(4, 5), "PER"),
        Mention(Span(9, 11), "MISC"),
    ]
    score = ClassificationScore()
    score_sentence_mentions(pred_mentions, ref_mentions, score)
    assert score.true_pos == 2
    assert score.false_pos == 1
    assert score.false_neg == 2
    assert score.type_scores == {
        "PER": ClassificationScore(true_pos=1),
        "LOC": ClassificationScore(false_neg=1),
        "MISC": ClassificationScore(false_neg=1, true_pos=1),
        "ORG": ClassificationScore(false_pos=1),
    }
    assert score.total_ref == 4
    assert score.total_pos == 3
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 4)
    # Note that we have already checked the precision and recall values
    assert score.f1 == pytest.approx(
        2 * (score.precision * score.recall) / (score.precision + score.recall)
    )


def test_classification_score_empty() -> None:
    score = ClassificationScore()
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_classification_score_update() -> None:
    score1 = ClassificationScore()
    score1.true_pos += 1
    score1.type_scores["PER"].true_pos += 1
    score1.false_pos += 1
    score1.type_scores["ORG"].false_pos += 1

    score2 = ClassificationScore()
    score2.false_pos += 1
    score2.type_scores["ORG"].false_pos += 1
    score2.false_neg += 1
    score2.type_scores["MISC"].false_neg += 1
    score2.true_pos += 4
    score2.type_scores["ORG"].true_pos += 4

    score1.update(score2)

    assert score1.true_pos == 5
    assert score1.false_pos == 2
    assert score1.false_neg == 1
    assert score1.type_scores == {
        "PER": ClassificationScore(true_pos=1),
        "ORG": ClassificationScore(true_pos=4, false_pos=2),
        "MISC": ClassificationScore(false_neg=1),
    }


def test_accuracy_score_empty() -> None:
    score = AccuracyScore()
    assert score.accuracy == 0.0
