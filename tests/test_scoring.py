from collections import Counter
from decimal import Decimal

import pytest

from seqscore.encoding import EncodingError
from seqscore.model import AnnotatedSequence, Mention, SequenceProvenance, Span
from seqscore.scoring import (
    AccuracyScore,
    ClassificationScore,
    TokenCountError,
    TokenMismatchError,
    TokensWithType,
    compute_scores,
    convert_score,
    score_label_sequences,
    score_sequence_label_accuracy,
    score_sequence_mentions,
)


def test_score_sentence_labels_correct() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    pred_labels = ref_labels[:]
    score = AccuracyScore()
    score_sequence_label_accuracy(pred_labels, ref_labels, score)
    assert score.total == 4
    assert score.hits == 4
    assert score.accuracy == 1.0


def test_score_sentence_labels_incorrect() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    pred_labels = ref_labels[:]
    pred_labels[2] = "B-LOC"
    score = AccuracyScore()
    score_sequence_label_accuracy(pred_labels, ref_labels, score)
    assert score.total == 4
    assert score.hits == 3
    assert score.accuracy == pytest.approx(3 / 4)


def test_score_sentence_labels_invalid() -> None:
    ref_labels = ["O", "B-ORG", "I-ORG", "O"]
    # Shorter predictions than reference
    pred_labels = ref_labels[:-1]
    with pytest.raises(ValueError):
        score_sequence_label_accuracy(pred_labels, ref_labels, AccuracyScore())


def test_score_sequence_mentions_correct() -> None:
    ref_mentions = [Mention(Span(0, 2), "PER"), Mention(Span(4, 5), "ORG")]
    pred_mentions = [Mention(Span(0, 2), "PER"), Mention(Span(4, 5), "ORG")]
    score = ClassificationScore()
    score_sequence_mentions(pred_mentions, ref_mentions, score)
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

    # Test that tokens are required for counting FP/FN
    with pytest.raises(ValueError):
        score_sequence_mentions(
            pred_mentions, ref_mentions, score, count_fp_fn_examples=True
        )


def test_score_sequence_mentions_incorrect1() -> None:
    ref_mentions = [
        Mention(Span(0, 2), "LOC"),
        Mention(Span(4, 5), "PER"),
        Mention(Span(7, 8), "MISC"),
        Mention(Span(9, 11), "MISC"),
    ]
    pred_mentions = [
        Mention(Span(0, 2), "ORG"),
        Mention(Span(4, 5), "PER"),
        Mention(
            Span(6, 7), "SPURIOUS"
        ),  # Note that this type isn't even in the reference
        Mention(Span(9, 11), "MISC"),
    ]
    score = ClassificationScore()
    score_sequence_mentions(pred_mentions, ref_mentions, score)
    assert score.true_pos == 2
    assert score.false_pos == 2
    assert score.false_neg == 2
    assert score.type_scores == {
        "PER": ClassificationScore(true_pos=1),
        "LOC": ClassificationScore(false_neg=1),
        "MISC": ClassificationScore(false_neg=1, true_pos=1),
        "ORG": ClassificationScore(false_pos=1),
        "SPURIOUS": ClassificationScore(false_pos=1),
    }
    assert score.total_ref == 4
    assert score.total_pos == 4
    assert score.precision == pytest.approx(2 / 4)
    assert score.recall == pytest.approx(2 / 4)
    # Note that we have already checked the precision and recall values
    assert score.f1 == pytest.approx(
        2 * (score.precision * score.recall) / (score.precision + score.recall)
    )

    # Run again and check counted fp/fn examples. We do this in a second pass so
    # we can cover both True/False cases for count_fp_fn_examples.
    score2 = ClassificationScore()
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    score_sequence_mentions(
        pred_mentions, ref_mentions, score2, count_fp_fn_examples=True, tokens=tokens
    )
    expected_false_pos = Counter(
        [
            TokensWithType(("a", "b"), "ORG"),
            TokensWithType(("g",), "SPURIOUS"),
        ]
    )
    expected_false_neg = Counter(
        [
            TokensWithType(("a", "b"), "LOC"),
            TokensWithType(("h",), "MISC"),
        ]
    )
    assert score2.false_pos_examples == expected_false_pos
    assert score2.false_neg_examples == expected_false_neg


def test_score_label_sequences_correct() -> None:
    ref_labels = [["O", "B-ORG", "I-ORG", "O"], ["B-PER", "I-PER"]]
    pred_labels = ref_labels[:]
    classification, accuracy = score_label_sequences(
        pred_labels, ref_labels, "BIO", repair=None
    )

    assert accuracy.total == 6
    assert accuracy.hits == 6
    assert accuracy.accuracy == 1.0

    assert classification.true_pos == 2
    assert classification.false_pos == 0
    assert classification.false_neg == 0
    assert classification.type_scores["ORG"] == ClassificationScore(true_pos=1)
    assert classification.type_scores["PER"] == ClassificationScore(true_pos=1)


def test_score_label_sequences_invalid_norepair() -> None:
    ref_labels = [["O", "B-ORG", "I-ORG", "O"], ["B-PER", "I-PER"]]
    pred_labels = [["O", "B-ORG", "I-ORG", "O"], ["I-PER", "I-PER"]]
    with pytest.raises(EncodingError):
        score_label_sequences(pred_labels, ref_labels, "BIO", repair=None)


def test_score_label_sequences_invalid_repair() -> None:
    ref_labels = [["O", "B-ORG", "I-ORG", "O"], ["B-PER", "I-PER"]]
    pred_labels = [["O", "I-ORG", "I-ORG", "O"], ["O", "I-PER"]]
    classification, accuracy = score_label_sequences(
        pred_labels, ref_labels, "BIO", repair="conlleval"
    )

    assert accuracy.total == 6
    assert accuracy.hits == 4
    assert accuracy.accuracy == 4 / 6

    assert classification.true_pos == 1
    assert classification.false_pos == 1
    assert classification.false_neg == 1
    assert classification.type_scores["ORG"] == ClassificationScore(true_pos=1)
    assert classification.type_scores["PER"] == ClassificationScore(
        false_pos=1, false_neg=1
    )


def test_score_label_sequences_different_lengths() -> None:
    ref_labels = [["O", "B-ORG", "I-ORG", "O"], ["B-PER", "I-PER"]]
    pred_labels = [["O", "B-ORG", "I-ORG", "O"]]
    with pytest.raises(ValueError):
        score_label_sequences(pred_labels, ref_labels, "BIO", repair=None)


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


def test_compute_scores() -> None:
    ref_labels = ("O", "B-ORG", "I-ORG", "O", "B-LOC")
    ref_mentions = (
        Mention(Span(1, 3), "ORG"),
        Mention(Span(4, 5), "LOC"),
    )
    pred_labels = ("O", "B-ORG", "I-ORG", "O", "B-ORG")
    pred_mentions = (
        Mention(Span(1, 3), "ORG"),
        Mention(Span(4, 5), "ORG"),
    )
    tokens = ("a", "b", "c", "d", "e")
    ref_sequence = AnnotatedSequence(
        tokens=tokens, labels=ref_labels, mentions=ref_mentions
    )
    pred_sequence = AnnotatedSequence(
        tokens=tokens, labels=pred_labels, mentions=pred_mentions
    )
    class_score, acc_score = compute_scores([[pred_sequence]], [[ref_sequence]])
    assert acc_score.accuracy == 4 / 5
    print(class_score)
    assert class_score.true_pos == 1
    assert class_score.false_pos == 1
    assert class_score.false_neg == 1


def test_token_count_error() -> None:
    ref_labels = ("O", "B-ORG", "I-ORG", "O")
    pred_labels = ("O", "B-ORG", "I-ORG", "O", "O")
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b", "c", "d"),
        labels=ref_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "b", "c", "d", "e"),
        labels=pred_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    with pytest.raises(TokenCountError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.ref_token_count == 4
    assert exc_info.value.pred_token_count == 5
    assert exc_info.value.ref_last_token == "d"
    assert exc_info.value.pred_last_token == "e"
    assert "was truncated" in str(exc_info.value)
    assert "outside label (O)" in str(exc_info.value)


def test_token_count_error_provenance_none_uses_fallback() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=None,
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "b", "c"),
        labels=("O", "B-ORG", "O"),
        mentions=(),
        provenance=None,
    )
    error = TokenCountError.from_sequences(ref_sequence, pred_sequence)
    assert error.line_num is None
    assert error.source is None
    assert "Token count mismatch" in str(error)
    assert " at line " not in str(error)


def test_differing_num_docs() -> None:
    ref_labels = ("O", "B-ORG")
    pred_labels = ("O", "B-LOC")
    tokens = ("a", "b")
    ref_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=ref_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=pred_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    with pytest.raises(ValueError):
        compute_scores([[pred_sequence]], [[ref_sequence], [ref_sequence]])


def test_differing_doc_length() -> None:
    ref_labels = ("O", "B-ORG")
    pred_labels = ("O", "B-LOC")
    tokens = ("a", "b")
    ref_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=ref_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=pred_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    with pytest.raises(ValueError):
        compute_scores([[pred_sequence]], [[ref_sequence, ref_sequence]])


def test_differing_pred_and_ref_tokens() -> None:
    ref_labels = ("O", "B-ORG")
    pred_labels = ("O", "B-LOC")
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=ref_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "c"),
        labels=pred_labels,
        mentions=(),
        provenance=SequenceProvenance(0, "test"),
    )
    with pytest.raises(TokenMismatchError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.ref_token == "b"
    assert exc_info.value.pred_token == "c"
    assert exc_info.value.differing_index == 1
    assert "First differing index: 1" in str(exc_info.value)


def test_token_count_error_provenance_with_source() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b", "c"),
        labels=("O", "B-ORG", "O"),
        mentions=(),
        provenance=SequenceProvenance(5, "document.txt"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=SequenceProvenance(5, "document.txt"),
    )
    with pytest.raises(TokenCountError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert "of document.txt" in str(exc_info.value)
    assert "line 5" in str(exc_info.value)


def test_token_count_error_last_token_ref_len1() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a",),
        labels=("O",),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    with pytest.raises(TokenCountError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.ref_last_token == "a"
    assert exc_info.value.pred_last_token == "b"
    assert "Last token of reference sequence: 'a'" in str(exc_info.value)


def test_token_count_error_last_token_pred_len1() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a",),
        labels=("O",),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    with pytest.raises(TokenCountError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.pred_last_token == "a"


def test_token_mismatch_error_first_differing_index() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b", "c", "d"),
        labels=("O", "B-ORG", "I-ORG", "O"),
        mentions=(),
        provenance=SequenceProvenance(3, "data.txt"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "x", "c", "d"),
        labels=("O", "B-LOC", "I-LOC", "O"),
        mentions=(),
        provenance=SequenceProvenance(3, "data.txt"),
    )
    with pytest.raises(TokenMismatchError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.differing_index == 1
    assert exc_info.value.ref_token == "b"
    assert exc_info.value.pred_token == "x"
    assert "First differing index: 1" in str(exc_info.value)
    assert "of data.txt" in str(exc_info.value)
    assert "line 3" in str(exc_info.value)


def test_token_mismatch_error_at_index_0() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    pred_sequence = AnnotatedSequence(
        tokens=("x", "b"),
        labels=("O", "B-LOC"),
        mentions=(),
        provenance=SequenceProvenance(1, "test"),
    )
    with pytest.raises(TokenMismatchError) as exc_info:
        compute_scores([[pred_sequence]], [[ref_sequence]])
    assert exc_info.value.differing_index == 0
    assert exc_info.value.ref_token == "a"
    assert exc_info.value.pred_token == "x"


def test_token_mismatch_error_provenance_none_uses_fallback() -> None:
    ref_sequence = AnnotatedSequence(
        tokens=("a", "b"),
        labels=("O", "B-ORG"),
        mentions=(),
        provenance=None,
    )
    pred_sequence = AnnotatedSequence(
        tokens=("a", "x"),
        labels=("O", "B-LOC"),
        mentions=(),
        provenance=None,
    )
    error = TokenMismatchError.from_sequences(ref_sequence, pred_sequence)
    assert error.line_num is None
    assert error.source is None
    assert "Tokens do not match" in str(error)
    assert " at line " not in str(error)


def test_compute_scores_matching_tokens_passes() -> None:
    tokens = ("a", "b", "c", "d")
    ref_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=("O", "B-ORG", "I-ORG", "O"),
        mentions=(Mention(Span(1, 3), "ORG"),),
    )
    pred_sequence = AnnotatedSequence(
        tokens=tokens,
        labels=("O", "B-ORG", "I-ORG", "O"),
        mentions=(Mention(Span(1, 3), "ORG"),),
    )
    classification, accuracy = compute_scores([[pred_sequence]], [[ref_sequence]])
    assert accuracy.accuracy == 1.0


def test_convert_score() -> None:
    # Check basic rounding up/down
    assert convert_score(0.92156, False) == Decimal("92.16")
    assert convert_score(0.92154, False) == Decimal("92.15")

    # Check half rounding
    # Note: due to inexact float representation, changing the test values
    # can lead to unexpected failures. If the final 5 is actually represented
    # as 49999 instead, it will cause rounding down.
    # See: https://docs.python.org/3/library/functions.html#round
    assert convert_score(0.03205, False) == Decimal("3.21")
    assert convert_score(0.03225, False) == Decimal("3.23")
    assert convert_score(0.02205, False) == Decimal("2.21")
    assert convert_score(0.02245, False) == Decimal("2.25")

    # Check that the number of decimal places is constant
    assert convert_score(1.0, False) == Decimal("100.00")
    assert convert_score(0.5, False) == Decimal("50.00")
    assert convert_score(0.0, False) == Decimal("0.00")

    # Check full precision
    assert convert_score(1 / 3, True) == 1 / 3
    assert convert_score(1 / 7, True) == 1 / 7
    assert convert_score(1 / 9, True) == 1 / 9
