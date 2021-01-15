import pytest

from seqscore.encoding import (
    BIO,
    EncodingError,
    LabeledSentence,
    Mention,
    SentenceMention,
    Span,
    get_encoding,
)


def test_bio_whole_sentence_mention() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a",), ("B-PER",))
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("a",), "PER"), Span(0, 1))
    ]

    sent2 = LabeledSentence(("a", "b"), ("B-PER", "I-PER"))
    assert decoder.decode_mentions(sent2) == [
        SentenceMention(Mention(("a", "b"), "PER"), Span(0, 2))
    ]


def test_bio_start_sentence_mention() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a", "b"), ("B-PER", "O"))
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("a",), "PER"), Span(0, 1))
    ]

    sent2 = LabeledSentence(("a", "b", "c"), ("B-PER", "I-PER", "O"))
    assert decoder.decode_mentions(sent2) == [
        SentenceMention(Mention(("a", "b"), "PER"), Span(0, 2))
    ]


def test_bio_end_sentence_mention() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a", "b"), ("O", "B-PER"))
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("b",), "PER"), Span(1, 2))
    ]

    sent2 = LabeledSentence(("a", "b", "c"), ("O", "B-PER", "I-PER"))
    assert decoder.decode_mentions(sent2) == [
        SentenceMention(Mention(("b", "c"), "PER"), Span(1, 3))
    ]


def test_bio_mid_sentence_mention() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a", "b", "c"), ("O", "B-PER", "O"))
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("b",), "PER"), Span(1, 2))
    ]

    sent2 = LabeledSentence(("a", "b", "c", "d"), ("O", "B-PER", "I-PER", "O"))
    assert decoder.decode_mentions(sent2) == [
        SentenceMention(Mention(("b", "c"), "PER"), Span(1, 3))
    ]


def test_bio_adjacent_mentions() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a", "b", "c", "d"), ("B-PER", "B-ORG", "I-ORG", "B-LOC"))
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("a",), "PER"), Span(0, 1)),
        SentenceMention(Mention(("b", "c"), "ORG"), Span(1, 3)),
        SentenceMention(Mention(("d",), "LOC"), Span(3, 4)),
    ]


def test_bio_non_adjacent_mentions() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(
        ("a", "b", "c", "d", "e", "f"), ("B-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC")
    )
    assert decoder.decode_mentions(sent1) == [
        SentenceMention(Mention(("a",), "PER"), Span(0, 1)),
        SentenceMention(Mention(("c", "d"), "ORG"), Span(2, 4)),
        SentenceMention(Mention(("f",), "LOC"), Span(5, 6)),
    ]


def test_bio_invalid_start() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a",), ("I-PER",))
    with pytest.raises(EncodingError):
        assert decoder.decode_mentions(sent1)


def test_bio_invalid_continue() -> None:
    decoder = BIO()

    sent1 = LabeledSentence(("a", "b"), ("B-PER", "I-LOC"))
    with pytest.raises(EncodingError):
        assert decoder.decode_mentions(sent1)


def test_get_bio() -> None:
    assert isinstance(get_encoding("BIO"), BIO)


def test_get_unknown_encoding() -> None:
    with pytest.raises(ValueError):
        get_encoding("FOO")
