import pytest

from seqscore.model import LabeledSequence, Mention, SequenceProvenance, Span


def test_span() -> None:
    assert len(Span(0, 1)) == 1
    assert len(Span(1, 2)) == 1
    assert len(Span(0, 2)) == 2

    with pytest.raises(ValueError):
        Span(-1, 0)

    with pytest.raises(ValueError):
        Span(0, 0)


def test_mention() -> None:
    m1 = Mention(Span(0, 1), "PER")
    assert m1.type == "PER"
    assert m1.span == Span(0, 1)
    assert len(m1) == 1

    with pytest.raises(ValueError):
        Mention(Span(0, 1), "")

    with pytest.raises(TypeError):
        # Intentionally incorrect type
        Mention(Span(0, 1), None)  # type: ignore


def test_labeled_sentence() -> None:
    s1 = LabeledSequence(
        ["a", "b"],
        ["B-PER", "I-PER"],
        provenance=SequenceProvenance(7, "test"),
    )
    assert s1.tokens == ("a", "b")
    assert s1[0] == "a"
    assert s1[0:2] == ("a", "b")
    assert list(s1) == ["a", "b"]
    assert s1.labels == ("B-PER", "I-PER")
    assert s1.provenance == SequenceProvenance(7, "test")
    assert str(s1) == "a/B-PER b/I-PER"
    assert s1.tokens_with_labels() == (("a", "B-PER"), ("b", "I-PER"))
    assert s1.span_tokens(Span(0, 1)) == ("a",)
    assert s1.mention_tokens(Mention(Span(0, 1), "PER")) == ("a",)

    s2 = LabeledSequence(s1.tokens, s1.labels)
    # Provenance not included in equality
    assert s1 == s2

    with pytest.raises(ValueError):
        # Mismatched length
        LabeledSequence(["a", "b"], ["B-PER"])

    with pytest.raises(ValueError):
        # Empty
        LabeledSequence([], [])

    with pytest.raises(ValueError):
        # Bad label
        LabeledSequence(["a"], [""])

    with pytest.raises(ValueError):
        # Bad token
        LabeledSequence([""], ["B-PER"])

    s2 = s1.with_mentions([Mention(Span(0, 2), "PER")])
    assert s2.mentions == (Mention(Span(0, 2), "PER"),)

    with pytest.raises(ValueError):
        # Mismatched length between tokens and other_fields
        LabeledSequence(["a", "b"], ["B-PER", "I-PER"], other_fields=[["DT"]])
