import pytest

from seqscore.model import LabeledSequence, Mention, SequenceProvenance, Span


def test_span() -> None:
    assert len(Span(0, 1)) == 1
    assert len(Span(1, 2)) == 1
    assert len(Span(0, 2)) == 2

    with pytest.raises(ValueError):
        Span(-2, -1)

    with pytest.raises(ValueError):
        Span(-1, 0)

    with pytest.raises(ValueError):
        Span(0, -1)

    with pytest.raises(ValueError):
        Span(0, 0)

    with pytest.raises(ValueError):
        Span(3, 1)


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
        tokens=("a", "b"),
        labels=("B-PER", "I-PER"),
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

    s2 = LabeledSequence(tokens=s1.tokens, labels=s1.labels)
    # Provenance not included in equality
    assert s1 == s2
    # Hashes identical for equal objects
    assert hash(s1) == hash(s2)
    # Equality fails for objects of other types
    assert s1 != ""

    with pytest.raises(ValueError):
        # Mismatched length
        LabeledSequence(tokens=("a", "b"), labels=("B-PER",))

    with pytest.raises(ValueError):
        # Empty
        LabeledSequence(tokens=(), labels=())

    with pytest.raises(ValueError) as err:
        # Bad label
        LabeledSequence(tokens=("a",), labels=("",))
    assert "Invalid label at sequence index 0: ''" in str(err.value)

    with pytest.raises(ValueError) as err:
        # Bad token
        LabeledSequence(tokens=("",), labels=("B-PER",))
    assert "Invalid token at sequence index 0: ''" in str(err.value)

    s2 = s1.with_mentions([Mention(Span(0, 2), "PER")])
    assert s2.mentions == (Mention(Span(0, 2), "PER"),)

    with pytest.raises(ValueError):
        # Mismatched length between tokens and other_fields
        LabeledSequence(
            tokens=("a", "b"), labels=("B-PER", "I-PER"), token_fields=(("DT",),)
        )
