import pytest
from seqscore_lib.model import LabeledSequence, Mention, SequenceProvenance, Span


def test_span() -> None:
    assert len(Span(start=0, end=1)) == 1
    assert len(Span(start=1, end=2)) == 1
    assert len(Span(start=0, end=2)) == 2

    with pytest.raises(ValueError):
        Span(start=-1, end=0)

    with pytest.raises(ValueError):
        Span(start=0, end=0)


def test_mention() -> None:
    m1 = Mention(span=Span(start=0, end=1), type="PER")
    assert m1.type == "PER"
    assert m1.span == Span(start=0, end=1)
    assert len(m1) == 1

    with pytest.raises(ValueError):
        Mention(span=Span(start=0, end=1), type="")

    with pytest.raises(TypeError):
        # Intentionally incorrect type
        Mention(Span(0, 1), None)  # type: ignore


def test_labeled_sentence() -> None:
    s1 = LabeledSequence(
        tokens=["a", "b"],
        labels=["B-PER", "I-PER"],
        provenance=SequenceProvenance(starting_line=7, source="test"),
    )
    assert s1.tokens == ("a", "b")
    assert s1[0] == "a"
    assert s1[0:2] == ("a", "b")
    assert list(s1.tokens) == ["a", "b"]
    assert s1.labels == ("B-PER", "I-PER")
    assert s1.provenance == SequenceProvenance(starting_line=7, source="test")
    assert str(s1) == "a/B-PER b/I-PER"
    assert s1.tokens_with_labels() == (("a", "B-PER"), ("b", "I-PER"))
    assert s1.span_tokens(Span(start=0, end=1)) == ("a",)
    assert s1.mention_tokens(Mention(span=Span(start=0, end=1), type="PER")) == ("a",)

    s2 = LabeledSequence(tokens=s1.tokens, labels=s1.labels)
    # Provenance not included in equality
    assert s1 == s2

    with pytest.raises(ValueError):
        # Mismatched length
        LabeledSequence(tokens=["a", "b"], labels=["B-PER"])

    with pytest.raises(ValueError):
        # Empty
        LabeledSequence(tokens=[], labels=[])

    with pytest.raises(ValueError):
        # Bad label
        LabeledSequence(tokens=["a"], labels=[""])

    with pytest.raises(ValueError):
        # Bad token
        LabeledSequence(tokens=[""], labels=["B-PER"])

    s2 = s1.with_mentions([Mention(span=Span(start=0, end=2), type="PER")])
    assert s2.mentions == (Mention(span=Span(start=0, end=2), type="PER"),)

    with pytest.raises(ValueError):
        # Mismatched length between tokens and other_fields
        LabeledSequence(
            tokens=["a", "b"],
            labels=["B-PER", "I-PER"],
            other_fields=[["DT"]],
        )
