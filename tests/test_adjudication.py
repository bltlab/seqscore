from collections import Counter

import pytest

from seqscore.adjudication import (
    AdjudicationResult,
    MentionCounts,
    adjudicate_documents,
    fleiss_kappa,
    ngrams,
    overlap,
)
from seqscore.model import LabeledSequence, Mention, SequenceProvenance, Span

DOCS_BY_ANNOTATOR = {
    0: [
        [
            LabeledSequence(
                ("The", "Boston", "Red", "Sox", "lost", "to", "Cleveland"),
                ("O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "B-ORG"),
                (Mention(Span(1, 4), "ORG"), Mention(Span(6, 7), "ORG")),
                provenance=SequenceProvenance(0, "test-case"),
            ),
            LabeledSequence(
                ("Boston", "and", "Cleveland", "are", "cities", "."),
                ("B-LOC", "O", "B-LOC", "O", "O", "O"),
                (Mention(Span(0, 1), "LOC"), Mention(Span(2, 3), "LOC")),
                provenance=SequenceProvenance(1, "test-case"),
            ),
        ]
    ],
    1: [
        [
            LabeledSequence(
                ("The", "Boston", "Red", "Sox", "lost", "to", "Cleveland"),
                ("O", "B-LOC", "O", "O", "O", "O", "B-LOC"),
                (Mention(Span(1, 2), "ORG"), Mention(Span(6, 7), "LOC")),
                provenance=SequenceProvenance(0, "test-case"),
            ),
            LabeledSequence(
                ("Boston", "and", "Cleveland", "are", "cities", "."),
                ("B-LOC", "O", "B-LOC", "O", "O", "O"),
                (Mention(Span(0, 1), "LOC"), Mention(Span(2, 3), "LOC")),
                provenance=SequenceProvenance(1, "test-case"),
            ),
        ]
    ],
    2: [
        [
            LabeledSequence(
                ("The", "Boston", "Red", "Sox", "lost", "to", "Cleveland"),
                ("B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC"),
                (Mention(Span(0, 4), "ORG"), Mention(Span(6, 7), "LOC")),
                provenance=SequenceProvenance(0, "test-case"),
            ),
            LabeledSequence(
                ("Boston", "and", "Cleveland", "are", "cities", "."),
                ("B-LOC", "O", "B-LOC", "O", "O", "O"),
                (Mention(Span(0, 1), "LOC"), Mention(Span(2, 3), "LOC")),
                provenance=SequenceProvenance(1, "test-case"),
            ),
        ]
    ],
}


def test_mention_type_counts():
    mention_type_counts = MentionCounts()
    mention_type_counts.put(("Boston", "Red", "Sox"), "ORG")
    mention_type_counts.put(("Boston", "Red", "Sox"), "ORG")
    mention_type_counts.put(("Boston",), "ORG")
    mention_type_counts.put(("Boston",), "LOC")
    mention_type_counts.put(("Boston",), "LOC")
    mention_type_counts.put(("Boston",), "LOC")

    # mention_type_counts.entropy("Boston")
    # mention_type_counts.most_common_types("Boston")


def test_ngrams():
    sent = ["The", "dog", "ate", "a", "taco", "."]
    unigrams = list(ngrams(sent, 1))
    bigrams = list(ngrams(sent, 2))
    trigrams = list(ngrams(sent, 3))

    assert len(unigrams) == 6
    assert len(bigrams) == 5
    assert len(trigrams) == 4
    assert trigrams[0] == ("The", "dog", "ate")
    assert trigrams[-1] == ("a", "taco", ".")
    assert bigrams[0] == ("The", "dog")
    assert bigrams[-1] == ("taco", ".")


def test_adjudicate():
    adjuication_result: AdjudicationResult = adjudicate_documents(DOCS_BY_ANNOTATOR)
    assert adjuication_result.overall_entity_counts.entropy(("Cleveland",)) == 0
    assert adjuication_result.overall_entity_counts.entropy(("Boston",)) == 0
    assert adjuication_result.overall_entity_counts.entropy(("lost", "to")) == 0
    assert 0.6365141682948128 == pytest.approx(
        adjuication_result.overall_entity_counts.entropy(("Red", "Sox"))
    )
    assert 0.6365141682948128 == pytest.approx(
        adjuication_result.overall_entity_counts.entropy(("The",))
    )
    disagreements = adjuication_result.disagreements

    assert (
        disagreements.disagreements_by_span + disagreements.disaggrements_by_type
        == disagreements.disagreements
    )
    assert disagreements.disagreements_by_span == 3

    assert disagreements.disaggrements_by_type == 1
    assert disagreements.majority_vote_sequences[0].labelsequence.labels == (
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-LOC",
    )
    assert disagreements.majority_vote_sequences[1].labelsequence.labels == (
        "B-LOC",
        "O",
        "B-LOC",
        "O",
        "O",
        "O",
    )


def test_fleiss_kappa():
    # Example from https://en.wikipedia.org/wiki/Fleiss%27_kappa
    units_to_labels = {
        (1, Span(1, 2)): Counter({"5": 14}),
        (2, Span(1, 2)): Counter({"2": 2, "3": 6, "4": 4, "5": 2}),
        (3, Span(1, 2)): Counter({"3": 3, "4": 5, "5": 6}),
        (4, Span(1, 2)): Counter({"2": 3, "3": 9, "4": 2, "5": 0}),
        (5, Span(1, 2)): Counter({"1": 2, "2": 2, "3": 8, "4": 1, "5": 1}),
        (6, Span(1, 2)): Counter({"1": 7, "2": 7}),
        (7, Span(1, 2)): Counter({"1": 3, "2": 2, "3": 6, "4": 3, "5": 0}),
        (8, Span(1, 2)): Counter({"1": 2, "2": 5, "3": 3, "4": 2, "5": 2}),
        (9, Span(1, 2)): Counter({"1": 6, "2": 5, "3": 2, "4": 1}),
        (10, Span(1, 2)): Counter({"2": 2, "3": 2, "4": 3, "5": 7}),
    }
    kappa = fleiss_kappa(units_to_labels, 14, set("12345"))
    assert kappa == pytest.approx(0.209930, rel=1e-4)


def test_overlap():
    s1 = Span(3, 7)
    s2 = Span(0, 1)
    s3 = Span(0, 3)
    s4 = Span(0, 4)
    s5 = Span(4, 5)
    s6 = Span(5, 8)
    s7 = Span(7, 9)
    s8 = Span(8, 9)
    assert not overlap(s1, s2)
    assert not overlap(s1, s3)
    assert overlap(s1, s4)
    assert overlap(s1, s5)
    assert overlap(s1, s6)
    assert not overlap(s1, s7)
    assert not overlap(s1, s8)
