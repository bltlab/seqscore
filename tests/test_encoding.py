from dataclasses import dataclass

import pytest

from seqscore.encoding import (
    _BIO,
    _BIOES,
    _IO,
    _IOB,
    SUPPORTED_INPUT_ENCODINGS,
    SUPPORTED_OUTPUT_ENCODINGS,
    EncodingError,
    _BILOUDialect,
    _BIOESDialect,
    _BMEOWDialect,
    _BMESDialect,
    _IOBLenient,
    get_encoding,
    get_output_encoding,
)
from seqscore.model import AnnotatedSequence, LabeledSequence, Mention, Span

FULL_SENTENCE_LABELS = {
    "IO": ["I-PER", "O", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-LOC"],
    "IOB": ["I-PER", "O", "I-ORG", "I-ORG", "B-ORG", "I-ORG", "I-ORG", "I-LOC"],
    "BIO": ["B-PER", "O", "B-ORG", "I-ORG", "B-ORG", "I-ORG", "I-ORG", "B-LOC"],
    "BIOES": ["S-PER", "O", "B-ORG", "E-ORG", "B-ORG", "I-ORG", "E-ORG", "S-LOC"],
    "BILOU": ["U-PER", "O", "B-ORG", "L-ORG", "B-ORG", "I-ORG", "L-ORG", "U-LOC"],
    "BMES": ["S-PER", "O", "B-ORG", "E-ORG", "B-ORG", "M-ORG", "E-ORG", "S-LOC"],
    "BMEOW": ["W-PER", "O", "B-ORG", "E-ORG", "B-ORG", "M-ORG", "E-ORG", "W-LOC"],
}
# IOB-lenient decodes valid IOB labels identically, so it shares the IOB labels
FULL_SENTENCE_LABELS["IOB-lenient"] = list(FULL_SENTENCE_LABELS["IOB"])
FULL_SENTENCE_MENTS = [
    Mention(Span(0, 1), "PER"),
    Mention(Span(2, 4), "ORG"),
    Mention(Span(4, 7), "ORG"),
    Mention(Span(7, 8), "LOC"),
]
# IO cannot faithfully encode this sentence, so there is just one org
FULL_SENTENCE_MENTS_IO = [
    Mention(Span(0, 1), "PER"),
    Mention(Span(2, 7), "ORG"),
    Mention(Span(7, 8), "LOC"),
]
# Map to sets of encodings that allow that state
VALID_ENCODING_STATES = {
    "B": {"IOB", "IOB-lenient", "BIO", "BIOES", "BILOU", "BMES", "BMEOW"},
    "I": {"IOB", "IOB-lenient", "BIO", "BIOES", "BILOU", "IO"},
    "O": {"IOB", "IOB-lenient", "IO", "BIO", "BIOES", "BILOU", "BMES", "BMEOW"},
    "E": {"BIOES", "BMES", "BMEOW"},
    "M": {"BMES", "BMEOW"},
    "L": {"BILOU"},
    "W": {"BMEOW"},
    "Z": {},
}


@dataclass
class EdgeTestSentence:
    name: str
    mentions: list[Mention]
    encoding_labels: list[tuple[list[str], list[str]]]


EDGE_TEST_SENTENCES = [
    EdgeTestSentence(
        "One token, one mention",
        [Mention(Span(0, 1), "PER")],
        [
            (["BIO"], ["B-PER"]),
            (["BIOES", "BMES"], ["S-PER"]),
            (["BILOU"], ["U-PER"]),
            (["BMEOW"], ["W-PER"]),
            (["IO", "IOB"], ["I-PER"]),
        ],
    ),
    EdgeTestSentence(
        "Two tokens, one mention covering them all",
        [Mention(Span(0, 2), "PER")],
        [
            (["BIO"], ["B-PER", "I-PER"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "L-PER"]),
            (["IO", "IOB"], ["I-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        "Three tokens, one mention covering them all",
        [Mention(Span(0, 3), "PER")],
        [
            (["BIO"], ["B-PER", "I-PER", "I-PER"]),
            (["BIOES"], ["B-PER", "I-PER", "E-PER"]),
            (["BMES", "BMEOW"], ["B-PER", "M-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "I-PER", "L-PER"]),
            (["IO", "IOB"], ["I-PER", "I-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        "Adjacent same-type one-token mentions",
        [Mention(Span(0, 1), "PER"), Mention(Span(1, 2), "PER")],
        [
            (["BIO"], ["B-PER", "B-PER"]),
            (["BIOES", "BMES"], ["S-PER", "S-PER"]),
            (["BILOU"], ["U-PER", "U-PER"]),
            (["BMEOW"], ["W-PER", "W-PER"]),
            # IO is not included because it cannot faithfully handle this
            (["IOB"], ["I-PER", "B-PER"]),
        ],
    ),
    EdgeTestSentence(
        "Adjacent different-type one-token mentions",
        [Mention(Span(0, 1), "PER"), Mention(Span(1, 2), "ORG")],
        [
            (["BIO"], ["B-PER", "B-ORG"]),
            (["BIOES", "BMES"], ["S-PER", "S-ORG"]),
            (["BILOU"], ["U-PER", "U-ORG"]),
            (["BMEOW"], ["W-PER", "W-ORG"]),
            (["IO", "IOB"], ["I-PER", "I-ORG"]),
        ],
    ),
    EdgeTestSentence(
        "Adjacent same-type two-token mentions",
        [Mention(Span(0, 2), "PER"), Mention(Span(2, 4), "PER")],
        [
            (["BIO"], ["B-PER", "I-PER", "B-PER", "I-PER"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER", "B-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "L-PER", "B-PER", "L-PER"]),
            # IO is not included because it cannot faithfully handle this
            (["IOB"], ["I-PER", "I-PER", "B-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        "Adjacent different-type two-token mentions",
        [Mention(Span(0, 2), "PER"), Mention(Span(2, 4), "ORG")],
        [
            (["BIO"], ["B-PER", "I-PER", "B-ORG", "I-ORG"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER", "B-ORG", "E-ORG"]),
            (["BILOU"], ["B-PER", "L-PER", "B-ORG", "L-ORG"]),
            (["IO", "IOB"], ["I-PER", "I-PER", "I-ORG", "I-ORG"]),
        ],
    ),
]


def test_basic_decoding() -> None:
    for encoding_name in SUPPORTED_INPUT_ENCODINGS:
        encoding = get_encoding(encoding_name)
        labels = FULL_SENTENCE_LABELS[encoding_name]
        mentions = (
            FULL_SENTENCE_MENTS_IO if encoding_name == "IO" else FULL_SENTENCE_MENTS
        )
        assert encoding.decode_labels(labels) == mentions


def test_basic_encoding() -> None:
    for encoding_name in SUPPORTED_OUTPUT_ENCODINGS:
        encoding = get_encoding(encoding_name)
        labels = FULL_SENTENCE_LABELS[encoding_name]
        mentions = (
            FULL_SENTENCE_MENTS_IO if encoding_name == "IO" else FULL_SENTENCE_MENTS
        )

        assert encoding.encode_mentions(mentions, len(labels)) == labels
        # Also test encoding sentence object, intentionally putting no mentions in the
        # sentence labels to make sure encoding using the mentions, not the labels
        sentence = AnnotatedSequence(
            tokens=tuple("a" for _ in labels),
            labels=tuple("O" for _ in labels),
            mentions=tuple(mentions),
        )
        assert encoding.encode_sequence(sentence) == labels


def test_round_trip() -> None:
    for encoding_name in SUPPORTED_OUTPUT_ENCODINGS:
        # Skip IO since it can't round-trip
        if encoding_name == "IO":
            continue

        encoding = get_encoding(encoding_name)
        labels = FULL_SENTENCE_LABELS[encoding_name]
        mentions = FULL_SENTENCE_MENTS

        # Encode, then decode
        out_labels = encoding.encode_mentions(mentions, len(labels))
        assert encoding.decode_labels(out_labels) == mentions

        # Decode, then encode
        out_mentions = encoding.decode_labels(labels)
        assert encoding.encode_mentions(out_mentions, len(labels)) == labels


def test_valid_states() -> None:
    all_encoding_names = set(SUPPORTED_INPUT_ENCODINGS)
    for state, valid_encoding_names in VALID_ENCODING_STATES.items():
        for encoding_name in all_encoding_names:
            encoding = get_encoding(encoding_name)
            if encoding_name in valid_encoding_names:
                assert encoding.is_valid_state(state)
            else:
                assert not encoding.is_valid_state(state)


def test_edge_case_encoding() -> None:
    for case in EDGE_TEST_SENTENCES:
        mentions = case.mentions
        for encoding_names, labels in case.encoding_labels:
            for encoding_name in encoding_names:
                encoding = get_encoding(encoding_name)
                assert encoding.encode_mentions(mentions, len(labels)) == labels


def test_get_encodings() -> None:
    enc = get_encoding("IO")
    assert isinstance(enc, _IO)
    assert enc.name == "IO"

    enc = get_encoding("IOB")
    assert isinstance(enc, _IOB)
    assert enc.name == "IOB"

    enc = get_encoding("IOB-lenient")
    assert isinstance(enc, _IOBLenient)
    assert enc.name == "IOB-lenient"
    # Names are matched case-insensitively
    assert get_encoding("iob-lenient") is enc

    enc = get_encoding("BIO")
    assert isinstance(enc, _BIO)
    assert enc.name == "BIO"

    # Test the dialects for BIOES and derivatives
    enc = get_encoding("BIOES")
    assert isinstance(enc, _BIOES)
    assert isinstance(enc.dialect, _BIOESDialect)
    assert enc.name == "BIOES"

    enc = get_encoding("BILOU")
    assert isinstance(enc, _BIOES)
    assert isinstance(enc.dialect, _BILOUDialect)
    assert enc.name == "BILOU"

    enc = get_encoding("BMES")
    assert isinstance(enc, _BIOES)
    assert isinstance(enc.dialect, _BMESDialect)
    assert enc.name == "BMES"

    enc = get_encoding("BMEOW")
    assert isinstance(enc, _BIOES)
    assert isinstance(enc.dialect, _BMEOWDialect)
    assert enc.name == "BMEOW"


def test_get_unknown_encoding() -> None:
    with pytest.raises(ValueError):
        get_encoding("FOO")


def test_split_label() -> None:
    # This logic is shared across all encodings, we just need any instantiable one
    encoding = get_encoding("BIO")

    assert encoding.split_label("O") == ("O", None)
    assert encoding.split_label("B-PER") == ("B", "PER")
    # Only splits the first delim
    assert encoding.split_label("I-ORG-CORP") == ("I", "ORG-CORP")

    with pytest.raises(EncodingError):
        assert encoding.split_label("B")

    with pytest.raises(EncodingError):
        assert encoding.split_label("O-ORG")

    with pytest.raises(EncodingError):
        assert encoding.split_label("")


def test_join_label() -> None:
    # This logic is shared across all encodings, we just need any instantiable one
    encoding = get_encoding("BIO")

    assert encoding.join_label("B", "PER") == "B-PER"
    assert encoding.join_label("O", None) == "O"

    with pytest.raises(AssertionError):
        encoding.join_label("B", None)

    with pytest.raises(AssertionError):
        encoding.join_label("O", "PER")


def test_labeled_sequence() -> None:
    # Test length mismatch
    with pytest.raises(ValueError):
        LabeledSequence(
            tokens=("a",) * 10,
            labels=("O",) * 9,
        )


def test_decode_bio_invalid_continue() -> None:
    decoder = get_encoding("BIO")
    sent1 = LabeledSequence(tokens=("a", "b"), labels=("B-PER", "I-LOC"))
    with pytest.raises(AssertionError):
        assert decoder.decode_sequence(sent1)


def test_decode_iob_invalid_begin() -> None:
    decoder = get_encoding("IOB")
    sent = LabeledSequence(tokens=("a", "b"), labels=("I-PER", "B-LOC"))
    with pytest.raises(AssertionError):
        assert decoder.decode_sequence(sent)


def test_decode_bioes_invalid_start() -> None:
    decoder = get_encoding("BIOES")
    sents = [
        LabeledSequence(tokens=("a",), labels=("I-PER",)),
        LabeledSequence(tokens=("a",), labels=("E-PER",)),
    ]
    for sent in sents:
        with pytest.raises(AssertionError):
            assert decoder.decode_sequence(sent)


def test_decode_bioes_invalid_end() -> None:
    decoder = get_encoding("BIOES")
    sents = [
        # Single-token mentions must start (and end) with S
        LabeledSequence(tokens=("a", "b"), labels=("B-PER", "S-PER")),
        # Multi-token mentions must end in E
        LabeledSequence(tokens=("a",), labels=("B-PER",)),
        LabeledSequence(tokens=("a", "b"), labels=("B-PER", "I-PER")),
        # Ends with wrong type
        LabeledSequence(tokens=("a", "b", "c"), labels=("B-PER", "I-PER", "E-ORG")),
        # Multi-token mentions cannot end in S
        LabeledSequence(tokens=("a", "b", "c"), labels=("B-PER", "I-PER", "S-PER")),
    ]
    for sent in sents:
        with pytest.raises(AssertionError):
            assert decoder.decode_sequence(sent)


def test_decode_bioes_invalid_continue() -> None:
    decoder = get_encoding("BIOES")
    sents = [
        # B must be followed by I or E of the same type
        LabeledSequence(tokens=("a", "b"), labels=("B-PER", "B-PER")),
        # Cannot change types mid-mention
        LabeledSequence(tokens=("a", "b"), labels=("B-PER", "E-ORG")),
        LabeledSequence(tokens=("a", "b", "c"), labels=("B-PER", "I-PER", "E-ORG")),
    ]
    for sent in sents:
        with pytest.raises(AssertionError):
            assert decoder.decode_sequence(sent)


# Labels valid under IOB-lenient but not under IOB, with the mentions they decode to
IOB_LENIENT_DECODE_CASES = [
    # B- after O is an ordinary mention start
    (["O", "B-PER", "O"], [Mention(Span(1, 2), "PER")]),
    # The start of a sequence counts as following O
    (["B-PER", "I-PER", "O"], [Mention(Span(0, 2), "PER")]),
    # B- of a different type ends the mention in progress and starts a new one
    (["I-PER", "B-ORG"], [Mention(Span(0, 1), "PER"), Mention(Span(1, 2), "ORG")]),
    (
        ["I-PER", "B-ORG", "I-ORG", "O"],
        [Mention(Span(0, 1), "PER"), Mention(Span(1, 3), "ORG")],
    ),
    # B- after B- of a different type does the same
    (["B-PER", "B-ORG"], [Mention(Span(0, 1), "PER"), Mention(Span(1, 2), "ORG")]),
]


def test_iob_lenient_decoding() -> None:
    encoding = get_encoding("IOB-lenient")
    for labels, mentions in IOB_LENIENT_DECODE_CASES:
        assert encoding.decode_labels(labels) == mentions


def test_iob_rejects_lenient_transitions() -> None:
    encoding = get_encoding("IOB")
    for labels, _ in IOB_LENIENT_DECODE_CASES:
        with pytest.raises(AssertionError):
            encoding.decode_labels(labels)


def test_iob_lenient_transitions() -> None:
    iob = get_encoding("IOB")
    lenient = get_encoding("IOB-lenient")
    for first, second in (("O", None), ("I", "PER"), ("B", "PER")):
        assert not iob.is_valid_transition(first, second, "B", "ORG")
        assert lenient.is_valid_transition(first, second, "B", "ORG")

    # Same-type transitions are unchanged
    assert iob.valid_same_type_transitions == lenient.valid_same_type_transitions

    # Every transition between I, B, and O labels is valid, but states from
    # other encodings are still rejected
    assert not lenient.is_valid_state("E")
    assert not lenient.is_valid_transition("O", None, "E", "PER")


def test_iob_lenient_matches_iob_on_valid_input() -> None:
    iob = get_encoding("IOB")
    lenient = get_encoding("IOB-lenient")

    all_labels = [FULL_SENTENCE_LABELS["IOB"]]
    for case in EDGE_TEST_SENTENCES:
        for encoding_names, labels in case.encoding_labels:
            if "IOB" in encoding_names:
                all_labels.append(labels)

    for labels in all_labels:
        assert lenient.decode_labels(labels) == iob.decode_labels(labels)


def test_iob_lenient_is_input_only() -> None:
    encoding = get_encoding("IOB-lenient")
    assert not encoding.supports_output

    with pytest.raises(EncodingError):
        encoding.encode_mentions(FULL_SENTENCE_MENTS, 8)

    with pytest.raises(EncodingError):
        get_output_encoding("IOB-lenient")

    # Encodings that support output are still accepted
    assert get_output_encoding("IOB") is get_encoding("IOB")


def test_iob_lenient_not_an_output_encoding_name() -> None:
    assert "IOB-lenient" in SUPPORTED_INPUT_ENCODINGS
    assert "IOB-lenient" not in SUPPORTED_OUTPUT_ENCODINGS
