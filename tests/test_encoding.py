import pytest
from pydantic import BaseModel

from seqscore.encoding import (
    _ENCODING_NAMES,
    BIO,
    BIOES,
    IO,
    IOB,
    SUPPORTED_ENCODINGS,
    BILOUDialect,
    BIOESDialect,
    BMEOWDialect,
    BMESDialect,
    EncodingError,
    get_encoding,
)
from seqscore.model import LabeledSequence, Mention, Span

FULL_SENTENCE_LABELS = {
    "IO": ["I-PER", "O", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-LOC"],
    "IOB": ["I-PER", "O", "I-ORG", "I-ORG", "B-ORG", "I-ORG", "I-ORG", "I-LOC"],
    "BIO": ["B-PER", "O", "B-ORG", "I-ORG", "B-ORG", "I-ORG", "I-ORG", "B-LOC"],
    "BIOES": ["S-PER", "O", "B-ORG", "E-ORG", "B-ORG", "I-ORG", "E-ORG", "S-LOC"],
    "BILOU": ["U-PER", "O", "B-ORG", "L-ORG", "B-ORG", "I-ORG", "L-ORG", "U-LOC"],
    "BMES": ["S-PER", "O", "B-ORG", "E-ORG", "B-ORG", "M-ORG", "E-ORG", "S-LOC"],
    "BMEOW": ["W-PER", "O", "B-ORG", "E-ORG", "B-ORG", "M-ORG", "E-ORG", "W-LOC"],
}
FULL_SENTENCE_MENTS = [
    Mention(span=Span(start=0, end=1), type="PER"),
    Mention(span=Span(start=2, end=4), type="ORG"),
    Mention(span=Span(start=4, end=7), type="ORG"),
    Mention(span=Span(start=7, end=8), type="LOC"),
]
# IO cannot faithfully encode this sentence, so there is just one org
FULL_SENTENCE_MENTS_IO = [
    Mention(span=Span(start=0, end=1), type="PER"),
    Mention(span=Span(start=2, end=7), type="ORG"),
    Mention(span=Span(start=7, end=8), type="LOC"),
]
# Map to sets of encodings that allow that state
VALID_ENCODING_STATES = {
    "B": {"IOB", "BIO", "BIOES", "BILOU", "BMES", "BMEOW"},
    "I": {"IOB", "BIO", "BIOES", "BILOU", "IO"},
    "O": {"IOB", "IO", "BIO", "BIOES", "BILOU", "BMES", "BMEOW"},
    "E": {"BIOES", "BMES", "BMEOW"},
    "M": {"BMES", "BMEOW"},
    "L": {"BILOU"},
    "W": {"BMEOW"},
    "Z": {},
}


class EdgeTestSentence(BaseModel):
    name: str
    mentions: list[Mention]
    encoding_labels: list[tuple[list[str], list[str]]]


EDGE_TEST_SENTENCES = [
    EdgeTestSentence(
        name="One token, one mention",
        mentions=[Mention(span=Span(start=0, end=1), type="PER")],
        encoding_labels=[
            (["BIO"], ["B-PER"]),
            (["BIOES", "BMES"], ["S-PER"]),
            (["BILOU"], ["U-PER"]),
            (["BMEOW"], ["W-PER"]),
            (["IO", "IOB"], ["I-PER"]),
        ],
    ),
    EdgeTestSentence(
        name="Two tokens, one mention covering them all",
        mentions=[Mention(span=Span(start=0, end=2), type="PER")],
        encoding_labels=[
            (["BIO"], ["B-PER", "I-PER"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "L-PER"]),
            (["IO", "IOB"], ["I-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        name="Three tokens, one mention covering them all",
        mentions=[Mention(span=Span(start=0, end=3), type="PER")],
        encoding_labels=[
            (["BIO"], ["B-PER", "I-PER", "I-PER"]),
            (["BIOES"], ["B-PER", "I-PER", "E-PER"]),
            (["BMES", "BMEOW"], ["B-PER", "M-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "I-PER", "L-PER"]),
            (["IO", "IOB"], ["I-PER", "I-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        name="Adjacent same-type one-token mentions",
        mentions=[
            Mention(span=Span(start=0, end=1), type="PER"),
            Mention(span=Span(start=1, end=2), type="PER"),
        ],
        encoding_labels=[
            (["BIO"], ["B-PER", "B-PER"]),
            (["BIOES", "BMES"], ["S-PER", "S-PER"]),
            (["BILOU"], ["U-PER", "U-PER"]),
            (["BMEOW"], ["W-PER", "W-PER"]),
            # IO is not included because it cannot faithfully handle this
            (["IOB"], ["I-PER", "B-PER"]),
        ],
    ),
    EdgeTestSentence(
        name="Adjacent different-type one-token mentions",
        mentions=[
            Mention(span=Span(start=0, end=1), type="PER"),
            Mention(span=Span(start=1, end=2), type="ORG"),
        ],
        encoding_labels=[
            (["BIO"], ["B-PER", "B-ORG"]),
            (["BIOES", "BMES"], ["S-PER", "S-ORG"]),
            (["BILOU"], ["U-PER", "U-ORG"]),
            (["BMEOW"], ["W-PER", "W-ORG"]),
            (["IO", "IOB"], ["I-PER", "I-ORG"]),
        ],
    ),
    EdgeTestSentence(
        name="Adjacent same-type two-token mentions",
        mentions=[
            Mention(span=Span(start=0, end=2), type="PER"),
            Mention(span=Span(start=2, end=4), type="PER"),
        ],
        encoding_labels=[
            (["BIO"], ["B-PER", "I-PER", "B-PER", "I-PER"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER", "B-PER", "E-PER"]),
            (["BILOU"], ["B-PER", "L-PER", "B-PER", "L-PER"]),
            # IO is not included because it cannot faithfully handle this
            (["IOB"], ["I-PER", "I-PER", "B-PER", "I-PER"]),
        ],
    ),
    EdgeTestSentence(
        name="Adjacent different-type two-token mentions",
        mentions=[
            Mention(span=Span(start=0, end=2), type="PER"),
            Mention(span=Span(start=2, end=4), type="ORG"),
        ],
        encoding_labels=[
            (["BIO"], ["B-PER", "I-PER", "B-ORG", "I-ORG"]),
            (["BIOES", "BMES", "BMEOW"], ["B-PER", "E-PER", "B-ORG", "E-ORG"]),
            (["BILOU"], ["B-PER", "L-PER", "B-ORG", "L-ORG"]),
            (["IO", "IOB"], ["I-PER", "I-PER", "I-ORG", "I-ORG"]),
        ],
    ),
]


def test_basic_decoding() -> None:
    for encoding_name in SUPPORTED_ENCODINGS:
        encoding = get_encoding(encoding_name)
        labels = FULL_SENTENCE_LABELS[encoding_name]
        mentions = (
            FULL_SENTENCE_MENTS_IO if encoding_name == "IO" else FULL_SENTENCE_MENTS
        )
        assert encoding.decode_labels(labels) == mentions


def test_basic_encoding() -> None:
    for encoding_name in SUPPORTED_ENCODINGS:
        encoding = get_encoding(encoding_name)
        labels = FULL_SENTENCE_LABELS[encoding_name]
        mentions = (
            FULL_SENTENCE_MENTS_IO if encoding_name == "IO" else FULL_SENTENCE_MENTS
        )

        assert encoding.encode_mentions(mentions, len(labels)) == labels
        # Also test encoding sentence object, intentionally putting no mentions in the
        # sentence labels to make sure encoding using the mentions, not the labels
        sentence = LabeledSequence(
            tokens=["a"] * len(labels),
            labels=["O"] * len(labels),
            mentions=mentions,
        )
        assert encoding.encode_sequence(sentence) == labels


def test_round_trip() -> None:
    for encoding_name in set(SUPPORTED_ENCODINGS) & set(SUPPORTED_ENCODINGS):
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
    all_encoding_names = set(_ENCODING_NAMES)
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
    assert isinstance(get_encoding("IO"), IO)
    assert isinstance(get_encoding("IOB"), IOB)
    assert isinstance(get_encoding("BIO"), BIO)

    # Test the dialects for BIOES and derivatives
    enc = get_encoding("BIOES")
    assert isinstance(enc, BIOES)
    assert isinstance(enc.dialect, BIOESDialect)

    enc = get_encoding("BILOU")
    assert isinstance(enc, BIOES)
    assert isinstance(enc.dialect, BILOUDialect)

    enc = get_encoding("BMES")
    assert isinstance(enc, BIOES)
    assert isinstance(enc.dialect, BMESDialect)

    enc = get_encoding("BMEOW")
    assert isinstance(enc, BIOES)
    assert isinstance(enc.dialect, BMEOWDialect)


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
            tokens=["a"] * 10,
            labels=["O"] * 9,
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
