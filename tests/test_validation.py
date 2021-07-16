from typing import Dict, List

import pytest
from attr import attrs

from seqscore.encoding import REPAIR_NONE, EncodingError, get_encoding
from seqscore.validation import validate_labels


@attrs(auto_attribs=True)
class RepairTest:
    original_labels: List[str]
    n_errors: int
    repaired_labels: Dict[str, List[str]]


BIO_REPAIRS = [
    RepairTest(
        ["I-PER"],
        1,
        {"conlleval": ["B-PER"], "discard": ["O"]},
    ),
    RepairTest(
        ["I-PER", "I-PER"],
        1,
        {"conlleval": ["B-PER", "I-PER"], "discard": ["O", "O"]},
    ),
    RepairTest(
        ["O", "I-PER", "I-PER"],
        1,
        {"conlleval": ["O", "B-PER", "I-PER"], "discard": ["O", "O", "O"]},
    ),
    RepairTest(
        ["B-ORG", "I-PER", "I-PER"],
        1,
        {"conlleval": ["B-ORG", "B-PER", "I-PER"], "discard": ["B-ORG", "O", "O"]},
    ),
    RepairTest(
        ["I-ORG", "I-PER", "I-PER"],
        2,
        {"conlleval": ["B-ORG", "B-PER", "I-PER"], "discard": ["O", "O", "O"]},
    ),
    RepairTest(
        ["O", "I-ORG", "I-PER", "I-ORG"],
        3,
        {"conlleval": ["O", "B-ORG", "B-PER", "B-ORG"], "discard": ["O", "O", "O", "O"]},
    ),
    RepairTest(
        ["O", "B-ORG", "B-PER", "I-PER"],
        0,
        {
            "conlleval": ["O", "B-ORG", "B-PER", "I-PER"],
            "discard": ["O", "B-ORG", "B-PER", "I-PER"],
        },
    ),
]
IOB_REPAIRS = [
    RepairTest(
        ["B-PER"],
        1,
        {"conlleval": ["I-PER"]},
    ),
    RepairTest(
        ["B-PER", "I-PER"],
        1,
        {"conlleval": ["I-PER", "I-PER"]},
    ),
    RepairTest(
        ["O", "B-PER", "I-PER"],
        1,
        {"conlleval": ["O", "I-PER", "I-PER"]},
    ),
    RepairTest(
        ["B-ORG", "B-PER", "I-PER"],
        2,
        {"conlleval": ["I-ORG", "I-PER", "I-PER"]},
    ),
    RepairTest(
        ["I-ORG", "B-PER", "I-PER"],
        1,
        {"conlleval": ["I-ORG", "I-PER", "I-PER"]},
    ),
    RepairTest(
        ["O", "I-ORG", "B-PER", "I-ORG"],
        1,
        {"conlleval": ["O", "I-ORG", "I-PER", "I-ORG"]},
    ),
    RepairTest(
        ["O", "B-ORG", "B-PER", "I-PER"],
        2,
        {
            "conlleval": ["O", "I-ORG", "I-PER", "I-PER"],
        },
    ),
    RepairTest(
        ["O", "B-ORG", "B-ORG", "I-PER"],
        1,
        {
            "conlleval": ["O", "I-ORG", "B-ORG", "I-PER"],
        },
    ),
]

REPAIRS = {
    "BIO": BIO_REPAIRS,
    "IOB": IOB_REPAIRS,
}


def test_repair() -> None:
    for encoding_name, repairs in REPAIRS.items():
        encoding = get_encoding(encoding_name)

        # Invalid repair method name
        with pytest.raises(ValueError):
            encoding.repair_labels(["O"], "unk")
        assert "unk" not in encoding.supported_repair_methods()

        for case in repairs:
            result = validate_labels(case.original_labels, encoding)
            assert len(result) == case.n_errors
            if case.n_errors:
                assert not result.is_valid()

            for method, repaired in case.repaired_labels.items():
                assert encoding.repair_labels(case.original_labels, method) == repaired

                # Check that using no repair method raises an error
                with pytest.raises(ValueError):
                    encoding.repair_labels(case.original_labels, REPAIR_NONE)


def test_validation_invalid_state() -> None:
    encoding = get_encoding("BIO")

    result = validate_labels(["O", "S-PER"], encoding)
    assert not result.is_valid()

    with pytest.raises(EncodingError):
        validate_labels(["OUTSIDE", "B-PER"], encoding)

    with pytest.raises(EncodingError):
        validate_labels(["O", "PER"], encoding)


def test_validation_errors() -> None:
    encoding = get_encoding("BIO")

    tokens = ["Dr.", "Jonas", "Salk"]
    line_nums = [7, 8, 9]
    labels = ["O", "I-PER", "I-PER"]
    result = validate_labels(labels, encoding, tokens=tokens, line_nums=line_nums)
    assert not result.is_valid()
    assert len(result) == 1
    assert (
        result.errors[0].msg
        == "Invalid transition 'O' -> 'I-PER' for token 'Jonas' on line 8"
    )

    tokens = ["foo"]
    line_nums = [7]
    labels = ["S-FOO"]
    result = validate_labels(labels, encoding, tokens=tokens, line_nums=line_nums)
    assert not result.is_valid()
    assert len(result) == 3
    assert (
        result.errors[0].msg
        == "Invalid state 'S' in label 'S-FOO' for token 'foo' on line 7"
    )
    assert (
        result.errors[1].msg
        == "Invalid transition 'O' -> 'S-FOO' for token 'foo' on line 7"
    )
    assert (
        result.errors[2].msg
        == "Invalid transition 'S-FOO' -> 'O' after token 'foo' on line 7 at end of sequence"
    )
