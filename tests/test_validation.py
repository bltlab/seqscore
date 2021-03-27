from typing import Dict, List

import pytest
from attr import attrs

from seqscore.encoding import REPAIR_NONE, SUPPORTED_REPAIR_METHODS, get_encoding
from seqscore.validation import validate_labels


@attrs(auto_attribs=True)
class RepairTest:
    original_labels: List[str]
    n_errors: int
    repaired_labels: Dict[str, List[str]]


BIO_REPAIRS = [
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


def test_repair() -> None:
    encoding = get_encoding("BIO")
    for case in BIO_REPAIRS:
        result = validate_labels(case.original_labels, encoding)
        assert len(result) == case.n_errors
        if case.n_errors:
            assert not result.is_valid()

        for method in SUPPORTED_REPAIR_METHODS:
            if method == REPAIR_NONE:
                # Cannot repair with method none
                with pytest.raises(ValueError):
                    encoding.repair_labels(case.original_labels, method)
            else:
                assert (
                    encoding.repair_labels(case.original_labels, method)
                    == case.repaired_labels[method]
                )

    # Invalid repair method name
    with pytest.raises(ValueError):
        encoding.repair_labels(["O"], "unk")


def test_validation_invalid_state() -> None:
    encoding = get_encoding("BIO")
    result = validate_labels(["O", "S-PER"], encoding)
    assert not result.is_valid()


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
        == "Invalid transition O -> I-PER for token 'Jonas' on line 8"
    )

    tokens = ["foo"]
    line_nums = [7]
    labels = ["S-FOO"]
    result = validate_labels(labels, encoding, tokens=tokens, line_nums=line_nums)
    assert not result.is_valid()
    assert len(result) == 3
    assert (
        result.errors[0].msg == "Invalid state S in label S-FOO for token 'foo' on line 7"
    )
    assert (
        result.errors[1].msg == "Invalid transition O -> S-FOO for token 'foo' on line 7"
    )
    assert (
        result.errors[2].msg
        == "Invalid transition S-FOO -> O after token 'foo' on line 7 at end of sequence"
    )
