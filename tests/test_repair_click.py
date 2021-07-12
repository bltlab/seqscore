from click.testing import CliRunner

from seqscore.scripts.seqscore import repair
from seqscore.util import files_match


def test_repair_BIO_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "tests/conll_annotation/invalid1.bio",
            "tests/invalid_BIO_repaired_conlleval.txt",
        ],
    )
    assert result.exit_code == 0
    assert (
        "Validation errors in sequence at line 7 of tests/conll_annotation/invalid1.bio:"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-ORG' for token 'University' on line 7"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-LOC' for token 'West' on line 12" in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-LOC' for token 'Pennsylvania' on line 15"
        in result.output
    )
    assert "Used method conlleval to repair:" in result.output
    assert (
        "Old: ('I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'I-LOC', 'I-LOC', 'O', 'I-LOC', 'O')"
        in result.output
    )
    assert (
        "New: ('B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert files_match(
        "tests/invalid_BIO_repaired_conlleval.txt", "tests/conll_annotation/minimal.bio"
    )


def test_repair_BIO_discard() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "discard",
            "tests/conll_annotation/invalid1.bio",
            "tests/invalid_BIO_repaired_discard.txt",
        ],
    )
    assert result.exit_code == 0
    assert (
        "Validation errors in sequence at line 7 of tests/conll_annotation/invalid1.bio:"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-ORG' for token 'University' on line 7"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-LOC' for token 'West' on line 12" in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-LOC' for token 'Pennsylvania' on line 15"
        in result.output
    )
    assert "Used method discard to repair:" in result.output
    assert (
        "Old: ('I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'I-LOC', 'I-LOC', 'O', 'I-LOC', 'O')"
        in result.output
    )
    assert "New: ('O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O')" in result.output
    assert files_match(
        "tests/invalid_BIO_repaired_discard.txt",
        "tests/conll_annotation/invalid1_BIO_discard.txt",
    )


def test_invalid_label() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair, ["--labels", "BIO", "tests/conll_annotation/invalid1.bioes", "temp.txt"]
    )
    assert result.exit_code != 0


def test_repair_none_raises_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "none",
            "tests/conll_annotation/invalid1.bio",
            "temp.txt",
        ],
    )
    assert result.exit_code != 0
