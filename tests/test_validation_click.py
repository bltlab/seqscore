import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import validate
from seqscore.util import normalize_str_with_path


def test_valid_bio() -> None:
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--labels", "BIO", os.path.join("tests", "conll_annotation", "minimal.bio")],
    )
    assert (
        result.output
        == "No errors found in 15 tokens, 2 sequences, and 1 document(s) in tests/conll_annotation/minimal.bio\n"
    )
    assert result.exit_code == 0


def test_valid_bioes() -> None:
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--labels", "BIOES", os.path.join("tests", "conll_annotation", "minimal.bioes")],
    )
    assert (
        result.output
        == "No errors found in 15 tokens, 2 sequences, and 1 document(s) in tests/conll_annotation/minimal.bioes\n"
    )
    assert result.exit_code == 0


def test_invalid_bio() -> None:
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--labels", "BIO", os.path.join("tests", "conll_annotation", "invalid1.bio")],
    )
    assert result.exit_code != 0
    assert (
        normalize_str_with_path(
            "Encountered 3 errors in 15 tokens, 2 sequences, and 1 document(s) in tests/conll_annotation/invalid1.bio"
        )
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


def test_invalid_bioes() -> None:
    runner = CliRunner()
    result = runner.invoke(
        validate,
        [
            "--labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "invalid1.bioes"),
        ],
    )
    assert result.exit_code != 0
    assert (
        normalize_str_with_path(
            "Encountered 9 errors in 25 tokens, 6 sequences, and 1 document(s) in tests/conll_annotation/invalid1.bioes"
        )
        in result.output
    )
    assert "Invalid transition 'I-ORG' -> 'O' for token 'is' on line 10" in result.output
    assert (
        "Invalid transition 'S-LOC' -> 'I-LOC' for token 'Philadelphia' on line 13"
        in result.output
    )
    assert "Invalid transition 'I-LOC' -> 'O' for token ',' on line 14" in result.output
    assert "Invalid transition 'B-LOC' -> 'O' for token '.' on line 16" in result.output
    assert (
        "Invalid transition 'I-ORG' -> 'O' after token 'Maryland' on line 20"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-ORG' for token 'Department' on line 22"
        in result.output
    )
    assert (
        "Invalid transition 'O' -> 'I-ORG' for token 'University' on line 26"
        in result.output
    )
    assert (
        "Invalid transition 'I-ORG' -> 'O' after token 'Maryland' on line 28"
        in result.output
    )
    assert (
        "Invalid transition 'B-LOC' -> 'O' after token 'Massachusetts' on line 30"
        in result.output
    )
