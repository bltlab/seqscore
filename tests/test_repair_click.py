import os
import tempfile
from typing import Optional

from click.testing import CliRunner

from seqscore.scripts.seqscore import repair
from seqscore.util import file_fields_match, normalize_str_with_path

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_repair_BIO_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--repair-method",
            "conlleval",
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join(TMP_DIR.name, "invalid_BIO_repaired_conlleval.txt"),
        ],
    )
    assert result.exit_code == 0
    assert (
        normalize_str_with_path(
            "Validation errors in sequence beginning at line 7 of tests/conll_annotation/invalid1.bio:"
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
    assert "Used method conlleval to repair:" in result.output
    assert (
        "Old: ('I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'I-LOC', 'I-LOC', 'O', 'I-LOC', 'O')"
        in result.output
    )
    assert (
        "New: ('B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert file_fields_match(
        os.path.join(TMP_DIR.name, "invalid_BIO_repaired_conlleval.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
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
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join(TMP_DIR.name, "invalid_BIO_repaired_discard.txt"),
        ],
    )
    assert result.exit_code == 0
    assert (
        normalize_str_with_path(
            "Validation errors in sequence beginning at line 7 of tests/conll_annotation/invalid1.bio:"
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
    assert "Used method discard to repair:" in result.output
    assert (
        "Old: ('I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'I-LOC', 'I-LOC', 'O', 'I-LOC', 'O')"
        in result.output
    )
    assert "New: ('O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O')" in result.output
    assert file_fields_match(
        os.path.join(TMP_DIR.name, "invalid_BIO_repaired_discard.txt"),
        os.path.join("tests", "conll_annotation", "invalid1_BIO_discard.txt"),
    )


def test_repair_BIO_first_type() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "first-type",
            os.path.join("tests", "conll_annotation", "invalid2.bio"),
            os.path.join(TMP_DIR.name, "invalid_BIO_repaired_first_type.txt"),
        ],
    )
    assert result.exit_code == 0
    assert (
        normalize_str_with_path(
            "Validation errors in sequence beginning at line 7 of tests/conll_annotation/invalid2.bio:"
        )
        in result.output
    )
    assert (
        "Invalid transition 'I-ORG' -> 'I-LOC' for token 'Pennsylvania' on line 9"
        in result.output
    )
    assert "Used method first-type to repair:" in result.output
    assert (
        "Old: ('B-ORG', 'I-ORG', 'I-LOC', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert (
        "New: ('B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert file_fields_match(
        os.path.join(TMP_DIR.name, "invalid_BIO_repaired_first_type.txt"),
        os.path.join("tests", "conll_annotation", "invalid2_BIO_first_type.txt"),
    )


def test_repair_BIO_last_type() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "last-type",
            os.path.join("tests", "conll_annotation", "invalid2.bio"),
            os.path.join(TMP_DIR.name, "invalid_BIO_repaired_last_type.txt"),
        ],
    )
    assert result.exit_code == 0
    assert (
        normalize_str_with_path(
            "Validation errors in sequence beginning at line 7 of tests/conll_annotation/invalid2.bio:"
        )
        in result.output
    )
    assert (
        "Invalid transition 'I-ORG' -> 'I-LOC' for token 'Pennsylvania' on line 9"
        in result.output
    )
    assert "Used method last-type to repair:" in result.output
    assert (
        "Old: ('B-ORG', 'I-ORG', 'I-LOC', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert (
        "New: ('B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')"
        in result.output
    )
    assert file_fields_match(
        os.path.join(TMP_DIR.name, "invalid_BIO_repaired_last_type.txt"),
        os.path.join("tests", "conll_annotation", "invalid2_BIO_last_type.txt"),
    )


def test_invalid_label() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "conlleval",
            os.path.join("tests", "conll_annotation", "invalid1.bioes"),
            os.path.join(TMP_DIR.name, "temp.txt"),
        ],
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
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join(TMP_DIR.name, "temp.txt"),
        ],
    )
    assert result.exit_code != 0


def test_repair_iob_lenient_rejected() -> None:
    runner = CliRunner()
    result = runner.invoke(
        repair,
        [
            "--repair-method",
            "conlleval",
            "--labels",
            "IOB-lenient",
            os.path.join("tests", "conll_annotation", "minimal_lenient.iob"),
            os.path.join(TMP_DIR.name, "lenient_repaired.txt"),
        ],
    )
    assert result.exit_code == 2
    assert "'--labels'" in result.output
