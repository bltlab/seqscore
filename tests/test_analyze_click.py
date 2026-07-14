import os
import tempfile
from typing import Optional

from click.testing import CliRunner

from seqscore.scripts.seqscore import analyze
from seqscore.util import file_lines_match

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_analyze_BIO_stdout() -> None:
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    expected_lines = open(
        os.path.join("tests", "test_files", "analyze_minimal_ref.tsv")
    ).read()
    assert result.stdout == expected_lines


def test_analyze_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "analyze_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "analyze_BIO_out.txt"),
        os.path.join("tests", "test_files", "analyze_minimal_ref.tsv"),
    )


def test_analyze_BIO_comma() -> None:
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            "--output-delim",
            ",",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "analyze_BIO_comma_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "analyze_BIO_comma_out.txt"),
        os.path.join("tests", "test_files", "analyze_minimal_ref.csv"),
    )
