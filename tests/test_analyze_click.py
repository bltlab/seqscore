import os
import tempfile
from typing import Optional

from click.testing import CliRunner

from seqscore.output import SUPPORTED_OUTPUT_FORMATS
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
            "--output-format",
            "delim",
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


def test_analyze_pretty_output() -> None:
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            "--output-format",
            "pretty",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    # GitHub table with the Mention/Type/Span/Sentence header, in insertion order.
    # Span is numeric so it is right-justified; the other columns are left-justified.
    assert (
        result.output
        == """| Mention                    | Type   |   Span | Sentence                                                            |
|----------------------------|--------|--------|---------------------------------------------------------------------|
| University of Pennsylvania | ORG    |    0-3 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| West Philadelphia          | LOC    |    5-7 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| Pennsylvania               | LOC    |    8-9 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
"""
    )


def test_analyze_default_stdout_pretty() -> None:
    # With no --output-file and no explicit format, stdout defaults to a table.
    # Span is numeric so it is right-justified; other columns are left-justified.
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
    assert (
        result.output
        == """| Mention                    | Type   |   Span | Sentence                                                            |
|----------------------------|--------|--------|---------------------------------------------------------------------|
| University of Pennsylvania | ORG    |    0-3 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| West Philadelphia          | LOC    |    5-7 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| Pennsylvania               | LOC    |    8-9 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
"""
    )


def test_analyze_default_file_delim() -> None:
    # With --output-file and no explicit format, output defaults to delimited
    assert TMP_DIR is not None
    out_path = os.path.join(TMP_DIR.name, "analyze_default_file_delim.txt")
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            "--output-file",
            out_path,
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    with open(out_path, encoding="utf8") as output:
        assert (
            output.read()
            == """Mention\tType\tSpan\tSentence
University of Pennsylvania\tORG\t0-3\tUniversity of Pennsylvania is in West Philadelphia , Pennsylvania .
West Philadelphia\tLOC\t5-7\tUniversity of Pennsylvania is in West Philadelphia , Pennsylvania .
Pennsylvania\tLOC\t8-9\tUniversity of Pennsylvania is in West Philadelphia , Pennsylvania .
"""
        )


def test_analyze_pretty_output_file() -> None:
    # --output-format pretty writes the table (not delimited) to the file
    assert TMP_DIR is not None
    out_path = os.path.join(TMP_DIR.name, "analyze_pretty_out.txt")
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            "--output-format",
            "pretty",
            "--output-file",
            out_path,
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    with open(out_path, encoding="utf8") as output:
        assert (
            output.read()
            == """| Mention                    | Type   |   Span | Sentence                                                            |
|----------------------------|--------|--------|---------------------------------------------------------------------|
| University of Pennsylvania | ORG    |    0-3 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| West Philadelphia          | LOC    |    5-7 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
| Pennsylvania               | LOC    |    8-9 | University of Pennsylvania is in West Philadelphia , Pennsylvania . |
"""
        )


def test_analyze_output_format_help() -> None:
    runner = CliRunner()
    result = runner.invoke(analyze, ["--help"])
    assert "--output-format" in result.output
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        assert fmt in result.output


def test_analyze_with_repair_method() -> None:
    # After repair, invalid1.bio analyzes the same as minimal.bio.
    runner = CliRunner()
    out_path = os.path.join(TMP_DIR.name, "analyze_repaired.txt")
    result = runner.invoke(
        analyze,
        [
            "--labels",
            "BIO",
            "--output-file",
            out_path,
            "--repair-method",
            "conlleval",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Used method conlleval to repair:" in result.stderr
    assert file_lines_match(
        out_path,
        os.path.join("tests", "test_files", "analyze_minimal_ref.tsv"),
    )
