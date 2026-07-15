import os
import tempfile
from typing import Optional

from click.testing import CliRunner

from seqscore.output import SUPPORTED_OUTPUT_FORMATS
from seqscore.scripts.seqscore import count
from seqscore.util import file_lines_match

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_count_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_stdout() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
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
        os.path.join("tests", "test_files", "count_minimal_ref.txt")
    ).read()
    assert result.stdout == expected_lines


def test_count_BIO_twofiles() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_annotation", "minimal2.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_twofiles_ref.txt"),
        debug=True,
    )


def test_count_BIO_tab1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-delim",
            "\t",  # Actual tab
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_tab2() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-delim",
            r"\t",  # Backlash and t
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_tab3() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-delim",
            "tab",  # Tab spelled out
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_comma() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-delim",
            ",",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref_comma.txt"),
    )


def test_count_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIOES",
            "--repair-method",
            "none",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIOES_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIOES_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "IO",
            "--repair-method",
            "none",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_IO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_IO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_invalid_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "conlleval",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_conlleval_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join(TMP_DIR.name, "count_BIO_conlleval_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
    )


def test_count_BIO_invalid_discard() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "discard",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            "--output-file",
            os.path.join(TMP_DIR.name, "count_BIO_discard_out.txt"),
        ],
    )
    assert result.exit_code == 0
    # all entities have invalid label sequences
    with open(
        os.path.join(TMP_DIR.name, "count_BIO_discard_out.txt"), encoding="utf8"
    ) as output:
        assert not output.readlines()


def test_count_pretty_output() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-format",
            "pretty",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    # Github table with a header, rows in most_common() order
    assert (
        result.output
        == """|   Count | Type   | Tokens                     |
|---------|--------|----------------------------|
|       1 | ORG    | University of Pennsylvania |
|       1 | LOC    | West Philadelphia          |
|       1 | LOC    | Pennsylvania               |
"""
    )


def test_count_default_stdout_pretty() -> None:
    # With no --output-file and no explicit format, stdout defaults to a table
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """|   Count | Type   | Tokens                     |
|---------|--------|----------------------------|
|       1 | ORG    | University of Pennsylvania |
|       1 | LOC    | West Philadelphia          |
|       1 | LOC    | Pennsylvania               |
"""
    )


def test_count_default_file_delim() -> None:
    # With --output-file and no explicit format, output defaults to delimited
    assert TMP_DIR is not None
    out_path = os.path.join(TMP_DIR.name, "count_default_file_delim.txt")
    runner = CliRunner()
    result = runner.invoke(
        count,
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
            == """1\tORG\tUniversity of Pennsylvania
1\tLOC\tWest Philadelphia
1\tLOC\tPennsylvania
"""
        )


def test_count_pretty_output_file() -> None:
    # --output-format pretty writes the table (not delimited) to the file
    assert TMP_DIR is not None
    out_path = os.path.join(TMP_DIR.name, "count_pretty_out.txt")
    runner = CliRunner()
    result = runner.invoke(
        count,
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
            == """|   Count | Type   | Tokens                     |
|---------|--------|----------------------------|
|       1 | ORG    | University of Pennsylvania |
|       1 | LOC    | West Philadelphia          |
|       1 | LOC    | Pennsylvania               |
"""
        )


def test_count_output_format_help() -> None:
    runner = CliRunner()
    result = runner.invoke(count, ["--help"])
    assert "--output-format" in result.output
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        assert fmt in result.output


def test_count_comma_delim_warning() -> None:
    # The non-tab delimiter warning fires when delimited output is actually used
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-format",
            "delim",
            "--output-delim",
            ",",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Warning" in result.stderr or "not recommended" in result.stderr


def test_count_no_delim_warning_when_pretty() -> None:
    # The delimiter is unused in a table, so no warning should be emitted
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            "--output-format",
            "pretty",
            "--output-delim",
            ",",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Warning" not in result.stderr
