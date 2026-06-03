import os
import tempfile
from typing import Optional

from click.testing import CliRunner

from seqscore.scripts.seqscore import convert
from seqscore.util import file_fields_match, file_lines_match

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_invalid_conversion_BIO() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "temp.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            output_path,
        ],
    )
    assert result.exit_code != 0


def test_invalid_conversion_BIOES() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "temp.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "invalid1.bioes"),
            output_path,
        ],
    )
    assert result.exit_code != 0


def test_BIO_to_BIOES() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoBIOES.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bioes"),
    )


def test_BIOES_to_BIO() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOEStoBIO.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_BIO_to_IO() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoIO.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "IO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.io"),
    )


def test_IO_to_BIO() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "IOtoBIO.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIO correctly but does in this case
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_BIO_to_IOB_fields() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoIOB.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "IOB",
            os.path.join("tests", "conll_annotation", "minimal_fields.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal_fields.iob"),
    )


def test_IOB_to_BIO_fields() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "IOBtoBIO.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IOB",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal_fields.iob"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal_fields.bio"),
    )


def test_IOB_to_BIO_fields_and_specified_indices() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "labels_not_last_col.bioes")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            "--label-index",
            "1",
            os.path.join("tests", "conll_annotation", "labels_not_last_col.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "labels_not_last_col.bioes"),
    )


def test_IO_to_BIOES() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "IOtoBIOES.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIOES correctly but does in this case
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bioes"),
    )


def test_BIOES_to_IO() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOEStoIO.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "IO",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.io"),
    )


def test_BIO_to_BIO_space_delim() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoBIO_space.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIO",
            "--output-delim",
            " ",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        output_path,
        os.path.join("tests", "test_files", "minimal_space_delim.txt"),
    )


def test_BIO_to_BIO_tab_spelled_out() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoBIO_tab_spelled_out.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIO",
            "--output-delim",
            "tab",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_BIO_to_BIO_tab_backslash_t() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "BIOtoBIO_tab_backslash_t.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIO",
            "--output-delim",
            "\\t",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        output_path,
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_diff_token_label_indices() -> None:
    runner = CliRunner()
    output_path = os.path.join(TMP_DIR.name, "diff_token_label_indices_BIOES.txt")
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            "--token-index",
            "1",
            "--label-index",
            "2",
            os.path.join("tests", "conll_annotation", "diff_token_label_indices.bio"),
            output_path,
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        output_path,
        os.path.join("tests", "conll_annotation", "diff_token_label_indices.bioes"),
    )
