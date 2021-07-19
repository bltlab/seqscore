import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import convert
from seqscore.util import file_fields_match


def test_invalid_conversion_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join("tests", "temp.txt"),
        ],
    )
    assert result.exit_code != 0


def test_invalid_conversion_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "invalid1.bioes"),
            os.path.join("tests", "temp.txt"),
        ],
    )
    assert result.exit_code != 0


def test_BIO_to_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "BIOtoBIOES.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        os.path.join("tests", "BIOtoBIOES.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bioes"),
    )


def test_BIOES_to_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            os.path.join("tests", "BIOEStoBIO.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        os.path.join("tests", "BIOEStoBIO.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_BIO_to_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "IO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "BIOtoIO.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        os.path.join("tests", "BIOtoIO.txt"),
        os.path.join("tests", "conll_annotation", "minimal.io"),
    )


def test_IO_to_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            os.path.join("tests", "IOtoBIO.txt"),
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIO correctly but does in this case
    assert file_fields_match(
        os.path.join("tests", "IOtoBIO.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_IO_to_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIOES",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            os.path.join("tests", "IOtoBIOES.txt"),
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIOES correctly but does in this case
    assert file_fields_match(
        os.path.join("tests", "IOtoBIOES.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bioes"),
    )


def test_BIOES_to_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIOES",
            "--output-labels",
            "IO",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            os.path.join("tests", "BIOEStoIO.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        os.path.join("tests", "BIOEStoIO.txt"),
        os.path.join("tests", "conll_annotation", "minimal.io"),
    )


def test_same_input_and_output_labels_raises_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "temp.txt",
        ],
    )
    assert result.exit_code != 0
