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
            "tests/conll_annotation/invalid1.bio",
            "tests/temp.txt",
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
            "tests/conll_annotation/invalid1.bioes",
            "tests/temp.txt",
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
            "tests/conll_annotation/minimal.bio",
            "tests/BIOtoBIOES.txt",
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match(
        "tests/BIOtoBIOES.txt", "tests/conll_annotation/minimal.bioes"
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
            "tests/conll_annotation/minimal.bioes",
            "tests/BIOEStoBIO.txt",
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match("tests/BIOEStoBIO.txt", "tests/conll_annotation/minimal.bio")


def test_BIO_to_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "IO",
            "tests/conll_annotation/minimal.bio",
            "tests/BIOtoIO.txt",
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match("tests/BIOtoIO.txt", "tests/conll_annotation/minimal.io")


def test_IO_to_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIO",
            "tests/conll_annotation/minimal.io",
            "tests/IOtoBIO.txt",
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIO correctly but does in this case
    assert file_fields_match("tests/IOtoBIO.txt", "tests/conll_annotation/minimal.bio")


def test_IO_to_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "IO",
            "--output-labels",
            "BIOES",
            "tests/conll_annotation/minimal.io",
            "tests/IOtoBIOES.txt",
        ],
    )
    assert result.exit_code == 0
    # conversion will not necessarily reproduce BIOES correctly but does in this case
    assert file_fields_match(
        "tests/IOtoBIOES.txt", "tests/conll_annotation/minimal.bioes"
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
            "tests/conll_annotation/minimal.bioes",
            "tests/BIOEStoIO.txt",
        ],
    )
    assert result.exit_code == 0
    assert file_fields_match("tests/BIOEStoIO.txt", "tests/conll_annotation/minimal.io")


def test_same_input_and_output_labels_raises_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        convert,
        [
            "--input-labels",
            "BIO",
            "--output-labels",
            "BIO",
            "tests/conll_annotation/minimal.bio",
            "temp.txt",
        ],
    )
    assert result.exit_code != 0
