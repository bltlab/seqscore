import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import count
from seqscore.util import file_lines_match


def test_count_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        count,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "count_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "count_BIO_out.txt"),
        os.path.join("tests", "test_files", "count_minimal_ref.txt"),
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
            os.path.join("tests", "count_BIOES_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "count_BIOES_out.txt"),
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
            os.path.join("tests", "count_IO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "count_IO_out.txt"),
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
            os.path.join("tests", "count_BIO_conlleval_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "count_BIO_conlleval_out.txt"),
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
            os.path.join("tests", "count_BIO_discard_out.txt"),
        ],
    )
    assert result.exit_code == 0
    # all entities have invalid label sequences
    with open(
        os.path.join("tests", "count_BIO_discard_out.txt"), encoding="utf8"
    ) as output:
        assert not output.readlines()
