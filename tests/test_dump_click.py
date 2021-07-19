import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import dump
from seqscore.util import file_lines_match


def test_dump_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "dump_BIO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "dump_BIO_out.txt"),
        os.path.join("tests", "test_files", "dump_minimal_ref.txt"),
    )


def test_dump_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIOES",
            "--repair-method",
            "none",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            os.path.join("tests", "dump_BIOES_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "dump_BIOES_out.txt"),
        os.path.join("tests", "test_files", "dump_minimal_ref.txt"),
    )


def test_dump_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "IO",
            "--repair-method",
            "none",
            os.path.join("tests", "conll_annotation", "minimal.io"),
            os.path.join("tests", "dump_IO_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "dump_IO_out.txt"),
        os.path.join("tests", "test_files", "dump_minimal_ref.txt"),
    )


def test_dump_BIO_invalid_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "conlleval",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join("tests", "dump_BIO_conlleval_out.txt"),
        ],
    )
    assert result.exit_code == 0
    assert file_lines_match(
        os.path.join("tests", "dump_BIO_conlleval_out.txt"),
        os.path.join("tests", "test_files", "dump_minimal_ref.txt"),
    )


def test_dump_BIO_invalid_discard() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "discard",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
            os.path.join("tests", "dump_BIO_discard_out.txt"),
        ],
    )
    assert result.exit_code == 0
    # all entities have invalid label sequences
    with open(
        os.path.join("tests", "dump_BIO_discard_out.txt"), encoding="utf8"
    ) as output:
        assert not output.readlines()
