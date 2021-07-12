from click.testing import CliRunner

from seqscore.scripts.seqscore import dump
from seqscore.util import dump_files_match


def test_dump_BIO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIO",
            "tests/conll_annotation/minimal.bio",
            "tests/dump_BIO_out.txt",
        ],
    )
    assert result.exit_code == 0
    assert dump_files_match("tests/dump_BIO_out.txt", "tests/dump_minimal_ref.txt")


def test_dump_BIOES() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIOES",
            "--repair-method",
            "none",
            "tests/conll_annotation/minimal.bioes",
            "tests/dump_BIOES_out.txt",
        ],
    )
    assert result.exit_code == 0
    assert dump_files_match("tests/dump_BIOES_out.txt", "tests/dump_minimal_ref.txt")


def test_dump_IO() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "IO",
            "--repair-method",
            "none",
            "tests/conll_annotation/minimal.io",
            "tests/dump_IO_out.txt",
        ],
    )
    assert result.exit_code == 0
    assert dump_files_match("tests/dump_IO_out.txt", "tests/dump_minimal_ref.txt")


def test_dump_BIO_invalid_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dump,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "conlleval",
            "tests/conll_annotation/invalid1.bio",
            "tests/dump_BIO_conlleval_out.txt",
        ],
    )
    assert result.exit_code == 0
    assert dump_files_match(
        "tests/dump_BIO_conlleval_out.txt", "tests/dump_minimal_ref.txt"
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
            "tests/conll_annotation/invalid1.bio",
            "tests/dump_BIO_discard_out.txt",
        ],
    )
    assert result.exit_code == 0
    # all entities have invalid label sequences
    with open("tests/dump_BIO_discard_out.txt", encoding="utf8") as output:
        assert not output.readlines()
