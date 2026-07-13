from click.testing import CliRunner

import seqscore
from seqscore.scripts.seqscore import cli

HELP_OUTPUT = "Usage: seqscore [OPTIONS] COMMAND [ARGS]..."


def test_seqscore_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"], prog_name="seqscore")
    assert result.exit_code == 0
    assert result.output.startswith(HELP_OUTPUT)


def test_seqscore_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"], prog_name="seqscore")
    assert result.exit_code == 0
    assert result.output == f"seqscore, version {seqscore.__version__}\n"
