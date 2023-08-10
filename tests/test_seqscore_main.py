import subprocess

import seqscore

HELP_OUTPUT = "Usage: seqscore [OPTIONS] COMMAND [ARGS]..."


def test_seqscore_help() -> None:
    result = subprocess.run(["seqscore", "--help"], capture_output=True, encoding="UTF-8")
    assert result.returncode == 0
    assert result.stdout.startswith(HELP_OUTPUT)


def test_seqscore_version() -> None:
    result = subprocess.run(
        ["seqscore", "--version"], capture_output=True, encoding="UTF-8"
    )
    assert result.returncode == 0
    assert result.stdout == f"seqscore, version {seqscore.__version__}\n"
