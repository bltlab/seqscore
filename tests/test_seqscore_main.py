import subprocess

import seqscore

HELP_OUTPUT = """Usage: seqscore [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  convert
  count
  process
  repair
  score
  summarize
  validate
"""


def test_seqscore_help() -> None:
    result = subprocess.run(["seqscore", "--help"], capture_output=True, encoding="UTF-8")
    assert result.returncode == 0
    assert result.stdout == HELP_OUTPUT


def test_seqscore_version() -> None:
    result = subprocess.run(
        ["seqscore", "--version"], capture_output=True, encoding="UTF-8"
    )
    assert result.returncode == 0
    assert result.stdout == f"seqscore, version {seqscore.__version__}\n"
