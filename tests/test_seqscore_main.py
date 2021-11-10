import subprocess

HELP_OUTPUT = """Usage: seqscore [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  convert
  count
  repair
  score
  validate
"""


def test_seqscore_help() -> None:
    result = subprocess.run(["seqscore", "--help"], capture_output=True, encoding="UTF-8")
    assert result.returncode == 0
    assert result.stdout == HELP_OUTPUT
