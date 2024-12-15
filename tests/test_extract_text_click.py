import tempfile
from pathlib import Path
from typing import List, Optional, Union

from click.testing import CliRunner

from seqscore.scripts.seqscore import extract_text

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None
MINIMAL_SENTENCES = [
    ["This", "is", "a", "sentence", "."],
    [
        "University",
        "of",
        "Pennsylvania",
        "is",
        "in",
        "West",
        "Philadelphia",
        ",",
        "Pennsylvania",
        ".",
    ],
]


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_single_file() -> None:
    runner = CliRunner()
    input_path = str(Path("tests") / "conll_annotation" / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.txt")
    result = runner.invoke(
        extract_text,
        [
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    actual_text = _read_tokenized_lines(output_path)
    assert actual_text == MINIMAL_SENTENCES


def test_multiple_files() -> None:
    runner = CliRunner()
    input_path = str(Path("tests") / "conll_annotation" / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.txt")
    result = runner.invoke(
        extract_text,
        [
            input_path,
            input_path,  # Put it again as a second file
            output_path,
        ],
    )
    assert result.exit_code == 0
    # It's the same sentences, but with a blank line between them for the document break
    expected_text = MINIMAL_SENTENCES[:]
    expected_text.append([""])
    expected_text.extend(MINIMAL_SENTENCES)

    actual_text = _read_tokenized_lines(output_path)
    assert actual_text == expected_text


def _read_tokenized_lines(path: Union[str, Path]) -> List[List[str]]:
    return [line.rstrip("\n").split(" ") for line in open(path, encoding="utf8")]
