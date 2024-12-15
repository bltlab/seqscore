import tempfile
from pathlib import Path
from typing import Optional

from click.testing import CliRunner

from seqscore.scripts.seqscore import process
from seqscore.util import file_fields_match

TMP_DIR: Optional[tempfile.TemporaryDirectory] = None
ANNOTATION_DIR = Path("tests", "conll_annotation")
TEST_FILES_DIR = Path("tests", "test_files")


def setup_module() -> None:
    """Create temporary directory used by tests."""
    global TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module() -> None:
    """Remove temporary directory used by tests."""
    TMP_DIR.cleanup()


def test_keep_types1() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--keep-types",
            "ORG",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Output will not have LOC since ORG was kept
    assert file_fields_match(TEST_FILES_DIR / "minimal_no_LOC.bio", output_path)


def test_keep_types2() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--keep-types",
            "LOC,ORG",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Input should be unchanged since all types were kept
    assert file_fields_match(input_path, output_path)


def test_remove_types1() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--remove-types",
            "LOC",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Output will not have LOC
    assert file_fields_match(TEST_FILES_DIR / "minimal_no_LOC.bio", output_path)


def test_remove_types2() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--remove-types",
            "MISC",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Input should be unchanged since MISC isn't in the data
    assert file_fields_match(input_path, output_path)


def test_remove_types3() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--remove-types",
            "LOC,ORG",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Output won't have any names since all types were removed
    assert file_fields_match(TEST_FILES_DIR / "minimal_no_names.bio", output_path)


def test_map_types1() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_LOC_GPE.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Output will have GPE instead of LOC
    assert file_fields_match(TEST_FILES_DIR / "minimal_GPE.bio", output_path)


def test_map_types2() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_NAME.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # All types will be NAME in output
    assert file_fields_match(TEST_FILES_DIR / "minimal_NAME.bio", output_path)


def test_map_types3() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_PERSON.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # Input will be unchanged since map doesn't affect LOC and ORG
    assert file_fields_match(input_path, output_path)


# TODO: Add a test for map_types with keep-types
# TODO: Add a test for map_types with remove-types


def test_map_types_invalid_map() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_bad_value.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    # Malformed map, dictionary value is a string and not a list
    assert result.exit_code != 0


def test_map_types_duplicate_mapping() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_bad_duplicate.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    # Malformed map, dictionary value is a string and not a list
    assert result.exit_code != 0


def test_keep_and_remove_types() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--keep-types",
            "LOC,ORG",
            "--remove-types",
            "MISC",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    # Can't specify both keep and remove
    assert result.exit_code != 0
