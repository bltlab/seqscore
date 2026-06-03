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


def test_map_types_remove_types() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_LOC_GPE.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--remove-types",
            "LOC",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # LOC will be mapped to GPE since mapping applies before removal
    assert file_fields_match(TEST_FILES_DIR / "minimal_GPE.bio", output_path)


def test_map_types_keep_types() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_LOC_GPE.json")
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            map_path,
            "--keep-types",
            "LOC",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 0
    # No names since LOC will be mapped to GPE and only LOC will be kept
    assert file_fields_match(TEST_FILES_DIR / "minimal_no_names.bio", output_path)


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
    assert result.exit_code == 2
    assert "Cannot specify both keep-types and remove-types" in result.output


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
    assert result.exit_code == 2
    assert (
        "Value 'LOC' in type map 'tests/test_files/map_bad_value.json' is not a list"
        in result.output
    )


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
    assert result.exit_code == 2
    assert "Multiple mappings specified for type 'LOC' in type map" in result.output


def test_no_operation() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 2
    assert (
        "Must specify at least one of keep-types, remove-types, or type-map"
        in result.output
    )


def test_keep_outside_type() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--keep-types",
            "O",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 2
    assert "Cannot specify the outside type O in keep/remove types" in result.output


def test_remove_outside_type() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--remove-types",
            "O",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 2
    assert "Cannot specify the outside type O in keep/remove types" in result.output


def test_type_map_missing_file() -> None:
    runner = CliRunner()
    input_path = str(ANNOTATION_DIR / "minimal.bio")
    output_path = str(Path(TMP_DIR.name) / "out.bio")
    result = runner.invoke(
        process,
        [
            "--type-map",
            "nonexistent_map.json",
            "--labels",
            "BIO",
            input_path,
            output_path,
        ],
    )
    assert result.exit_code == 2
    assert "Could not open type map file 'nonexistent_map.json'" in result.output


def test_type_map_invalid_json() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_invalid_json.json")
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
    assert result.exit_code == 2
    assert (
        "Type map provided in file 'tests/test_files/map_invalid_json.json' is not valid JSON"
        in result.output
    )


def test_type_map_not_dict() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_not_dict.json")
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
    assert result.exit_code == 2
    assert (
        "Type map provided in file 'tests/test_files/map_not_dict.json' is not a dictionary"
        in result.output
    )


def test_type_map_empty_key() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_empty_key.json")
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
    assert result.exit_code == 2
    assert (
        "Key '' in type map 'tests/test_files/map_empty_key.json' is not a non-empty string"
        in result.output
    )


def test_type_map_outside_key() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_outside_key.json")
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
    assert result.exit_code == 2
    assert (
        "Key 'O' in type map 'tests/test_files/map_outside_key.json' is the outside type O"
        in result.output
    )


def test_type_map_empty_value() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_empty_value.json")
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
    assert result.exit_code == 2
    assert (
        "Value '' in type map 'tests/test_files/map_empty_value.json' is not a non-empty string"
        in result.output
    )


def test_type_map_outside_value() -> None:
    runner = CliRunner()
    map_path = str(TEST_FILES_DIR / "map_outside_value.json")
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
    assert result.exit_code == 2
    assert (
        "Value 'O' in type map 'tests/test_files/map_outside_value.json' is the outside type O"
        in result.output
    )
