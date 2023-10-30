import os

from seqscore.util import file_fields_match, file_lines_match, tuplify_strs


def test_tuplify_strs() -> None:
    strs = ["a", "b", "c"]
    tup = tuplify_strs(strs)
    assert tup == ("a", "b", "c")


def test_identical_files() -> None:
    assert file_fields_match(
        os.path.join("tests", "test_files", "minimal_bio_copy.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )
    assert file_lines_match(
        os.path.join("tests", "test_files", "minimal_bio_copy.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_empty_file() -> None:
    assert not file_fields_match(
        os.path.join("tests", "test_files", "empty.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )
    assert not file_lines_match(
        os.path.join("tests", "test_files", "empty.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_differing_whitespace() -> None:
    assert file_fields_match(
        os.path.join("tests", "test_files", "space_delim.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )
    assert not file_lines_match(
        os.path.join("tests", "test_files", "space_delim.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )


def test_differing_file_fields() -> None:
    assert not file_fields_match(
        os.path.join("tests", "conll_annotation", "minimal.bio"),
        os.path.join("tests", "conll_annotation", "invalid1.bio"),
    )
    assert not file_lines_match(
        os.path.join("tests", "conll_annotation", "minimal.bio"),
        os.path.join("tests", "conll_annotation", "invalid1.bio"),
    )


def test_extra_line() -> None:
    assert not file_fields_match(
        os.path.join("tests", "test_files", "minimal_bio_extra_line.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )
    assert not file_lines_match(
        os.path.join("tests", "test_files", "minimal_bio_extra_line.txt"),
        os.path.join("tests", "conll_annotation", "minimal.bio"),
    )
