import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import summarize


def test_summarize_bio_onedoc() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """File 'tests/conll_annotation/minimal.bio' contains 1 document(s) and 2 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
"""
    )


def test_summarize_bio_onedoc_quiet() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--quiet",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
"""
    )


def test_summarize_iob_twodoc() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "IOB",
            os.path.join("tests", "conll_annotation", "minimal_fields.iob"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """File 'tests/conll_annotation/minimal_fields.iob' contains 2 document(s) and 2 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
"""
    )


def test_summarize_bio_twofiles() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_annotation", "minimal2.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """File 'tests/conll_annotation/minimal.bio' contains 1 document(s) and 2 sentences
File 'tests/conll_annotation/minimal2.bio' contains 1 document(s) and 2 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       5 |
| ORG           |       2 |
"""
    )
