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
        == """File 'tests/conll_annotation/minimal.bio' contains 1 document(s) with the following mentions:
| Entity Type   |   Count |
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
        == """File 'tests/conll_annotation/minimal_fields.iob' contains 2 document(s) with the following mentions:
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
"""
    )
