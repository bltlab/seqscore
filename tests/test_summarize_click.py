import os

from click.testing import CliRunner

from seqscore.output import SUPPORTED_OUTPUT_FORMATS
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
| TOTAL         |       3 |
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
| TOTAL         |       3 |
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
| TOTAL         |       3 |
"""
    )


def test_summarize_iob_twodoc_ignore_doc_boundaries() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "IOB",
            "--ignore-document-boundaries",
            os.path.join("tests", "conll_annotation", "minimal_fields.iob"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """File 'tests/conll_annotation/minimal_fields.iob' contains 1 document(s) and 2 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
| TOTAL         |       3 |
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
Total 2 document(s) and 4 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       5 |
| ORG           |       2 |
| TOTAL         |       7 |
"""
    )


def test_summarize_delim_output_quiet() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--quiet",
            "--output-format",
            "delim",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """Entity Type\tCount
LOC\t2
ORG\t1
TOTAL\t3
"""
    )


def test_summarize_delim_output_info_lines() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--output-format",
            "delim",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    # Without --quiet, the info line precedes the delimited table
    assert (
        result.output
        == """File 'tests/conll_annotation/minimal.bio' contains 1 document(s) and 2 sentences
Entity Type\tCount
LOC\t2
ORG\t1
TOTAL\t3
"""
    )


def test_summarize_table_format_plain() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--table-format",
            "plain",
            "--quiet",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """Entity Type      Count
LOC                  2
ORG                  1
TOTAL                3
"""
    )


def test_summarize_output_format_help() -> None:
    runner = CliRunner()
    result = runner.invoke(summarize, ["--help"])
    assert "--output-format" in result.output
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        assert fmt in result.output


def test_summarize_invalid_table_format_rejected() -> None:
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--table-format",
            "not_a_real_format",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--table-format'" in result.output


def test_summarize_file_delim_default() -> None:
    # --output-file with no explicit format defaults to delimited; the info line
    # stays on stdout so the file holds only report data
    runner = CliRunner()
    input_path = os.path.abspath(os.path.join("tests", "conll_annotation", "minimal.bio"))
    with runner.isolated_filesystem():
        result = runner.invoke(
            summarize,
            ["--labels", "BIO", "--output-file", "out.tsv", input_path],
        )
        assert result.exit_code == 0
        assert "contains 1 document(s) and 2 sentences" in result.output
        with open("out.tsv", encoding="utf8") as output:
            assert (
                output.read()
                == """Entity Type\tCount
LOC\t2
ORG\t1
TOTAL\t3
"""
            )


def test_summarize_file_pretty() -> None:
    # --output-format pretty writes the table (not delimited) to the file
    runner = CliRunner()
    input_path = os.path.abspath(os.path.join("tests", "conll_annotation", "minimal.bio"))
    with runner.isolated_filesystem():
        result = runner.invoke(
            summarize,
            [
                "--labels",
                "BIO",
                "--output-format",
                "pretty",
                "--output-file",
                "out.txt",
                input_path,
            ],
        )
        assert result.exit_code == 0
        with open("out.txt", encoding="utf8") as output:
            assert (
                output.read()
                == """| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
| TOTAL         |       3 |
"""
            )


def test_summarize_comma_delim_warning() -> None:
    # The non-tab delimiter warning fires when delimited output is actually used
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--output-format",
            "delim",
            "--output-delim",
            ",",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Warning" in result.stderr or "not recommended" in result.stderr


def test_summarize_with_repair_method() -> None:
    # After repair, invalid1.bio summarizes the same as minimal.bio.
    runner = CliRunner()
    result = runner.invoke(
        summarize,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "conlleval",
            os.path.join("tests", "conll_annotation", "invalid1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Used method conlleval to repair:" in result.stderr
    assert (
        result.stdout
        == """File 'tests/conll_annotation/invalid1.bio' contains 1 document(s) and 2 sentences
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
| TOTAL         |       3 |
"""
    )
