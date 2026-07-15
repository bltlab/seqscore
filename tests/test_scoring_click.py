import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import analyze, count, score, summarize


def test_score_correct_labels() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t100.00\t100.00\t100.00\t3\t3\t3" in result.output
    assert "LOC\t100.00\t100.00\t100.00\t2\t2\t2" in result.output
    assert "ORG\t100.00\t100.00\t100.00\t1\t1\t1" in result.output


def test_score_no_predictions() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join("tests", "conll_predictions", "incorrect1_nopredictions.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t0.00\t0.00\t0.00\t3\t0\t0" in result.output
    assert "LOC\t0.00\t0.00\t0.00\t2\t0\t0" in result.output
    assert "ORG\t0.00\t0.00\t0.00\t1\t0\t0" in result.output


def test_score_incorrect_default_format() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        "| ALL    |       50.00 |    66.67 |  57.14 |           3 |           4 |         2 |"
        in result.output
    )
    assert (
        "| LOC    |       33.33 |    50.00 |  40.00 |           2 |           3 |         1 |"
        in result.output
    )
    assert (
        "| ORG    |      100.00 |   100.00 | 100.00 |           1 |           1 |         1 |"
        in result.output
    )


def test_score_incorrect_conlleval_format() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "conlleval",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        "processed 15 tokens with 3 phrases; found: 4 phrases; correct: 2."
        in result.output
    )
    assert (
        "accuracy:  93.33%; precision:  50.00%; recall:  66.67%; FB1:  57.14"
        in result.output
    )
    assert "LOC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3" in result.output
    assert "ORG: precision: 100.00%; recall: 100.00%; FB1: 100.00  1" in result.output


def test_score_invalid_sequence_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--repair-method",
            "conlleval",
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join(
                "tests", "conll_predictions", "correct1_improper_sequence_pred.txt"
            ),
        ],
    )
    assert result.exit_code == 0
    assert "Used method conlleval to repair:" in result.output
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t100.00\t100.00\t100.00\t3\t3\t3" in result.output
    assert "LOC\t100.00\t100.00\t100.00\t2\t2\t2" in result.output
    assert "ORG\t100.00\t100.00\t100.00\t1\t1\t1" in result.output


def test_score_invalid_sequence_discard() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "discard",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join(
                "tests", "conll_predictions", "correct1_improper_sequence_pred.txt"
            ),
        ],
    )
    assert result.exit_code == 0
    assert "Used method discard to repair:" in result.output
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t100.00\t66.67\t80.00\t3\t2\t2" in result.output
    assert "LOC\t100.00\t100.00\t100.00\t2\t2\t2" in result.output
    assert "ORG\t0.00\t0.00\t0.00\t1\t0\t0" in result.output


def test_score_invalid_sequence_none() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--repair-method",
            "none",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join(
                "tests", "conll_predictions", "correct1_improper_sequence_pred.txt"
            ),
        ],
    )
    assert result.exit_code == 1
    assert "Invalid transition 'O' -> 'I-ORG'" in str(result.exception)


def test_score_valid_incorrect_sequence() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--repair-method",
            "conlleval",
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t50.00\t66.67\t57.14\t3\t4\t2" in result.output
    assert "LOC\t33.33\t50.00\t40.00\t2\t3\t1" in result.output
    assert "ORG\t100.00\t100.00\t100.00\t1\t1\t1" in result.output


def test_score_entity_type_not_in_reference() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join(
                "tests", "conll_predictions", "incorrect_type_not_in_reference.bio"
            ),
        ],
    )
    assert result.exit_code == 0
    output_fields = [line.split("\t") for line in result.output.rstrip("\n").split("\n")]
    assert output_fields == [
        ["Type", "Precision", "Recall", "F1", "Reference", "Predicted", "Correct"],
        ["LOC", "100.00", "100.00", "100.00", "2", "2", "2"],
        ["ORG", "100.00", "100.00", "100.00", "1", "1", "1"],
        ["SPURIOUS", "0.00", "0.00", "0.00", "0", "1", "0"],
        ["ALL", "75.00", "100.00", "85.71", "3", "4", "3"],
    ]


def test_score_invalid_labels() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bioes"),
            "--output-format",
            "delim",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 1
    assert "The above labels are not valid for the chunk encoding BIO." in str(
        result.exception
    )


def test_score_multiple_files() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "SE\tALL\tNA\tNA\t21.43\tNA\tNA\tNA" in result.output
    assert "Mean\tALL\tNA\tNA\t78.57\tNA\tNA\tNA" in result.output


def test_score_multiple_files_pretty() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_predictions", "correct1.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert os.path.join("tests", "conll_predictions", "correct1.bio") in result.output
    assert os.path.join("tests", "conll_predictions", "incorrect1.bio") in result.output
    assert "Summary" in result.output
    assert "| ALL    |     78.57 | 21.43 |           3 |" in result.output
    assert "| LOC    |     70.00 | 30.00 |           2 |" in result.output
    assert "| ORG    |    100.00 |  0.00 |           1 |" in result.output


def test_score_error_counts_single_file() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--error-counts",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    # Ordering within the same count is deterministic and determined by
    # the token string
    assert (
        result.output
        == """|   Count | Error   | Type   | Tokens            |
|---------|---------|--------|-------------------|
|       1 | FP      | LOC    | Philadelphia      |
|       1 | FP      | LOC    | West              |
|       1 | FN      | LOC    | West Philadelphia |
"""
    )


def test_score_error_counts_delim_format() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "delim",
            "--error-counts",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert (
        result.output
        == """Count\tError\tType\tTokens
1\tFP\tLOC\tPhiladelphia
1\tFP\tLOC\tWest
1\tFN\tLOC\tWest Philadelphia
"""
    )


def test_score_error_counts_multiple_files() -> None:
    # Cannot use error-counts with multiple files
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_predictions", "correct1.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
            "--error-counts",
        ],
    )
    assert result.exit_code == 2
    assert "Cannot use error-counts with multiple files to be scored" in result.output


def test_score_full_precision_not_delim() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--full-precision",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
        ],
    )
    assert result.exit_code == 2
    assert "Can only use full-precision with output-format delim" in result.output


def test_score_error_counts_conlleval_format() -> None:
    # Cannot use error-counts with conlleval format
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "conlleval",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
            "--error-counts",
        ],
    )
    assert result.exit_code == 2
    assert "Cannot use error-counts with multiple files to be scored" in result.output


def test_score_prediction_shorter_than_reference() -> None:
    # The prediction file's first sentence is truncated (4 tokens) relative to
    # the reference (5 tokens), so scoring fails with a token count mismatch.
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            os.path.join("tests", "conll_predictions", "correct1_truncated_sentence.bio"),
        ],
    )
    assert result.exit_code == 1
    exception_text = str(result.exception)
    assert "Token count mismatch at line 1" in exception_text
    assert (
        "Reference sequence contains 5 tokens; predicted sequence contains 4."
        in exception_text
    )
    assert "Last token of reference sequence: '.'" in exception_text
    assert "Last token of predicted sequence: 'sentence'" in exception_text
    assert "was truncated" in exception_text


def test_score_uses_output_format_with_conlleval() -> None:
    runner = CliRunner()
    result = runner.invoke(score, ["--help"])
    assert "--output-format" in result.output
    assert "--score-format" not in result.output
    assert "conlleval" in result.output


def test_report_commands_have_output_format() -> None:
    runner = CliRunner()
    for cmd, name in [(summarize, "summarize"), (count, "count"), (analyze, "analyze")]:
        result = runner.invoke(cmd, ["--help"])
        assert result.exit_code == 0, f"{name} --help failed"
        assert "--output-format" in result.output, f"{name} missing --output-format"
        assert "--table-format" in result.output, f"{name} missing --table-format"


def _score_table_format(table_format: str) -> str:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--table-format",
            table_format,
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 0
    return result.output


def test_score_table_format_pretty_not_centered() -> None:
    # The "pretty" table format centers by default; the score table must keep
    # numeric columns right-justified instead.
    output = _score_table_format("pretty")
    assert "| ALL  |     50.00 |  66.67 |  57.14 |" in output


def test_score_table_format_grid() -> None:
    # "grid" uses row separators made of '=' and '-'
    output = _score_table_format("grid")
    assert "+====" in output


def test_score_table_format_invalid_rejected() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--table-format",
            "nonexistent",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--table-format'" in result.output


def test_score_multiple_files_conlleval_rejected() -> None:
    # conlleval format is inherently single-file
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--output-format",
            "conlleval",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code == 2
    assert "Cannot use the conlleval output format with multiple files" in result.output


def test_score_output_file_default_delim() -> None:
    # --output-file with no explicit format defaults to delimited; the file holds
    # only report data and stdout stays empty
    runner = CliRunner()
    ref = os.path.abspath(os.path.join("tests", "conll_annotation", "minimal.bio"))
    pred = os.path.abspath(os.path.join("tests", "conll_predictions", "incorrect1.bio"))
    with runner.isolated_filesystem():
        result = runner.invoke(
            score,
            ["--labels", "BIO", "--reference", ref, "--output-file", "out.tsv", pred],
        )
        assert result.exit_code == 0
        assert result.output == ""
        with open("out.tsv", encoding="utf8") as output:
            assert (
                output.read()
                == """Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect
LOC\t33.33\t50.00\t40.00\t2\t3\t1
ORG\t100.00\t100.00\t100.00\t1\t1\t1
ALL\t50.00\t66.67\t57.14\t3\t4\t2
"""
            )


def test_score_output_file_pretty() -> None:
    # --output-format pretty writes the table (not delimited) to the file
    runner = CliRunner()
    ref = os.path.abspath(os.path.join("tests", "conll_annotation", "minimal.bio"))
    pred = os.path.abspath(os.path.join("tests", "conll_predictions", "incorrect1.bio"))
    with runner.isolated_filesystem():
        result = runner.invoke(
            score,
            [
                "--labels",
                "BIO",
                "--reference",
                ref,
                "--output-format",
                "pretty",
                "--output-file",
                "out.txt",
                pred,
            ],
        )
        assert result.exit_code == 0
        with open("out.txt", encoding="utf8") as output:
            content = output.read()
        assert "| Type   |   Precision |" in content
        assert "| ALL    |       50.00 |    66.67 |  57.14 |" in content
