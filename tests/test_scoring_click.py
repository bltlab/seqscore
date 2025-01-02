import glob
import os

from click.testing import CliRunner

from seqscore.scripts.seqscore import score


def test_score_correct_labels() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--score-format",
            "delim",
            os.path.join("tests", "conll_predictions", "correct1.bio"),
        ],
    )
    assert result.exit_code == 0
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in result.output
    assert "ALL\t100.00\t100.00\t100.00\t3\t3\t3" in result.output
    assert "LOC\t100.00\t100.00\t100.00\t2\t2\t2" in result.output
    assert "ORG\t100.00\t100.00\t100.00\t1\t1\t1" in result.output


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
            "--score-format",
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
            "--score-format",
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
            "--score-format",
            "delim",
            os.path.join(
                "tests", "conll_predictions", "correct1_improper_sequence_pred.txt"
            ),
        ],
    )
    assert result.exit_code != 0


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
            "--score-format",
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
            "--score-format",
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
        ["ALL", "75.00", "100.00", "85.71", "3", "4", "3"],
        ["LOC", "100.00", "100.00", "100.00", "2", "2", "2"],
        ["ORG", "100.00", "100.00", "100.00", "1", "1", "1"],
        ["SPURIOUS", "0.00", "0.00", "0.00", "0", "1", "0"],
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
            "--score-format",
            "delim",
            os.path.join("tests", "conll_predictions", "incorrect1.bio"),
        ],
    )
    assert result.exit_code != 0


def test_score_multiple_files() -> None:
    runner = CliRunner()
    result = runner.invoke(
        score,
        [
            "--labels",
            "BIO",
            "--reference",
            os.path.join("tests", "conll_annotation", "minimal.bio"),
            "--score-format",
            "delim",
        ]
        + glob.glob(os.path.join("tests", "conll_predictions", "*1.bio")),
    )
    assert result.exit_code == 0
    assert "SD\tALL\tNA\tNA\t30.30\tNA\tNA\tNA" in result.output
    assert "Mean\tALL\tNA\tNA\t78.57\tNA\tNA\tNA" in result.output
