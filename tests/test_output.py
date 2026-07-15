import io

import pytest

from seqscore.encoding import get_encoding
from seqscore.model import AnnotatedSequence
from seqscore.output import report_scores, write_report

_TOKENS = ("Alice", "visited", "Paris")


def _doc(labels: list[str]) -> list[list[AnnotatedSequence]]:
    # A single-document, single-sequence corpus with the given labels
    seq = AnnotatedSequence.from_tokens_and_labels(_TOKENS, labels, get_encoding("BIO"))
    return [[seq]]


def test_write_report_defaults_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # With no file argument, output goes to stdout
    write_report(
        ("Count", "Type"),
        [("1", "LOC"), ("2", "ORG")],
        output_format="delim",
        delim="\t",
        delim_header=False,
    )
    assert capsys.readouterr().out == "1\tLOC\n2\tORG\n"


def test_write_report_pretty_justification_no_centering() -> None:
    # The "pretty" table format centers by default; write_report must override
    # that so numeric columns are right-justified, others left-justified, and
    # nothing is centered.
    out = io.StringIO()
    write_report(
        ("Type", "Span"),
        [["ORG", "0-3"], ["LOCATION", "10-12"]],
        output_format="pretty",
        delim="\t",
        table_format="pretty",
        file=out,
        numeric_columns=(1,),
    )
    assert (
        out.getvalue()
        == """+----------+-------+
| Type     |  Span |
+----------+-------+
| ORG      |   0-3 |
| LOCATION | 10-12 |
+----------+-------+
"""
    )


def test_write_report_delim_no_header() -> None:
    out = io.StringIO()
    write_report(
        ("Count", "Type"),
        [("1", "LOC"), ("2", "ORG")],
        output_format="delim",
        delim="\t",
        file=out,
        delim_header=False,
    )
    assert out.getvalue() == "1\tLOC\n2\tORG\n"


def test_report_scores_single_file_delim(capsys: pytest.CaptureFixture[str]) -> None:
    ref = _doc(["B-PER", "O", "B-LOC"])
    pred = _doc(["B-PER", "O", "O"])
    report_scores([("pred.bio", pred)], ref, output_format="delim", delim="\t")
    out = capsys.readouterr().out
    assert "Type\tPrecision\tRecall\tF1\tReference\tPredicted\tCorrect" in out
    assert "ALL\t100.00\t50.00\t66.67\t2\t1\t1" in out
    assert "LOC\t0.00\t0.00\t0.00\t1\t0\t0" in out
    assert "PER\t100.00\t100.00\t100.00\t1\t1\t1" in out


def test_report_scores_conlleval_rejects_multiple_files() -> None:
    ref = _doc(["B-PER", "O", "B-LOC"])
    pred = _doc(["B-PER", "O", "O"])
    with pytest.raises(ValueError, match="conlleval format is not supported"):
        report_scores(
            [("a.bio", pred), ("b.bio", pred)],
            ref,
            output_format="conlleval",
            delim="\t",
        )


def test_report_scores_error_counts_rejects_multiple_files() -> None:
    ref = _doc(["B-PER", "O", "B-LOC"])
    pred = _doc(["B-PER", "O", "O"])
    with pytest.raises(ValueError, match="single prediction file"):
        report_scores(
            [("a.bio", pred), ("b.bio", pred)],
            ref,
            output_format="pretty",
            delim="\t",
            error_counts=True,
        )
