import io

import pytest
from tabulate import SEPARATING_LINE

from seqscore.output import write_report


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


def test_write_report_delim_skips_separating_line() -> None:
    # SEPARATING_LINE is a table-only sentinel; in delim mode it must be skipped
    # rather than joined (which would raise or emit a bogus row).
    out = io.StringIO()
    write_report(
        ("Entity Type", "Count"),
        [("LOC", 2), SEPARATING_LINE, ("TOTAL", 2)],
        output_format="delim",
        delim="\t",
        file=out,
    )
    assert (
        out.getvalue()
        == """Entity Type\tCount
LOC\t2
TOTAL\t2
"""
    )


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
