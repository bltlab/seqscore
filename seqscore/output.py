import sys
from typing import Collection, Sequence, TextIO

from tabulate import SEPARATING_LINE, tabulate

FORMAT_PRETTY = "pretty"
FORMAT_CONLLEVAL = "conlleval"
FORMAT_DELIM = "delim"
SUPPORTED_SCORE_FORMATS = (FORMAT_PRETTY, FORMAT_CONLLEVAL, FORMAT_DELIM)

SUPPORTED_OUTPUT_FORMATS = (FORMAT_PRETTY, FORMAT_DELIM)


def write_report(
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    output_format: str,
    delim: str,
    table_format: str = "github",
    file: TextIO | None = None,
    delim_header: bool = True,
    numeric_columns: Collection[int] = (),
) -> None:
    if file is None:
        file = sys.stdout
    if output_format == FORMAT_PRETTY:
        # Justify explicitly so the result is deterministic regardless of the
        # table format (some, like "pretty", center by default) and regardless
        # of tabulate's content-based type sniffing: numeric columns are
        # right-justified, all others left-justified, nothing centered.
        colalign = tuple(
            "right" if i in numeric_columns else "left" for i in range(len(header))
        )
        print(
            tabulate(
                list(rows),
                list(header),
                tablefmt=table_format,
                intfmt=",",
                colalign=colalign,
            ),
            file=file,
        )
    else:
        if delim_header:
            print(delim.join(str(c) for c in header), file=file)
        for row in rows:
            if row is SEPARATING_LINE:
                continue
            print(delim.join(str(c) for c in row), file=file)
