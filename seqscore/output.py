import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from math import sqrt
from statistics import mean, stdev
from typing import Any, DefaultDict, TextIO

from tabulate import tabulate

from seqscore.model import AnnotatedSequence
from seqscore.scoring import (
    AccuracyScore,
    ClassificationScore,
    compute_scores,
    convert_score,
)

ALL_TYPES = "ALL"

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
    floatfmt: str = "",
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
                floatfmt=floatfmt,
                colalign=colalign,
            ),
            file=file,
        )
    else:
        if delim_header:
            print(delim.join(str(c) for c in header), file=file)
        for row in rows:
            print(delim.join(str(c) for c in row), file=file)


def _join_delim(items: Iterable[Any], delim: str) -> str:
    return delim.join(str(item) for item in items)


def format_output_conlleval(
    class_scores: ClassificationScore,
    acc_scores: AccuracyScore,
) -> str:
    """Format output like conlleval.pl.

    Example:
    processed 15 tokens with 3 phrases; found: 4 phrases; correct: 2.
    accuracy:  93.33%; precision:  50.00%; recall:  66.67%; FB1:  57.14
                  LOC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3
                  ORG: precision: 100.00%; recall: 100.00%; FB1: 100.00  1
    """
    n_phrases = class_scores.true_pos + class_scores.false_neg
    lines = [
        f"processed {acc_scores.total} tokens with {n_phrases} phrases; "
        + f"found: {class_scores.total_pos} phrases; correct: {class_scores.true_pos}.",
        f"accuracy: {100 * acc_scores.accuracy:6.2f}%; "
        + f"precision: {100 * class_scores.precision:6.2f}%; "
        + f"recall: {100 * class_scores.recall:6.2f}%; "
        + f"FB1: {100 * class_scores.f1:6.2f}",
    ]
    # Add lines for each type
    for type_name, score in sorted(class_scores.type_scores.items()):
        lines.append(
            f"{type_name.rjust(17)}: "  # This is the width that conlleval uses
            + f"precision: {100 * score.precision:6.2f}%; "
            + f"recall: {100 * score.recall:6.2f}%; "
            + f"FB1: {100 * score.f1:6.2f}  {score.total_pos}"
        )
    return "\n".join(lines)


def format_output_table(
    class_scores: ClassificationScore,
    full_precision: bool,
) -> tuple[list[str], list[list[Any]]]:
    header = [
        "Type",
        "Precision",
        "Recall",
        "F1",
        "Reference",
        "Predicted",
        "Correct",
    ]
    rows: list[list[Any]] = []
    # One line per type, then the ALL row last so it reads like a total
    for type_name, score in sorted(class_scores.type_scores.items()):
        rows.append(
            [
                type_name,
                convert_score(score.precision, full_precision),
                convert_score(score.recall, full_precision),
                convert_score(score.f1, full_precision),
                score.total_ref,
                score.total_pos,
                score.true_pos,
            ]
        )
    rows.append(
        [
            ALL_TYPES,
            convert_score(class_scores.precision, full_precision),
            convert_score(class_scores.recall, full_precision),
            convert_score(class_scores.f1, full_precision),
            class_scores.total_ref,
            class_scores.total_pos,
            class_scores.true_pos,
        ]
    )
    return header, rows


def _f1_mean_stderr(
    all_class_scores: Sequence[ClassificationScore],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the mean and standard error of F1 per type (and for ALL) across files."""
    type_f1s: DefaultDict[str, list[float]] = defaultdict(list)
    for class_scores in all_class_scores:
        for entity_type, entity_score in class_scores.type_scores.items():
            type_f1s[entity_type].append(entity_score.f1)

    means = {entity_type: mean(f1s) for entity_type, f1s in type_f1s.items()}
    stderrs = {
        entity_type: stdev(f1s) / sqrt(len(f1s)) for entity_type, f1s in type_f1s.items()
    }
    all_f1s = [score.f1 for score in all_class_scores]
    means[ALL_TYPES] = mean(all_f1s)
    stderrs[ALL_TYPES] = stdev(all_f1s) / sqrt(len(all_f1s))
    return means, stderrs


def _report_error_counts(
    class_scores: ClassificationScore,
    output_format: str,
    delim: str,
    table_format: str,
    file: TextIO,
) -> None:
    header = ["Count", "Error", "Type", "Tokens"]
    combined_counts: Counter[tuple[str, str, str]] = Counter()
    for counter, error_type in zip(
        (class_scores.false_pos_examples, class_scores.false_neg_examples),
        ("FP", "FN"),
    ):
        for item, count in counter.items():
            combined_counts[(error_type, item.type, " ".join(item.tokens))] = count

    # Sort by count descending, breaking ties on the token string
    rows = [
        [count, error_type, mention_type, token_str]
        for (error_type, mention_type, token_str), count in sorted(
            combined_counts.items(), key=lambda item: (-item[1], item[0][2])
        )
    ]
    write_report(
        header,
        rows,
        output_format=output_format,
        delim=delim,
        numeric_columns=(0,),
        table_format=table_format,
        file=file,
    )


def report_scores(
    pred_docs_by_file: Sequence[tuple[str, Sequence[Sequence[AnnotatedSequence]]]],
    ref_docs: Sequence[Sequence[AnnotatedSequence]],
    *,
    output_format: str,
    delim: str,
    error_counts: bool = False,
    full_precision: bool = False,
    table_format: str = "github",
    file: TextIO | None = None,
) -> None:
    """Score and present results for one or more prediction files.

    Operates only on already-ingested documents; does no file I/O and does not
    import from conll.py.
    """
    if file is None:
        file = sys.stdout

    if output_format not in SUPPORTED_SCORE_FORMATS:
        raise ValueError(f"Unrecognized output format: {output_format}")

    multi_files = len(pred_docs_by_file) > 1

    all_class_scores: list[ClassificationScore] = []
    all_acc_scores: list[AccuracyScore] = []
    for _, pred_docs in pred_docs_by_file:
        class_scores, acc_scores = compute_scores(
            pred_docs, ref_docs, count_fp_fn_examples=error_counts
        )
        all_class_scores.append(class_scores)
        all_acc_scores.append(acc_scores)

    if error_counts:
        if multi_files:
            raise ValueError(
                "Outputting error counts is only available for a single prediction file"
            )
        if output_format == FORMAT_CONLLEVAL:
            raise ValueError(
                f"Format {output_format!r} is not supported with error counts"
            )
        _report_error_counts(
            all_class_scores[0], output_format, delim, table_format, file
        )
        return

    if output_format == FORMAT_CONLLEVAL:
        if multi_files:
            raise ValueError(
                "conlleval format is not supported when scoring multiple files"
            )
        print(format_output_conlleval(all_class_scores[0], all_acc_scores[0]), file=file)
        return

    if output_format == FORMAT_DELIM:
        lines: list[str] = []
        for idx, (fname, _) in enumerate(pred_docs_by_file):
            header, rows = format_output_table(all_class_scores[idx], full_precision)
            if idx == 0:
                lines.append(delim.join(["File"] + header if multi_files else header))
            for row in rows:
                lines.append(_join_delim([fname] + row if multi_files else row, delim))

        if multi_files:
            means, stderrs = _f1_mean_stderr(all_class_scores)
            for entity_type, value in stderrs.items():
                lines.append(
                    _join_delim(
                        [
                            "SE",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(value, full_precision),
                            "NA",
                            "NA",
                            "NA",
                        ],
                        delim,
                    )
                )
            for entity_type, value in means.items():
                lines.append(
                    _join_delim(
                        [
                            "Mean",
                            entity_type,
                            "NA",
                            "NA",
                            convert_score(value, full_precision),
                            "NA",
                            "NA",
                            "NA",
                        ],
                        delim,
                    )
                )
        print("\n".join(lines), file=file)
        return

    # Pretty
    if full_precision:
        raise ValueError("Cannot use full_precision with pretty formatting")

    for idx, (fname, _) in enumerate(pred_docs_by_file):
        if multi_files:
            print(fname, file=file)
        header, rows = format_output_table(all_class_scores[idx], full_precision)
        write_report(
            header,
            rows,
            output_format=FORMAT_PRETTY,
            delim=delim,
            numeric_columns=(1, 2, 3, 4, 5, 6),
            floatfmt="6.2f",
            table_format=table_format,
            file=file,
        )
        if multi_files and idx != len(pred_docs_by_file) - 1:
            print(file=file)

    if multi_files:
        means, stderrs = _f1_mean_stderr(all_class_scores)
        ref_scores = all_class_scores[0]
        summary_header = ["Type", "Mean F1", "SE", "Reference"]
        summary_rows: list[list[Any]] = []
        for entity_type in sorted(means):
            if entity_type == ALL_TYPES:
                continue
            summary_rows.append(
                [
                    entity_type,
                    means[entity_type] * 100,
                    stderrs[entity_type] * 100,
                    ref_scores.type_scores[entity_type].total_ref,
                ]
            )
        summary_rows.append(
            [
                ALL_TYPES,
                means[ALL_TYPES] * 100,
                stderrs[ALL_TYPES] * 100,
                ref_scores.total_ref,
            ]
        )
        print(file=file)
        print("Summary", file=file)
        write_report(
            summary_header,
            summary_rows,
            output_format=FORMAT_PRETTY,
            delim=delim,
            numeric_columns=(1, 2, 3),
            floatfmt="6.2f",
            table_format=table_format,
            file=file,
        )
