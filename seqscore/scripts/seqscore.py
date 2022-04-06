import sys
from typing import Callable, Counter, List, Tuple

import click

from seqscore.conll import (
    SUPPORTED_SCORE_FORMATS,
    ingest_conll_file,
    repair_conll_file,
    score_conll_files,
    validate_conll_file,
    write_docs_using_encoding,
)
from seqscore.encoding import REPAIR_NONE, SUPPORTED_ENCODINGS, SUPPORTED_REPAIR_METHODS


# This is tested by a subprocess call in test_seqscore_main so coverage will miss it
@click.group()
def cli():  # pragma: no cover
    pass


def _input_file_options() -> List[Callable]:
    return [
        click.option("--file-encoding", default="UTF-8", show_default=True),
        click.option("--ignore-comment-lines", is_flag=True),
        click.option(
            "--ignore-document-boundaries/--use-document-boundaries", default=True
        ),
    ]


def _single_input_file_arguments(func: Callable) -> Callable:
    # In order they can be used on the command line
    decorators = [
        click.argument("file", type=click.Path(dir_okay=False)),
    ] + _input_file_options()
    # Need to apply these backwards to match decorator application order
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


def _multi_input_file_arguments(func: Callable) -> Callable:
    # In order they can be used on the command line
    decorators = [
        click.argument("file", type=click.Path(dir_okay=False), nargs=-1, required=True),
    ] + _input_file_options()
    # Need to apply these backwards to match decorator application order
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


def _repair_option() -> Callable:
    return click.option(
        "--repair-method",
        type=click.Choice(SUPPORTED_REPAIR_METHODS),
        show_default=True,
    )


def _labels_option() -> Callable:
    return click.option("--labels", required=True, type=click.Choice(SUPPORTED_ENCODINGS))


def _quiet_option() -> Callable:
    return click.option(
        "--quiet", "-q", is_flag=True, help="do not log the repairs performed"
    )


def _normalize_tab(s: str) -> str:
    # Clean up the string r"\t" if it's been given
    return s.replace(r"\t", "\t")


@cli.command()
@_single_input_file_arguments
@_labels_option()
def validate(
    file: str,
    labels: str,
    file_encoding: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
):
    result = validate_conll_file(
        file,
        labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )
    if result.errors:
        print(
            f"Encountered {len(result.errors)} errors in {result.n_tokens} tokens, "
            + f"{result.n_sequences} sequences, and {result.n_docs} documents in {file}"
        )
        print("\n".join(err.msg for err in result.errors))
        sys.exit(1)
    else:
        print(
            f"No errors found in {result.n_tokens} tokens, {result.n_sequences} sequences, "
            + f"and {result.n_docs} documents in {file}"
        )


@cli.command()
@_single_input_file_arguments
@click.argument("output_file")
@_repair_option()
@_labels_option()
@click.option("--output-delim", default=" ", help="[default: space")
@_quiet_option()
def repair(
    file: str,
    output_file: str,
    labels: str,
    file_encoding: str,
    repair_method: str,
    output_delim: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        raise ValueError("Cannot repair if 'none' is specified as repair strategy")

    repair_conll_file(
        file,
        output_file,
        labels,
        repair_method,
        file_encoding,
        output_delim,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        quiet=quiet,
    )


@cli.command()
@_single_input_file_arguments
@click.argument("output_file")
@click.option("--output-delim", default=" ", help="[default: space]")
@click.option("--input-labels", required=True, type=click.Choice(SUPPORTED_ENCODINGS))
@click.option("--output-labels", required=True, type=click.Choice(SUPPORTED_ENCODINGS))
def convert(
    file: str,
    output_file: str,
    file_encoding: str,
    output_delim: str,
    input_labels: str,
    output_labels: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
):
    if input_labels == output_labels:
        raise ValueError("Cannot repair if 'none' is specified as repair strategy")

    docs = ingest_conll_file(
        file,
        input_labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )

    write_docs_using_encoding(
        docs, output_labels, file_encoding, output_delim, output_file
    )


@cli.command()
@_single_input_file_arguments
@click.argument("output_file")
@_repair_option()
@_labels_option()
@click.option(
    "--delim",
    default="\t",
    help="the delimiter to be used for output (has no effect on input) [default: tab]",
)
@_quiet_option()
def count(
    file: str,
    file_encoding: str,
    output_file: str,
    labels: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    delim: str,
    repair_method: str,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        repair_method = None

    delim = _normalize_tab(delim)
    if delim != "\t":
        print(
            "Warning: Using a delimiter other than tab is not recommended as fields are not quoted",
            file=sys.stderr,
        )

    docs = ingest_conll_file(
        file,
        labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        repair=repair_method,
        quiet=quiet,
    )

    counts: Counter[Tuple[str, Tuple[str, ...]]] = Counter()
    for doc in docs:
        for sequence in doc:
            for mention in sequence.mentions:
                key = (mention.type, sequence.mention_tokens(mention))
                counts[key] += 1

    with open(output_file, "w", encoding=file_encoding) as output:
        for item, count in counts.most_common():
            print(delim.join((str(count), item[0], " ".join(item[1]))), file=output)


@cli.command()
@_multi_input_file_arguments
@click.option("--reference", required=True)
@_labels_option()
@_repair_option()
@click.option(
    "--score-format",
    default="pretty",
    type=click.Choice(SUPPORTED_SCORE_FORMATS),
    show_default=True,
)
@click.option(
    "--delim",
    default="\t",
    help="the delimiter to be used for delimited output (has no effect on input) [default: tab]",
)
@_quiet_option()
def score(
    file: List[str],  # Name is "file" to make sense on the command line, but it's a list
    file_encoding: str,
    labels: str,
    *,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    reference: str,
    score_format: str,
    delim: str,
    repair_method: str,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        repair_method = None

    delim = _normalize_tab(delim)

    score_conll_files(
        file,
        reference,
        labels,
        repair_method,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        output_format=score_format,
        delim=delim,
        quiet=quiet,
    )


# This is tested by a subprocess call in test_seqscore_main so coverage will miss it
if __name__ == "__main__":  # pragma: no cover
    cli()
