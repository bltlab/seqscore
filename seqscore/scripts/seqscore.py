from typing import Callable, Counter, Tuple

import click

from seqscore.conll import (
    SUPPORTED_SCORE_FORMATS,
    ingest_conll_file,
    repair_conll_file,
    score_conll_files,
    validate_conll_file,
    write_docs_using_encoding,
)
from seqscore.encoding import (
    DECODING_SUPPORTED_ENCODINGS,
    ENCODING_SUPPORTED_ENCODINGS,
    REPAIR_NONE,
    SUPPORTED_REPAIR_METHODS,
)
from seqscore.validation import VALIDATION_SUPPORTED_ENCODINGS


@click.group()
def cli():
    pass


def _input_file_arguments(func: Callable) -> Callable:
    # In order they can be used on the command line
    decorators = [
        click.argument("file", type=click.Path(dir_okay=False)),
        click.option("--file-encoding", default="UTF-8"),
        click.option("--ignore-comment-lines", is_flag=True),
        click.option(
            "--ignore-document-boundaries/--use-document-boundaries", default=True
        ),
    ]
    # Need to apply these backwards to match decorator application order
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


def _repair_option() -> Callable:
    return click.option(
        "--repair-method",
        default="conlleval",
        type=click.Choice(SUPPORTED_REPAIR_METHODS),
    )


@cli.command()
@_input_file_arguments
@click.option(
    "--labels", required=True, type=click.Choice(VALIDATION_SUPPORTED_ENCODINGS)
)
def validate(
    file: str,
    labels: str,
    file_encoding: str,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
):
    validate_conll_file(
        file,
        labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )


@cli.command()
@_input_file_arguments
@click.argument("output_file")
@_repair_option()
@click.option("--output-delim", default=" ")
def repair(
    file: str,
    output_file: str,
    file_encoding: str,
    repair: str,
    output_delim: str,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
):
    if repair == REPAIR_NONE:
        raise ValueError("Cannot repair if 'none' is specified as repair strategy")

    repair_conll_file(
        file,
        output_file,
        repair,
        file_encoding,
        output_delim,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
    )


@cli.command()
@_input_file_arguments
@click.argument("output_file")
@click.option("--output-delim", default=" ")
@click.option(
    "--input-labels", required=True, type=click.Choice(DECODING_SUPPORTED_ENCODINGS)
)
@click.option(
    "--output-labels", required=True, type=click.Choice(ENCODING_SUPPORTED_ENCODINGS)
)
def convert(
    file: str,
    output_file: str,
    file_encoding: str,
    output_delim: str,
    input_labels: str,
    output_labels: str,
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
@_input_file_arguments
@click.argument("output_file")
@_repair_option()
@click.option("--labels", required=True, type=click.Choice(DECODING_SUPPORTED_ENCODINGS))
@click.option("--delim", default="\t")
def dump(
    file: str,
    file_encoding: str,
    output_file: str,
    labels: str,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    delim: str,
    repair: str,
):
    if repair == REPAIR_NONE:
        repair = None

    docs = ingest_conll_file(
        file,
        labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        repair=repair,
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
@click.argument("file", type=click.Path(dir_okay=False))
@click.option("--reference", required=True)
@click.option("--ignore-comment-lines", is_flag=True)
@click.option("--ignore-document-boundaries/--use-document-boundaries", default=True)
@_repair_option()
@click.option(
    "--score-format", default="pretty", type=click.Choice(SUPPORTED_SCORE_FORMATS)
)
@click.option("--delim", default="\t")
def score(
    file: str,
    file_encoding: str,
    ignore_document_boundaries: bool,
    ignore_comment_lines: bool,
    reference: str,
    display: str,
    delim: str,
    repair: str,
):
    if repair == REPAIR_NONE:
        repair = None

    score_conll_files(
        file,
        reference,
        repair,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        ignore_comment_lines=ignore_comment_lines,
        output_format=display,
        delim=delim,
    )


if __name__ == "__main__":
    cli()
