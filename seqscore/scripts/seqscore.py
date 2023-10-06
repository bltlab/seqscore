import json
import sys
from typing import Callable, Counter, Dict, List, Optional, Set, Tuple

import click
from tabulate import tabulate

import seqscore
from seqscore.conll import (
    FORMAT_DELIM,
    SUPPORTED_SCORE_FORMATS,
    ingest_conll_file,
    repair_conll_file,
    score_conll_files,
    validate_conll_file,
    write_docs_using_encoding,
)
from seqscore.encoding import (
    DEFAULT_OUTSIDE,
    REPAIR_NONE,
    SUPPORTED_ENCODINGS,
    SUPPORTED_REPAIR_METHODS,
)
from seqscore.processing import modify_types


# Set up a click command group
@click.group(
    help=f"Provides scoring and analysis tools for NER/chunking files (version {seqscore.__version__})"
)
@click.version_option(seqscore.__version__)
# This is tested by a subprocess call in test_seqscore_main so coverage will miss it
def cli():  # pragma: no cover
    pass


# Argument helpers for commands
def _input_file_options() -> List[Callable]:
    return [
        click.option("--file-encoding", default="UTF-8", show_default=True),
        click.option("--parse-comment-lines", is_flag=True),
        click.option(
            "--ignore-document-boundaries/--use-document-boundaries", default=False
        ),
    ]


def _single_input_file_arguments(func: Callable) -> Callable:
    # In the order they can be used on the command line
    decorators = [
        click.argument("file", type=click.Path(dir_okay=False)),
    ] + _input_file_options()
    # Need to apply these backwards to match decorator application order
    for decorator in decorators[::-1]:
        func = decorator(func)
    return func


def _multi_input_file_arguments(func: Callable) -> Callable:
    # In the order they can be used on the command line
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
        default=REPAIR_NONE,
        show_default=True,
    )


def _repair_required_option() -> Callable:
    return click.option(
        "--repair-method",
        type=click.Choice(SUPPORTED_REPAIR_METHODS),
    )


def _labels_option() -> Callable:
    return click.option("--labels", required=True, type=click.Choice(SUPPORTED_ENCODINGS))


def _labels_option_default_bio() -> Callable:
    return click.option(
        "--labels",
        default="BIO",
        show_default=True,
        type=click.Choice(SUPPORTED_ENCODINGS),
    )


def _quiet_option() -> Callable:
    return click.option(
        "--quiet",
        "-q",
        is_flag=True,
        help="do not log the repairs performed and supress other non-critical messages",
    )


@cli.command(help="validate labels")
@_multi_input_file_arguments
@_labels_option()
@_quiet_option()
def validate(
    file: List[str],  # Name is "file" to make sense on the command line, but it's a list
    labels: str,
    file_encoding: str,
    *,
    ignore_document_boundaries: bool,
    parse_comment_lines: bool,
    quiet: bool,
):
    error = False
    for each_file in file:
        result = validate_conll_file(
            each_file,
            labels,
            file_encoding,
            ignore_document_boundaries=ignore_document_boundaries,
            parse_comment_lines=parse_comment_lines,
        )
        if result.errors:
            print(
                f"Encountered {len(result.errors)} errors in {result.n_tokens} tokens, "
                + f"{result.n_sequences} sequences, and {result.n_docs} document(s) in {each_file}"
            )
            print("\n".join(err.msg for err in result.errors))
            error = True
        elif not quiet:
            print(
                f"No errors found in {result.n_tokens} tokens, {result.n_sequences} sequences, "
                + f"and {result.n_docs} document(s) in {each_file}"
            )

    if error:
        sys.exit(1)


@cli.command(help="repair invalid label transitions")
@_single_input_file_arguments
@click.argument("output_file")
@_repair_required_option()
@_labels_option()
@click.option("--output-delim", default=" ", help="[default: space]")
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
    parse_comment_lines: bool,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        raise ValueError(f"Cannot repair with repair strategy {repr(repair_method)}")

    repair_conll_file(
        file,
        output_file,
        labels,
        repair_method,
        file_encoding,
        output_delim,
        ignore_document_boundaries=ignore_document_boundaries,
        parse_comment_lines=parse_comment_lines,
        quiet=quiet,
    )


@cli.command(help="convert between mention encodings")
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
    parse_comment_lines: bool,
):
    if input_labels == output_labels:
        raise ValueError("Conversion requires different input and output labels")

    docs = ingest_conll_file(
        file,
        input_labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        parse_comment_lines=parse_comment_lines,
    )

    write_docs_using_encoding(
        docs, output_labels, file_encoding, output_delim, output_file
    )


@cli.command(help="transform entity types by keeping/removing/mapping")
@_single_input_file_arguments
@click.argument("output_file")
@_labels_option()
@click.option(
    "--keep-types",
    default="",
    help="entity types to keep, comma-separated [example: PER,LOC,ORG]",
)
@click.option(
    "--remove-types",
    default="",
    help="entity types to remove, comma-separated [example: MISC,DATE]",
)
@click.option(
    "--type-map",
    type=click.Path(dir_okay=False),
    help="a JSON file containing types to be modified, in the format of a dict with keys as the target type and values as the source type [example file: {'MISC': ['WorkOfArt', 'Event']}]",
)
@click.option("--output-delim", default=" ", help="[default: space]")
def process(
    file: str,
    output_file: str,
    file_encoding: str,
    output_delim: str,
    labels: str,
    keep_types: str,
    remove_types: str,
    type_map: str,
    *,
    ignore_document_boundaries: bool,
    parse_comment_lines: bool,
):
    keep_types_set = _parse_type_list(keep_types)
    remove_types_set = _parse_type_list(remove_types)
    type_map_dict: Dict[str, List[str]] = _load_type_map(type_map, file_encoding)

    if keep_types_set and remove_types_set:
        raise ValueError("Cannot specify both keep-types and remove-types")

    if not keep_types_set and not remove_types_set and not type_map:
        raise ValueError(
            "Must specify at least one of keep-types, remove-types, or type-map"
        )

    docs = ingest_conll_file(
        file,
        labels,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        parse_comment_lines=parse_comment_lines,
    )

    mod_docs = modify_types(docs, keep_types_set, remove_types_set, type_map_dict)

    write_docs_using_encoding(mod_docs, labels, file_encoding, output_delim, output_file)


@cli.command(help="show counts for all the mentions contained in a file")
@_multi_input_file_arguments
@click.argument("output_file")
@_repair_option()
@_labels_option_default_bio()
@click.option(
    "--delim",
    default="\t",
    help="the delimiter to be used for output (has no effect on input) [default: tab]",
)
@_quiet_option()
def count(
    file: List[str],  # Name is "file" to make sense on the command line, but it's a list
    file_encoding: str,
    output_file: str,
    labels: str,
    *,
    ignore_document_boundaries: bool,
    parse_comment_lines: bool,
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

    counts: Counter[Tuple[str, Tuple[str, ...]]] = Counter()
    for each_file in file:
        docs = ingest_conll_file(
            each_file,
            labels,
            file_encoding,
            ignore_document_boundaries=ignore_document_boundaries,
            parse_comment_lines=parse_comment_lines,
            repair=repair_method,
            quiet=quiet,
        )

        for doc in docs:
            for sequence in doc:
                for mention in sequence.mentions:
                    key = (mention.type, sequence.mention_tokens(mention))
                    counts[key] += 1

    with open(output_file, "w", encoding=file_encoding) as output:
        for item, item_count in counts.most_common():
            print(delim.join((str(item_count), item[0], " ".join(item[1]))), file=output)


@cli.command(help="show counts of the documents, sentences, and entity types")
@_multi_input_file_arguments
@_repair_option()
@_labels_option_default_bio()
@_quiet_option()
def summarize(
    file: List[str],  # Name is "file" to make sense on the command line, but it's a list
    file_encoding: str,
    labels: str,
    *,
    ignore_document_boundaries: bool,
    parse_comment_lines: bool,
    repair_method: str,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        repair_method = None

    type_counts: Counter[str] = Counter()
    total_documents = 0
    total_sentences = 0
    for each_file in file:
        docs = ingest_conll_file(
            each_file,
            labels,
            file_encoding,
            ignore_document_boundaries=ignore_document_boundaries,
            parse_comment_lines=parse_comment_lines,
            repair=repair_method,
            quiet=quiet,
        )

        for doc in docs:
            for sequence in doc:
                for mention in sequence.mentions:
                    type_counts[mention.type] += 1

        if not quiet:
            # Count sentences
            sentence_count = sum(len(doc) for doc in docs)
            print(
                f"File {repr(each_file)} contains {len(docs)} document(s) and {sentence_count} sentences"
            )
            total_documents += len(docs)
            total_sentences += sentence_count

    if not quiet and len(file) > 1:
        print(f"Total {total_documents} document(s) and {total_sentences} sentences")

    header = ["Entity Type", "Count"]
    rows = sorted(type_counts.items())
    print(tabulate(rows, header, tablefmt="github", floatfmt="6.2f"))


@cli.command(help="score a file and report performance or an error count table")
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
@click.option(
    "--error-counts",
    is_flag=True,
    help="whether to output counts of false positives and negatives instead of scores",
)
@click.option(
    "--full-precision",
    is_flag=True,
    help="whether to output floating values at full precision instead of rounding half even at two decimal places",
)
@_quiet_option()
def score(
    file: List[str],  # Name is "file" to make sense on the command line, but it's a list
    file_encoding: str,
    labels: str,
    *,
    ignore_document_boundaries: bool,
    parse_comment_lines: bool,
    reference: str,
    score_format: str,
    delim: str,
    repair_method: str,
    error_counts: bool,
    full_precision: bool,
    quiet: bool,
):
    if repair_method == REPAIR_NONE:
        repair_method = None

    if full_precision and score_format != FORMAT_DELIM:
        raise ValueError(f"Can only use full-precision with score-format {FORMAT_DELIM}")

    if error_counts and len(file) > 1:
        raise ValueError("Cannot use error-counts with multiple files to be scored")

    delim = _normalize_tab(delim)

    score_conll_files(
        file,
        reference,
        labels,
        repair_method,
        file_encoding,
        ignore_document_boundaries=ignore_document_boundaries,
        parse_comment_lines=parse_comment_lines,
        output_format=score_format,
        delim=delim,
        error_counts=error_counts,
        full_precision=full_precision,
        quiet=quiet,
    )


def _normalize_tab(s: str) -> str:
    if s == "tab":
        return "\t"
    else:
        # Clean up the string r"\t" if it's been given
        return s.replace(r"\t", "\t")


def _parse_type_list(types: str) -> Set[str]:
    # Remove any whitespace we got in the types somehow
    split_types = [t.strip() for t in types.split(",") if t.strip()]
    # Check for outside type
    for entity_type in split_types:
        if entity_type == DEFAULT_OUTSIDE:
            raise ValueError(
                f"Cannot specify the outside type {DEFAULT_OUTSIDE} in keep/remove types"
            )
    return set(split_types)


def _load_type_map(
    type_map_path: Optional[str], file_encoding: str
) -> Dict[str, List[str]]:
    if not type_map_path:
        return {}

    try:
        with open(type_map_path, encoding=file_encoding) as file:
            type_map = json.load(file)
    except FileNotFoundError as err:
        raise ValueError(f"Could not open type map file {repr(type_map_path)}") from err
    except json.decoder.JSONDecodeError as err:
        raise ValueError(
            f"Type map provided in file {repr(type_map_path)} is not valid JSON"
        ) from err

    # Validate types
    if not isinstance(type_map, dict):
        raise ValueError(
            f"Type map provided in file {repr(type_map_path)} is not a dictionary"
        )

    for from_type, to_types in type_map.items():
        if not isinstance(from_type, str) or not from_type:
            raise ValueError(
                f"Key {repr(from_type)} in type map {repr(type_map_path)} is not a non-empty string"
            )
        if from_type == DEFAULT_OUTSIDE:
            raise ValueError(
                f"Key {repr(from_type)} in type map {repr(type_map_path)} is the outside type {DEFAULT_OUTSIDE}"
            )

        if not isinstance(to_types, list):
            raise ValueError(
                f"Value {repr(to_types)} in type map {repr(type_map_path)} is not a list"
            )

        for to_type in to_types:
            if not isinstance(to_type, str) or not to_type:
                raise ValueError(
                    f"Value {repr(to_type)} in type map {repr(type_map_path)} is not a non-empty string"
                )
            if to_type == DEFAULT_OUTSIDE:
                raise ValueError(
                    f"Value {repr(to_type)} in type map {repr(type_map_path)} is the outside type {DEFAULT_OUTSIDE}"
                )

    return type_map


# This is tested by a subprocess call in test_seqscore_main so coverage will miss it
if __name__ == "__main__":  # pragma: no cover
    cli()
