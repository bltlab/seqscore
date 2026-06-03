import os

import pytest

from seqscore.conll import (
    FORMAT_CONLLEVAL,
    FORMAT_DELIM,
    FORMAT_PRETTY,
    LineSpec,
    score_conll_files,
)

REFERENCE = os.path.join("tests", "conll_annotation", "minimal.bio")
CORRECT1 = os.path.join("tests", "conll_predictions", "correct1.bio")
INCORRECT1 = os.path.join("tests", "conll_predictions", "incorrect1.bio")


def _score(
    pred_files: list[str],
    output_format: str,
    error_counts: bool = False,
    full_precision: bool = False,
) -> None:
    score_conll_files(
        pred_files,
        REFERENCE,
        mention_encoding_name="BIO",
        repair=None,
        file_encoding="utf-8",
        line_spec=LineSpec(0, -1),
        ignore_document_boundaries=False,
        parse_comment_lines=False,
        delim="\t",
        output_format=output_format,
        error_counts=error_counts,
        full_precision=full_precision,
    )


def test_score_error_counts_multiple_files() -> None:
    with pytest.raises(
        ValueError,
        match="Outputting error counts is only available for a single prediction file",
    ):
        _score([CORRECT1, INCORRECT1], FORMAT_DELIM, error_counts=True)


def test_score_error_counts_conlleval_format() -> None:
    with pytest.raises(
        ValueError,
        match=f"Format {repr(FORMAT_CONLLEVAL)} is not supported with error counts",
    ):
        _score([CORRECT1], FORMAT_CONLLEVAL, error_counts=True)


def test_score_full_precision_pretty_format() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot use full_precision with pretty formatting",
    ):
        _score([CORRECT1], FORMAT_PRETTY, full_precision=True)


def test_score_unrecognized_format() -> None:
    with pytest.raises(
        ValueError,
        match="Unrecognized output format: bogus",
    ):
        _score([CORRECT1], "bogus")
