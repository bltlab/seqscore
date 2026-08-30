from pathlib import Path

import pytest

from seqscore.conll import (
    DOCSTART,
    CoNLLFormatError,
    CoNLLIngester,
    LineSpec,
    _CoNLLToken,
    ingest_conll_file,
    write_docs_raw,
    write_docs_using_encoding,
)
from seqscore.encoding import REPAIR_NONE, EncodingError, get_encoding
from seqscore.model import AnnotatedSequence, LabeledSequence
from seqscore.util import file_fields_match
from seqscore.validation import InvalidLabelError

BIO = get_encoding("BIO")
LINE_SPEC = LineSpec(0, -1)


def test_parse_comments_true() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC, allow_comment_lines=True)
    comments_path = Path("tests") / "test_files" / "minimal_comments.bio"
    with comments_path.open(encoding="utf8") as file:
        documents = ingester.ingest(file, "test", REPAIR_NONE)

    assert len(documents) == 1
    sequences = documents[0]
    assert len(sequences) == 4
    assert sequences[0].comment == "#"
    assert sequences[1].comment == "# Comment"
    assert sequences[2].comment == "# Three fields"
    assert sequences[3].comment == "# Now four fields\n# And a second line"

    first_sent = sequences[0]

    assert first_sent[0] == "This"
    assert first_sent[7] == "#"
    assert first_sent[8] == "##"
    assert first_sent[9] == "#1"


def test_comment_before_docstart_stays_with_docstart() -> None:
    # A comment preceding a DOCSTART belongs to the document boundary and must
    # not be carried onto the first sentence of the document.
    ingester = CoNLLIngester(BIO, LINE_SPEC, allow_comment_lines=True)
    path = Path("tests") / "test_files" / "minimal_comments_docstart.bio"
    with path.open(encoding="utf8") as file:
        documents = ingester.ingest(file, "test", REPAIR_NONE)

    # The lone sentence keeps only its own comment, not the document comment.
    assert len(documents) == 1
    sequences = documents[0]
    assert len(sequences) == 1
    assert sequences[0].tokens == ("Hello",)
    assert sequences[0].comment == "# Sentence note"

    # The DOCSTART sequence itself carries the document comment.
    with path.open(encoding="utf8") as file:
        parsed = list(ingester._parse_file(file, "test", parse_comments=True))
    docstart_seq, docstart_comment = parsed[0]
    assert docstart_seq[0].is_docstart
    assert docstart_comment == "# Document ID: 7"


@pytest.mark.parametrize(
    "filename",
    [
        "minimal_comments.bio",
        "minimal_comments_1.bio",
        "minimal_comments_2.bio",
        "minimal_comments_3.bio",
        "minimal_comments_4.bio",
    ],
)
def test_comments_round_trip(filename: str, tmp_path: Path) -> None:
    # Ingesting a file with comments and writing it back out reproduces the
    # original file exactly, including single- and multi-line comments.
    input_path = Path("tests") / "test_files" / filename
    docs = ingest_conll_file(
        input_path,
        "BIO",
        "UTF-8",
        LINE_SPEC,
        ignore_document_boundaries=False,
        allow_comment_lines=True,
    )
    output_path = tmp_path / filename
    write_docs_using_encoding(docs, "BIO", "UTF-8", "\t", LINE_SPEC, output_path)
    assert output_path.read_text(encoding="utf8") == input_path.read_text(encoding="utf8")


def test_parse_comments_false() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    comments_path = Path("tests") / "test_files" / "minimal_comments_1.bio"
    with comments_path.open(encoding="utf8") as file:
        # err1 needs to not be reused below because the exception is a different type
        with pytest.raises(CoNLLFormatError) as err1:
            ingester.ingest(file, "test", REPAIR_NONE)
        assert (
            str(err1.value)
            == "Line 1 of test does not appear to be delimited and begins with #. Perhaps you want to use the --allow-comment-lines flag? Line contents: '#'"
        )

    comments_path = Path("tests") / "test_files" / "minimal_comments_2.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            ingester.ingest(file, "test", REPAIR_NONE)

    comments_path = Path("tests") / "test_files" / "minimal_comments_3.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            ingester.ingest(file, "test", REPAIR_NONE)

    comments_path = Path("tests") / "test_files" / "minimal_comments_4.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            ingester.ingest(file, "test", REPAIR_NONE)


def test_invalid_token_leading_space() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests") / "test_files" / "minimal_bio_empty_token.txt"
    with path.open(encoding="utf8") as file:
        with pytest.raises(ValueError) as err:
            ingester.ingest(file, "test", REPAIR_NONE)

    assert str(err.value) == "Invalid token '' on line 9 of test"


def test_bad_docstart() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests") / "test_files" / "minimal_bad_docstart.bio"
    with path.open(encoding="utf8") as file:
        with pytest.raises(CoNLLFormatError) as err:
            ingester.ingest(file, str(path), REPAIR_NONE)

    assert (
        str(err.value)
        == "Encountered -DOCSTART- at line 4 of tests/test_files/minimal_bad_docstart.bio in the middle of a sequence"
    )


def test_check_sequence() -> None:
    tokens = [
        _CoNLLToken(DOCSTART, "O", True, 0, ()),
        _CoNLLToken("Hello", "O", True, 0, ()),
    ]
    with pytest.raises(ValueError):
        CoNLLIngester._check_sequence(tokens)


def test_no_delims() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests") / "test_files" / "minimal_no_delims.bio"
    with path.open(encoding="utf8") as file:
        with pytest.raises(CoNLLFormatError) as err:
            ingester.ingest(file, str(path), REPAIR_NONE)

    assert (
        str(err.value)
        == "Line 1 of tests/test_files/minimal_no_delims.bio is not delimited by space or tab: 'ThisO'"
    )


def test_validate_with_docstart() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC, ignore_document_boundaries=False)
    # Check two variants, one with docstart in its own sentence and another with
    # docstart at the start of the sentence
    for filename in ("minimal_docstart1.bio", "minimal_docstart2.bio"):
        path = Path("tests") / "test_files" / filename
        with path.open(encoding="utf8") as file:
            ingester.validate(
                file,
                str(path),
            )


def test_repair_bad_name() -> None:
    path = Path("tests") / "conll_annotation" / "minimal.bio"
    with pytest.raises(ValueError) as err:
        ingest_conll_file(
            str(path),
            "BIOES",
            "UTF-8",
            LINE_SPEC,
            repair="conlleval",
            ignore_document_boundaries=False,
            allow_comment_lines=False,
        )

    assert str(err.value).startswith(
        "Cannot repair mention encoding BIOES using method conlleval."
    )


def test_bad_label1() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests") / "test_files" / "bad_label1.bio"
    with path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError) as err:
            ingester.ingest(file, str(path), repair=REPAIR_NONE)

    assert str(err.value).startswith(
        "Could not parse label 'GPE' on line 4 of tests/test_files/bad_label1.bio during validation"
    )


def test_bad_label2() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests") / "test_files" / "bad_label2.bio"
    with path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError) as err:
            ingester.ingest(file, str(path), repair=REPAIR_NONE)

    assert str(err.value).startswith(
        "Could not parse label 'OUT' on line 1 of tests/test_files/bad_label2.bio during validation"
    )


def test_write_docs_no_orig_fields(tmp_path: Path) -> None:
    sent1 = AnnotatedSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
        mentions=(),
    )
    sent2 = AnnotatedSequence.from_tokens_and_labels(
        (
            "University",
            "of",
            "Pennsylvania",
            "is",
            "in",
            "West",
            "Philadelphia",
            ",",
            "Pennsylvania",
            ".",
        ),
        ("B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC", "O"),
        BIO,
    )
    docs = [[sent1], [sent2]]
    output_file = tmp_path / "out.bio"
    write_docs_using_encoding(docs, "BIO", "utf-8", "\t", LINE_SPEC, output_file)
    assert file_fields_match(
        output_file, Path("tests") / "test_files" / "minimal_docstart1.bio", debug=True
    )


def test_write_docs_using_encoding_single_doc_no_docstart_by_default(
    tmp_path: Path,
) -> None:
    sent = AnnotatedSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
        mentions=(),
    )
    output_file = tmp_path / "out.bio"
    write_docs_using_encoding([[sent]], "BIO", "utf-8", " ", LINE_SPEC, output_file)
    assert DOCSTART not in output_file.read_text()


def test_write_docs_using_encoding_single_doc_always_write_docstart(
    tmp_path: Path,
) -> None:
    sent = AnnotatedSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
        mentions=(),
    )
    output_file = tmp_path / "out.bio"
    write_docs_using_encoding(
        [[sent]],
        "BIO",
        "utf-8",
        " ",
        LINE_SPEC,
        output_file,
        always_write_docstart=True,
    )
    text = output_file.read_text()
    assert text.count(DOCSTART) == 1
    assert text == "-DOCSTART- O\n\nThis O\nis O\na O\nsentence O\n. O\n\n"


def test_write_docs_raw_single_doc_no_docstart_by_default(tmp_path: Path) -> None:
    sent = LabeledSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
    )
    output_file = tmp_path / "out.bio"
    write_docs_raw([[sent]], "utf-8", " ", LINE_SPEC, output_file)
    assert DOCSTART not in output_file.read_text()


def test_write_docs_raw_single_doc_always_write_docstart(tmp_path: Path) -> None:
    sent = LabeledSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
    )
    output_file = tmp_path / "out.bio"
    write_docs_raw(
        [[sent]], "utf-8", " ", LINE_SPEC, output_file, always_write_docstart=True
    )
    text = output_file.read_text()
    assert text.count(DOCSTART) == 1
    assert text == "-DOCSTART- O\n\nThis O\nis O\na O\nsentence O\n. O\n\n"


def test_write_docs_raw_outside_label(
    tmp_path: Path,
) -> None:
    sent = LabeledSequence(tokens=("A",), labels=("O",))
    output_file = tmp_path / "out.bio"
    write_docs_raw(
        [[sent]],
        "utf-8",
        " ",
        LINE_SPEC,
        output_file,
        outside_label="NONE",
        always_write_docstart=True,
    )
    assert output_file.read_text().startswith("-DOCSTART- NONE\n")


def test_write_docs_raw_invalid_labels(
    tmp_path: Path,
) -> None:
    # Completely invalid labels are written without error
    sent = LabeledSequence(tokens=("A", "B", "C"), labels=("B-", "B-AAA", "I-YYY"))
    output_file = tmp_path / "out.bio"
    write_docs_raw(
        [[sent]],
        "utf-8",
        " ",
        LINE_SPEC,
        output_file,
    )
    assert output_file.read_text().startswith("A B-\nB B-AAA\nC I-YYY")


def test_write_docs_using_encoding_multi_doc_always_write_docstart(
    tmp_path: Path,
) -> None:
    sent1 = AnnotatedSequence(
        tokens=("This", "is", "a", "sentence", "."),
        labels=("O", "O", "O", "O", "O"),
        mentions=(),
    )
    sent2 = AnnotatedSequence(
        tokens=("Another", "sentence", "."),
        labels=("O", "O", "O"),
        mentions=(),
    )
    docs = [[sent1], [sent2]]
    default_file = tmp_path / "default.bio"
    forced_file = tmp_path / "forced.bio"
    write_docs_using_encoding(docs, "BIO", "utf-8", " ", LINE_SPEC, default_file)
    write_docs_using_encoding(
        docs, "BIO", "utf-8", " ", LINE_SPEC, forced_file, always_write_docstart=True
    )
    # Setting always_write_docstart=True has no effect.
    # DOCSTART is always written once per document.
    assert default_file.read_text() == forced_file.read_text()


_INVALID1_INPUT = "tests/conll_annotation/invalid1.bio"


def _ingest_invalid1_repaired(quiet: bool) -> list:
    return ingest_conll_file(
        _INVALID1_INPUT,
        "BIO",
        "UTF-8",
        LINE_SPEC,
        repair="conlleval",
        ignore_document_boundaries=False,
        allow_comment_lines=False,
        quiet=quiet,
    )


def test_ingest_conll_file_repair_suppresses_output_when_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # quiet suppresses the repair diagnostics but does not change the output.
    docs_no_quiet = _ingest_invalid1_repaired(quiet=False)
    assert "Used method conlleval to repair:" in capsys.readouterr().err

    docs_quiet = _ingest_invalid1_repaired(quiet=True)
    assert "Used method conlleval to repair:" not in capsys.readouterr().err

    assert len(docs_no_quiet) == len(docs_quiet) == 1
    for seq_no_quiet, seq_quiet in zip(docs_no_quiet[0], docs_quiet[0]):
        assert seq_no_quiet.tokens == seq_quiet.tokens
        assert seq_no_quiet.labels == seq_quiet.labels


def test_ingest_conll_file_empty_input_raises() -> None:
    # A completely empty file is rejected rather than producing no documents.
    with pytest.raises(CoNLLFormatError, match="contains no sequences"):
        ingest_conll_file(
            "tests/test_files/empty.txt",
            "BIO",
            "UTF-8",
            LINE_SPEC,
            repair=None,
            ignore_document_boundaries=False,
            allow_comment_lines=False,
            quiet=False,
        )


def test_validate_conll_file_empty_input_raises() -> None:
    # A completely empty file is rejected by validate as well.
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests/test_files/empty.txt")
    with pytest.raises(CoNLLFormatError, match="contains no sequences"):
        ingester.validate(path.open(encoding="utf8"), str(path))


def test_docstart_only_raises() -> None:
    # A trailing DOCSTART with no sequences is an empty document and is rejected
    # by both ingest and validate.
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests/test_files/docstart_only.txt")
    with pytest.raises(CoNLLFormatError, match="with no sequences following it"):
        ingester.ingest(path.open(encoding="utf8"), str(path), REPAIR_NONE)
    with pytest.raises(CoNLLFormatError, match="with no sequences following it"):
        ingester.validate(path.open(encoding="utf8"), str(path))


def test_docstart_with_no_sequences_before_next_docstart_raises() -> None:
    # A DOCSTART directly followed by another DOCSTART leaves an empty document.
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    path = Path("tests/test_files/docstart_empty_doc.txt")
    with pytest.raises(CoNLLFormatError, match="with no sequences before the"):
        ingester.ingest(path.open(encoding="utf8"), str(path), REPAIR_NONE)


@pytest.mark.parametrize(
    "filename",
    [
        "minimal_comments.bio",
        "minimal_comments_1.bio",
        "minimal_comments_2.bio",
        "minimal_comments_3.bio",
        "minimal_comments_4.bio",
    ],
)
def test_discard_comments(filename: str, tmp_path: Path) -> None:
    # Writing with discard_comments drops the comment lines and leaves the tokens
    # and sequence breaks untouched
    input_path = Path("tests") / "test_files" / filename
    docs = ingest_conll_file(
        input_path,
        "BIO",
        "UTF-8",
        LINE_SPEC,
        ignore_document_boundaries=False,
        allow_comment_lines=True,
    )
    output_path = tmp_path / filename
    write_docs_using_encoding(
        docs, "BIO", "UTF-8", "\t", LINE_SPEC, output_path, discard_comments=True
    )

    output_lines = output_path.read_text(encoding="utf8").splitlines()
    input_lines = input_path.read_text(encoding="utf8").splitlines()
    assert not any(line.startswith("#") and "\t" not in line for line in output_lines)
    # Every non-comment line survives in order, tokens included
    assert output_lines == [
        line for line in input_lines if not (line.startswith("#") and "\t" not in line)
    ]


def test_write_docs_iob_lenient_rejected(tmp_path: Path) -> None:
    docs = ingest_conll_file(
        Path("tests") / "conll_annotation" / "minimal_lenient.iob",
        "IOB-lenient",
        "UTF-8",
        LINE_SPEC,
        ignore_document_boundaries=False,
        allow_comment_lines=False,
    )
    with pytest.raises(EncodingError):
        write_docs_using_encoding(
            docs,
            "IOB-lenient",
            "UTF-8",
            "\t",
            LINE_SPEC,
            tmp_path / "out.txt",
        )
