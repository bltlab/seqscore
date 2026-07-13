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
from seqscore.encoding import REPAIR_NONE, get_encoding
from seqscore.model import AnnotatedSequence, LabeledSequence
from seqscore.util import file_fields_match
from seqscore.validation import InvalidLabelError

BIO = get_encoding("BIO")
LINE_SPEC = LineSpec(0, -1)


def test_parse_comments_true() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC, parse_comment_lines=True)
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


def test_parse_comments_false() -> None:
    ingester = CoNLLIngester(BIO, LINE_SPEC)
    comments_path = Path("tests") / "test_files" / "minimal_comments_1.bio"
    with comments_path.open(encoding="utf8") as file:
        # err1 needs to not be reused below because the exception is a different type
        with pytest.raises(CoNLLFormatError) as err1:
            ingester.ingest(file, "test", REPAIR_NONE)
        assert (
            str(err1.value)
            == "Line 1 of test does not appear to be delimited and begins with #. Perhaps you want to use the --parse-comment-lines flag? Line contents: '#'"
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
            parse_comment_lines=False,
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
