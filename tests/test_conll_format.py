from pathlib import Path

import pytest

from seqscore.conll import CoNLLFormatError, CoNLLIngester, LineSpec
from seqscore.encoding import REPAIR_NONE, get_encoding
from seqscore.validation import InvalidLabelError


def test_parse_comments_true() -> None:
    mention_encoding = get_encoding("BIO")
    line_spec = LineSpec(0, 1)
    ingester = CoNLLIngester(mention_encoding, line_spec, parse_comment_lines=True)
    comments_path = Path("tests") / "test_files" / "minimal_comments.bio"
    with comments_path.open(encoding="utf8") as file:
        documents = list(ingester.ingest(file, "test", REPAIR_NONE))

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
    mention_encoding = get_encoding("BIO")
    line_spec = LineSpec(0, 1)
    ingester = CoNLLIngester(mention_encoding, line_spec)

    comments_path = Path("tests") / "test_files" / "minimal_comments_1.bio"
    with comments_path.open(encoding="utf8") as file:
        # err1 needs to not be reused below because the exception is a different type
        with pytest.raises(CoNLLFormatError) as err1:
            list(ingester.ingest(file, "test", REPAIR_NONE))
        assert (
            str(err1.value)
            == "Line 1 of test does not appear to be delimited and begins with #. Perhaps you want to use the --parse-comment-lines flag? Line contents: '#'"
        )

    comments_path = Path("tests") / "test_files" / "minimal_comments_2.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            list(ingester.ingest(file, "test", REPAIR_NONE))

    comments_path = Path("tests") / "test_files" / "minimal_comments_3.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            list(ingester.ingest(file, "test", REPAIR_NONE))

    comments_path = Path("tests") / "test_files" / "minimal_comments_4.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError):
            list(ingester.ingest(file, "test", REPAIR_NONE))


def test_invalid_token_leading_space() -> None:
    mention_encoding = get_encoding("BIO")
    line_spec = LineSpec(0, -1)
    ingester = CoNLLIngester(mention_encoding, line_spec)

    path = Path("tests") / "test_files" / "minimal_bio_empty_token.txt"
    with path.open(encoding="utf8") as file:
        with pytest.raises(ValueError) as err:
            list(ingester.ingest(file, "test", REPAIR_NONE))

    assert str(err.value) == "Invalid token '' on line 9 of test"
