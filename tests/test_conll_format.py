from pathlib import Path

import pytest

from seqscore.conll import CoNLLFormatError, CoNLLIngester
from seqscore.encoding import REPAIR_NONE, get_encoding
from seqscore.validation import InvalidLabelError


def test_parse_comments_true() -> None:
    mention_encoding = get_encoding("BIO")
    ingester = CoNLLIngester(mention_encoding, parse_comment_lines=True)
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
    ingester = CoNLLIngester(mention_encoding)

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
        with pytest.raises(InvalidLabelError) as err:
            list(ingester.ingest(file, "test", REPAIR_NONE))
        assert (
            str(err.value)
            == "Could not parse label 'Comment' on line 1 of test during validation: Label 'Comment' does not have a state and entity type but is not outside ('O'). Expected the label to be of a format like '<STATE>-<ENTITY_TYPE>'. The first token '#' of this sentence starts with '#'. If it's a comment, consider enabling --parse-comment-lines."
        )

    comments_path = Path("tests") / "test_files" / "minimal_comments_3.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError) as err:
            list(ingester.ingest(file, "test", REPAIR_NONE))
        assert (
            str(err.value)
            == "Could not parse label 'fields' on line 1 of test during validation: Label 'fields' does not have a state and entity type but is not outside ('O'). Expected the label to be of a format like '<STATE>-<ENTITY_TYPE>'. The first token '#' of this sentence starts with '#'. If it's a comment, consider enabling --parse-comment-lines."
        )

    comments_path = Path("tests") / "test_files" / "minimal_comments_4.bio"
    with comments_path.open(encoding="utf8") as file:
        with pytest.raises(InvalidLabelError) as err:
            list(ingester.ingest(file, "test", REPAIR_NONE))
        assert (
            str(err.value)
            == "Could not parse label 'fields' on line 1 of test during validation: Label 'fields' does not have a state and entity type but is not outside ('O'). Expected the label to be of a format like '<STATE>-<ENTITY_TYPE>'. The first token '#' of this sentence starts with '#'. If it's a comment, consider enabling --parse-comment-lines."
        )
