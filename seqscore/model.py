from collections.abc import Iterable, Iterator, Sequence
from itertools import repeat
from typing import Any, Optional, Union, overload

from attr import Attribute, attrib, attrs

from seqscore.util import (
    tuplify_optional_nested_strs,
    tuplify_strs,
    validator_nonempty_str,
)


def _validator_nonnegative(_inst: Any, _attr: Attribute, value: Any) -> None:
    if value < 0:
        raise ValueError(f"Negative value: {repr(value)}")


def _tuplify_mentions(
    mentions: Iterable["Mention"],
) -> tuple["Mention", ...]:
    return tuple(mentions)


@attrs(frozen=True, slots=True)
class Span:
    start: int = attrib(validator=_validator_nonnegative)
    end: int = attrib(validator=_validator_nonnegative)

    def __attrs_post_init__(self) -> None:
        if not self.end > self.start:
            raise ValueError(
                f"End of span ({self.end}) must be greater than start ({self.start}"
            )

    def __len__(self) -> int:
        return self.end - self.start


@attrs(frozen=True, slots=True)
class Mention:
    span: Span = attrib()
    type: str = attrib(validator=validator_nonempty_str)

    def __len__(self) -> int:
        return len(self.span)

    def with_type(self, new_type: str) -> "Mention":
        return Mention(self.span, new_type)


@attrs(frozen=True, slots=True)
class SequenceProvenance:
    starting_line: int = attrib()
    source: Optional[str] = attrib()


@attrs(frozen=True, slots=True)
class LabeledSequence(Sequence[str]):
    tokens: tuple[str, ...] = attrib(converter=tuplify_strs)
    labels: tuple[str, ...] = attrib(converter=tuplify_strs)
    mentions: tuple[Mention, ...] = attrib(default=(), converter=_tuplify_mentions)
    other_fields: Optional[tuple[tuple[str, ...], ...]] = attrib(
        default=None, kw_only=True, converter=tuplify_optional_nested_strs
    )
    provenance: Optional[SequenceProvenance] = attrib(
        default=None, eq=False, kw_only=True
    )
    comment: Optional[str] = attrib(default=None, eq=False, kw_only=True)

    def __attrs_post_init__(self) -> None:
        # TODO: Check for overlapping mentions

        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"Tokens ({len(self.tokens)}) and labels ({len(self.labels)}) "
                "must be of the same length"
            )
        if not self.tokens:
            raise ValueError("Tokens and labels must be non-empty")

        if self.other_fields and len(self.tokens) != len(self.other_fields):
            raise ValueError(
                f"Tokens ({len(self.tokens)}) and other_fields ({len(self.other_fields)}) "
                "must be of the same length"
            )

        for label in self.labels:
            # Labels cannot be None or an empty string
            if not label:
                raise ValueError(f"Invalid label: {repr(label)}")

        for token in self.tokens:
            # Labels cannot be None or an empty string
            if not token:
                raise ValueError(f"Invalid token: {repr(token)}")

    def with_mentions(self, mentions: Sequence[Mention]) -> "LabeledSequence":
        return LabeledSequence(
            self.tokens, self.labels, mentions, provenance=self.provenance
        )

    @overload
    def __getitem__(self, index: int) -> str:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[str, tuple[str, ...]]:
        return self.tokens[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def __len__(self) -> int:
        # Guaranteed that labels and tokens are same length by construction
        return len(self.tokens)

    def __str__(self) -> str:
        return " ".join(
            "/".join((token, label)) for token, label in zip(self.tokens, self.labels)
        )

    def tokens_with_labels(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.tokens, self.labels))

    def tokens_with_other_fields(
        self,
    ) -> tuple[tuple[str, Optional[tuple[str, ...]]], ...]:
        if self.other_fields:
            return tuple(zip(self.tokens, self.other_fields))
        else:
            return tuple(zip(self.tokens, repeat(None)))

    def span_tokens(self, span: Span) -> tuple[str, ...]:
        return self.tokens[span.start : span.end]

    def mention_tokens(self, mention: Mention) -> tuple[str, ...]:
        return self.span_tokens(mention.span)
