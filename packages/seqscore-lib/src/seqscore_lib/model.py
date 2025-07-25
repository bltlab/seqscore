from collections.abc import Iterable, Sequence
from itertools import repeat
from typing import Annotated, Any, Optional, Union, overload

from pydantic import BaseModel, BeforeValidator, Field


def _validator_nonnegative(value: Any) -> Any:
    if value < 0:
        raise ValueError(f"Negative value: {repr(value)}")
    else:
        return value


def _tuplify_mentions(
    mentions: Iterable["Mention"],
) -> tuple["Mention", ...]:
    return tuple(mentions)


class Span(BaseModel, frozen=True):
    start: Annotated[int, BeforeValidator(_validator_nonnegative)]
    end: Annotated[int, BeforeValidator(_validator_nonnegative)]

    def model_post_init(self, context: Any) -> None:
        if not self.end > self.start:
            raise ValueError(
                f"End of span ({self.end}) must be greater than start ({self.start}"
            )

    def __len__(self) -> int:
        return self.end - self.start


class Mention(BaseModel, frozen=True):
    span: Span
    type: str = Field(min_length=1, description="Must be a non-empty string")

    def __len__(self) -> int:
        return len(self.span)

    def with_type(self, new_type: str) -> "Mention":
        return Mention(span=self.span, type=new_type)


class SequenceProvenance(BaseModel, frozen=True):
    starting_line: int
    source: Optional[str]


class LabeledSequence(BaseModel, frozen=True):
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    mentions: tuple[Mention, ...] = tuple()
    other_fields: Optional[tuple[tuple[str, ...], ...]] = None
    provenance: Optional[SequenceProvenance] = None
    comment: Optional[str] = None

    def model_post_init(self, context: Any) -> None:
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
            tokens=self.tokens,
            labels=self.labels,
            mentions=tuple(mentions),
            provenance=self.provenance,
        )

    # Pydantic doesn't support excluding certain fields when it generates
    # its default `__hash__` method when `frozen=True`
    # To get around that limitation, define custom `__hash__` method
    def __hash__(self) -> int:
        # Do not hash `provenance` and `comment`
        return hash((self.tokens, self.labels, self.mentions, self.other_fields))

    # Do not check eq with `provenance` and `comment` fields
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabeledSequence):
            return NotImplemented

        return (
            self.tokens == other.tokens
            and self.labels == other.labels
            and self.mentions == other.mentions
            and self.other_fields == other.other_fields
        )

    @overload
    def __getitem__(self, index: int) -> str:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[str, tuple[str, ...]]:
        return self.tokens[i]

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
