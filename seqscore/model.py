from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Union,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from seqscore.encoding import Encoding  # pragma: no cover

__all__ = [
    "AnnotatedSequence",
    "LabeledSequence",
    "Mention",
    "SequenceProvenance",
    "Span",
    "TokenSequence",
]


@dataclass(frozen=True, slots=True)
class Span:
    """A token index range [start, end) within a sequence.

    The start index is inclusive, and the end index is exclusive.
    Both start and end must be non-negative, and end must be
    greater than start. Zero-length spans are not allowed.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Start ({self.start}) cannot be negative")

        if self.end < 0:
            raise ValueError(f"End ({self.end}) cannot be negative")

        if not self.end > self.start:
            raise ValueError(
                f"End of span ({self.end}) must be greater than start ({self.start}"
            )

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Mention:
    """A typed span representing a named entity or chunk.

    Combines a Span with a string entity type.
    """

    span: Span
    type: str

    def __post_init__(self) -> None:
        if not isinstance(self.type, str):
            raise TypeError(f"Expected str for type, got {type(self.type).__name__}")
        if not self.type:
            raise ValueError(f"Empty string for type: {repr(self.type)}")

    def __len__(self) -> int:
        return len(self.span)

    def with_type(self, new_type: str) -> "Mention":
        """Return a new Mention with the same span and the provided type."""
        return Mention(self.span, new_type)


@dataclass(frozen=True, slots=True)
class SequenceProvenance:
    """Origin of a sequence, with a starting line and optional source name."""

    starting_line: int
    source: Optional[str]


@runtime_checkable
class TokenSequence(Protocol):
    """Protocol for a sequence of tokens with optional metadata."""

    @property
    def tokens(self) -> tuple[str, ...]: ...

    @property
    def token_fields(self) -> Optional[tuple[tuple[str, ...], ...]]: ...

    @property
    def provenance(self) -> Optional[SequenceProvenance]: ...

    @property
    def comment(self) -> Optional[str]: ...

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, i: Union[int, slice]) -> Union[str, tuple[str, ...]]: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...

    def span_tokens(self, span: Span) -> tuple[str, ...]: ...

    def tokens_with_fields(
        self,
    ) -> tuple[tuple[str, Optional[tuple[str, ...]]], ...]: ...


class _SequenceBase(BaseModel, Sequence[str]):
    """Shared base for token sequences with labels."""

    model_config = ConfigDict(frozen=True)

    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    token_fields: Optional[tuple[tuple[str, ...], ...]] = None
    provenance: Optional[SequenceProvenance] = None
    comment: Optional[str] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "_SequenceBase":
        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"Tokens ({len(self.tokens)}) and labels ({len(self.labels)}) "
                "must be of the same length"
            )
        if not self.tokens:
            raise ValueError("Tokens and labels must be non-empty")

        if self.token_fields and len(self.tokens) != len(self.token_fields):
            raise ValueError(
                f"Tokens ({len(self.tokens)}) and token_fields ({len(self.token_fields)}) "
                "must be of the same length"
            )

        for idx, label in enumerate(self.labels):
            if not label:
                raise ValueError(f"Invalid label at sequence index {idx}: {repr(label)}")

        for idx, token in enumerate(self.tokens):
            if not token:
                raise ValueError(f"Invalid token at sequence index {idx}: {repr(token)}")

        return self

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, i: Union[int, slice]) -> Union[str, tuple[str, ...]]:
        return self.tokens[i]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return " ".join(
            "/".join((token, label)) for token, label in zip(self.tokens, self.labels)
        )

    def tokens_with_labels(self) -> tuple[tuple[str, str], ...]:
        """Return a tuple of (token, label) tuples."""
        return tuple(zip(self.tokens, self.labels))

    def tokens_with_fields(
        self,
    ) -> tuple[tuple[str, Optional[tuple[str, ...]]], ...]:
        """Return a tuple of (token, token_fields) pairs, with None if token_fields is absent."""
        if self.token_fields:
            return tuple(zip(self.tokens, self.token_fields))
        else:
            return tuple(zip(self.tokens, repeat(None)))

    def span_tokens(self, span: Span) -> tuple[str, ...]:
        """Return the tokens included in the given span."""
        return self.tokens[span.start : span.end]


class LabeledSequence(_SequenceBase):
    """A sequence of tokens with their labels.

    This class only contains labels and tokens. For mentions, use
    AnnotatedSequence.

    Equality and hashing are defined using only the tokens and labels.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabeledSequence):
            return False
        return self.tokens == other.tokens and self.labels == other.labels

    def __hash__(self) -> int:
        return hash((self.tokens, self.labels))


class AnnotatedSequence(_SequenceBase):
    """A sequence of tokens with labels and decoded mentions.

    Equality and hashing are defined using only the tokens, labels, and mentions.
    """

    mentions: tuple[Mention, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotatedSequence):
            return False
        return (
            self.tokens == other.tokens
            and self.labels == other.labels
            and self.mentions == other.mentions
        )

    def __hash__(self) -> int:
        return hash((self.tokens, self.labels, self.mentions))

    def with_mentions(self, mentions: Sequence[Mention]) -> "AnnotatedSequence":
        """Return a copy of this sequence with different mentions."""
        return AnnotatedSequence(
            tokens=self.tokens,
            labels=self.labels,
            mentions=tuple(mentions),
            token_fields=self.token_fields,
            provenance=self.provenance,
            comment=self.comment,
        )

    def mention_tokens(self, mention: Mention) -> tuple[str, ...]:
        """Return the tokens included in the given mention."""
        return self.span_tokens(mention.span)

    @classmethod
    def from_tokens_and_labels(
        cls,
        tokens: Sequence[str],
        labels: Sequence[str],
        encoding: "Encoding",
        **kwargs: Any,
    ) -> "AnnotatedSequence":
        """Create an AnnotatedSequence by decoding mentions from labels using the given encoding."""
        mentions = encoding.decode_labels(labels)
        return cls(
            tokens=tuple(tokens), labels=tuple(labels), mentions=tuple(mentions), **kwargs
        )
