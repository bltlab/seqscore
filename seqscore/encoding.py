from abc import abstractmethod
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)

from attr import Attribute, attrib, attrs, validators

REPAIR_CONLL = "conlleval"
REPAIR_DISCARD = "discard"
REPAIR_NONE = "none"
SUPPORTED_REPAIRS = (REPAIR_CONLL, REPAIR_DISCARD, REPAIR_NONE)


def _validator_nonnegative(_inst: Any, _attr: Attribute, value: Any) -> None:
    if value < 0:
        raise ValueError(f"Negative value: {repr(value)}")


# Instantiate in advance for _validator_optional_nonempty_str
_optional_instance_of_str = validators.optional(validators.instance_of(str))


def _validator_optional_nonempty_str(_inst: Any, attr: Attribute, value: Any) -> None:
    # Check type
    _optional_instance_of_str(value, attr, value)
    # Check string isn't empty
    if not value:
        raise ValueError(f"Empty string: {repr(value)}")


# Type-specific implementations to work around type checker limitations. No, writing these as
# generic functions with type variables does not satisfy all type checkers.
def _tuplify_strs(strs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(strs)


def _tuplify_mentions(
    mentions: Iterable["Mention"],
) -> Tuple["Mention", ...]:
    return tuple(mentions)


@attrs(frozen=True, slots=True)
class Span:
    start: int = attrib(validator=_validator_nonnegative)
    end: int = attrib(validator=_validator_nonnegative)


@attrs(frozen=True, slots=True)
class Mention:
    span: Span = attrib()
    type: str = attrib()


@attrs(frozen=True, slots=True)
class SentenceProvenance:
    starting_line: int = attrib()
    source: Optional[str] = attrib()


@attrs(frozen=True, slots=True)
class LabeledSentence(Sequence[str]):
    tokens: Tuple[str, ...] = attrib(converter=_tuplify_strs)
    labels: Tuple[str, ...] = attrib(converter=_tuplify_strs)
    mentions: Tuple[Mention, ...] = attrib(default=(), converter=_tuplify_mentions)
    provenance: Optional[SentenceProvenance] = attrib(
        default=None, eq=False, kw_only=True
    )

    def __attrs_post_init__(self):
        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"Tokens ({len(self.tokens)}) and labels ({len(self.labels)}) "
                "must be of the same length"
            )
        if not self.tokens:
            raise ValueError("Tokens and labels must be non-empty")

        for label in self.labels:
            # Labels cannot be None or an empty string
            if not label:
                raise ValueError(f"Invalid label: {repr(label)}")

        for token in self.tokens:
            # Labels cannot be None or an empty string
            if not token:
                raise ValueError(f"Invalid token: {repr(token)}")

    @overload
    def __getitem__(self, index: int) -> str:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: slice) -> Tuple[str, ...]:
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[str, Tuple[str, ...]]:
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

    def tokens_with_labels(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(zip(self.tokens, self.labels))

    def span_tokens(self, span: Span) -> Tuple[str, ...]:
        return self.tokens[span.start : span.end]

    def mention_tokens(self, mention: Mention) -> Tuple[str, ...]:
        return self.span_tokens(mention.span)


@attrs
class _EncoderToken:
    entity_type: Optional[str] = attrib(validator=_validator_optional_nonempty_str)
    begin: bool = attrib(default=False, kw_only=True)
    inside: bool = attrib(default=False, kw_only=True)
    end: bool = attrib(default=False, kw_only=True)
    only: bool = attrib(default=False, kw_only=True)

    def __attrs_post_init__(self) -> None:
        # Make sure that exactly one of the flags is set
        count = (
            self.begin,
            self.inside,
            self.end,
            self.only,
        ).count(True)
        if count != 1:
            raise ValueError(
                f"Exactly one token flag should be set, found {count}: {repr(self)}"
            )


class Encoding(Protocol):
    label_delim: str = "-"
    outside: str = "O"
    begin: Optional[str]
    inside: Optional[str]
    end: Optional[str]
    only: Optional[str]

    valid_same_type_transitions: AbstractSet[Tuple[str, str]]
    valid_different_type_transitions: AbstractSet[Tuple[str, str]]

    def split_label(self, label: str) -> Tuple[str, Optional[str]]:
        splits = label.split(self.label_delim)
        if len(splits) == 1:
            return (label, None)
        elif len(splits) == 2:
            # Manually unpack just to appease type checking
            state, entity_type = splits
            return (state, entity_type)
        else:
            raise ValueError("Cannot parse label {!r}".format(label))

    def join_label(self, state: str, entity_type: str) -> str:
        if entity_type:
            return state + self.label_delim + entity_type
        else:
            assert state == self.outside
            return state

    def is_valid_transition(
        self,
        first_state: str,
        first_type: Optional[str],
        second_state: str,
        second_type: Optional[str],
    ) -> bool:
        transition = (first_state, second_state)
        if first_type == second_type:
            return transition in self.valid_same_type_transitions
        else:
            return transition in self.valid_different_type_transitions

    @abstractmethod
    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def encode_mentions(
        self, sentence: LabeledSentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def decode_mentions(self, sentence: LabeledSentence) -> List[Mention]:
        raise NotImplementedError


class EncodingError(Exception):
    pass


@attrs
class MentionBuilder:
    tokens: Tuple[str, ...] = attrib(converter=_tuplify_strs)

    start_idx: Optional[int] = attrib(default=None, init=False)
    entity_type: Optional[str] = attrib(default=None, init=False)

    def start_mention(self, start_idx: int, entity_type: str) -> None:
        # Check arguments
        assert start_idx >= 0
        assert entity_type

        # Check state
        if self.start_idx is not None:
            raise EncodingError(
                f"Mention has already been started at index {self.start_idx}"
            )
        if self.entity_type is not None:
            raise EncodingError(
                f"Mention has already been started with type {self.entity_type}"
            )

        self.start_idx = start_idx
        self.entity_type = entity_type

    def end_mention(self, end_idx: int) -> Mention:
        # Since end index is exclusive, cannot be zero
        assert end_idx > 0

        # Check state
        if self.start_idx is None:
            raise ValueError("No mention start index")
        if self.entity_type is None:
            raise ValueError("No mention entity type")

        mention = Mention(Span(self.start_idx, end_idx), self.entity_type)

        self.start_idx = None
        self.entity_type = None

        return mention

    def in_mention(self) -> bool:
        return self.start_idx is not None


class IO(Encoding):
    def __init__(self):
        self.inside = "I"

        self.begin = None
        self.end = None
        self.only = None

        self.valid_same_type_transitions = frozenset((("I", "I"), ("O", "O")))
        self.valid_different_type_transitions = frozenset(
            (("I", "I"), ("O", "I"), ("I", "O"))
        )

    def encode_mentions(
        self, sentence: LabeledSentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        raise NotImplementedError

    def decode_mentions(self, sentence: LabeledSentence) -> List[Mention]:
        raise NotImplementedError

    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        raise NotImplementedError


class BIO(IO):
    def __init__(self):
        super().__init__()
        self.begin = "B"

        self.valid_same_type_transitions = frozenset(
            (("B", "I"), ("B", "B"), ("I", "I"), ("I", "B"), ("O", "O"))
        )
        self.valid_different_type_transitions = frozenset(
            (("B", "B"), ("B", "O"), ("I", "B"), ("I", "O"), ("O", "B"))
        )

    def encode_mentions(
        self, sentence: LabeledSentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        raise NotImplementedError

    def decode_mentions(self, sentence: LabeledSentence) -> List[Mention]:
        mentions: List[Mention] = []
        builder = MentionBuilder(sentence.tokens)

        # We define this just to make it clear it will be defined regardless of the loop running,
        # even though it's guaranteed to run since sentences cannot be empty by construction.
        idx = 0

        for idx, (token, label) in enumerate(zip(sentence.tokens, sentence.labels)):
            state, entity_type = self.split_label(label)

            # End mention if needed. This is independent of whether we choose to begin a new one.
            # We end a mention if we are in a mention and the current state is not continue.
            if builder.in_mention() and state != self.inside:
                mentions.append(builder.end_mention(idx))

            # Begin a mention if needed
            if state == self.begin:
                builder.start_mention(idx, entity_type)
            # Check for valid continuation
            elif state == self.inside:
                if entity_type != builder.entity_type:
                    if builder.entity_type:
                        raise EncodingError(
                            f"Illegal use of {label} to continue {builder.entity_type}"
                        )
                    else:
                        raise EncodingError(f"Illegal use of {label} to begin a mention")
                # Check state
                assert builder.in_mention()
            # No action needed for outside (since ending mentions is mentioned above) other than
            # checking state.
            elif state == self.outside:
                assert not builder.in_mention()

        # Finish the last mention if needed
        if builder.in_mention():
            mentions.append(builder.end_mention(idx + 1))

        assert not builder.in_mention()

        return mentions

    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        # All of this is essentially the same as validation, but the labels can change during
        # iteration, so the design is slightly different.

        # Treat sentence as if preceded by "O"
        prev_label = self.outside
        prev_state, prev_entity_type = self.split_label(prev_label)

        # Range loop since we will modify the labels
        repaired_labels = list(labels)
        for idx in range(len(repaired_labels)):
            label = repaired_labels[idx]

            state, entity_type = self.split_label(label)
            if not self.is_valid_transition(
                prev_state, prev_entity_type, state, entity_type
            ):
                # For BIO, this can only happen when the current label has a type
                assert entity_type
                if method == REPAIR_CONLL:
                    # Treat this as the beginning of a new chunk
                    state = self.begin
                elif method == REPAIR_DISCARD:
                    # Treat this as O
                    state = self.outside
                    entity_type = None
                else:
                    raise ValueError(f"Unrecognized repair method: {method}")

                label = self.join_label(state, entity_type)
                repaired_labels[idx] = label

            prev_label, prev_state, prev_entity_type = (
                label,
                state,
                entity_type,
            )

        # Since BIO cannot have an illegal end-of sentence transition, no need to check
        return repaired_labels


class BIOES(BIO):
    def __init__(self):
        super().__init__()
        self.end = "E"
        self.only = "S"

        self.valid_same_type_transitions = frozenset(
            (
                ("B", "I"),
                ("B", "E"),
                ("B", "B"),
                ("B", "S"),
                ("I", "I"),
                ("I", "E"),
                ("E", "B"),
                ("E", "S"),
                ("O", "O"),
            )
        )
        self.valid_different_type_transitions = frozenset(
            (
                ("E", "B"),
                ("E", "O"),
                ("S", "S"),
                ("S", "B"),
                ("S", "O"),
                ("S", "S"),
                ("O", "B"),
                ("O", "S"),
            )
        )

    def decode_mentions(self, sentence: LabeledSentence) -> List[Mention]:
        raise NotImplementedError

    def encode_mentions(
        self, sentence: LabeledSentence, mentions: Sequence[Mention]
    ) -> Sequence[str]:
        raise NotImplementedError


# Declared mid-file so it can refer to classes in file
_ENCODING_NAMES: Dict[str, Encoding] = {
    "BIO": BIO(),
    "IO": IO(),
    "BIOES": BIOES(),
}
VALIDATION_SUPPORTED_ENCODINGS: Sequence[str] = tuple(sorted(_ENCODING_NAMES))
DECODING_SUPPORTED_ENCODINGS = ("BIO",)


def get_encoding(name: str) -> Encoding:
    name = name.upper()
    if name in _ENCODING_NAMES:
        return _ENCODING_NAMES[name]
    else:
        raise ValueError(f"Unknown encoder {repr(name)}")


@attrs
class ValidationError:
    msg: str = attrib()
    label: str = attrib()
    type: str = attrib()
    state: str = attrib()
    token: str = attrib()
    line_num: int = attrib()


@attrs
class ValidationResult:
    errors: Sequence[ValidationError] = attrib()
    n_tokens: int = attrib()
    repaired_labels: Optional[Tuple[str, ...]] = attrib(
        converter=_tuplify_strs, default=()
    )

    def is_valid(self) -> bool:
        return not self.errors

    def __len__(self):
        return self.n_tokens


def validate_sentence(
    tokens: Sequence[str],
    labels: Sequence[str],
    line_nums: Sequence[int],
    encoding: Encoding,
    *,
    repair: Optional[str] = None,
) -> ValidationResult:
    if not (len(tokens) == len(labels) == len(line_nums)):
        raise ValueError("Tokens, labels, and line numbers must be the same length")
    if not tokens:
        raise ValueError("Cannot validate empty sequences")

    errors: List[ValidationError] = []

    # Treat sentence as if preceded by "O"
    prev_label = encoding.outside
    prev_state, prev_entity_type = encoding.split_label(prev_label)
    # We initialize these to avoid warnings about them being uninitialized if the loop doesn't
    # run, but since we have checked for an empty sequence, the loop is guaranteed to run.
    token, label, line_num = "DUMMY_TOKEN", "DUMMY_LABEL", -1
    prev_token = token
    for token, label, line_num in zip(tokens, labels, line_nums):
        state, entity_type = encoding.split_label(label)
        if not encoding.is_valid_transition(
            prev_state, prev_entity_type, state, entity_type
        ):
            msg = (
                f"Invalid transition {prev_label} -> {label} for token {repr(token)} "
                + f"on line {line_num}"
            )
            errors.append(
                ValidationError(msg, label, entity_type, state, token, line_num)
            )
        prev_label, prev_state, prev_entity_type, prev_token = (
            label,
            state,
            entity_type,
            token,
        )

    # Treat sentence as if followed by "O"
    label = encoding.outside
    state, entity_type = encoding.split_label(label)
    if not encoding.is_valid_transition(prev_state, prev_entity_type, state, entity_type):
        msg = (
            f"Invalid transition {prev_label} -> {label} "
            + f"after token {prev_token} on line {line_num} at end of sentence"
        )
        errors.append(
            ValidationError(
                msg, prev_label, prev_entity_type, prev_state, prev_token, line_num
            )
        )

    if errors and repair:
        repaired_labels = encoding.repair_labels(labels, repair)
        return ValidationResult(errors, len(tokens), repaired_labels)
    else:
        return ValidationResult(errors, len(tokens))
