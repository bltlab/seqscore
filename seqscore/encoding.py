from abc import abstractmethod
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

from attr import Factory, attrib, attrs
from typing_extensions import Protocol

from seqscore.model import LabeledSequence, Mention, Span

REPAIR_CONLL = "conlleval"
REPAIR_DISCARD = "discard"
REPAIR_NONE = "none"
SUPPORTED_REPAIR_METHODS = (REPAIR_CONLL, REPAIR_DISCARD, REPAIR_NONE)


class EncodingDialect(Protocol):
    label_delim: str
    outside: str
    begin: Optional[str]
    inside: Optional[str]
    end: Optional[str]
    single: Optional[str]


class BIOESDialect(EncodingDialect):
    def __init__(self):
        self.label_delim = "-"
        self.begin = "B"
        self.inside = "I"
        self.outside = "O"
        self.end = "E"
        self.single = "S"


class BILOUDialect(BIOESDialect):
    def __init__(self):
        super().__init__()
        self.end = "L"
        self.single = "U"


class BMESDialect(BIOESDialect):
    def __init__(self):
        super().__init__()
        self.inside = "M"


class BMEOWDialect(BMESDialect):
    def __init__(self):
        super().__init__()
        self.single = "W"


class Encoding(Protocol):
    dialect: EncodingDialect

    valid_same_type_transitions: AbstractSet[Tuple[str, str]]
    valid_different_type_transitions: AbstractSet[Tuple[str, str]]

    def split_label(self, label: str) -> Tuple[str, Optional[str]]:
        splits = label.split(self.dialect.label_delim, maxsplit=1)
        if len(splits) == 1:
            if label != self.dialect.outside:
                raise EncodingError(
                    f"Label {repr(label)} does not have a state and entity type "
                    + f"but is not outside ({repr(self.dialect.outside)})"
                )
            return (label, None)
        elif len(splits) == 2:
            # Manually unpack just to appease type checking
            state, entity_type = splits
            if state == self.dialect.outside:
                raise EncodingError(
                    f"Label {repr(label)} has an entity type but is outside"
                )
            return (state, entity_type)
        else:  # pragma: no cover
            # Since maxsplit=1 for split, this is unreachable
            raise EncodingError(f"Cannot parse label {repr(label)}")

    def join_label(self, state: str, entity_type: Optional[str]) -> str:
        if entity_type:
            assert (
                state != self.dialect.outside
            ), "Entity type must be None for outside state"
            return state + self.dialect.label_delim + entity_type
        else:
            assert (
                state == self.dialect.outside
            ), "Entity type cannot be None for non-outside states"
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
    def is_valid_state(self, state: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def encode_mentions(
        self, mentions: Sequence[Mention], sequence_length: int
    ) -> Sequence[str]:
        raise NotImplementedError

    def encode_sequence(
        self,
        sequence: LabeledSequence,
    ) -> Sequence[str]:
        return self.encode_mentions(sequence.mentions, len(sequence))

    @abstractmethod
    def decode_labels(self, labels: Sequence[str]) -> List[Mention]:
        raise NotImplementedError

    def decode_sequence(self, sequence: LabeledSequence) -> List[Mention]:
        return self.decode_labels(sequence.labels)

    def supported_repair_methods(self) -> Tuple[str, ...]:
        raise NotImplementedError


class EncodingError(Exception):
    pass


class IO(Encoding):
    def __init__(self, dialect: EncodingDialect):
        self.dialect: EncodingDialect = dialect

        inside = dialect.inside
        outside = dialect.outside

        self.valid_same_type_transitions = frozenset(
            ((inside, inside), (outside, outside))
        )
        self.valid_different_type_transitions = frozenset(
            ((inside, inside), (outside, inside), (inside, outside))
        )

        self._valid_states = {inside, outside}

    def is_valid_state(self, state: str) -> bool:
        return state in self._valid_states

    def encode_mentions(
        self, mentions: Sequence[Mention], sequence_length: int
    ) -> Sequence[str]:
        inside = self.dialect.inside
        outside = self.dialect.outside
        output_labels = [outside] * sequence_length

        for mention in mentions:
            span = mention.span
            label = self.join_label(inside, mention.type)
            output_labels[span.start] = label
            for idx in range(span.start + 1, span.end):
                output_labels[idx] = label

        return output_labels

    def repair_labels(self, labels: Sequence[str], method: str) -> Sequence[str]:
        raise NotImplementedError

    def decode_labels(self, labels: Sequence[str]) -> List[Mention]:
        builder = _MentionBuilder()

        inside = self.dialect.inside
        outside = self.dialect.outside

        for idx, label in enumerate(labels):
            state, entity_type = self.split_label(label)

            if state == inside:
                if builder.in_mention():
                    if entity_type != builder.entity_type:
                        # End mention, start new one
                        builder.end_mention(idx)
                        builder.start_mention(idx, entity_type)
                    # Otherwise, nothing to do, just continue
                else:
                    # Begin new mention
                    builder.start_mention(idx, entity_type)
            else:
                assert state == outside
                # End previous mention if needed
                if builder.in_mention():
                    builder.end_mention(idx)

        # Finish the last mention if needed
        if builder.in_mention():
            builder.end_mention(len(labels))

        assert not builder.in_mention()
        return builder.mentions


class IOB(Encoding):
    def __init__(self, dialect: EncodingDialect):
        self.dialect = dialect

        inside = dialect.inside
        outside = dialect.outside
        begin = dialect.begin

        self.valid_same_type_transitions = frozenset(
            (
                (begin, begin),
                (begin, inside),
                (inside, inside),
                (inside, begin),
                (outside, outside),
            )
        )
        self.valid_different_type_transitions = frozenset(
            (
                # You might think I->B is allowed for different types, but it isn't.
                # Correctly-encoded IOB only uses B for I-X to B-X (same-type) transitions.
                (begin, inside),
                (begin, outside),
                (inside, inside),
                (inside, outside),
                (outside, inside),
            )
        )

        self._valid_states = {inside, outside, begin}

    def is_valid_state(self, state: str) -> bool:
        return state in self._valid_states

    def decode_labels(self, labels: Sequence[str]) -> List[Mention]:
        builder = _MentionBuilder()

        inside = self.dialect.inside
        outside = self.dialect.outside
        begin = self.dialect.begin

        for idx, label in enumerate(labels):
            state, entity_type = self.split_label(label)

            if state == begin:
                # Begin only allowed if previous entity type is the same as current
                if not builder.in_mention() or (
                    builder.in_mention() and entity_type != builder.entity_type
                ):
                    raise EncodingError(
                        "Begin only allowed after a mention of the same type"
                    )
                builder.end_mention(idx)
                builder.start_mention(idx, entity_type)
            elif state == inside:
                if builder.in_mention():
                    if entity_type != builder.entity_type:
                        # End mention, start new one
                        builder.end_mention(idx)
                        builder.start_mention(idx, entity_type)
                    # Otherwise, nothing to do, just continue
                else:
                    # Begin new mention
                    builder.start_mention(idx, entity_type)
            else:
                assert state == outside
                # End previous mention if needed
                if builder.in_mention():
                    builder.end_mention(idx)

        # Finish the last mention if needed
        if builder.in_mention():
            builder.end_mention(len(labels))

        assert not builder.in_mention()
        return builder.mentions

    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        if method == REPAIR_NONE:
            raise ValueError(f"Cannot perform repair with method {repr(method)}")

        if method != REPAIR_CONLL:
            raise ValueError(f"Only repair method {REPAIR_CONLL} is supported for IOB")

        begin = self.dialect.begin
        inside = self.dialect.inside

        # Treat sequence as if preceded by outside
        prev_label = self.dialect.outside
        prev_state, prev_entity_type = self.split_label(prev_label)

        # Range loop since we will modify the labels during iteration
        repaired_labels = list(labels)
        for idx in range(len(repaired_labels)):
            label = repaired_labels[idx]

            state, entity_type = self.split_label(label)
            if not self.is_valid_transition(
                prev_state, prev_entity_type, state, entity_type
            ):
                # The only invalid transition is O-B or mismatched type I-B or B-B. In all cases,
                # the solution is changing B to I.
                assert state == begin
                assert entity_type
                state = inside

                label = self.join_label(state, entity_type)
                repaired_labels[idx] = label

            prev_label, prev_state, prev_entity_type = (
                label,
                state,
                entity_type,
            )

        # Since IOB cannot have an illegal end-of sequence transition, no need to check
        return repaired_labels

    def encode_mentions(
        self, mentions: Sequence[Mention], sequence_length: int
    ) -> Sequence[str]:
        begin = self.dialect.begin
        inside = self.dialect.inside
        outside = self.dialect.outside
        output_labels = [outside] * sequence_length

        last_end: Optional[int] = None
        last_type: Optional[str] = None
        for mention in mentions:
            span = mention.span

            start_state = (
                begin if span.start == last_end and mention.type == last_type else inside
            )
            start_label = self.join_label(start_state, mention.type)
            output_labels[span.start] = start_label

            inside_label = self.join_label(inside, mention.type)
            for idx in range(span.start + 1, span.end):
                output_labels[idx] = inside_label

            last_end = span.end
            last_type = mention.type

        return output_labels

    def supported_repair_methods(self) -> Tuple[str, ...]:
        return (REPAIR_CONLL,)


class BIO(Encoding):
    def __init__(self, dialect: EncodingDialect):
        self.dialect = dialect

        inside = dialect.inside
        outside = dialect.outside
        begin = dialect.begin

        self.valid_same_type_transitions = frozenset(
            (
                (begin, inside),
                (begin, begin),
                (inside, inside),
                (inside, begin),
                (outside, outside),
            )
        )
        self.valid_different_type_transitions = frozenset(
            (
                (begin, begin),
                (begin, outside),
                (inside, begin),
                (inside, outside),
                (outside, begin),
            )
        )
        self._valid_states = {begin, inside, outside}

    def is_valid_state(self, state: str) -> bool:
        return state in self._valid_states

    def encode_mentions(
        self, mentions: Sequence[Mention], sequence_length: int
    ) -> Sequence[str]:
        begin = self.dialect.begin
        inside = self.dialect.inside
        outside = self.dialect.outside
        output_labels = [outside] * sequence_length

        for mention in mentions:
            span = mention.span
            start_label = self.join_label(begin, mention.type)
            output_labels[span.start] = start_label

            inside_label = self.join_label(inside, mention.type)
            for idx in range(span.start + 1, span.end):
                output_labels[idx] = inside_label

        return output_labels

    def decode_labels(self, labels: Sequence[str]) -> List[Mention]:
        builder = _MentionBuilder()

        begin = self.dialect.begin
        inside = self.dialect.inside
        outside = self.dialect.outside

        # We define this just to make it clear it will be defined regardless of the loop running,
        # even though it's guaranteed to run since sequences cannot be empty by construction.
        idx = 0

        for idx, label in enumerate(labels):
            state, entity_type = self.split_label(label)

            # End mention if needed. This is independent of whether we choose to begin a new one.
            # We end a mention if we are in a mention and the current state is not continue.
            if builder.in_mention() and state != inside:
                builder.end_mention(idx)

            # Begin a mention if needed
            if state == begin:
                builder.start_mention(idx, entity_type)
            # Check for valid continuation
            elif state == inside:
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
            elif state == outside:
                assert not builder.in_mention()

        # Finish the last mention if needed
        if builder.in_mention():
            builder.end_mention(idx + 1)

        assert not builder.in_mention()
        return builder.mentions

    def repair_labels(
        self,
        labels: Sequence[str],
        method: str,
    ) -> Sequence[str]:
        if method == REPAIR_NONE:
            raise ValueError(f"Cannot perform repair with method {repr(method)}")

        if method not in SUPPORTED_REPAIR_METHODS:
            raise ValueError(f"Unknown repair method {repr(method)}")

        begin = self.dialect.begin
        outside = self.dialect.outside

        # Treat sequence as if preceded by outside
        prev_label = outside
        prev_state, prev_entity_type = self.split_label(prev_label)

        # Range loop since we will modify the labels during iteration
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
                    state = begin
                elif method == REPAIR_DISCARD:
                    # Treat this as O
                    state = outside
                    entity_type = None
                else:  # pragma: no cover
                    # We can only hit this if we add something to SUPPORTED_REPAIR_METHODS but
                    # fail to create a case for it
                    raise ValueError(f"Unrecognized repair method: {method}")

                label = self.join_label(state, entity_type)
                repaired_labels[idx] = label

            prev_label, prev_state, prev_entity_type = (
                label,
                state,
                entity_type,
            )

        # Since BIO cannot have an illegal end-of sequence transition, no need to check
        return repaired_labels

    def supported_repair_methods(self) -> Tuple[str, ...]:
        return (REPAIR_CONLL, REPAIR_DISCARD)


class BIOES(Encoding):
    def __init__(self, dialect: EncodingDialect):
        self.dialect = dialect

        begin = dialect.begin
        inside = dialect.inside
        outside = dialect.outside
        end = dialect.end
        single = dialect.single

        self.valid_same_type_transitions = frozenset(
            (
                (begin, inside),
                (begin, end),
                (begin, begin),
                (begin, single),
                (inside, inside),
                (inside, end),
                (end, begin),
                (end, single),
                (single, begin),
                (single, single),
                (outside, outside),
            )
        )
        self.valid_different_type_transitions = frozenset(
            (
                (end, begin),
                (end, outside),
                (end, single),
                (single, begin),
                (single, outside),
                (single, single),
                (outside, begin),
                (outside, single),
            )
        )

        self._valid_states = {begin, inside, outside, end, single}

    def is_valid_state(self, state: str) -> bool:
        return state in self._valid_states

    def repair_labels(self, labels: Sequence[str], method: str) -> Sequence[str]:
        raise NotImplementedError

    def decode_labels(self, labels: Sequence[str]) -> List[Mention]:
        builder = _MentionBuilder()

        begin = self.dialect.begin
        inside = self.dialect.inside
        outside = self.dialect.outside
        end = self.dialect.end
        single = self.dialect.single

        for idx, label in enumerate(labels):
            state, entity_type = self.split_label(label)

            if state == single:
                assert not builder.in_mention()
                # Begin and end a mention
                builder.start_mention(idx, entity_type)
                builder.end_mention(idx + 1)
            elif state == begin:
                assert not builder.in_mention()
                builder.start_mention(idx, entity_type)
            elif state == end:
                assert builder.in_mention()
                builder.end_mention(idx + 1)
            elif state == inside:
                # Nothing to do but check state
                assert builder.in_mention()
            else:
                # Nothing to do but check state
                assert state == outside
                assert not builder.in_mention()

        # Since mentions are ended by single or end, we can't still be in a mention at the end
        assert not builder.in_mention()

        return builder.mentions

    def encode_mentions(
        self, mentions: Sequence[Mention], sequence_length: int
    ) -> Sequence[str]:
        begin = self.dialect.begin
        inside = self.dialect.inside
        end = self.dialect.end
        single = self.dialect.single
        outside = self.dialect.outside
        output_labels = [outside] * sequence_length

        for mention in mentions:
            span = mention.span

            if len(mention) == 1:
                output_labels[span.start] = self.join_label(single, mention.type)
            else:
                start_label = self.join_label(begin, mention.type)
                output_labels[span.start] = start_label

                for idx in range(span.start + 1, span.end):
                    # span.end is exclusive, so the index of the final label is -1
                    state = end if idx == span.end - 1 else inside
                    output_labels[idx] = self.join_label(state, mention.type)

        return output_labels


# Declared mid-file so it can refer to classes in file
_ENCODING_NAMES: Dict[str, Encoding] = {
    "BIO": BIO(BIOESDialect()),
    "BIOES": BIOES(BIOESDialect()),
    "BILOU": BIOES(BILOUDialect()),
    "BMES": BIOES(BMESDialect()),
    "BMEOW": BIOES(BMEOWDialect()),
    "IO": IO(BIOESDialect()),
    "IOB": IOB(BIOESDialect()),
}
# Note that the ordering here is what will appear on the command line options
# All are supported for encoding and decoding, but in theory things could change
SUPPORTED_ENCODINGS = tuple(_ENCODING_NAMES)


def get_encoding(name: str) -> Encoding:
    name = name.upper()
    if name in _ENCODING_NAMES:
        return _ENCODING_NAMES[name]
    else:
        raise ValueError(f"Unknown encoder {repr(name)}")


@attrs
class _MentionBuilder:
    start_idx: Optional[int] = attrib(default=None, init=False)
    entity_type: Optional[str] = attrib(default=None, init=False)
    mentions: List[Mention] = attrib(default=Factory(list), init=False)

    def start_mention(self, start_idx: int, entity_type: str) -> None:
        # Check arguments
        assert start_idx >= 0
        assert entity_type

        # Check state
        assert (
            self.start_idx is None
        ), f"Mention has already been started at index {self.start_idx}"
        assert (
            self.entity_type is None
        ), f"Mention has already been started with type {self.entity_type}"

        self.start_idx = start_idx
        self.entity_type = entity_type

    def end_mention(self, end_idx: int) -> None:
        # Since end index is exclusive, cannot be zero
        assert end_idx > 0

        # Check state
        assert self.start_idx is not None, "No mention start index"
        assert self.entity_type is not None, "No mention entity type"

        mention = Mention(Span(self.start_idx, end_idx), self.entity_type)
        self.mentions.append(mention)

        self.start_idx = None
        self.entity_type = None

    def in_mention(self) -> bool:
        return self.start_idx is not None
