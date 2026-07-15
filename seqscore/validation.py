from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from seqscore.encoding import Encoding, EncodingError


@dataclass
class ValidationError:
    msg: str
    label: str
    type: str
    state: str
    token: Optional[str] = None
    line_num: Optional[int] = None
    source_name: Optional[str] = None


class InvalidStateError(ValidationError):
    pass


class InvalidTransitionError(ValidationError):
    pass


class InvalidLabelError(EncodingError):
    def __init__(self, label: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label: str = label


@dataclass
class SequenceValidationResult:
    errors: tuple[ValidationError, ...]
    n_tokens: int
    repaired_labels: tuple[str, ...] = ()

    def is_valid(self) -> bool:
        return not self.errors

    def invalid_state_errors(self) -> list[InvalidStateError]:
        return [error for error in self.errors if isinstance(error, InvalidStateError)]

    def __len__(self) -> int:
        return len(self.errors)


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[ValidationError, ...]
    n_tokens: int
    n_sequences: int
    n_docs: int


def validate_labels(
    labels: Sequence[str],
    encoding: Encoding,
    *,
    repair: Optional[str] = None,
    tokens: Optional[Sequence[str]] = None,
    line_nums: Optional[Sequence[int]] = None,
    source_name: Optional[str] = None,
) -> SequenceValidationResult:
    assert not tokens or len(tokens) == len(labels), (
        "Tokens and labels must be the same length"
    )
    assert not line_nums or len(line_nums) == len(labels), (
        "Line numbers and labels must be the same length"
    )

    # Validate tokens if supplied
    if tokens:
        for idx, tok in enumerate(tokens):
            if not tok:
                line_msg = f" on line {line_nums[idx]}" if line_nums else ""
                source_msg = f" of {source_name}" if source_name else ""
                raise ValueError(f"Invalid token {repr(tok)}{line_msg}{source_msg}")

    errors: list[ValidationError] = []
    outside = encoding.dialect.outside

    # Treat sequence as if preceded by outside
    prev_label = outside
    prev_state, prev_entity_type = encoding.split_label(prev_label)

    # Enumerate so we can look up tokens and labels if needed
    for idx, label in enumerate(labels):
        try:
            state, entity_type = encoding.split_label(label)
        except EncodingError as e:
            line_msg = f" on line {line_nums[idx]}" if line_nums else ""
            source_msg = f" of {source_name}" if source_name else ""
            raise InvalidLabelError(
                label,
                f"Could not parse label {repr(label)}{line_msg}{source_msg} during validation: "
                + str(e)
                + " Use the --label-index argument if the label is not the last field.",
            ) from e

        if not encoding.is_valid_state(state):
            msg = f"Invalid state {repr(state)} in label {repr(label)}"
            if tokens:
                token = tokens[idx]
                msg += f" for token {repr(token)}"
            else:
                token = None

            if line_nums:
                line_num = line_nums[idx]
                msg += f" on line {line_num}"
            else:
                line_num = None

            if source_name:
                msg += f" of {source_name}"

            errors.append(
                InvalidStateError(
                    msg, label, entity_type, state, token, line_num, source_name
                )
            )

        if not encoding.is_valid_transition(
            prev_state, prev_entity_type, state, entity_type
        ):
            msg = f"Invalid transition {repr(prev_label)} -> {repr(label)}"
            if tokens:
                token = tokens[idx]
                msg += f" for token {repr(token)}"
            else:
                token = None

            if line_nums:
                line_num = line_nums[idx]
                msg += f" on line {line_num}"
            else:
                line_num = None

            if source_name:
                msg += f" of {source_name}"

            errors.append(
                InvalidTransitionError(
                    msg, label, entity_type, state, token, line_num, source_name
                )
            )
        prev_label, prev_state, prev_entity_type = (
            label,
            state,
            entity_type,
        )

    # Treat sequence as if followed by outside
    label = outside
    state, entity_type = encoding.split_label(label)
    if not encoding.is_valid_transition(prev_state, prev_entity_type, state, entity_type):
        msg = f"Invalid transition {repr(prev_label)} -> {repr(label)}"
        if tokens:
            token = tokens[-1]
            msg += f" after token {repr(token)}"
        else:
            token = None

        if line_nums:
            line_num = line_nums[-1]
            msg += f" on line {line_num}"
        else:
            line_num = None

        msg += " at end of sequence"

        errors.append(
            InvalidTransitionError(
                msg, prev_label, prev_entity_type, prev_state, token, line_num
            )
        )

    if errors and repair:
        repaired_labels = encoding.repair_labels(labels, repair)
        return SequenceValidationResult(
            tuple(errors), len(labels), tuple(repaired_labels)
        )
    else:
        return SequenceValidationResult(tuple(errors), len(labels))
