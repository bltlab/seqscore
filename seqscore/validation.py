from typing import List, Optional, Sequence, Tuple

from attr import attrib, attrs

from seqscore.encoding import _ENCODING_NAMES, Encoding
from seqscore.util import tuplify_strs

# All encodings can be validated
VALIDATION_SUPPORTED_ENCODINGS: Sequence[str] = tuple(_ENCODING_NAMES)


@attrs
class ValidationError:
    msg: str = attrib()
    label: str = attrib()
    type: str = attrib()
    state: str = attrib()
    token: Optional[str] = attrib()
    line_num: Optional[int] = attrib()


@attrs
class ValidationResult:
    errors: Sequence[ValidationError] = attrib()
    n_tokens: int = attrib()
    repaired_labels: Optional[Tuple[str, ...]] = attrib(
        converter=tuplify_strs, default=()
    )
    tokens: Optional[Tuple[str, ...]] = attrib(converter=tuplify_strs, default=None)
    labels: Optional[Tuple[str, ...]] = attrib(converter=tuplify_strs, default=None)
    line_nums: Optional[Tuple[int, ...]] = attrib(default=None)

    def is_valid(self) -> bool:
        return not self.errors

    def __len__(self):
        return len(self.errors)


def validate_labels(
    labels: Sequence[str],
    encoding: Encoding,
    *,
    repair: Optional[str] = None,
    tokens: Optional[Sequence[str]] = None,
    line_nums: Optional[Sequence[int]] = None,
) -> ValidationResult:
    assert not tokens or len(tokens) == len(
        labels
    ), "Tokens and labels must be the same length"
    assert not line_nums or len(line_nums) == len(
        labels
    ), "Line numbers and labels must be the same length"

    errors: List[ValidationError] = []
    outside = encoding.dialect.outside

    # Treat sequence as if preceded by outside
    prev_label = outside
    prev_state, prev_entity_type = encoding.split_label(prev_label)

    # Enumerate so we can look up tokens and labels if needed
    for idx, label in enumerate(labels):
        state, entity_type = encoding.split_label(label)

        if not encoding.is_valid_state(state):
            msg = f"Invalid state {state} in label {label}"
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

            errors.append(
                ValidationError(msg, label, entity_type, state, token, line_num)
            )

        if not encoding.is_valid_transition(
            prev_state, prev_entity_type, state, entity_type
        ):
            msg = f"Invalid transition {prev_label} -> {label}"
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

            errors.append(
                ValidationError(msg, label, entity_type, state, token, line_num)
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
        msg = f"Invalid transition {prev_label} -> {label}"
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
            ValidationError(
                msg, prev_label, prev_entity_type, prev_state, token, line_num
            )
        )

    if errors and repair:
        repaired_labels = encoding.repair_labels(labels, repair)
        return ValidationResult(errors, len(labels), repaired_labels,
                                tokens=tokens, labels=labels, line_nums=line_nums
                                )
    else:
        return ValidationResult(errors, len(labels),
                                tokens=tokens, labels=labels, line_nums=line_nums
                                )
