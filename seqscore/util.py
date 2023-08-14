import os
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

from attr import Attribute, validators

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]


# Type-specific implementations to work around type checker limitations. No, writing these as
# generic functions with type variables does not satisfy all type checkers.
def tuplify_strs(strs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(strs)


def tuplify_optional_nested_strs(
    items: Optional[Iterable[Iterable[str]]],
) -> Optional[Tuple[Tuple[str, ...], ...]]:
    if items is not None:
        return tuple(tuple(item) for item in items)
    else:
        return None


def file_fields_match(path1: PathType, path2: PathType, *, debug=False) -> bool:
    """Return whether the whitespace-delimited fields of two files are identical."""
    with open(path1, encoding="utf8") as f1, open(path2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1 is None or l2 is None or l1.split() != l2.split():
                if debug:  # pragma: no cover
                    print("Non-matching lines:")
                    print(repr(l1))
                    print(repr(l2))
                return False
        return True


def file_lines_match(path1: PathType, path2: PathType, debug=False) -> bool:
    """Return whether lines of two files are identical ignoring line endings."""
    with open(path1, encoding="utf8") as f1, open(path2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1 is None or l2 is None or l1.rstrip("\r\n") != l2.rstrip("\r\n"):
                if debug:  # pragma: no cover
                    print("Lines differ:")
                    print(l1.strip() if l1 else l1)
                    print(l2.strip() if l2 else l2)
                return False
        return True


def normalize_str_with_path(s: str) -> str:
    """Normalize the OS path separator to '/'."""
    return s.replace(os.path.sep, "/")


# Instantiate in advance for _validator_nonempty_str
_instance_of_str = validators.instance_of(str)


def validator_nonempty_str(_inst: Any, attr: Attribute, value: Any) -> None:
    # Check type
    _instance_of_str(value, attr, value)
    # Check string isn't empty
    if not value:
        raise ValueError(f"Empty string: {repr(value)}")
