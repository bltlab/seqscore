import os
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]


# Type-specific implementation to work around type checker limitations. No, writing this as a
# generic function with type variables does not satisfy all type checkers.
def tuplify_strs(strs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(strs)


def file_fields_match(path1: PathType, path2: PathType) -> bool:
    """Return whether the whitespace-delimited fields of two files are identical."""
    with open(path1, encoding="utf8") as f1, open(path2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1 is None or l2 is None or l1.split() != l2.split():
                return False
        return True


def file_lines_match(path1: PathType, path2: PathType) -> bool:
    """Return whether lines of two files are identical ignoring line endings."""
    with open(path1, encoding="utf8") as f1, open(path2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1 is None or l2 is None or l1.rstrip("\r\n") != l2.rstrip("\r\n"):
                return False
        return True


def normalize_str_with_path(s: str) -> str:
    """Normalize the OS path separator to '/'."""
    return s.replace(os.path.sep, "/")
