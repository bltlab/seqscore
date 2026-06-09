import os
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import Union

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]


def file_fields_match(path1: PathType, path2: PathType, *, debug: bool = False) -> bool:
    """Return whether the whitespace-delimited fields of two files are identical."""
    with open(path1, encoding="utf8") as f1, open(path2, encoding="utf8") as f2:
        line_count = 1
        for l1, l2 in zip_longest(f1, f2):
            if l1 is None or l2 is None or l1.split() != l2.split():
                if debug:  # pragma: no cover
                    print(f"Failed to match at line {line_count}:")
                    print(repr(l1))
                    print(repr(l2))
                return False
            line_count += 1
        return True


def file_lines_match(path1: PathType, path2: PathType, debug: bool = False) -> bool:
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
