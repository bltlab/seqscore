from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import Iterable, Tuple, Union

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]


# Type-specific implementation to work around type checker limitations. No, writing this as a
# generic function with type variables does not satisfy all type checkers.
def tuplify_strs(strs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(strs)


# Test whether two files have the same tokens and labels, disregarding whether they are separated by space or tab.
def files_match(file1, file2) -> bool:
    with open(file1, encoding="utf8") as f1, open(file2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1.split() != l2.split():
                return False
        return True


# Test whether dump output files match
def dump_files_match(file1, file2) -> bool:
    with open(file1, encoding="utf8") as f1, open(file2, encoding="utf8") as f2:
        for l1, l2 in zip_longest(f1, f2):
            if l1 != l2:
                return False
        return True
