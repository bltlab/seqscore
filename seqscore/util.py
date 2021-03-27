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
