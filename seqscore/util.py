from os import PathLike
from pathlib import Path
from typing import Union

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]
