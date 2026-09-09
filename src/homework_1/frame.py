import numpy as np

from dataclasses import dataclass
from typing import Any

from column import Column
from dtypes import DataType

@dataclass(frozen=True)
class DataFrame:
    columns: tuple[Column, ...]
    colnames: dict[str, int]
    schema: list[DataType | None]
    dims: tuple[int, int]
    metadata: list[dict[str, Any]]

