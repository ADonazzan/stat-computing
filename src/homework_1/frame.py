import numpy as np

from dataclasses import dataclass
from typing import Any

from column import Column
from dtypes import DataType, guess

@dataclass(frozen=True)
class DataFrame:
    columns: tuple[Column, ...]
    colnames: dict[str, int]
    schema: list[DataType | None]
    dims: tuple[int, int]
    metadata: list[dict[str, Any]]

    @classmethod
    def from_columns (cls, cols: dict[str, Column]):
        columns = tuple(cols.values())
        colnames = {name: i for i, name in enumerate(cols)}
        schema = tuple(c.data_type for c in columns)
        nrows = len(columns[0]) if columns else 0
        df = cls(columns, colnames, schema, (len(columns), nrows), tuple({} for _ in columns))
        # df.validate()
        return df

    def validate(self) -> None:
        raise NotImplementedError



if __name__ == "__main__":
    col1 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    col2 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    col3 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)

    df = DataFrame.from_columns({"a": col1, "b": col2, "c": col3})