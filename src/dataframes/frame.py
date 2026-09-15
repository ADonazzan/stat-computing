import numpy as np

from dataclasses import dataclass, replace
from typing import Any

from dataframes.column import Column
from dataframes.dtypes import DataType, guess

@dataclass(frozen=True)
class DataFrame:
    columns: tuple[Column, ...]
    _colnames: dict[str, int]
    schema: list[DataType | None]
    _dims: tuple[int, int]
    metadata: list[dict[str, Any]]

    @classmethod
    def _make(cls, columns, colnames, schema, metadata) -> "DataFrame":
        nrows = len(columns[0]) if columns else 0
        df = cls(tuple(columns), colnames, tuple(schema),
                (len(columns), nrows), tuple(metadata))
        df.validate()
        return df

    @classmethod
    def from_columns (cls, cols: dict[str, Column]):
        columns = list(cols.values())
        colnames = {name: i for i, name in enumerate(cols.keys())}
        schema = [col.data_type for col in columns]
        metadata = [{} for _ in columns]
        df = cls._make(columns, colnames, schema, metadata)
        return df

    @classmethod
    def from_list(cls, data: list[list[Any]], colnames: list[str]|None = None):
        if colnames is None:
            colnames = [f"x_{i}" for i in range(len(data))]
        if len(data) != len(colnames):
            raise ValueError("Number of columns and column names must match.")
        columns = [Column.from_list(col) for col in data]
        schema = [col.data_type for col in columns]
        colnames_dict = {name: i for i, name in enumerate(colnames)}
        metadata = [{} for _ in columns]
        df = cls._make(columns, colnames_dict, schema, metadata)
        return df
        
    def validate(self) -> None:
        len_cols = [len(col) for col in self.columns]
        if len(set(len_cols)) != 1:
            raise ValueError("All columns must have the same number of rows.")
        if self._dims != (len(self.columns), len_cols[0]):
            raise ValueError("dims out of sync with columns")
        for name, i in self._colnames.items():
            col = self.columns[i]
            if not isinstance(col, Column):
                raise TypeError(f"Expected Column, got {type(col)}.")
            if self.schema[i] is not None and col.data_type != self.schema[i]:
                raise ValueError(f"Column {name} has type {col.data_type}, schema says {self.schema[i]}")

    def get_col(self, col: str|int) -> Column:
        if isinstance(col, str):
            if col not in self._colnames:
                raise KeyError(f"No column named {col}")
            return self.columns[self._colnames[col]]
        if not 0 <= col < len(self.columns):
            raise IndexError(f"Column index {col} out of bounds")
        return self.columns[col]

    def set_col(self, name: str, values, data_type: DataType | None = None):
        nrows = self._dims[1]
        if isinstance(values, Column):
            col = values
        elif isinstance(values, (list, tuple, np.ndarray)):
            col = Column.from_list(values, data_type=data_type)
        else:
            col = self.fill_col(values, nrows, data_type)   # scalar

        if self.columns and len(col) != nrows:
            raise ValueError(f"length {len(col)} does not match nrows {nrows}")

        # replace column if exists
        if name in self._colnames:
            i = self._colnames[name]
            columns = self.columns[:i] + (col,) + self.columns[i+1:]
            schema  = self.schema[:i]  + (col.data_type,) + self.schema[i+1:]
            return replace(self, columns=columns, schema=schema) # use replace for immutability

        # append
        columns  = self.columns + (col,)
        schema   = self.schema + (col.data_type,)
        colnames = {**self._colnames, name: len(self.columns)}
        return replace(self, columns=columns, schema=schema, _colnames=colnames, 
                       _dims=(len(columns), nrows), metadata=self.metadata + ({},))

    def get_value(self, col, row: int):
        return self.get_col(col)[row]

    def set_value(self, col, row: int, value):
        raise NotImplementedError()

    def drop_col(self, name: str):
        i = self._colnames[name]
        columns  = self.columns[:i] + self.columns[i+1:]
        schema   = self.schema[:i]  + self.schema[i+1:]
        metadata = self.metadata[:i] + self.metadata[i+1:]
        colnames = {n: (j if j < i else j - 1)
                    for n, j in self._colnames.items() if n != name}
        return replace(self, columns=columns, schema=schema, metadata=metadata,
                    _colnames=colnames, _dims=(len(columns), self._dims[1]))

    def rename_col(self, old: str, new: str):
        if old not in self._colnames:
            raise KeyError(f"No column named {old}")
        if new in self._colnames:
            raise KeyError(f"Column name {new} already exists")
        idx = self._colnames[old]
        names = list(self._colnames.keys())
        names[idx] = new
        colnames = {name: i for i, name in enumerate(names)}
        return replace(self, _colnames=colnames)

    def retype_col(self, name: str, t: DataType):
        idx = self._colnames[name]
        for value in self.columns[idx].values:
            if value is not None and not isinstance(value, t):
                raise ValueError(f"Cannot retype column {name} to {t}, value {value} is incompatible")
            

        self.schema[idx] = t




    def dims(self):
        return self._dims

    def colnames(self) -> list[str]:
        return [name for name in self._colnames]
    
    def col_idxs(self, col_names: list[str]) -> list[int]:
        for name in col_names:
            if name not in self._colnames:
                raise ValueError(f"Column name '{name}' not found in DataFrame.")
        return [self._colnames[name] for name in col_names]

    def col_name(self, col_idx: int) -> str:
        if col_idx < 0 or col_idx >= len(self.columns):
            raise IndexError(f"Column index '{col_idx}' out of bounds.")
        return list(self._colnames.keys())[col_idx]

    @staticmethod
    def fill_col(value, length: int, data_type: DataType | None = None) -> Column:
        return Column.from_list([value] * length, data_type=data_type)


if __name__ == "__main__":
    col1 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    col2 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    col3 = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)

    df = DataFrame.from_columns({"a": col1, "b": col2, "c": col3})