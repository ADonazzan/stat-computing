import polars as pl
from dataframes.column import Column
from dataframes.dtypes import DataType, guess, is_missing, PARSERS
from dataframes.frame import DataFrame

def read_csv(path, header: bool = True, skip: int = 0) -> DataFrame:
    raw = pl.read_csv(path, schema_overrides=pl.String, has_header=False,
                      skip_rows=skip)
    rows = raw.rows()
    if header:
        names, rows = list(rows[0]), rows[1:]
    else:
        names = [f"x_{i}" for i in range(raw.width)]

    cols = {}
    for i, name in enumerate(names):
        strings = [r[i] for r in rows]
        t = guess(strings)
        typed = [None if s is None or is_missing(s) else PARSERS[t](s) for s in strings]
        cols[name] = Column.from_list(typed, data_type=t)
    return DataFrame.from_columns(cols)