import polars as pl

from dataframes.column import Column
from dataframes.dtypes import DataType, guess, is_missing, parse_int, parse_float, parse_bool
from dataframes.frame import DataFrame

PARSERS = {
    DataType.INT64:   parse_int,
    DataType.FLOAT64: parse_float,
    DataType.BOOLEAN: parse_bool,
    DataType.STRING:  lambda s: s,
}


def _dedupe(names: list[str]) -> list[str]:
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def read_csv(path, header: bool = True, skip: int = 0) -> DataFrame:
    raw = pl.read_csv(path, schema_overrides=pl.String,
                      has_header=False, skip_rows=skip)
    rows = raw.rows()
    if header:
        names, rows = _dedupe([str(x) for x in rows[0]]), rows[1:]
    else:
        names = [f"x_{i}" for i in range(raw.width)]

    cols = {name: Column.from_strings([r[i] for r in rows])
            for i, name in enumerate(names)}
    return DataFrame.from_columns(cols)