import polars as pl

from dataframes.column import Column
from dataframes.frame import DataFrame


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
    raw = pl.read_csv(path, infer_schema=False, has_header=False, skip_rows=skip)
    rows = raw.rows()
    if header:
        names, rows = _dedupe([str(x) for x in rows[0]]), rows[1:]
    else:
        names = [f"x_{i}" for i in range(raw.width)]

    cols = {name: Column.from_strings([r[i] for r in rows])
            for i, name in enumerate(names)}
    return DataFrame.from_columns(cols)