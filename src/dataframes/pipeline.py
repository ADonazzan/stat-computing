from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from dataframes.column import Column
from dataframes.frame import DataFrame
from dataframes.expr import ColumnExpression, evaluate


@dataclass(frozen=True)
class Step:
    """store information for one step in pipeline"""
    kind: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def render(self) -> str:
        parts = [a.render() if isinstance(a, ColumnExpression) else repr(a) for a in self.args]
        parts += [f"{k}={v.render() if isinstance(v, ColumnExpression) else v}" for k, v in self.kwargs.items()]
        return f"{self.kind}({', '.join(parts)})"
    

@dataclass(frozen=True)
class DataFramePipeline:
    source: DataFrame | None = None
    steps: tuple[Step, ...] = ()

    def __rshift__(self, other):
        if isinstance(other, DataFramePipeline):
            return _extend(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        if isinstance(other, DataFrame):
            return replace(self, source=other)
        if isinstance(other, DataFramePipeline):
            return _extend(other, self)
        return NotImplemented

    def render(self) -> str:
        head = "<data>" if self.source is None else f"DataFrame{self.source.dims()}"
        return "\n  |> ".join([head] + [s.render() for s in self.steps])

    def execute(self) -> DataFrame:
        if self.source is None:
            raise ValueError("Missing source DataFrame")
        df = self.source
        for step in self.steps:
            df = EXECUTORS[step.kind](df, *step.args, **step.kwargs)
        return df


def _extend(first: DataFramePipeline, second: DataFramePipeline) -> DataFramePipeline:
    return DataFramePipeline(source=first.source or second.source,
                             steps=first.steps + second.steps)

def _op(kind: str):
    def make(*args, **kwargs):
        return DataFramePipeline(steps=(Step(kind, args, kwargs),))
    make.__name__ = kind
    return make

# user facing operations:
select = _op("select")
exclude = _op("exclude")
keep = _op("keep")
remove = _op("remove")
derive = _op("derive")

def _exec_select(df: DataFrame, *names: str) -> DataFrame:
    return DataFrame.from_columns({n: df.get_col(n) for n in names})

def _exec_exclude(df: DataFrame, *names: str) -> DataFrame:
    keep_names = [n for n in df.colnames() if n not in names]
    return _exec_select(df, *keep_names)

def _exec_keep(df: DataFrame, predicate: ColumnExpression) -> DataFrame:
    values, mask = evaluate(df, predicate)
    take = np.asarray(values, dtype=bool) & np.asarray(mask)
    idx = np.flatnonzero(take)
    return DataFrame.from_columns(
        {n: df.get_col(n).take(idx) for n in df.colnames()})


def _exec_remove(df: DataFrame, predicate: ColumnExpression) -> DataFrame:
    values, mask = evaluate(df, predicate)
    take = ~(np.asarray(values, dtype=bool) & np.asarray(mask))
    idx = np.flatnonzero(take)
    return DataFrame.from_columns(
        {n: df.get_col(n).take(idx) for n in df.colnames()})


def _exec_derive(df: DataFrame, **exprs) -> DataFrame:
    for name, expr in exprs.items():
        values, mask = evaluate(df, expr)
        df = df.set_col(name, Column.from_arrays(values, mask, df.dims()[1]))
    return df

EXECUTORS = {
    "select":  _exec_select,
    "exclude": _exec_exclude,
    "keep":    _exec_keep,
    "remove":  _exec_remove,
    "derive":  _exec_derive
}