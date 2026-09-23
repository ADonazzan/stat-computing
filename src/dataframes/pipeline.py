from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from dataframes.column import Column
from dataframes.expr import ColumnExpression, evaluate
from dataframes.frame import DataFrame


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
            return self._extend(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        if isinstance(other, DataFrame):
            return replace(self, source=other)
        if isinstance(other, DataFramePipeline):
            return self._extend(other, self)
        return NotImplemented

    def render(self) -> str:
        head = "<data>" if self.source is None else f"DataFrame{self.source.dims()}"
        return "\n  |> ".join([head] + [s.render() for s in self.steps])

    def execute(self, source=None) -> DataFrame:
        if self.source is None and source is None:
            raise ValueError("Missing source DataFrame")
        df = self.source or source
        for step in self.steps:
            df = EXECUTORS[step.kind](df, *step.args, **step.kwargs)
        return df

    
    def optimize(self, predicates: bool = True, projections: bool = True) -> "DataFramePipeline":
        from dataframes.optimize import push_down, FILTERS, PROJECTIONS
        steps = self.steps
        if predicates:
            steps = push_down(steps, FILTERS)
        if projections:
            steps = push_down(steps, PROJECTIONS)
        return replace(self, steps=tuple(steps))

    @staticmethod
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
groupBy = _op("groupby")
ungroup = _op("ungroup")
aggregate = _op("aggregate")


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

def _exec_groupby(df: DataFrame, *names: str) -> DataFrame:
    for n in names:
        if n not in df._colnames:
            raise KeyError(f"No column named {n}")

    cols = [df.get_col(n) for n in names]
    index: dict[tuple, list[int]] = {}
    for i in range(df.dims()[1]):
        key = tuple(col[i] for col in cols)
        index.setdefault(key, []).append(i)

    return replace(df, groups={"keys": tuple(names), "index": index})


def _exec_ungroup(df: DataFrame) -> DataFrame:
    return replace(df, groups={})

def _exec_aggregate(df: DataFrame, **aggs) -> DataFrame:
    keys = df.groups["keys"]
    groups = df.groups["index"]
    out = {}

    # one column per grouping variable, one row per group
    for j, name in enumerate(keys):
        out[name] = Column.from_list([k[j] for k in groups])

    # one column per requested aggregation
    for newname, (colname, fn) in aggs.items():
        src = df.get_col(colname)
        vals = []
        for idx in groups.values():
            present = [src[i] for i in idx if src[i] is not None]
            vals.append(fn(present) if present else None)
        out[newname] = Column.from_list(vals)

    return DataFrame.from_columns(out)

EXECUTORS = {
    "select":       _exec_select,
    "exclude":      _exec_exclude,
    "keep":         _exec_keep,
    "remove":       _exec_remove,
    "derive":       _exec_derive,
    "groupby":      _exec_groupby,
    "ungroup":      _exec_ungroup,
    "aggregate":    _exec_aggregate
}