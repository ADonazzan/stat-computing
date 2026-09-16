import pytest

from dataframes.column import Column
from dataframes.expr import c, evaluate
from dataframes.frame import DataFrame

def test_evaluate_expression():
    df = DataFrame.from_columns({
        "a": Column.from_list([1, None, 3]),
        "t": Column.from_list([2, 3, 4]),
    })
    vals, mask = evaluate(df, c("a") * c("t") + 3)
    assert list(mask) == [True, False, True]