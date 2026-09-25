import pytest

from dataframes import DataFrame, c, evaluate


@pytest.fixture
def df():
    return DataFrame.from_list(
        [[1, 2, None, 4], [10, None, 30, 40]],
        colnames=["a", "b"],
    )


def test_arithmetic_and_null_propagation(df):
    expr = 100 - c("a") * c("b")

    values, valid = evaluate(df, expr)

    assert valid.tolist() == [True, False, False, True]
    assert values[valid].tolist() == [90, -60]
    assert str(expr) == "(100 - (a * b))"
    assert expr.cols_used() == {"a", "b"}
    assert expr.is_rowwise()


def test_comparison_and_boolean_expression(df):
    expr = (c("a") > 2) & (c("b") >= 30)

    values, valid = evaluate(df, expr)

    assert valid.tolist() == [True, False, False, True]
    assert values[valid].tolist() == [False, True]