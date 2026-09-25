import pytest

from dataframes.column import Column
from dataframes.dtypes import DataType


def test_instantiate():
    col = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    assert col.data_type == DataType.INT64
    assert col.values.tolist() == [1,2,3,4,5]

def test_strings():
    col = Column.from_list(["plural", None, "anecdote", "is", "data"])
    assert col.data_type is DataType.STRING
    assert [col[i] for i in range(5)] == ["plural", None, "anecdote", "is", "data"]
    assert list(col.offsets) == [0, 6, 6, 14, 16, 20]

    sub = col.slice(1, 4)
    assert [sub[i] for i in range(3)] == [None, "anecdote", "is"]

def test_slice():
    col = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    sliced = col.slice(1, 4)
    assert sliced.values.tolist() == [2,3,4]
    assert sliced.n == 3
    assert sliced.data_type == DataType.INT64


def test_slice_with_validity():
    vals = [1, None, 3, None, 5]
    col = Column.from_list(vals, data_type=DataType.INT64)
    sliced = col.slice(1, 4)
    assert [sliced[i] for i in range(len(sliced))] == [None, 3, None]
    assert list(sliced._valid_mask()) == [False, True, False]
    assert sliced.n == 3
    assert sliced.data_type == DataType.INT64
    # Check validity mask
    valid_mask = sliced._valid_mask()
    assert valid_mask.tolist() == [False, True, False]


def test_slice_out_of_bounds():
    col = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    sliced = col.slice(-10, 10)
    assert sliced.values.tolist() == [1,2,3,4,5]
    assert sliced.n == 5


def test_guess_data_type():
    col = Column.from_list((1, 2, 3))
    assert col.data_type == DataType.INT64

    col_float = Column.from_list((1.0, 2.0, 3.0))
    assert col_float.data_type == DataType.FLOAT64

    col_str = Column.from_list(("a", "b", "c"))
    assert col_str.data_type == DataType.STRING

    col_mixed = Column.from_list((1, "b", 3.0))
    assert col_mixed.data_type == DataType.STRING

    col_bool = Column.from_list((True, False, True))
    assert col_bool.data_type == DataType.BOOLEAN


def test_with_value_fills_missing():
    col = Column.from_list([1, None, 3])
    new = col.with_value(1, 42)
    assert new[1] == 42
    assert col[1] is None

def test_with_value_leaves_neighbours_alone():
    col = Column.from_list(list(range(10)))
    new = col.with_value(3, None)
    assert [new[i] for i in range(10)] == [0, 1, 2, None, 4, 5, 6, 7, 8, 9]