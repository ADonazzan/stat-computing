import numpy as np
import pytest

from dataframes.column import Column
from dataframes.dtypes import DataType
from dataframes.frame import DataFrame


@pytest.fixture
def simple():
    """3 x 5 frame, no missing values."""
    return DataFrame.from_columns({
        "a": Column.from_list([1, 2, 3, 4, 5]),
        "b": Column.from_list([10, 20, 30, 40, 50]),
        "c": Column.from_list([1.5, 2.5, 3.5, 4.5, 5.5]),
    })

@pytest.fixture
def missing():
    """Missing values at indices 0, 3, 6"""
    vals = [None if i % 3 == 0 else i for i in range(20)]
    return DataFrame.from_columns({"x": Column.from_list(vals)})


def test_from_columns_dims(simple):
    assert simple.dims() == (3, 5)


def test_from_columns_names_in_order(simple):
    assert simple.colnames() == ["a", "b", "c"]


def test_from_columns_schema_matches_columns(simple):
    assert simple.schema == (DataType.INT64, DataType.INT64, DataType.FLOAT64)


def test_from_list_auto_names():
    df = DataFrame.from_list([[1, 2], [3, 4]])
    assert df.colnames() == ["x_0", "x_1"]


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError):
        DataFrame.from_list([[1, 2], [3, 4]], colnames=["only_one"])


def test_ragged_columns_rejected():
    with pytest.raises(ValueError):
        DataFrame.from_columns({
            "a": Column.from_list([1, 2, 3]),
            "b": Column.from_list([1, 2]),
        })

# lookups
def test_col_idxs(simple):
    assert simple.col_idxs(["c", "a"]) == [2, 0]


def test_col_idxs_unknown_name(simple):
    with pytest.raises((ValueError, KeyError)):
        simple.col_idxs(["nope"])


def test_col_name_roundtrip(simple):
    for i, name in enumerate(simple.colnames()):
        assert simple.col_name(i) == name


def test_col_name_out_of_bounds(simple):
    with pytest.raises(IndexError):
        simple.col_name(99)


def test_get_col_by_name_and_index(simple):
    assert simple.get_col("b") is simple.get_col(1)


def test_get_col_unknown(simple):
    with pytest.raises(KeyError):
        simple.get_col("nope")

# fill_col tests
def test_fill_col_length_and_value():
    col = DataFrame.fill_col(7, 4)
    assert len(col) == 4
    assert [col[i] for i in range(4)] == [7, 7, 7, 7]


def test_fill_col_type_override():
    col = DataFrame.fill_col(7, 3, DataType.FLOAT64)
    assert col.data_type is DataType.FLOAT64


# set col
def test_set_col_replace_leaves_original_untouched(simple):
    df2 = simple.set_col("b", [9, 9, 9, 9, 9])
    assert simple.get_col("b")[0] == 10
    assert df2.get_col("b")[0] == 9


def test_set_col_shares_untouched_columns(simple):
    """The structural-sharing claim: other columns are the SAME objects."""
    df2 = simple.set_col("b", [9, 9, 9, 9, 9])
    assert df2.columns[0] is simple.columns[0]
    assert df2.columns[2] is simple.columns[2]


def test_set_col_append_grows_frame(simple):
    df2 = simple.set_col("d", [0, 0, 0, 0, 0])
    assert df2.dims() == (4, 5)
    assert df2.colnames() == ["a", "b", "c", "d"]
    assert simple.dims() == (3, 5)


def test_set_col_scalar_broadcasts(simple):
    df2 = simple.set_col("const", 42)
    col = df2.get_col("const")
    assert len(col) == 5 and col[3] == 42


def test_set_col_wrong_length_rejected(simple):
    with pytest.raises(ValueError):
        simple.set_col("bad", [1, 2])


def test_set_col_updates_schema(simple):
    df2 = simple.set_col("a", [1.5, 2.5, 3.5, 4.5, 5.5])
    assert df2.schema[0] is DataType.FLOAT64
    assert simple.schema[0] is DataType.INT64

def test_missing_values_readable(missing):
    col = missing.get_col("x")
    assert col[0] is None
    assert col[1] == 1


def test_validity_mask_matches(missing):
    col = missing.get_col("x")
    expected = [i % 3 != 0 for i in range(20)]
    assert list(col._valid_mask()) == expected

def test_set_value_roundtrip_and_immutability():
    df = DataFrame.from_columns({
        "a": Column.from_list([1, None, 3]),
        "b": Column.from_list([10, 20, 30]),
    })
    df2 = df.set_value("a", 1, 42)

    assert df2.get_value("a", 1) == 42
    assert df.get_value("a", 1) is None      # original frame unchanged
    assert df2.columns[1] is df.columns[1]   # untouched column is shared


def test_slice_zero_copy(simple):
    sub = simple.slice(1, 4)
    assert sub.dims() == (3, 3)
    assert sub.get_value("a", 0) == 2

def test_set_value_leaves_original(simple):
    df2 = simple.set_value("a", 2, 99)
    assert simple.get_value("a", 2) == 3 and df2.get_value("a", 2) == 99

def test_set_value_type_mismatch_raises(simple):
    with pytest.raises(ValueError):
        simple.set_value("a", 0, "hello")

def test_retype_int_to_string(simple):
    df2 = simple.retype_col("a", DataType.STRING)
    assert df2.schema[0] is DataType.STRING
    assert df2.get_value("a", 0) == "1"

def test_retype_string_to_int():
    df = DataFrame.from_columns({"x": Column.from_list(["1", "2", "3"])})
    df2 = df.retype_col("x", DataType.INT64)
    assert df2.get_value("x", 0) == 1