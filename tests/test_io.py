import pytest

from dataframes.dtypes import DataType
from dataframes.io import read_csv

def test_import(): 
    df = read_csv("data/pa-flights/airlines.csv")
    assert df.get_col("carrier").data_type == "string"
    assert df.dims() == (2, 16)

    df = read_csv("data/pa-flights/flights.csv")
    nrows = df.dims()[1]

    assert df.get_col("tailnum").data_type is DataType.STRING
    assert df.get_col("dep_delay").data_type is DataType.INT64

    mask = df.get_col("dep_delay")._valid_mask()
    assert len(mask) == nrows
    assert 0 < mask.sum() < nrows