import numpy as np
import pytest

from dataframes import Column, DataFrame, c, call, select, keep, derive, read_csv


@pytest.fixture
def flights_test():
    return DataFrame.from_columns({
        "carrier":   Column.from_list(["AA", "AA", "DL", "UA"]),
        "dep_delay": Column.from_list([5, 10, None, -5]),
        "distance":  Column.from_list([1000, 1500, 2000, 2500]),
    })

def kinds(p):
    return [s.kind for s in p.steps]


def as_lists(df):
    return {n: [df.get_col(n)[i] for i in range(len(df.get_col(n)))] for n in df.colnames()}

def assert_same(p):
    a, b = p.execute(), p.optimize().execute()
    assert a.colnames() == b.colnames()
    assert as_lists(a) == as_lists(b)

def test_keep_blocked_by_non_rowwise_derive(flights_test):
    demean = lambda x: x - np.mean(x)
    p = flights_test >> derive(z=call(demean, c("distance"))) >> keep(c("carrier") == "AA")
    assert kinds(p.optimize()) == ["derive", "keep"]
    assert_same(p)

def test_select_optimizations(flights_test):
    p = (flights_test
         >> derive(km=c("distance") * 1.609)
         >> select("carrier", "km")
         >> keep(c("carrier") == "AA"))
    assert kinds(p.optimize()) == ["select", "keep", "derive", "select"]
    assert_same(p)


def test_real_data():
    flights = read_csv("data/pa-flights/flights.csv")
    p = (flights
         >> derive(km=c("distance") * 1.609)
         >> select("carrier", "km")
         >> keep(c("carrier") == "AA"))
    assert_same(p)