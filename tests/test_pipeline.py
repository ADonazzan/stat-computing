import pytest

from dataframes import Column, c, DataFrame, select, exclude, keep, remove, derive
from dataframes.pipeline import DataFramePipeline


@pytest.fixture
def flights():
    return DataFrame.from_columns({
        "carrier":   Column.from_list(["AA", "AA", "DL", "UA"]),
        "dep_delay": Column.from_list([5, 10, None, -5]),
        "distance":  Column.from_list([1000, 1500, 2000, 2500]),
    })

def test_pipeline_defers(flights):
    p = flights >> select("carrier", "distance")
    assert isinstance(p, DataFramePipeline)
    assert len(p.steps) == 1
    assert p.source is flights


def test_render_shows_pipeline(flights):
    p = flights >> select("carrier") >> keep(c("carrier") == "AA")
    text = p.render()
    assert "select" in text and "keep" in text and "==" in text


def test_full_pipeline(flights):
    out = (flights
           >> select("carrier", "dep_delay", "distance")
           >> keep(c("carrier") == "AA")
           >> derive(km=c("distance") * 1.609)
           ).execute()
    assert out.colnames() == ["carrier", "dep_delay", "distance", "km"]
    assert out.dims()[1] == 2