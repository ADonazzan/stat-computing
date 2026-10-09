from itertools import count

from swag.aggregators import concat_agg, mean_agg
from swag.stream import make_stream
from swag.window import swag_stream


def test_swag_stream_concatenation():
    source = make_stream("abcdefgh")
    results = swag_stream(concat_agg, 3, source)

    assert source.is_active()
    assert source.is_active()

    assert next(results) == "abc"
    assert list(results) == [
        "bcd", "cde", "def", "efg", "fgh"
    ]

    assert not source.is_active()


def test_swag_stream_infinite_input():
    consumed = []

    def source():
        for x in count(1):
            consumed.append(x)
            yield x

    results = swag_stream(mean_agg, 3, make_stream(source()))

    assert consumed == []

    assert next(results) == 2.0
    assert consumed == [1, 2, 3]

    assert next(results) == 3.0
    assert consumed == [1, 2, 3, 4]

    assert next(results) == 4.0
    assert consumed == [1, 2, 3, 4, 5]