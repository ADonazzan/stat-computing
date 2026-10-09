from itertools import count, islice

import numpy as np
import pytest

from swag.aggregators import (concat_agg, mean_agg, reg_agg, std_agg, 
                              sum_agg, mean_group_agg, sum_group_agg)
from swag.window import swag, swag_stream, swag_group


def test_sum():
    assert swag(sum_agg, 3, [0, 10, 20, 30, 40, 50]) == [
        30, 60, 90, 120
    ]


def test_mean():
    values = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5]
    assert swag(mean_agg, 5, values) == pytest.approx(
        [3.0, 3.8, 4.4, 4.8, 5.0, 5.0, 5.0]
    )


def test_concatenation():
    assert swag(concat_agg, 4, list("abcdefgh")) == [
        "abcd", "bcde", "cdef", "defg", "efgh"
        ]


def test_standard_deviation():
    values = [1.0, 4.0, 2.0, 8.0, 5.0]
    expected = [
        np.std(values[i:i + 3], ddof=0)
        for i in range(len(values) - 2)
    ]
    assert swag(std_agg, 3, values) == pytest.approx(expected)


def test_regression():
    values = [(0.0, 1.0), (1.0, 4.0), (2.0, 2.0), (3.0, 7.0)]

    for start, actual in enumerate(swag(reg_agg, 3, values)):
        x, y = np.array(values[start:start + 3]).T
        design = np.column_stack([np.ones(len(x)), x])
        expected, *_ = np.linalg.lstsq(design, y, rcond=None)

        # Coefficient order: intercept, slope.
        assert actual == pytest.approx(expected)


def test_infinite_stream():
    averages = swag_stream(mean_agg, 3, count(1))

    assert list(islice(averages, 5)) == pytest.approx(
        [2.0, 3.0, 4.0, 5.0, 6.0]
    )


def test_window_of_one_still_annotates_and_finalizes():
    assert swag(std_agg, 1, [2.0, 5.0, 9.0]) == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_group_sum_matches_monoid():
    values = [1.0, 4.0, 2.0, 8.0, 5.0]

    for window in [1, 3, 5, 10]:
        expected = swag(sum_agg, window, values)
        actual = swag_group(sum_group_agg, window, values)

        assert actual == pytest.approx(expected), f"window={window}"


def test_group_mean_matches_monoid():
    values = [1.0, 4.0, 2.0, 8.0, 5.0]

    for window in [1, 3, 5, 10]:
        expected = swag(mean_agg, window, values)
        actual = swag_group(mean_group_agg, window, values)

        assert actual == pytest.approx(expected), f"window={window}"