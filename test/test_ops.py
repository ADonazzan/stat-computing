import math
import pytest

from streams import *

def test_transforms():
    nat = integers_from(0)
    assert take(map(lambda x: x * x, nat), 5) == [0, 1, 4, 9, 16]
    assert take(keep(nat, lambda x: x % 2 == 0), 3) == [0, 2, 4]
    assert take(remove(nat, lambda x: x % 2 == 0), 3) == [1, 3, 5]
    assert take(map_maybe(lambda x: x // 2 if x % 2 == 0 else None, nat), 3) == [0, 1, 2]
    assert take(zip_with(lambda a, b: a + b, nat, from_seq([10, 20])), math.inf) == [10, 21]
    assert take(mask(nat, cycle([True, False])), 3) == [0, 2, 4]

def test_zip_with_reiterable():
    s = zip_with(lambda a, b: a * b, integers_from(1), integers_from(1))
    assert take(s, 3) == take(s, 3) == [1, 4, 9]

def test_prefixes():
    nat = integers_from(0)
    assert take(drop(nat, 3), 2) == [3, 4]
    assert take(drop_while(nat, lambda x: x < 5), 2) == [5, 6]
    assert take(take_while(nat, lambda x: x < 3), math.inf) == [0, 1, 2]
    assert take(take_until(nat, lambda x: x == 3), math.inf) == [0, 1, 2]
    assert at(nat, 7) == 7
    first, rest = split_at(nat, 2)
    assert first == [0, 1] and head(rest) == 2
    assert take(slice(nat, 2, 10, 3), math.inf) == [2, 5, 8]
    with pytest.raises(IndexError):
        at(from_seq([1]), 5)

def test_combine():
    assert take(chain(from_seq([1, 2]), from_seq([3])), math.inf) == [1, 2, 3]
    assert take(interleave(repeat(0), repeat(1)), 4) == [0, 1, 0, 1]
    assert take(interleave(from_seq([1]), from_seq([2, 3])), math.inf) == [1, 2, 3]
    assert take(chunked(integers_from(0), 3), 2) == [(0, 1, 2), (3, 4, 5)]
    assert take(chunked(integers_from(0), 3, 1), 2) == [(0, 1, 2), (1, 2, 3)]
    assert take(chunked(integers_from(0), 2, 3), 2) == [(0, 1), (3, 4)]
    groups = group_by(from_seq([1, 1, 2, 3, 3]), lambda x: x)
    assert [take(g, math.inf) for g in take(groups, math.inf)] == [[1, 1], [2], [3, 3]]
    assert take(cumsum(integers_from(1)), 4) == [1, 3, 6, 10]
    assert take(diff(map(lambda x: x * x, integers_from(0))), 4) == [1, 3, 5, 7]

def test_merge():
    evens, odds = scale(2, integers_from(0)), map(lambda x: 2 * x + 1, integers_from(0))
    assert take(merge(evens, odds), 6) == [0, 1, 2, 3, 4, 5]
    assert take(merge(from_seq([1, 4]), from_seq([2, 3, 5])), math.inf) == [1, 2, 3, 4, 5]
    assert take(merge(from_seq([1, 2]), empty), math.inf) == [1, 2]
    longest_first = lambda a, b: len(a) >= len(b)
    assert take(merge(from_seq(["ccc", "a"]), from_seq(["bb"]), longest_first), math.inf) == ["ccc", "bb", "a"]

def test_merge_ties_take_first_stream():
    s, t = from_seq([(1, "s")]), from_seq([(1, "t")])
    assert take(merge(s, t, lambda a, b: a[0] <= b[0]), math.inf) == [(1, "s"), (1, "t")]
