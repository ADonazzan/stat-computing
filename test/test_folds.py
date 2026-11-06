import math
import pytest
from streams import *

plus = Fold(identity, lambda acc, x: acc + x, lambda: 0, identity)
biggest = Fold1(identity, max, identity)


def test_basic_folds():
    s = from_seq([3, 1, 4, 1, 5])
    assert fold(count, s) == 5
    assert fold1(biggest, s) == 5
    seen = []
    assert fold(drain, map(seen.append, s)) is None
    assert seen == [3, 1, 4, 1, 5]

def test_scan_is_partial_sums():
    assert take(scan(plus, integers_from(1)), 4) == [0, 1, 3, 6]
    assert take(scan(collect(2), integers_from(0)), math.inf) == [[], [0], [0, 1]]

def test_scan1_running_max():
    assert take(scan1(biggest, from_seq([3, 1, 4, 1, 5])), math.inf) == [3, 3, 4, 4, 5]
    assert take(scan1(biggest, empty), math.inf) == []

def test_fold1_empty_raises():
    with pytest.raises(ValueError):
        fold1(biggest, empty)

def test_puzzle1_take_as_fold():
    nat = integers_from(0)
    assert fold(collect(5), nat) == take(nat, 5) == [0, 1, 2, 3, 4]
    assert fold(collect(0), nat) == []

def test_fold_terminates_early_on_infinite_stream():
    # Would hang forever without Reduced.
    assert fold(collect(3), repeat(7)) == [7, 7, 7]

def test_fold_is_reusable():
    f = collect()
    s = from_seq([1, 2])
    assert fold(f, s) == [1, 2]
    assert fold(f, s) == [1, 2]   # fails if init were a shared []

def test_dimap_lmap_rmap():
    words = from_seq(["a", "bb", "ccc"])
    assert fold(lmap(len, collect()), words) == [1, 2, 3]
    assert fold(rmap(lambda n: n * 10, count), words) == 30
    assert fold(dimap(len, sum, collect()), words) == 6
    assert fold1(lmap(abs, biggest), from_seq([-5, 3])) == 5   # works on Fold1

def test_defer_resolve():
    paused = fold(defer(count), from_seq([1, 2]))
    paused = fold(defer(paused), from_seq([3]))
    assert resolve(paused) == 3

def test_defer_does_not_share_accumulator():
    paused = fold(defer(collect()), from_seq([1, 2]))
    assert fold(paused, from_seq([3])) == [1, 2, 3]
    assert fold(paused, from_seq([4])) == [1, 2, 4]   # fails without copy.copy

def test_defer_fold1_becomes_fold():
    paused = fold1(defer(biggest), from_seq([1, 5]))
    assert isinstance(paused, Fold)
    assert fold(paused, from_seq([3, 9])) == 9
    assert resolve(paused) == 5

def test_tee():
    assert fold(tee(count, collect()), from_seq([1, 2, 3])) == (3, [1, 2, 3])

def test_tee_stops_when_all_done():
    # Infinite stream: only terminates if tee handles Reduced per fold.
    assert fold(tee(collect(2), collect(3)), integers_from(0)) == ([0, 1], [0, 1, 2])