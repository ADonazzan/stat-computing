import math
import pytest

from streams import *


def test_head_tail():
    s = cons(1, cons(2, cons(3)))
    assert head(s) == 1
    assert take(tail(s), 5) == [2, 3]

def test_tail_is_lazy():
    t = tail(empty)
    with pytest.raises(ValueError):
        list(t) 

def test_cons_lazy_tail():
    def integers_from(n):
        return cons(n, lambda: integers_from(n + 1))
    assert take(integers_from(0), 5) == [0, 1, 2, 3, 4]

def test_empty():
    assert list(empty) == []

def test_is_empty():
    assert is_empty(empty)
    assert is_empty(Stream(lambda: iter([])))
    assert not is_empty(cons(1))

def test_integers_from_is_reiterable():
    s = integers_from(10)
    assert take(s, 5) == [10, 11, 12, 13, 14]
    assert take(s, 3) == [10, 11, 12]   # same stream, same values

def test_integers_from_deep():
    assert take(integers_from(0), 10_000)[-1] == 9_999   # no RecursionError

def test_repeat_iterate_cycle():
    assert take(repeat(8), 3) == [8, 8, 8]
    assert take(iterate(lambda x: 2 * x, 1), 5) == [1, 2, 4, 8, 16]
    assert take(cycle([1, 2, 3]), 7) == [1, 2, 3, 1, 2, 3, 1]
    assert cycle([]) is empty

def test_from_seq():
    assert take(from_seq([1, 2, 3]), math.inf) == [1, 2, 3]
    s = from_seq(x * x for x in range(4))   # one-shot generator
    assert take(s, math.inf) == [0, 1, 4, 9]
    assert take(s, math.inf) == [0, 1, 4, 9]  # still works the 2nd time