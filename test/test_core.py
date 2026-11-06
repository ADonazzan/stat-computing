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