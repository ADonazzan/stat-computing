import pytest
from fractions import Fraction
from streams import *


def poly(*coeffs):
    return chain(from_seq(coeffs), repeat(0))


def test_inject_scale_add_sub():
    assert take(inject(3), 4) == [3, 0, 0, 0]
    nat = integers_from(0)
    assert take(scale(2, nat), 3) == [0, 2, 4]
    assert take(add(nat, nat), 3) == [0, 2, 4]
    assert take(sub(nat, repeat(1)), 3) == [-1, 0, 1]

def test_convolve():
    assert take(convolve(poly(1, 1), poly(1, 1)), 4) == [1, 2, 1, 0]   # (1+z)^2
    assert take(convolve(repeat(1), repeat(1)), 4) == [1, 2, 3, 4]     # 1/(1-z)^2

def test_shuffle():
    # sum_k C(n,k) = 2^n   (e^x * e^x = e^{2x} as exponential gen. functions)
    assert take(shuffle(repeat(1), repeat(1)), 5) == [1, 2, 4, 8, 16]

def test_reciprocal():
    assert take(reciprocal(poly(1, -1)), 4) == [1, 1, 1, 1]    # 1/(1-z)
    assert take(reciprocal(repeat(1)), 4) == [1, -1, 0, 0]     # back to 1-z

def test_reciprocal_exact():
    s = map(Fraction, integers_from(2))                       # 2, 3, 4, ...
    assert take(convolve(s, reciprocal(s)), 6) == [1, 0, 0, 0, 0, 0]

def test_reciprocal_needs_nonzero_head():
    with pytest.raises(ZeroDivisionError):
        take(reciprocal(poly(0, 1)), 1)

def test_power():
    assert take(power(poly(1, 1), 3), 5) == [1, 3, 3, 1, 0]
    assert take(power(poly(1, 1), 0), 2) == [1, 0]
    assert take(power(poly(1, -1), -2), 4) == [1, 2, 3, 4]   # 1/(1-z)^2