import math
import operator

from streams.core import Stream, _MISSING, cons
from streams.primitives import repeat
from streams.ops import map, zip_with


def inject(a) -> Stream:
    return cons(a, repeat(0))


def scale(a, s: Stream) -> Stream:
    return map(lambda x: a * x, s)


def add(s: Stream, t: Stream) -> Stream:
    return zip_with(operator.add, s, t)


def sub(s: Stream, t: Stream) -> Stream:
    return zip_with(operator.sub, s, t)


def _products(s: Stream, t: Stream, weight):
    """Shared for convolve / shuffle: (s*t)_n = sum_k weight(n,k) s_k t_{n-k}."""
    def gen():
        si, ti = iter(s), iter(t)
        xs, ys = [], []
        while True:
            x, y = next(si, _MISSING), next(ti, _MISSING)
            if x is _MISSING or y is _MISSING:
                return
            xs.append(x)
            ys.append(y)
            n = len(xs) - 1
            yield sum(weight(n, k) * xs[k] * ys[n - k] for k in range(n + 1))
    return Stream(gen)


def convolve(s: Stream, t: Stream) -> Stream:
    return _products(s, t, lambda n, k: 1)


def shuffle(s: Stream, t: Stream) -> Stream:
    return _products(s, t, math.comb)


def reciprocal(s: Stream) -> Stream:
    def gen():
        xs, ts = [], []
        for x in s:
            xs.append(x)
            n = len(xs) - 1
            if n == 0:
                if x == 0:
                    raise ZeroDivisionError("reciprocal needs s(0) != 0")
                t = 1 / x
            else:
                t = -sum(xs[k] * ts[n - k] for k in range(1, n + 1)) / xs[0]
            ts.append(t)
            yield t
    return Stream(gen)


def power(s: Stream, n: int) -> Stream:
    if n < 0:
        return power(reciprocal(s), -n)
    result = inject(1)
    for _ in range(n):
        result = convolve(result, s)
    return result