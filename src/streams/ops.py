import math
import operator
from collections.abc import Callable, Hashable
from streams.core import Stream, _MISSING, tail, take
from streams.primitives import from_seq

def map[A, B](f: Callable[[A], B], s: Stream[A]) -> Stream[B]:
    def gen():
        for x in s:
            yield f(x)
    return Stream(gen)


def keep[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    def gen():
        for x in s:
            if pred(x):
                yield x
    return Stream(gen)


def remove[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    return keep(s, lambda x: not pred(x))


def map_maybe[A, B](f: Callable[[A], B | None], s: Stream[A]) -> Stream[B]:
    def gen():
        for x in s:
            y = f(x)
            if y is not None:
                yield y
    return Stream(gen)


def zip_with(f: Callable, *streams: Stream) -> Stream:
    """Stops as soon as the shortest stream ends."""
    def gen():
        its = [iter(s) for s in streams]
        while True:
            vals = []
            for it in its:
                v = next(it, _MISSING)
                if v is _MISSING:
                    return
                vals.append(v)
            yield f(*vals)
    return Stream(gen)


def mask[A](s1: Stream[A], s2: Stream) -> Stream[A]:
    return map(lambda p: p[0], keep(zip_with(lambda a, b: (a, b), s1, s2), lambda p: p[1]))

def drop[A](s: Stream[A], n: int) -> Stream[A]:
    def gen():
        it = iter(s)
        for _ in range(n):
            if next(it, _MISSING) is _MISSING:
                return
        yield from it
    return Stream(gen)


def drop_while[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    def gen():
        it = iter(s)
        for x in it:
            if not pred(x):
                yield x
                break
        yield from it
    return Stream(gen)


def take_while[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    def gen():
        for x in s:
            if not pred(x):
                return
            yield x
    return Stream(gen)


def take_until[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    return take_while(s, lambda x: not pred(x))


def drop_until[A](s: Stream[A], pred: Callable[[A], bool]) -> Stream[A]:
    return drop_while(s, lambda x: not pred(x))


def at[A](s: Stream[A], n: int) -> A:
    for x in drop(s, n):
        return x
    raise IndexError(f"Stream has no element {n}")


def split_at[A](s: Stream[A], n: int) -> tuple[list[A], Stream[A]]:
    return take(s, n), drop(s, n)


def slice[A](s: Stream[A], n0: int = 0, n1: int | float = math.inf, dn: int = 1) -> Stream[A]:
    def gen():
        for i, x in enumerate(s):
            if i >= n1:
                return
            if i >= n0 and (i - n0) % dn == 0:
                yield x
    return Stream(gen)

def is_stream(x) -> bool:
    return isinstance(x, Stream)


def chain[A](*streams: Stream[A]) -> Stream[A]:
    def gen():
        for s in streams:
            yield from s
    return Stream(gen)


def interleave[A](*streams: Stream[A]) -> Stream[A]:
    def gen():
        its = [iter(s) for s in streams]
        while its:
            alive = []
            for it in its:
                v = next(it, _MISSING)
                if v is not _MISSING:
                    yield v
                    alive.append(it)
            its = alive
    return Stream(gen)


def merge[A](s: Stream[A], t: Stream[A], ordering: Callable[[A, A], bool] = operator.le) -> Stream[A]:
    def gen():
        si, ti = iter(s), iter(t)
        a, b = next(si, _MISSING), next(ti, _MISSING)
        while a is not _MISSING and b is not _MISSING:
            if ordering(a, b):
                yield a
                a = next(si, _MISSING)
            else:
                yield b
                b = next(ti, _MISSING)
        if a is not _MISSING:
            yield a
            yield from si
        if b is not _MISSING:
            yield b
            yield from ti
    return Stream(gen)


def chunked[A](s: Stream[A], n: int, m: int | None = None) -> Stream[tuple[A, ...]]:
    """Windows of size n, starting every m items (overlapping if m < n). Drops a short last chunk."""
    m = n if m is None else m
    def gen():
        buf: list[A] = []
        skip = 0
        for x in s:
            if skip:
                skip -= 1
                continue
            buf.append(x)
            if len(buf) == n:
                yield tuple(buf)
                if m < n:
                    buf = buf[m:]
                else:
                    buf, skip = [], m - n
    return Stream(gen)


def group_by[A](s: Stream[A], f: Callable[[A], Hashable]) -> Stream[Stream[A]]:
    def gen():
        group: list[A] = []
        key = _MISSING
        for x in s:
            k = f(x)
            if group and k != key:
                yield from_seq(group)
                group = []
            key = k
            group.append(x)
        if group:
            yield from_seq(group)
    return Stream(gen)


def cumsum(s: Stream) -> Stream:
    def gen():
        total = _MISSING
        for x in s:
            total = x if total is _MISSING else total + x
            yield total
    return Stream(gen)


def diff(s: Stream) -> Stream:
    return zip_with(lambda a, b: b - a, s, tail(s))

