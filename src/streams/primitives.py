from collections.abc import Callable, Iterable, Iterator
from streams.core import Stream, empty, memo

def integers_from(n: int) -> Stream[int]:
    def gen():
        i = n
        while True:
            yield i
            i += 1
    return Stream(gen)


def repeat[A](a: A) -> Stream[A]:
    def gen():
        while True:
            yield a
    return Stream(gen)


def iterate[A](f: Callable[[A], A], a: A) -> Stream[A]:
    def gen():
        x = a
        while True:
            yield x
            x = f(x)
    return Stream(gen)


def from_seq[A](seq: Iterable[A]) -> Stream[A]:
    if isinstance(seq, Stream):
        return seq
    if isinstance(seq, Iterator):
        return memo(Stream(lambda: seq))
    items = tuple(seq)
    return Stream(lambda: iter(items))


def cycle[A](seq: Iterable[A]) -> Stream[A]:
    items = tuple(seq)
    if not items:
        return empty
    def gen():
        while True:
            yield from items
    return Stream(gen)