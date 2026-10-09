from collections import deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

V = TypeVar("V")


class Stream(Iterator[V]):
    def __init__(self, source: Iterable[V]):
        self._source = iter(source)
        self._buffer: deque[V] = deque()
        self._finished = False

    def is_active(self) -> bool:
        if self._buffer:
            return True
        if self._finished:
            return False
        try:
            self._buffer.append(next(self._source))
        except StopIteration:
            self._finished = True
            return False
        return True

    def __next__(self) -> V:
        if not self.is_active():
            raise StopIteration
        return self._buffer.popleft()

    def __iter__(self) -> "Stream[V]":
        return self


def make_stream(source: Iterable[V]) -> Stream[V]:
    return Stream(source)