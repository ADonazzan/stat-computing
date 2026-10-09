
from collections import deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

from swag.aggregators import Agg, Group
from swag.stack import AggStack

V = TypeVar("V")
A = TypeVar("A")


def swag_stream(agg: Agg[V, A], window: int, source: Iterable[V]) -> Iterator[V]:

    def generate() -> Iterator[V]:
        front = AggStack(agg.monoid)
        back = AggStack(agg.monoid)

        for x in source:
            back.push_back(agg.annotate(x))

            if len(front) + len(back) < window:
                continue

            total = agg.monoid.op(front.total(), back.total())
            result = agg.finalize(total)

            if not front:
                while back:
                    front.push_front(back.pop())

            front.pop()
            yield result
            
    return generate()


def swag(agg: Agg[V, A], window: int, source: Iterable[V]) -> list[V]:
    return list(swag_stream(agg, window, source))


def swag_group_stream(agg: Agg[V, A], window: int, source: Iterable[V]) -> Iterator[V]:
    group = agg.monoid

    values: deque[A] = deque()
    total = group.munit

    for x in source:
        value = agg.annotate(x)

        if len(values) == window:
            oldest = values.popleft()
            total = group.op(group.inverse(oldest), total)

        values.append(value)
        total = group.op(total, value)

        if len(values) == window:
            yield agg.finalize(total)


def swag_group(agg: Agg[V, A], window: int, source: Iterable[V]) -> list[V]:
    return list(swag_group_stream(agg, window, source))