from collections.abc import Iterable, Iterator
from typing import TypeVar

from swag.aggregators import Agg
from swag.stack import AggStack


V = TypeVar("V")
A = TypeVar("A")


def swag_stream(agg: Agg[V, A], window: int, source: Iterable[V]) -> Iterator[V]:
    if window < 1:
        raise ValueError("window must be positive")
    
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