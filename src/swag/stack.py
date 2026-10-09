from dataclasses import dataclass
from typing import Generic, TypeVar

from swag.aggregators import Monoid, mean_agg, concat_agg 


A = TypeVar("A")


@dataclass(frozen=True)
class StackEntry(Generic[A]):
    value: A
    total: A


class AggStack(Generic[A]):
    def __init__(self, monoid: Monoid[A]):
        self.monoid = monoid
        self.data: list[StackEntry[A]] = []

    def __len__(self) -> int:
        return len(self.data)

    def total(self) -> A:
        if not self.data:
            return self.monoid.unit
        return self.data[-1].total

    def push_back(self, x: A) -> None:
        total = self.monoid.op(self.total(), x)
        self.data.append(StackEntry(x, total))

    def push_front(self, x: A) -> None:
        total = self.monoid.op(x, self.total())
        self.data.append(StackEntry(x, total))

    def pop(self) -> A:
        return self.data.pop().value