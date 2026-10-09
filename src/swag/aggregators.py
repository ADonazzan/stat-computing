from dataclasses import dataclass
from operator import add
from typing import Callable, Generic, TypeVar

import numpy as np


V = TypeVar("V")
A = TypeVar("A")

@dataclass(frozen=True)
class Monoid(Generic[A]):
    munit: A
    op: Callable[[A, A], A] = add


@dataclass(frozen=True)
class Agg(Generic[V, A]):
    monoid: Monoid[A]
    annotate: Callable[[V], A]
    finalize: Callable[[A], V]


def identity(x: A) -> A:
    return x


def basic(monoid: Monoid[A]) -> Agg[A, A]:
    return Agg(
        monoid=monoid,
        annotate=identity,
        finalize=identity,
    )

sum_agg = basic(Monoid(0.0))
concat_agg = basic(Monoid(""))

# mean
def mean_finalize(a) -> float:
    total, count = a

    if count == 0:
        raise ValueError("Mean requires a nonempty window")

    return float(total / count)


mean_agg = Agg(
    monoid=Monoid(np.zeros(2)),
    annotate=lambda x: np.array([x, 1.0]),
    finalize=mean_finalize,
)

# standard deviation
def std_finalize(a) -> float:
    sx, sxx, count = a

    if count == 0:
        raise ValueError("Standard deviation requires a nonempty window")

    variance = sxx / count - (sx / count) ** 2
    return float(np.sqrt(variance))


std_agg = Agg(
    monoid=Monoid(np.zeros(3)),
    annotate=lambda x: np.array([x, x * x, 1.0]),
    finalize=std_finalize,
)

# linear regression
def reg_annotate(xy: tuple[float, float]):
    x, y = xy
    return np.array([x, y, x * x, x * y, 1.0])


def reg_finalize(a) -> tuple[float, float]:
    sx, sy, sxx, sxy, count = a

    if count == 0:
        raise ValueError("Regression requires a nonempty window")

    denominator = count * sxx - sx * sx
    if denominator == 0:
        raise ValueError("Regression requires variation in x")

    slope = (count * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / count

    return float(intercept), float(slope)


reg_agg = Agg(
    monoid=Monoid(np.zeros(5)),
    annotate=reg_annotate,
    finalize=reg_finalize,
)