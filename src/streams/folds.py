import copy
import math

from collections.abc import Callable
from dataclasses import  dataclass, replace
from typing import Any

from streams.core import Stream


def identity(x):
    return x


@dataclass(frozen=True, slots=True)
class Fold[V, W, R]:
    wrap: Callable[[V], W]
    step: Callable[[Any, W], Any]
    init: Callable[[], Any]
    done: Callable[[Any], R]


@dataclass(frozen=True, slots=True)
class Fold1[V, W, R]:
    wrap: Callable[[V], W]
    step: Callable[[W, W], W]
    done: Callable[[W], R]


@dataclass(frozen=True, slots=True)
class Reduced[A]:
    value: A

def fold[V, R](fld: Fold[V, Any, R], s: Stream[V]) -> R:
    acc = fld.init()
    if isinstance(acc, Reduced):
        return fld.done(acc.value)
    for x in s:
        acc = fld.step(acc, fld.wrap(x))
        if isinstance(acc, Reduced):
            return fld.done(acc.value)
    return fld.done(acc)


def scan[V, R](fld: Fold[V, Any, R], s: Stream[V]) -> Stream[R]:
    def gen():
        acc = fld.init()
        if isinstance(acc, Reduced):
            yield fld.done(acc.value)
            return
        yield fld.done(acc)
        for x in s:
            acc = fld.step(acc, fld.wrap(x))
            if isinstance(acc, Reduced):
                yield fld.done(acc.value)
                return
            yield fld.done(acc)
    return Stream(gen)


def fold1[V, R](fld: Fold1[V, Any, R], s: Stream[V]) -> R:
    it = iter(s)
    for first in it:
        acc = fld.wrap(first)
        break
    else:
        raise ValueError("fold1 of an empty stream")
    for x in it:
        acc = fld.step(acc, fld.wrap(x))
        if isinstance(acc, Reduced):
            return fld.done(acc.value)
    return fld.done(acc)


def scan1[V, R](fld: Fold1[V, Any, R], s: Stream[V]) -> Stream[R]:
    def gen():
        it = iter(s)
        for first in it:
            acc = fld.wrap(first)
            break
        else:
            return
        yield fld.done(acc)
        for x in it:
            acc = fld.step(acc, fld.wrap(x))
            if isinstance(acc, Reduced):
                yield fld.done(acc.value)
                return
            yield fld.done(acc)
    return Stream(gen)


def collect(n: int | float = math.inf) -> Fold:
    def step(acc, x):
        acc.append(x)
        return Reduced(acc) if len(acc) >= n else acc
    return Fold(identity, step, lambda: Reduced([]) if n <= 0 else [], list)

count = Fold(identity, lambda acc, _: acc + 1, lambda: 0, identity)

drain = Fold(identity, lambda acc, _: acc, lambda: None, identity)

def dimap(f, g, fld):
    """(v' -> v) -> (r -> r') -> Fold v w r -> Fold v' w r'"""
    wrap, done = fld.wrap, fld.done
    return replace(fld, wrap=lambda v: wrap(f(v)), done=lambda acc: g(done(acc)))


def lmap(f, fld):
    return dimap(f, identity, fld)


def rmap(g, fld):
    return dimap(identity, g, fld)

def defer(fld):
    """Fold v w r -> Fold v w (acc -> Fold v w r)"""
    def resume(acc):
        return Fold(fld.wrap, fld.step, lambda: copy.copy(acc), fld.done)
    return replace(fld, done=resume)


def resolve(fld: Fold):
    acc = fld.init()
    return fld.done(acc.value if isinstance(acc, Reduced) else acc)

def _unreduce(acc):
    return acc.value if isinstance(acc, Reduced) else acc


def tee(*folds: Fold) -> Fold:
    def wrap(v):
        return tuple(f.wrap(v) for f in folds)

    def init():
        accs = tuple(f.init() for f in folds)
        return Reduced(accs) if all(isinstance(a, Reduced) for a in accs) else accs

    def step(accs, ws):
        new = tuple(
            a if isinstance(a, Reduced) else f.step(a, w)
            for f, a, w in zip(folds, accs, ws)
        )
        return Reduced(new) if all(isinstance(a, Reduced) for a in new) else new

    def done(accs):
        return tuple(f.done(_unreduce(a)) for f, a in zip(folds, accs))

    return Fold(wrap, step, init, done)