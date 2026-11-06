from collections.abc import Callable, Iterable, Iterator


_MISSING = object()
_REPR_LIMIT = 5

class Stream[A]:
    __slots__ = ("_gen_fn",)

    def __init__(self, gen_fn:Callable[[], Iterator[A]]):
        if not callable(gen_fn):
            raise TypeError(f"Streams need a callable generator function, got {type(gen_fn).__name__}")
        self._gen_fn = gen_fn

    def __iter__(self) -> Iterator[A]:
        return self._gen_fn()

    def __repr__(self) -> str:
        items = take(self, _REPR_LIMIT + 1)
        shown = ", ".join(repr(x) for x in items[:_REPR_LIMIT])
        if len(items) > _REPR_LIMIT:
            return f"Stream({shown}, ...)"
        return f"Stream({shown})"


def _nothing():
    return iter(())
empty: Stream = Stream(_nothing)

def cons[A](a:A, rest:Stream[A] | Callable[[], Stream[A]] = empty) -> Stream[A]:
    def gen():
        yield a
        yield from (rest() if callable(rest) else rest)
    return Stream(gen)

def head[A](s:Stream[A]) -> A:
    for x in s:
        return x
    raise ValueError("Stream is empty")

def tail[A](s:Stream[A]) -> Stream[A]:
    def gen():
        it = iter(s)
        if next(it, _MISSING) is _MISSING:
            raise ValueError("Stream is empty")
        yield from it
    return Stream(gen)

def take[A](s:Stream, n:int|float) -> list[A]:
    out: list[A] = []
    if n <= 0:
        return out
    for x in s:
        out.append(x)
        if len(out) >= n:
            break
    return out

def is_empty(s: Stream) -> bool:
    return s is empty or next(iter(s), _MISSING) is _MISSING


class Delayed[A]:
    __slots__ = ("_thunk", "_value", "_forced")

    def __init__(self, thunk: Callable[[], A]):
        self._thunk = thunk
        self._value: A | None = None
        self._forced = False

    def __call__(self) -> A:
        if not self._forced:
            self._value = self._thunk()
            self._forced = True
            self._thunk = None
        return self._value


def delay[A](thunk: Callable[[], A]) -> Delayed[A]:
    return Delayed(thunk)


def force[A](d: Callable[[], A]) -> A:
    return d()

def memo[A](s: Stream[A]) -> Stream[A]:
    cache: list[A] = []
    source: Iterator[A] | None = None
    exhausted = False

    def gen():
        nonlocal source, exhausted
        i = 0
        while True:
            if i < len(cache):
                yield cache[i]
                i += 1
            elif exhausted:
                return
            else:
                if source is None:
                    source = iter(s)
                x = next(source, _MISSING)
                if x is _MISSING:
                    exhausted = True
                    return
                cache.append(x)

    return Stream(gen)