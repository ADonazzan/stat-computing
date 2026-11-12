import socket as _socket
from collections import OrderedDict
from pathlib import Path

from streams.core import Stream


class Region:
    """Owns the files opened while it is active and closes them all on exit.
    With `max_open`, at most that many are open at a time: the least recently
    read one is closed and reopened at its saved position when needed again."""

    def __init__(self, max_open: int | None = None):
        self.max_open = max_open
        self._open: OrderedDict = OrderedDict()   # reader -> file, least recently read first
        self._offsets: dict = {}                  # reader -> position of a file closed by the pool
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed = True
        for f in self._open.values():
            f.close()
        self._open.clear()

    def _file(self, reader, path):
        if self._closed:
            raise RuntimeError("stream read outside its region")
        if reader in self._open:
            self._open.move_to_end(reader)
            return self._open[reader]
        if self.max_open is not None and len(self._open) >= self.max_open:
            oldest, f = self._open.popitem(last=False)
            self._offsets[oldest] = f.tell()
            f.close()
        f = open(path, encoding="utf-8")
        f.seek(self._offsets.pop(reader, 0))
        self._open[reader] = f
        return f

    def lines(self, path):
        reader = object()
        try:
            while line := self._file(reader, path).readline():
                yield line
        finally:
            self._offsets.pop(reader, None)
            if reader in self._open:
                self._open.pop(reader).close()


def line_from(path: str | Path, region: Region | None = None) -> Stream[str]:
    if region is not None:
        return Stream(lambda: region.lines(path))
    def gen():
        with open(path, encoding="utf-8") as f:
            yield from f
    return Stream(gen)


def display(s: Stream, fmt=str, file=None) -> Stream:
    def gen():
        for x in s:
            print(fmt(x), file=file)
            yield x
    return Stream(gen)


def sink(s: Stream, path: str | Path, fmt=str) -> Stream:
    def gen():
        with open(path, "w", encoding="utf-8") as f:
            for x in s:
                print(fmt(x), file=f)
                yield x
    return Stream(gen)


def socket(s: Stream, host: str = "127.0.0.1", port: int = 65432) -> Stream[str]:
    def gen():
        with _socket.create_connection((host, port)) as conn, \
             conn.makefile("rw", encoding="utf-8", newline="\n") as f:
            for x in s:
                f.write(str(x).rstrip("\n") + "\n")
                f.flush()
                reply = f.readline()
                if not reply:          # server closed the connection
                    return
                yield reply.rstrip("\n")
    return Stream(gen)