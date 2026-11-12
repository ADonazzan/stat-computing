import socket as _socket
from pathlib import Path

from streams.core import Stream


def line_from(path: str | Path) -> Stream[str]:
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