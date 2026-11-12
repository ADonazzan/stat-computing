import math
import socketserver
import threading

import pytest

from streams import *


def test_line_from(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\ntwo\nthree\n")
    s = line_from(path)
    assert take(s, math.inf) == ["one\n", "two\n", "three\n"]
    assert take(s, 1) == ["one\n"]   # reopens the file on each run

def test_line_from_opens_lazily(tmp_path):
    s = line_from(tmp_path / "missing.txt")   # no error yet
    with pytest.raises(FileNotFoundError):
        head(s)

def test_display(capsys):
    s = display(from_seq([1, 2, 3]), fmt=lambda x: f"<{x}>")
    assert capsys.readouterr().out == ""                    # nothing printed until run
    assert take(s, 2) == [1, 2]
    assert capsys.readouterr().out == "<1>\n<2>\n"

def test_sink(tmp_path):
    out = tmp_path / "out.txt"
    assert fold(drain, sink(from_seq(["a", "b"]), out)) is None
    assert out.read_text() == "a\nb\n"

def test_socket():
    class Upper(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                self.wfile.write(line.upper())

    with socketserver.TCPServer(("127.0.0.1", 0), Upper) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        replies = socket(from_seq(["hello", "streams"]), port=port)
        assert take(replies, math.inf) == ["HELLO", "STREAMS"]
        assert take(replies, 1) == ["HELLO"]   # new connection per run
        server.shutdown()
