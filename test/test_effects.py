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

@pytest.fixture
def files(tmp_path):
    paths = [tmp_path / f"{name}.txt" for name in "abcde"]
    for p in paths:
        p.write_text("".join(f"{p.stem}{i}\n" for i in range(4)))
    return paths

@pytest.fixture
def opened(monkeypatch):
    """Every file opened during the test, and the largest number open at once."""
    handles, peak = [], [0]
    real_open = open
    def spy(*args, **kwargs):
        f = real_open(*args, **kwargs)
        handles.append(f)
        peak[0] = max(peak[0], sum(not h.closed for h in handles))
        return f
    monkeypatch.setattr("builtins.open", spy)
    return handles, peak

def test_region_pool_limits_open_files(files, opened):
    handles, peak = opened
    expected = take(interleave(*[line_from(p) for p in files]), math.inf)
    assert peak[0] == 5                     # without a region: one handle per file
    peak[0] = 0
    with Region(max_open=2) as region:
        s = interleave(*[line_from(p, region) for p in files])
        assert take(s, math.inf) == expected
        assert take(s, 7) == expected[:7]
    assert peak[0] == 2
    assert all(h.closed for h in handles)

def test_region_closes_on_exit(files, opened):
    handles, _ = opened
    with Region() as region:
        it = iter(line_from(files[0], region))
        assert next(it) == "a0\n"
        assert not handles[0].closed        # reader still alive, file still open
    assert handles[0].closed
    with pytest.raises(RuntimeError):
        next(it)

def test_region_closes_on_error(files, opened):
    handles, _ = opened
    with pytest.raises(ZeroDivisionError):
        with Region() as region:
            lines = [line_from(p, region) for p in files]
            take(map(lambda line: 1 / 0, interleave(*lines)), 1)
    assert handles and all(h.closed for h in handles)

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
