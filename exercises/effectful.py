import glob
import operator
import re

from streams import (Fold, Region, chain, count, display, drain, fold, identity,
                     integers_from, interleave, keep, line_from, lmap, sink, tee, zip_with)


# grep
def grep(pattern, paths, region=None):
    """Stream of (file, line number, line) for lines matching `pattern`."""
    rx = re.compile(pattern)

    def matches_in(path):
        numbered = zip_with(lambda n, line: (str(path), n, line),
                            integers_from(1), line_from(path, region))
        return keep(numbered, lambda t: rx.search(t[2]))

    return chain(*[matches_in(p) for p in paths])


def run_grep(pattern, paths, *, filename=True, line_number=True, out=None):
    """Print (or write to file `out`) the matches, formatted like grep."""
    def fmt(match):
        path, n, line = match
        fields = [path] * filename + [str(n)] * line_number + [line.rstrip("\n")]
        return ":".join(fields)

    matches = grep(pattern, paths)
    shown = display(matches, fmt) if out is None else sink(matches, out, fmt)
    fold(drain, shown)


# wc
total = Fold(identity, operator.add, lambda: 0, identity)

count_lines = count
count_words = lmap(lambda line: len(line.split()), total)
count_chars = lmap(len, total)

wc_fold = tee(count_lines, count_words, count_chars)


def wc(path):
    """(lines, words, characters) of a file, in one pass."""
    return fold(wc_fold, line_from(path))

if __name__ == "__main__":
    from streams import from_seq, socket, take

    run_grep(r"^def ", ["src/streams/core.py"])
    run_grep(r"^def ", ["src/streams/core.py"], filename=False)
    run_grep(r"^def ", ["src/streams/core.py"], out="matches.txt")
    print("wc README.md ->", wc("README.md"))

    # Many files read side by side, never more than two open at a time
    sources = sorted(glob.glob("src/streams/*.py"))
    with Region(max_open=2) as region:
        mixed = interleave(*[line_from(p, region) for p in sources])
        print(len(sources), "files interleaved,", fold(count, mixed), "lines")
        fold(drain, display(grep(r"^class ", sources, region), lambda m: f"{m[0]}:{m[1]}:{m[2].rstrip()}"))

    try:
        print(take(socket(from_seq(["hello", "streams"])), 5))
    except ConnectionRefusedError:
        print("conn refused")