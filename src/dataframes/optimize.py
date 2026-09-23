from __future__ import annotations
from dataclasses import replace

from dataframes.pipeline import Step

FILTERS = {"keep", "remove"}
PROJECTIONS = {"select", "exclude"}

def derive_io(der: Step) -> tuple[set[str], set[str]]:
    """columns a derive reads from its input, columns it creates"""
    reads, creates = set(), set()
    for name, e in der.kwargs.items():
        reads = reads | (e.cols_used() - creates)
        creates.add(name)
    return reads, creates


def can_move_before(step: Step, prev: Step) -> bool:
    """Check if `step` runs before `prev` without changing the result"""
    if step.kind in FILTERS:
        if prev.kind in PROJECTIONS:
            return True
        if prev.kind == "derive":
            _, creates = derive_io(prev)
            return (not step.args[0].cols_used() & creates                  # the filter doesn't read anything the derive creates (not of an empty set)
                    and all(e.is_rowwise() for e in prev.kwargs.values()))  # and every expression in the derive is row-wise
        return False

    if step.kind == "select" and prev.kind in FILTERS:
        return prev.args[0].cols_used() <= set(step.args)   # True if the columns the filter reads is a subset of the columns the select keeps

    if step.kind == "exclude":
        dropped = set(step.args)
        if prev.kind in FILTERS:
            return not dropped & prev.args[0].cols_used()
        if prev.kind == "derive":
            reads, creates = derive_io(prev)
            return not dropped & (reads | creates)

    return False
        

def select_before_derive(sel: Step, der: Step) -> Step:
    """Builds the extra select that goes in front of a derive"""
    reads, creates = derive_io(der)
    names = [n for n in sel.args if n not in creates]
    names += sorted(reads - set(names))
    return Step("select", tuple(names))


def push_down(steps, kinds: set[str]) -> list[Step]:
    """Move every step whose kind is in `kinds` as early as it can go"""
    out: list[Step] = []
    for step in steps:
        out.append(step)
        if step.kind not in kinds:
            continue
        i = len(out) - 1
        while i > 0:
            prev, cur = out[i - 1], out[i]
            if can_move_before(cur, prev):
                out[i - 1], out[i] = cur, prev
            elif cur.kind == "select" and prev.kind == "derive":
                out.insert(i - 1, select_before_derive(cur, prev))
            else:
                break
            i -= 1
    return out