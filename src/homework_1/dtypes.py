import re

from enum import StrEnum, auto
from typing import Any

class DataType(StrEnum):
    BOOLEAN = auto()
    INT64 = auto()
    FLOAT64 = auto()
    STRING = auto()
    CATEGORY = auto()
    ORDINAL = auto()
    DATE = auto()
    DATETIME = auto()
    UUID = auto()

NA_STRINGS = frozenset(["", "na", "n/a", "null", "none", "nan", "."])
NUMERIC = [DataType.INT64, DataType.FLOAT64]

INT_RE = re.compile(r"[+-]?\d+")
FLOAT_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
BOOL_RE = re.compile(r"\btrue\b|\bfalse\b|\bt\b|\bf\b|\byes\b|\bno\b|\by\b|\bn\b")

BOOL_MAP = {
    "true": True,  "false": False,
    "t":    True,  "f":     False,
    "yes":  True,  "no":    False,
    "y":    True,  "n":     False,
}


def is_missing(s: str) -> bool:
    return s.strip().lower() in NA_STRINGS

def parse_int(s: str) -> int|None:
    return int(s) if INT_RE.fullmatch(s) else None

def parse_float(s: str) -> float|None:
    return float(s) if FLOAT_RE.fullmatch(s) else None

def parse_bool(s: str) -> bool|None:
    s = s.lower()
    if BOOL_RE.fullmatch(s):
        return BOOL_MAP[s]
    return None

INFER_ORDER = [
    (DataType.BOOLEAN, parse_bool),
    (DataType.INT64, parse_int),
    (DataType.FLOAT64, parse_float),
    (DataType.STRING, lambda s: s)
]

def infer(s:str) -> DataType:
    s = s.strip()
    for dtype, parser in INFER_ORDER:
        if parser(s) is not None:
            return dtype
    return DataType.STRING


def join(a: DataType, b: DataType) -> DataType:
    if a == b:
        return a
    if a in NUMERIC and b in NUMERIC:
        return max(a, b, key=NUMERIC.index)
    if {a, b} == {DataType.DATE, DataType.DATETIME}:
        return DataType.DATETIME
    return DataType.STRING  


def guess(values: list[str]) -> DataType | None:
    seen = None
    for s in values:
        if is_missing(s):
            continue
        t = infer(s)
        seen = t if seen is None else join(seen, t)
        if seen == DataType.STRING:
            return seen
    return seen 


if __name__ == "__main__":
    test_list = ["1", "5", "4"]
    print(guess(test_list))

