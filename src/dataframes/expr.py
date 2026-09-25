import operator

import numpy as np

from abc    import ABC, abstractmethod
from enum   import StrEnum, auto
from typing import Any

class Op(StrEnum):
    PLUS = auto()
    TIMES = auto()
    EQ = auto()

class Op(StrEnum):
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIV = auto()
    POW = auto()

    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    AND = auto()
    OR = auto()


op_lookup = {
    Op.PLUS: "+",
    Op.MINUS: "-",
    Op.TIMES: "*",
    Op.DIV: "/",
    Op.POW: "**",
    Op.EQ: "==",
    Op.NE: "!=",
    Op.LT: "<",
    Op.LE: "<=",
    Op.GT: ">",
    Op.GE: ">=",
    Op.AND: "&",
    Op.OR: "|",
}

op_functions = {
    Op.PLUS: operator.add,
    Op.MINUS: operator.sub,
    Op.TIMES: operator.mul,
    Op.DIV: operator.truediv,
    Op.POW: operator.pow,
    Op.EQ: operator.eq,
    Op.NE: operator.ne,
    Op.LT: operator.lt,
    Op.LE: operator.le,
    Op.GT: operator.gt,
    Op.GE: operator.ge,
    Op.AND: np.logical_and,
    Op.OR: np.logical_or,
}

class ColumnExpression(ABC):
    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.render()

    def cols_used(self) -> set[str]:
        """Columns this expression reads."""
        return set()

    def is_rowwise(self) -> bool:
        """Row i of the result depends only on row i of the input."""
        return True

    def _binary(self, op, other):
        return BinaryOpExpression(op, self, as_expression(other))

    def _reverse_binary(self, op, other):
        return BinaryOpExpression(op, as_expression(other), self)

    def __add__(self, other):
        return self._binary(Op.PLUS, other)

    def __radd__(self, other):
        return self._reverse_binary(Op.PLUS, other)

    def __sub__(self, other):
        return self._binary(Op.MINUS, other)

    def __rsub__(self, other):
        return self._reverse_binary(Op.MINUS, other)

    def __mul__(self, other):
        return self._binary(Op.TIMES, other)

    def __rmul__(self, other):
        return self._reverse_binary(Op.TIMES, other)

    def __truediv__(self, other):
        return self._binary(Op.DIV, other)

    def __rtruediv__(self, other):
        return self._reverse_binary(Op.DIV, other)

    def __pow__(self, other):
        return self._binary(Op.POW, other)

    def __rpow__(self, other):
        return self._reverse_binary(Op.POW, other)

    def __eq__(self, other):
        return self._binary(Op.EQ, other)

    def __ne__(self, other):
        return self._binary(Op.NE, other)

    def __lt__(self, other):
        return self._binary(Op.LT, other)

    def __le__(self, other):
        return self._binary(Op.LE, other)

    def __gt__(self, other):
        return self._binary(Op.GT, other)

    def __ge__(self, other):
        return self._binary(Op.GE, other)

    def __and__(self, other):
        return self._binary(Op.AND, other)

    def __rand__(self, other):
        return self._reverse_binary(Op.AND, other)

    def __or__(self, other):
        return self._binary(Op.OR, other)

    def __ror__(self, other):
        return self._reverse_binary(Op.OR, other)

    def __bool__(self):
        raise TypeError(
            "Column expressions have no single truth value. "
            "Use &, |, and ~ with parentheses instead of and, or, and not."
        )

    
class ConstantExpression(ColumnExpression):
    def __init__(self, const:int|float):
        self.const = const

    def render(self) -> str:
        return str(self.const)

    def evaluate(self, df=None):
        return self.const, True


class NamedColumnExpression(ColumnExpression):
    def __init__(self, name:str):
        self.name = name

    def render(self) -> str:
        return str(self.name)

    def evaluate(self, df=None):
        col = df[self.name]
        return col.array, col._valid_mask()

    def cols_used(self) -> set[str]:
        return {self.name}

class BinaryOpExpression(ColumnExpression):
    def __init__(self, op, c1, c2):
        self.op = op
        self.c1 = c1
        self.c2 = c2
        
    def render(self) -> str:
        return f"({self.c1.render()} {op_lookup[self.op]} {self.c2.render()})"

    def evaluate(self, df=None):
        lv, lm = evaluate(df, self.c1)
        rv, rm = evaluate(df, self.c2)
        lv, rv = np.asarray(lv), np.asarray(rv)
        try:
            result = op_functions[self.op](lv, rv)
        except ValueError:
            if lv.size != rv.size:
                raise ValueError(
                    f"Columns have different lengths: "
                    f"{self.c1.render()} = {lv.size}, {self.c2.render()} = {rv.size}")
            raise
        return result, lm & rm

    def cols_used(self) -> set[str]:
        return self.c1.cols_used() | self.c2.cols_used()

    def is_rowwise(self) -> bool:
        return self.c1.is_rowwise() and self.c2.is_rowwise()

class ObjectExpression(ColumnExpression):
    def __init__(self, value):
        self.value = value

    def render(self):
        if self.value is None:
            return "NULL"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)

    def evaluate(self, df=None):
        return self.value, True

class FunCallExpression(ColumnExpression):
    def __init__(self, fn, *args: ColumnExpression):
        self.fn = fn
        self.args = args

    def render(self):
        arguments = [arg.render() for arg in self.args]
        inner = ", ".join(arguments)
        return f"{self.fn.__name__}({inner})"

    def evaluate(self, df=None):
        pairs = [evaluate(df, arg) for arg in self.args]
        values = [v for v, _ in pairs]
        mask = True
        for _, m in pairs:
            mask = mask & m
        return self.fn(*values), mask

    def cols_used(self) -> set[str]:
        return set().union(*(a.cols_used() for a in self.args))

    def is_rowwise(self) -> bool:
        fn_ok = isinstance(self.fn, np.ufunc) or getattr(self.fn, "rowwise", False)
        return fn_ok and all(a.is_rowwise() for a in self.args)


def as_expression(value: Any) -> ColumnExpression:
    if isinstance(value, ColumnExpression):
        return value
    if isinstance(value, bool):
        return ObjectExpression(value)
    if isinstance(value, int):
        return ConstantExpression(value)
    if isinstance(value, float):
        return ConstantExpression(value)
    return ObjectExpression(value)

def c(name: str) -> ColumnExpression:
    if not isinstance(name, str):
        raise ValueError(f"Column name should be a string, got {type(name)}")
    return NamedColumnExpression(name)

def call(func, *args):
    return FunCallExpression(func, *[as_expression(a) for a in args])

def rowwise(fn):
    """Mark a user function as acting entrywise, so the optimizer may push filters past it."""
    fn.rowwise = True
    return fn

def evaluate(df, expr):
    return expr.evaluate(df)
