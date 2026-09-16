import operator

import numpy as np

from abc    import ABC, abstractmethod
from enum   import StrEnum, auto
from typing import Any

class Op(StrEnum):
    PLUS = auto()
    TIMES = auto()
    EQ = auto()

op_lookup = {Op.PLUS: "+", Op.TIMES: "*", Op.EQ: "=="}
op_functions = {Op.PLUS: operator.add, Op.TIMES: operator.mul, Op.EQ: operator.eq}

class ColumnExpression(ABC):
    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.render()

    def __add__(self, other: Any) -> "ColumnExpression":
        return BinaryOpExpression(Op.PLUS, self, as_expression(other))
    
    def __radd__(self, other):
        return BinaryOpExpression(Op.PLUS, as_expression(other), self)

    def __mul__(self, other):
        return BinaryOpExpression(Op.TIMES, self, as_expression(other))

    def __rmul__ (self, other):
        return BinaryOpExpression(Op.TIMES, as_expression(other), self)

    def __eq__(self, other):
        return BinaryOpExpression(Op.EQ, self, as_expression(other))
    

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

def evaluate(df, expr):
    return expr.evaluate(df)

def optimize(expr):
    raise NotImplementedError
    

if __name__ == "__main__":
    print(call(max, c("a"), 3, c("t")))
    print(2 * c("a") + 3 * call(max, c("a"), 3, c("t")))

    test_df = {"a": [1,2,3], "t": [2,3,4]}
    print(evaluate(test_df, c("a") * c("t") + 3))

# Optimization: constant folding
