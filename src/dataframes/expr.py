import operator

import numpy as np

from abc    import ABC, abstractmethod
from enum   import StrEnum, auto
from typing import Any

class Op(StrEnum):
    PLUS = auto()
    TIMES = auto()

op_lookup = {Op.PLUS: "+", Op.TIMES: "*"}
op_functions = {Op.PLUS: operator.add, Op.TIMES: operator.mul}

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


class ConstantExpression(ColumnExpression):
    def __init__(self, const:int|float):
        self.const = const

    def render(self) -> str:
        return str(self.const)

    def evaluate(self, df=None):
        return self.const


class NamedColumnExpression(ColumnExpression):
    def __init__(self, name:str):
        self.name = name

    def render(self) -> str:
        return str(self.name)

    def evaluate(self, df=None):
        return df[self.name]

class BinaryOpExpression(ColumnExpression):
    def __init__(self, op, c1, c2):
        self.op = op
        self.c1 = c1
        self.c2 = c2
        
    def render(self) -> str:
        return f"({self.c1.render()} {op_lookup[self.op]} {self.c2.render()})"

    def evaluate(self, df=None):
        left = evaluate(df, self.c1)
        right = evaluate(df, self.c2)

        left = np.asarray(left)
        right = np.asarray(right)

        try:
            result = op_functions[self.op](left, right)
        except ValueError as e:
            if left.size != right.size:
                raise ValueError(f"Columns have different lenght: {self.c1.render()} = {left.size}, {self.c2.render()} = {right.size}")
            else:
                raise ValueError(e)
            
        return result

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
        return self.value

class FunCallExpression(ColumnExpression):
    def __init__(self, fn, *args: ColumnExpression):
        self.fn = fn
        self.args = args

    def render(self):
        arguments = [arg.render() for arg in self.args]
        inner = ", ".join(arguments)
        return f"{self.fn.__name__}({inner})"

    def evaluate(self, df=None):
        arguments = [evaluate(df, arg) for arg in self.args]
        return self.fn(*arguments)


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
