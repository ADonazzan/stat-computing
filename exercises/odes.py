import math
import operator

from streams import Fold, Stream, add, at, identity, map, memo, scale, scan


# integrate: s_n = s_0 + sum_{i<n} x_i dt
def integrate(x, dt, initial_value=0):
    """`x` is a stream or a thunk returning one (forced only when the second term is needed)."""
    running_sum = Fold(lambda v: v * dt, operator.add, lambda: initial_value, identity)
    delayed = Stream(lambda: iter(x() if callable(x) else x))
    return scan(running_sum, delayed)


# First order: y' = f(y), y(0) = y0
def solve_diffeq1(f, y0, dt):
    y = memo(integrate(lambda: dy, dt, y0))
    dy = map(f, y)
    return y


# Second order, constant coefficients: y'' = a y' + b y, y(0) = y0, y'(0) = dy0
def solve_diffeq2(a, b, y0, dy0, dt):
    y = memo(integrate(lambda: dy, dt, y0))
    dy = memo(integrate(lambda: ddy, dt, dy0))
    ddy = add(scale(a, dy), scale(b, y))
    return y


if __name__ == "__main__":
    dt = 0.001
    steps = round(1 / dt)

    # y' = y, y(0) = 1  ->  y(1) = e
    print("e      ~", at(solve_diffeq1(lambda y: y, 1, dt), steps), " exact", math.e)

    # y' = -2y, y(0) = 1  ->  y(1) = exp(-2)
    print("exp(-2)~", at(solve_diffeq1(lambda y: -2 * y, 1, dt), steps), " exact", math.exp(-2))

    # y'' = -y, y(0) = 1, y'(0) = 0  ->  y = cos t
    cos = solve_diffeq2(0, -1, 1, 0, dt)
    print("cos(1) ~", at(cos, steps), " exact", math.cos(1))
    print("y(2pi) ~", at(cos, round(2 * math.pi / dt)), " exact 1")

    # y'' = -y' - y (damped), compared with the closed form at t = 1
    w = math.sqrt(3) / 2
    exact = math.exp(-0.5) * (math.cos(w) + math.sin(w) / (2 * w))
    print("damped ~", at(solve_diffeq2(-1, -1, 1, 0, dt), steps), " exact", exact)
