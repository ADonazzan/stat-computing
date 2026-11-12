from streams import cons, head, integers_from, map, merge, tail, take


def by_sum(p, q):
    return (p[0] + p[1], p[0]) <= (q[0] + q[1], q[0])


# All pairs (s_i, t_j) ordered by s_i + t_j; s and t must be infinite and increasing.
def pairs(s, t):
    s0 = head(s)
    first_row = map(lambda b: (s0, b), tail(t))
    return cons((s0, head(t)), lambda: merge(first_row, pairs(tail(s), t), by_sum))


if __name__ == "__main__":
    nat = integers_from(0)
    print(take(pairs(nat, nat), 10))

    squares = map(lambda n: n * n, integers_from(1))
    cubes = map(lambda n: n ** 3, integers_from(1))
    print([(a, b, a + b) for a, b in take(pairs(squares, cubes), 8)])

    # The convolve and shuffle products are in src/streams/numeric.py:
    # (st)_n sums s_i t_j over the n-th anti-diagonal of pairs(nat, nat).
