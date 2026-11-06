
import math

from streams import at, chunked, cumsum, head, integers_from, iterate, map, take

# pi/4
pi_4_series = cumsum(map(lambda n: (-1) ** n / (2 * n + 1), integers_from(0)))

# ln(2)
ln2_series = cumsum(map(lambda n: (-1) ** n / (n + 1), integers_from(0)))

# euler_improve
def _euler_step(prev, cur, nxt):
    """s'_n = s_{n+1} - (s_{n+1} - s_n)^2 / (s_{n+1} - 2 s_n + s_{n-1})"""
    denom = nxt - 2 * cur + prev
    return nxt if denom == 0 else nxt - (nxt - cur) ** 2 / denom


def euler_improve(s):
    return map(lambda w: _euler_step(*w), chunked(s, 3, 1))


if __name__ == "__main__":
    pi_tableau = iterate(euler_improve, pi_4_series)
    ln2_tableau = iterate(euler_improve, ln2_series)

    print("pi, 8th partial sum:      ", 4 * at(pi_4_series, 7))
    print("pi, 8th Euler-improved:   ", 4 * at(euler_improve(pi_4_series), 7))
    print("pi, tableau heads:        ", [4 * x for x in take(map(head, pi_tableau), 8)])
    print("pi, tableau second elems: ", [4 * x for x in take(map(lambda s: at(s, 1), pi_tableau), 8)])

    print("ln2, tableau heads:       ", take(map(head, ln2_tableau), 8))