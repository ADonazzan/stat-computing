import math
from fractions import Fraction

from streams import (chain, convolve, diff, from_seq, integers_from, iterate, map,
                     power, reciprocal, repeat, take)


def poly(coeffs):
    return chain(from_seq(coeffs), repeat(0))


# Reciprocal: solve st = 1 --------
# t_0 = 1/s_0 and, for n >= 1, sum_k s_k t_{n-k} = 0  =>  t_n = -(1/s_0) sum_{k>=1} s_k t_{n-k}
s = map(Fraction, integers_from(1))            # 1 + 2z + 3z^2 + ... = 1/(1-z)^2
t = reciprocal(s)                              # so t should be (1-z)^2 = 1 - 2z + z^2


# Probability generating functions --------
def bernoulli_pgf(p):
    return poly([1 - p, p])


def pgf_of_sum(*pgfs):
    """PGF of a sum of independent variables = product of their PGFs."""
    total = poly([1])
    for g in pgfs:
        total = convolve(total, g)
    return total


# Puzzle 102 --------
known = [1, 2, 4, 8, 16, 31, 57, 99, 163]

# Difference table; n = first row that is all zeros (Delta^n a = 0*)
table = take(iterate(diff, from_seq(known)), len(known))
n = next(k for k, row in enumerate(table) if all(x == 0 for x in take(row, math.inf)))

# (1-z)^n A(z) = P(z)
one_minus_z = poly([1, -1])
P = take(convolve(power(one_minus_z, n), poly(known)), n)
a = convolve(poly(P), power(one_minus_z, -n))

if __name__ == "__main__":
    print("t    =", [str(x) for x in take(t, 6)])
    print("s*t  =", [str(x) for x in take(convolve(s, t), 6)])

    p = Fraction(1, 3)
    binomial = power(bernoulli_pgf(p), 4)
    print("Binomial(4, 1/3) pmf:", [str(x) for x in take(binomial, 5)])
    print("closed form         :", [str(math.comb(4, k) * p ** k * (1 - p) ** (4 - k)) for k in range(5)])
    mixed = pgf_of_sum(bernoulli_pgf(0.5), bernoulli_pgf(0.2), bernoulli_pgf(0.9))
    print("Bern(.5)+Bern(.2)+Bern(.9) pmf:", take(mixed, 4))

    print("n =", n, " P =", P)
    print(take(a, 12))