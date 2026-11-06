import math

from streams import chain, convolve, diff, from_seq, iterate, power, repeat, take


def poly(coeffs):
    return chain(from_seq(coeffs), repeat(0))


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
    print("n =", n, " P =", P)
    print(take(a, 12))