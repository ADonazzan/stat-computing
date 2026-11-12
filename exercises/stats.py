import math
import random
import statistics

from streams import Fold, Stream, chunked, drop, from_seq, identity, map, map_maybe, scan, slice, take


# --- Draws from distributions ----------------------------------------------
def uniforms(seed=None, lo=0.0, hi=1.0) -> Stream[float]:
    def gen():
        rng = random.Random(seed)
        while True:
            yield lo + (hi - lo) * rng.random()
    return Stream(gen)


def _box_muller(u1: float, u2: float) -> float:
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)


def _normal(u) -> float:
    """Standard normal from two uniforms (Box-Muller)"""
    return _box_muller(next(u), next(u))


def _gamma(shape: float, u) -> float:
    """Gamma(shape, rate=1) from uniforms (rejection sampler)."""
    if shape < 1:
        return _gamma(shape + 1, u) * next(u) ** (1 / shape)
    d = shape - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        z = _normal(u)
        v = (1 + c * z) ** 3
        if v > 0 and math.log(1 - next(u)) < z * z / 2 + d - d * v + d * math.log(v):
            return d * v


# --- Rejection sampling -----------------------------------------------------
def rejection_sample(density, lo, hi, bound, u: Stream[float]) -> Stream[float]:
    """iid draws from `density` (need not be normalised) on [lo, hi], with density <= bound."""
    def propose(pair):
        u1, u2 = pair
        x = lo + (hi - lo) * u1
        return x if u2 * bound <= density(x) else None
    return map_maybe(propose, chunked(u, 2))


# --- Metropolis-Hastings: y_i ~ N(theta, sigma2), theta ~ N(m0, s0_sq), sigma2 known ---
def mh_normal(data, u: Stream[float], *, sigma2=1.0, m0=0.0, s0_sq=100.0,
              step=1.0, theta_init=0.0) -> Stream[float]:
    ys = tuple(data)

    def log_post(theta):
        return (-sum((y - theta) ** 2 for y in ys) / (2 * sigma2)
                - (theta - m0) ** 2 / (2 * s0_sq))

    def move(state, us):
        theta, lp = state
        u1, u2, u3 = us
        proposal = theta + step * _box_muller(u1, u2)
        lp_new = log_post(proposal)
        if math.log(1 - u3) < lp_new - lp:
            return proposal, lp_new
        return state

    chain = Fold(identity, move, lambda: (theta_init, log_post(theta_init)), lambda st: st[0])
    return scan(chain, chunked(u, 3))


# --- Gibbs: y_i ~ N(mu, sigma2), mu ~ N(m0, s0_sq), sigma2 ~ InvGamma(a0, b0) ---
def gibbs_normal(data, u: Stream[float], *, m0=0.0, s0_sq=100.0, a0=2.0, b0=2.0,
                 sigma2_init=1.0) -> Stream[tuple[float, float]]:
    ys = tuple(data)
    n, total = len(ys), sum(ys)

    def gen():
        it = iter(u)
        sigma2 = sigma2_init
        while True:
            # mu | sigma2, y ~ Normal
            v = 1 / (1 / s0_sq + n / sigma2)
            m = v * (m0 / s0_sq + total / sigma2)
            mu = m + math.sqrt(v) * _normal(it)
            # sigma2 | mu, y ~ InvGamma(a0 + n/2, b0 + SS/2)
            ss = sum((y - mu) ** 2 for y in ys)
            sigma2 = (b0 + ss / 2) / _gamma(a0 + n / 2, it)
            yield mu, sigma2

    return Stream(gen)


# --- Lowess: local linear fit, tricube weights, k-point windows -------------
def lowess(points: Stream[tuple[float, float]], k: int, derivative=False) -> Stream[tuple[float, float]]:
    """Smoothed (x, y) at the centre of each window of k consecutive points (x increasing,
    k odd). With derivative=True the second component is the local slope instead."""
    def fit(window):
        x0 = window[k // 2][0]
        width = max(x0 - window[0][0], window[-1][0] - x0)
        sw = sx = sy = sxx = sxy = 0.0
        for x, y in window:
            d = x - x0
            w = (1 - abs(d / width) ** 3) ** 3
            sw, sx, sy, sxx, sxy = sw + w, sx + w * d, sy + w * y, sxx + w * d * d, sxy + w * d * y
        slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
        level = (sy - slope * sx) / sw
        return x0, slope if derivative else level
    return map(fit, chunked(points, k, 1))


if __name__ == "__main__":
    # Rejection sampling from Beta(2, 5): mean 2/7, variance 10/392
    beta = rejection_sample(lambda x: x * (1 - x) ** 4, 0, 1, 0.082, uniforms(seed=2))
    xs = take(beta, 20000)
    print("Beta(2,5) mean, var  :", statistics.mean(xs), statistics.variance(xs),
          " exact", 2 / 7, 10 / 392)

    # Metropolis-Hastings against the closed-form posterior
    rng = random.Random(3)
    obs = [rng.gauss(1.5, 2) for _ in range(50)]
    post_var = 1 / (1 / 100 + len(obs) / 4)
    post_mean = post_var * sum(obs) / 4
    thetas = take(drop(mh_normal(obs, uniforms(seed=4), sigma2=4.0, step=0.7), 1000), 20000)
    print("MH posterior mean,var:", statistics.mean(thetas), statistics.variance(thetas),
          " exact", post_mean, post_var)
    moves = sum(a != b for a, b in zip(thetas, thetas[1:])) / len(thetas)
    print("MH acceptance rate   :", moves)

    # Gibbs for unknown mean and variance
    rng = random.Random(0)
    data = [rng.gauss(5, 2) for _ in range(500)]          # true mu = 5, sigma2 = 4

    chain = gibbs_normal(data, uniforms(seed=1))
    draws = take(slice(drop(chain, 2000), dn=5), 5000)  # burn-in, then thin
    mus, sigma2s = zip(*draws)

    print("sample mean, var     :", statistics.mean(data), statistics.variance(data))
    print("posterior mean mu    :", statistics.mean(mus))
    print("posterior mean sigma2:", statistics.mean(sigma2s))

    # Lowess on noisy sin(x): compare with sin and, for the slope, cos
    rng = random.Random(5)
    grid = [i * 0.02 for i in range(315)]
    noisy = from_seq([(x, math.sin(x) + rng.gauss(0, 0.2)) for x in grid])
    smooth = take(lowess(noisy, 61), math.inf)
    slopes = take(lowess(noisy, 61, derivative=True), math.inf)
    print("lowess points        :", len(smooth), "of", len(grid))
    print("max |fit - sin|      :", max(abs(y - math.sin(x)) for x, y in smooth))
    print("max |slope - cos|    :", max(abs(d - math.cos(x)) for x, d in slopes))
