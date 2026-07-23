"""Pure-Python Student's t-distribution CDF and inverse CDF (ppf).

Replaces `scipy.stats.t.cdf` / `scipy.stats.t.ppf` with no third-party
dependencies, so this works fine under a free-threaded (no-GIL) build of
Python where scipy wheels aren't available yet.

Implementation notes
---------------------
The t-distribution CDF is expressed via the regularized incomplete beta
function I_x(a, b):

    F(t) = 1 - 0.5 * I_x(df/2, 1/2)   for t >= 0
    F(t) =       0.5 * I_x(df/2, 1/2)  for t <  0

    where x = df / (df + t**2)

I_x is evaluated with the standard continued-fraction method (Numerical
Recipes, "betacf"), which is accurate to ~1e-14 for the parameter ranges
used here. The inverse (ppf) is obtained by bisecting on `t_cdf`, which is
monotonic, so it's simple and robust even though it's not the fastest
possible approach. For benchmark-summary purposes (this module is called
a handful of times per report, not per-sample), that's more than fast
enough.
"""
from __future__ import annotations

import math

__all__ = ["t_cdf", "t_ppf"]

_MAXIT = 200
_EPS = 3.0e-16
_FPMIN = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b), for x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    """CDF of the Student's t-distribution with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x_val = df / (df + x * x)
    ib = _betainc(df / 2.0, 0.5, x_val)
    if x >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile function) of the Student's t-distribution.

    Solved by bisection since `t_cdf` is monotonic in `x`; plenty fast for
    the handful of calls this module makes per report.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    # Expand bracket until t_cdf(lo) < p < t_cdf(hi)
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(mid)):
            break
    return (lo + hi) / 2.0