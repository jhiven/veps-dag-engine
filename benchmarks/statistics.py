"""Statistical summary, bootstrap confidence interval, paired comparison, and TOST functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "SummaryStats",
    "PairedStats",
    "TOSTResult",
    "calculate_summary_stats",
    "calculate_paired_stats",
    "calculate_tost_equivalence",
]


@dataclass(frozen=True, slots=True)
class SummaryStats:
    count: int
    min_value: float
    max_value: float
    mean: float
    std_dev: float
    median: float
    p95: float
    p99: float
    iqr: float
    ci_95_lower: float
    ci_95_upper: float


@dataclass(frozen=True, slots=True)
class PairedStats:
    count: int
    mean_diff: float
    median_diff: float
    absolute_diff: float
    relative_diff: float
    ci_95_lower: float
    ci_95_upper: float


@dataclass(frozen=True, slots=True)
class TOSTResult:
    equivalent: bool
    p_value: float
    t_stat_lower: float
    t_stat_upper: float
    margin: float
    mean_diff: float
    ci_95_lower: float
    ci_95_upper: float


def _bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    n = len(data)
    if n <= 1:
        val = float(stat_fn(data)) if n == 1 else 0.0
        return val, val
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=(n_resamples, n), replace=True)
    resamples = data[idx]
    stats_list = [stat_fn(resamples[i]) for i in range(n_resamples)]
    stats_arr = np.asarray(stats_list, dtype=np.float64)
    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.percentile(stats_arr, alpha * 100.0))
    high = float(np.percentile(stats_arr, (1.0 - alpha) * 100.0))
    return low, high


def calculate_summary_stats(values: Sequence[float] | np.ndarray) -> SummaryStats:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return SummaryStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median_val = float(np.median(arr))
    p95_val = float(np.percentile(arr, 95.0))
    p99_val = float(np.percentile(arr, 99.0))
    p75 = float(np.percentile(arr, 75.0))
    p25 = float(np.percentile(arr, 25.0))
    iqr_val = p75 - p25

    def med_fn(a: np.ndarray) -> float:
        return float(np.median(a))

    ci_lower, ci_upper = _bootstrap_ci(arr, med_fn)

    return SummaryStats(
        count=n,
        min_value=min_val,
        max_value=max_val,
        mean=mean_val,
        std_dev=std_val,
        median=median_val,
        p95=p95_val,
        p99=p99_val,
        iqr=iqr_val,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )


def calculate_paired_stats(
    a_values: Sequence[float] | np.ndarray,
    b_values: Sequence[float] | np.ndarray,
) -> PairedStats:
    a_arr = np.asarray(a_values, dtype=np.float64)
    b_arr = np.asarray(b_values, dtype=np.float64)
    if len(a_arr) != len(b_arr):
        raise ValueError("Paired samples must have identical length.")

    n = len(a_arr)
    if n == 0:
        return PairedStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    diffs = b_arr - a_arr
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    mean_a = float(np.mean(a_arr))

    abs_diff = mean_diff
    rel_diff = (mean_diff / mean_a) if mean_a != 0 else 0.0

    def mean_fn(d: np.ndarray) -> float:
        return float(np.mean(d))

    ci_lower, ci_upper = _bootstrap_ci(diffs, mean_fn)

    return PairedStats(
        count=n,
        mean_diff=mean_diff,
        median_diff=median_diff,
        absolute_diff=abs_diff,
        relative_diff=rel_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )


def calculate_tost_equivalence(
    baseline_values: Sequence[float] | np.ndarray,
    treatment_values: Sequence[float] | np.ndarray,
    relative_margin: float = 0.01,
) -> TOSTResult:
    base = np.asarray(baseline_values, dtype=np.float64)
    treat = np.asarray(treatment_values, dtype=np.float64)
    n = len(base)
    if n < 2:
        return TOSTResult(False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    diffs = treat - base
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    margin = abs(relative_margin * float(np.mean(base)))

    t_stat_lower = (mean_diff - (-margin)) / se_diff if se_diff > 0 else 0.0
    cdf_lower = float(np.float64(stats.t.cdf(t_stat_lower, df=n - 1)))
    p_val_lower = float(1.0 - cdf_lower)

    t_stat_upper = (margin - mean_diff) / se_diff if se_diff > 0 else 0.0
    cdf_upper = float(np.float64(stats.t.cdf(t_stat_upper, df=n - 1)))
    p_val_upper = float(1.0 - cdf_upper)

    p_value = max(p_val_lower, p_val_upper)
    equivalent = bool(p_value < 0.05)

    def mean_fn(d: np.ndarray) -> float:
        return float(np.mean(d))

    ci_lower, ci_upper = _bootstrap_ci(diffs, mean_fn)

    return TOSTResult(
        equivalent=equivalent,
        p_value=p_value,
        t_stat_lower=float(t_stat_lower),
        t_stat_upper=float(t_stat_upper),
        margin=margin,
        mean_diff=mean_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )
