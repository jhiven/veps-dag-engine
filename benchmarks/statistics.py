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
    p_value_lower: float
    p_value_upper: float
    t_stat_lower: float
    t_stat_upper: float
    margin: float
    relative_margin_percent: float
    mean_diff: float
    mean_relative_diff_percent: float
    ci_90_lower: float
    ci_90_upper: float
    ci_95_lower: float
    ci_95_upper: float
    bootstrap_ci_95_lower: float
    bootstrap_ci_95_upper: float


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

    # Parametric 95 % CI for the mean (t-distribution), consistent with
    # the mean and std_dev reported in the same row.
    if n >= 2:
        se = std_val / np.sqrt(n)
        critical = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = mean_val - critical * se
        ci_upper = mean_val + critical * se
    else:
        ci_lower = mean_val
        ci_upper = mean_val

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
    """Two one-sided test (TOST) for paired equivalence.

    Uses a paired t-distribution for both p-values and confidence intervals
    so that the equivalence decision and the reported intervals are
    internally consistent.  Bootstrap CIs are also computed and stored
    separately for diagnostic use.
    """
    base = np.asarray(baseline_values, dtype=np.float64)
    treat = np.asarray(treatment_values, dtype=np.float64)
    n = len(base)
    if n < 2:
        return TOSTResult(
            False, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, relative_margin * 100.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    diffs = treat - base
    mean_base = float(np.mean(base))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    margin = abs(relative_margin * mean_base)

    if se_diff == 0:
        # All paired differences are identical (and may all be zero).
        # The t-test is degenerate; decide equivalence directly from
        # whether the constant difference lies within the margin.
        equivalent = bool(-margin < mean_diff < margin)
        p_val_lower = 0.0 if mean_diff > -margin else 1.0
        p_val_upper = 0.0 if mean_diff < margin else 1.0
        p_value = max(p_val_lower, p_val_upper)
        t_stat_lower = 0.0
        t_stat_upper = 0.0
        ci_90_lower = mean_diff
        ci_90_upper = mean_diff
        ci_95_lower = mean_diff
        ci_95_upper = mean_diff
    else:
        df = n - 1

        t_stat_lower = (mean_diff - (-margin)) / se_diff
        t_stat_lower_val = float(t_stat_lower)
        cdf_lower: float = stats.t.cdf(t_stat_lower_val, df=df)
        p_val_lower = 1.0 - cdf_lower

        t_stat_upper = (margin - mean_diff) / se_diff
        t_stat_upper_val = float(t_stat_upper)
        cdf_upper: float = stats.t.cdf(t_stat_upper_val, df=df)
        p_val_upper = float(1.0 - cdf_upper)

        p_value = max(p_val_lower, p_val_upper)
        equivalent = bool(p_value < 0.05)

        # Parametric confidence intervals from the t-distribution
        critical_90 = float(stats.t.ppf(0.95, df=df))
        ci_90_lower = mean_diff - critical_90 * se_diff
        ci_90_upper = mean_diff + critical_90 * se_diff

        critical_95 = float(stats.t.ppf(0.975, df=df))
        ci_95_lower = mean_diff - critical_95 * se_diff
        ci_95_upper = mean_diff + critical_95 * se_diff

    mean_rel_diff_pct = (mean_diff / mean_base * 100.0) if mean_base != 0 else 0.0

    # Bootstrap CIs for diagnostic comparison
    def mean_fn(d: np.ndarray) -> float:
        return float(np.mean(d))

    boot_95_lower, boot_95_upper = _bootstrap_ci(diffs, mean_fn, confidence_level=0.95)

    return TOSTResult(
        equivalent=equivalent,
        p_value=p_value,
        p_value_lower=p_val_lower,
        p_value_upper=p_val_upper,
        t_stat_lower=float(t_stat_lower),
        t_stat_upper=float(t_stat_upper),
        margin=margin,
        relative_margin_percent=relative_margin * 100.0,
        mean_diff=mean_diff,
        mean_relative_diff_percent=mean_rel_diff_pct,
        ci_90_lower=ci_90_lower,
        ci_90_upper=ci_90_upper,
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        bootstrap_ci_95_lower=boot_95_lower,
        bootstrap_ci_95_upper=boot_95_upper,
    )
