"""Tests for statistical calculation utilities."""

from __future__ import annotations

import numpy as np
from benchmarks.statistics import (
    calculate_paired_stats,
    calculate_summary_stats,
    calculate_tost_equivalence,
)


def test_calculate_summary_stats() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = calculate_summary_stats(data)
    assert stats.count == 5
    assert stats.min_value == 10.0
    assert stats.max_value == 50.0
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.ci_95_lower <= stats.median <= stats.ci_95_upper


def test_calculate_paired_stats() -> None:
    a = [100.0, 105.0, 110.0, 115.0, 120.0]
    b = [102.0, 106.0, 111.0, 116.0, 122.0]
    paired = calculate_paired_stats(a, b)
    assert paired.count == 5
    assert paired.mean_diff == 1.4
    assert paired.ci_95_lower <= paired.mean_diff <= paired.ci_95_upper


def test_tost_equivalence() -> None:
    np.random.seed(42)
    base = np.random.normal(5_000_000, 100, 30)
    treat = base + np.random.normal(0, 10, 30)  # extremely close to base

    res = calculate_tost_equivalence(base, treat, relative_margin=0.01)
    assert res.equivalent is True
    assert res.p_value < 0.05
