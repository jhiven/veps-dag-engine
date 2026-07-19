"""Summary generation for benchmark runs."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

from benchmarks.reporting.load import BenchmarkArtifactData
from benchmarks.statistics import (
    calculate_summary_stats,
    calculate_tost_equivalence,
)

__all__ = ["generate_summary_csv"]


def generate_summary_csv(data: BenchmarkArtifactData, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows: list[list[str]] = []
    headers = [
        "metric_category",
        "group_key",
        "count",
        "mean",
        "std_dev",
        "median",
        "p95",
        "p99",
        "ci_95_lower",
        "ci_95_upper",
        "additional_info",
    ]
    rows.append(headers)

    # 1. Steady-State Summaries
    steady_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in data.steady_state_samples:
        key = (sample.topology, sample.workload_id, sample.implementation)
        steady_by_key[key].append(sample.normalized_ns_per_frame)

    for (top, work, impl), vals in steady_by_key.items():
        stats = calculate_summary_stats(vals)
        rows.append([
            "steady_state",
            f"{top}:{work}:{impl}",
            str(stats.count),
            f"{stats.mean:.2f}",
            f"{stats.std_dev:.2f}",
            f"{stats.median:.2f}",
            f"{stats.p95:.2f}",
            f"{stats.p99:.2f}",
            f"{stats.ci_95_lower:.2f}",
            f"{stats.ci_95_upper:.2f}",
            "",
        ])

    # 2. TOST for 5ms workload
    for top in ("linear_5", "branch_merge_9"):
        hc_vals = steady_by_key.get((top, "approximately_5_ms", "hard_coded"), [])
        ver_vals = steady_by_key.get((top, "approximately_5_ms", "versioned_compiled"), [])
        if hc_vals and ver_vals and len(hc_vals) == len(ver_vals):
            tost = calculate_tost_equivalence(hc_vals, ver_vals, relative_margin=0.01)
            rows.append([
                "tost_equivalence_5ms",
                f"{top}:hard_coded_vs_versioned",
                str(len(hc_vals)),
                f"{tost.mean_diff:.2f}",
                "0.0",
                f"{tost.mean_diff:.2f}",
                "0.0",
                "0.0",
                f"{tost.ci_95_lower:.2f}",
                f"{tost.ci_95_upper:.2f}",
                f"equivalent={tost.equivalent}; p_value={tost.p_value:.4f}; margin={tost.margin:.2f}",
            ])

    # 3. Reconfiguration Summaries
    reconfig_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for rsample in data.reconfiguration_samples:
        key = (rsample.scenario_id, rsample.edit_type, rsample.baseline)
        if rsample.request_to_effect_ns is not None:
            reconfig_by_key[key].append(float(rsample.request_to_effect_ns))

    for (scen, edit, base), vals in reconfig_by_key.items():
        stats = calculate_summary_stats(vals)
        rows.append([
            "reconfiguration_req_to_effect",
            f"{scen}:{edit}:{base}",
            str(stats.count),
            f"{stats.mean:.2f}",
            f"{stats.std_dev:.2f}",
            f"{stats.median:.2f}",
            f"{stats.p95:.2f}",
            f"{stats.p99:.2f}",
            f"{stats.ci_95_lower:.2f}",
            f"{stats.ci_95_upper:.2f}",
            "",
        ])

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
