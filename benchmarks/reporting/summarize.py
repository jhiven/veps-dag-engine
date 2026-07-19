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

    # 2. Reconfiguration Summaries (req_to_effect, max_output_gap, old_plan_frames)
    reconfig_effect_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    reconfig_gap_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    reconfig_frames_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for rsample in data.reconfiguration_samples:
        key = (rsample.scenario_id, rsample.edit_type, rsample.baseline)
        if rsample.request_to_effect_ns is not None:
            reconfig_effect_key[key].append(float(rsample.request_to_effect_ns))
        if rsample.maximum_output_gap_ns is not None:
            reconfig_gap_key[key].append(float(rsample.maximum_output_gap_ns))
        reconfig_frames_key[key].append(float(rsample.old_plan_frames_admitted_after_request_before_commit))

    for (scen, edit, base), vals in reconfig_effect_key.items():
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

    for (scen, edit, base), vals in reconfig_gap_key.items():
        stats = calculate_summary_stats(vals)
        rows.append([
            "reconfiguration_max_output_gap",
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

    for (scen, edit, base), vals in reconfig_frames_key.items():
        stats = calculate_summary_stats(vals)
        rows.append([
            "reconfiguration_old_plan_frames_before_commit",
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

    # 3. Dedicated TOST Summary CSV
    tost_output_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "tost_summary.csv")
    tost_headers = [
        "comparison",
        "topology",
        "workload",
        "sample_size",
        "mean_diff_ns",
        "mean_relative_diff_pct",
        "equivalence_margin_pct",
        "ci_90_lower_ns",
        "ci_90_upper_ns",
        "ci_95_lower_ns",
        "ci_95_upper_ns",
        "p_value_lower",
        "p_value_upper",
        "tost_p_value",
        "equivalent",
    ]

    tost_rows: list[list[str]] = [tost_headers]

    comparisons = (
        ("static_compiled_vs_versioned_compiled", "static_compiled", "versioned_compiled"),
        ("hard_coded_vs_static_compiled", "hard_coded", "static_compiled"),
        ("hard_coded_vs_versioned_compiled", "hard_coded", "versioned_compiled"),
    )

    for comp_name, base_impl, treat_impl in comparisons:
        for top in ("linear_5", "branch_merge_9"):
            for work in ("minimal", "approximately_1_ms", "approximately_5_ms"):
                base_vals = steady_by_key.get((top, work, base_impl), [])
                treat_vals = steady_by_key.get((top, work, treat_impl), [])
                if base_vals and treat_vals and len(base_vals) == len(treat_vals):
                    tost = calculate_tost_equivalence(base_vals, treat_vals, relative_margin=0.01)
                    tost_rows.append([
                        comp_name,
                        top,
                        work,
                        str(len(base_vals)),
                        f"{tost.mean_diff:.2f}",
                        f"{tost.mean_relative_diff_percent:.4f}%",
                        f"{tost.relative_margin_percent:.2f}%",
                        f"{tost.ci_90_lower:.2f}",
                        f"{tost.ci_90_upper:.2f}",
                        f"{tost.ci_95_lower:.2f}",
                        f"{tost.ci_95_upper:.2f}",
                        f"{tost.p_value_lower:.4f}",
                        f"{tost.p_value_upper:.4f}",
                        f"{tost.p_value:.4f}",
                        str(tost.equivalent),
                    ])

    with open(tost_output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(tost_rows)
