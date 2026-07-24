"""Statistical summarization and planned contrast analysis for VEPS component ablation (E5, Sections 4 & 5)."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from benchmarks.model import (
    ABLATION_CONTRAST_HEADERS,
    ABLATION_STRATIFIED_HEADERS,
    AblationSampleRow,
)
from benchmarks.storage import read_ablation_rows

__all__ = [
    "AblationContrastResult",
    "AblationGroupSummary",
    "BimodalVariantDSummary",
    "apply_holm_correction",
    "calculate_ablation_summaries",
    "generate_ablation_summary_files",
]


@dataclass(frozen=True, slots=True)
class AblationGroupSummary:
    scenario_id: str
    edit_type: str
    variant: str
    variant_name: str
    metric: str
    count: int
    mean: float
    std_dev: float
    median: float
    p95: float
    ci_95_lower: float
    ci_95_upper: float


@dataclass(frozen=True, slots=True)
class AblationContrastResult:
    contrast_id: str  # A_vs_B, C_vs_D, A_vs_C, B_vs_D
    contrast_name: str
    scenario_id: str
    family: str  # preparation_placement, retirement_policy, transition_outcome
    metric: str
    count: int
    median_left: float
    median_right: float
    paired_median_difference: float
    paired_median_ratio: float
    paired_log_ratio_mean: float
    bootstrap_ci_95_lower: float
    bootstrap_ci_95_upper: float
    raw_p_value: float
    holm_p_value: float


@dataclass(frozen=True, slots=True)
class BimodalVariantDSummary:
    scenario_id: str
    frame_group: str  # "zero_frames" (== 0) vs "one_or_more_frames" (>= 1)
    count: int
    request_to_effect_median_ns: float
    request_to_effect_p95_ns: float
    transition_output_gap_median_ns: float
    transition_output_gap_p95_ns: float
    total_synchronous_median_ns: float
    total_synchronous_p95_ns: float
    offpath_preparation_median_ns: float
    offpath_preparation_p95_ns: float
    deferred_retirement_median_ns: float
    deferred_retirement_p95_ns: float


def apply_holm_correction(p_values: Sequence[float]) -> list[float]:
    """Apply Holm-Bonferroni correction over a sequence of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: list[tuple[int, float]] = []
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = (m - rank) * p_val
        running_max = max(running_max, adj)
        running_max = min(1.0, running_max)
        adjusted.append((orig_idx, running_max))
    adjusted.sort(key=lambda x: x[0])
    return [a[1] for a in adjusted]


def _bootstrap_paired_median_diff_ci(
    a_arr: np.ndarray, b_arr: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    n = len(a_arr)
    if n <= 1:
        diff = float(np.median(b_arr - a_arr)) if n == 1 else 0.0
        return diff, diff
    rng = np.random.default_rng(42)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(float(np.median(b_arr[idx] - a_arr[idx])))
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(diffs, alpha * 100.0))
    high = float(np.percentile(diffs, (1.0 - alpha) * 100.0))
    return low, high


def calculate_ablation_summaries(
    rows: Sequence[AblationSampleRow],
) -> tuple[
    list[AblationGroupSummary],
    list[AblationContrastResult],
    list[BimodalVariantDSummary],
]:
    # 1. Group Summaries
    by_group: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_group[(r.scenario_id, r.edit_type, r.variant, r.variant_name, "total_synchronous_ns")].append(float(r.total_synchronous_ns))
        by_group[(r.scenario_id, r.edit_type, r.variant, r.variant_name, "request_to_effect_ns")].append(float(r.request_to_effect_ns))
        by_group[(r.scenario_id, r.edit_type, r.variant, r.variant_name, "transition_output_gap_ns")].append(float(r.transition_output_gap_ns))
        by_group[(r.scenario_id, r.edit_type, r.variant, r.variant_name, "old_plan_frames_admitted_after_request_before_commit")].append(float(r.old_plan_frames_admitted_after_request_before_commit))
        by_group[(r.scenario_id, r.edit_type, r.variant, r.variant_name, "retirement_duration_ns")].append(float(r.retirement_duration_ns))

    group_summaries: list[AblationGroupSummary] = []
    for (scen, edit, var, var_name, metric), vals in sorted(by_group.items()):
        arr = np.asarray(vals, dtype=np.float64)
        n = len(arr)
        if n == 0:
            continue
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        med_v = float(np.median(arr))
        p95_v = float(np.percentile(arr, 95.0))
        if n >= 2 and std_v > 0:
            se = std_v / math.sqrt(n)
            crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low, ci_high = mean_v - crit * se, mean_v + crit * se
        else:
            ci_low, ci_high = mean_v, mean_v
        group_summaries.append(
            AblationGroupSummary(
                scenario_id=scen,
                edit_type=edit,
                variant=var,
                variant_name=var_name,
                metric=metric,
                count=n,
                mean=mean_v,
                std_dev=std_v,
                median=med_v,
                p95=p95_v,
                ci_95_lower=ci_low,
                ci_95_upper=ci_high,
            )
        )

    # 2. Planned Contrasts (Section 4)
    # Block keys: scenario_id, block_id, repetition, request_phase_offset_ns, candidate_graph_fingerprint, preparation_workload_id
    row_map: dict[tuple[str, str, int, int | None, str, str, str], AblationSampleRow] = {}
    for r in rows:
        row_map[(r.scenario_id, r.block_id, r.repetition, r.request_phase_offset_ns, r.candidate_graph_fingerprint, r.preparation_workload_id, r.variant)] = r

    planned_contrasts = [
        ("A_vs_B", "Offpath Prep (Sync Retire)", "variant_a", "variant_b"),
        ("C_vs_D", "Offpath Prep (Deferred Retire)", "variant_c", "variant_d"),
        ("A_vs_C", "Deferred Retire (Sync Prep)", "variant_a", "variant_c"),
        ("B_vs_D", "Deferred Retire (Offpath Prep)", "variant_b", "variant_d"),
    ]

    metric_families = {
        "preparation_placement": ("total_synchronous_ns", "synchronous_preparation_ns", "publication_ns"),
        "retirement_policy": ("total_synchronous_ns", "synchronous_retirement_ns", "deferred_retirement_ns"),
        "transition_outcome": ("request_to_effect_ns", "transition_output_gap_ns", "old_plan_frames_admitted_after_request_before_commit"),
    }

    contrast_results: list[AblationContrastResult] = []
    scenarios = sorted({r.scenario_id for r in rows})

    for scen in scenarios:
        for family_name, metrics in metric_families.items():
            family_raw_p_values: list[float] = []
            family_entries: list[tuple[str, str, str, int, float, float, float, float, float, float, float, float]] = []

            for metric in metrics:
                for cid, cname, var_left, var_right in planned_contrasts:
                    left_rows = [r for r in rows if r.scenario_id == scen and r.variant == var_left]

                    paired_left: list[float] = []
                    paired_right: list[float] = []

                    for rl in left_rows:
                        rr = row_map.get(
                            (
                                scen,
                                rl.block_id,
                                rl.repetition,
                                rl.request_phase_offset_ns,
                                rl.candidate_graph_fingerprint,
                                rl.preparation_workload_id,
                                var_right,
                            )
                        )
                        if rr is not None:
                            val_l = getattr(rl, metric)
                            val_r = getattr(rr, metric)
                            if val_l is not None and val_r is not None:
                                paired_left.append(float(val_l))
                                paired_right.append(float(val_r))

                    n_pair = len(paired_left)
                    if n_pair < 2:
                        continue

                    arr_left = np.asarray(paired_left, dtype=np.float64)
                    arr_right = np.asarray(paired_right, dtype=np.float64)
                    diffs = arr_right - arr_left

                    med_l = float(np.median(arr_left))
                    med_r = float(np.median(arr_right))
                    paired_med_diff = float(np.median(diffs))
                    paired_med_ratio = (med_r / med_l) if med_l != 0 else 1.0
                    ratios = arr_right / np.where(arr_left == 0, 1e-9, arr_left)
                    log_ratios = np.log(np.maximum(1e-9, ratios))
                    paired_log_ratio_mean = float(np.mean(log_ratios))

                    boot_low, boot_high = _bootstrap_paired_median_diff_ci(arr_left, arr_right)

                    if np.all(diffs == 0):
                        raw_p = 1.0
                    else:
                        try:
                            res = stats.wilcoxon(diffs)
                            raw_p = float(res.pvalue)
                        except Exception:
                            raw_p = 1.0

                    family_raw_p_values.append(raw_p)
                    family_entries.append(
                        (
                            cid,
                            cname,
                            metric,
                            n_pair,
                            med_l,
                            med_r,
                            paired_med_diff,
                            paired_med_ratio,
                            paired_log_ratio_mean,
                            boot_low,
                            boot_high,
                            raw_p,
                        )
                    )

            if family_raw_p_values:
                holm_ps = apply_holm_correction(family_raw_p_values)
                for entry, holm_p in zip(family_entries, holm_ps, strict=False):
                    cid, cname, metric, n_pair, med_l, med_r, pdiff, pratio, plog_mean, blow, bhigh, raw_p = entry
                    contrast_results.append(
                        AblationContrastResult(
                            contrast_id=cid,
                            contrast_name=cname,
                            scenario_id=scen,
                            family=family_name,
                            metric=metric,
                            count=n_pair,
                            median_left=med_l,
                            median_right=med_r,
                            paired_median_difference=pdiff,
                            paired_median_ratio=pratio,
                            paired_log_ratio_mean=plog_mean,
                            bootstrap_ci_95_lower=blow,
                            bootstrap_ci_95_upper=bhigh,
                            raw_p_value=raw_p,
                            holm_p_value=holm_p,
                        )
                    )

    # 3. Stratified Variant D Summary (Section 5)
    bimodal_results: list[BimodalVariantDSummary] = []
    variant_d_rows = [r for r in rows if r.variant == "variant_d"]
    vd_groups: dict[tuple[str, str], list[AblationSampleRow]] = defaultdict(list)
    for r in variant_d_rows:
        grp = "zero_frames" if r.old_plan_frames_admitted_after_request_before_commit == 0 else "one_or_more_frames"
        vd_groups[(r.scenario_id, grp)].append(r)

    for (scen, grp), r_list in sorted(vd_groups.items()):
        req_effs = np.asarray([float(r.request_to_effect_ns) for r in r_list], dtype=np.float64)
        trans_gaps = np.asarray([float(r.transition_output_gap_ns) for r in r_list], dtype=np.float64)
        tot_syncs = np.asarray([float(r.total_synchronous_ns) for r in r_list], dtype=np.float64)
        offpaths = np.asarray([float(r.offpath_preparation_ns or 0) for r in r_list], dtype=np.float64)
        def_rets = np.asarray([float(r.deferred_retirement_ns or 0) for r in r_list], dtype=np.float64)
        n = len(r_list)

        def _med_p95(arr: np.ndarray) -> tuple[float, float]:
            if len(arr) == 0:
                return 0.0, 0.0
            return float(np.median(arr)), float(np.percentile(arr, 95.0))

        m_req, p_req = _med_p95(req_effs)
        m_gap, p_gap = _med_p95(trans_gaps)
        m_sync, p_sync = _med_p95(tot_syncs)
        m_prep, p_prep = _med_p95(offpaths)
        m_ret, p_ret = _med_p95(def_rets)

        bimodal_results.append(
            BimodalVariantDSummary(
                scenario_id=scen,
                frame_group=grp,
                count=n,
                request_to_effect_median_ns=m_req,
                request_to_effect_p95_ns=p_req,
                transition_output_gap_median_ns=m_gap,
                transition_output_gap_p95_ns=p_gap,
                total_synchronous_median_ns=m_sync,
                total_synchronous_p95_ns=p_sync,
                offpath_preparation_median_ns=m_prep,
                offpath_preparation_p95_ns=p_prep,
                deferred_retirement_median_ns=m_ret,
                deferred_retirement_p95_ns=p_ret,
            )
        )

    return group_summaries, contrast_results, bimodal_results


def generate_ablation_summary_files(csv_path: str, output_dir: str) -> None:
    """Read existing ablation CSV file and generate summary CSVs without running benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    rows = read_ablation_rows(csv_path)
    group_summaries, contrast_results, bimodal_results = calculate_ablation_summaries(rows)

    # 1. Group Summaries CSV
    group_csv_path = os.path.join(output_dir, "ablation-summary.csv")
    with open(group_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "edit_type", "variant", "variant_name", "metric",
            "count", "mean", "std_dev", "median", "p95", "ci_95_lower", "ci_95_upper",
        ])
        for g in group_summaries:
            writer.writerow([
                g.scenario_id, g.edit_type, g.variant, g.variant_name, g.metric,
                g.count, f"{g.mean:.2f}", f"{g.std_dev:.2f}", f"{g.median:.2f}",
                f"{g.p95:.2f}", f"{g.ci_95_lower:.2f}", f"{g.ci_95_upper:.2f}",
            ])

    # 2. Planned Contrasts CSV (Section 4)
    contrast_csv_path = os.path.join(output_dir, "ablation-contrasts.csv")
    contrast_summary_path = os.path.join(output_dir, "ablation-contrast-summary.csv")

    for path in (contrast_csv_path, contrast_summary_path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ABLATION_CONTRAST_HEADERS)
            for c in contrast_results:
                writer.writerow([
                    c.scenario_id,
                    c.contrast_id,
                    c.family,
                    c.metric,
                    c.count,
                    f"{c.median_left:.2f}",
                    f"{c.median_right:.2f}",
                    f"{c.paired_median_difference:.2f}",
                    f"{c.paired_median_ratio:.4f}",
                    f"{c.paired_log_ratio_mean:.4f}",
                    f"{c.bootstrap_ci_95_lower:.2f}",
                    f"{c.bootstrap_ci_95_upper:.2f}",
                    f"{c.raw_p_value:.4e}",
                    f"{c.holm_p_value:.4e}",
                ])

    # 3. Stratified Variant D CSV (Section 5)
    stratified_csv_path = os.path.join(output_dir, "ablation-variant-d-stratified-summary.csv")
    legacy_stratified_path = os.path.join(output_dir, "ablation-stratified-summary.csv")

    for path in (stratified_csv_path, legacy_stratified_path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ABLATION_STRATIFIED_HEADERS)
            for b in bimodal_results:
                writer.writerow([
                    b.scenario_id,
                    b.frame_group,
                    b.count,
                    f"{b.request_to_effect_median_ns:.2f}",
                    f"{b.request_to_effect_p95_ns:.2f}",
                    f"{b.transition_output_gap_median_ns:.2f}",
                    f"{b.transition_output_gap_p95_ns:.2f}",
                    f"{b.total_synchronous_median_ns:.2f}",
                    f"{b.total_synchronous_p95_ns:.2f}",
                    f"{b.offpath_preparation_median_ns:.2f}",
                    f"{b.offpath_preparation_p95_ns:.2f}",
                    f"{b.deferred_retirement_median_ns:.2f}",
                    f"{b.deferred_retirement_p95_ns:.2f}",
                ])
