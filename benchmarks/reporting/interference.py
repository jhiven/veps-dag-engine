"""Interference summary and contrast generation for E6 active-path interference campaign (Sections 6, 7, 8, 9)."""

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
    INTERFERENCE_AGGREGATE_CONTRAST_HEADERS,
    INTERFERENCE_CONTRAST_HEADERS,
    InterferenceSampleRow,
)
from benchmarks.storage import read_interference_rows

__all__ = [
    "InterferenceAggregateContrastRow",
    "InterferenceCategorySummary",
    "InterferenceContrastRow",
    "calculate_interference_contrasts",
    "calculate_interference_summaries",
    "generate_interference_summary_file",
]


@dataclass(frozen=True, slots=True)
class InterferenceCategorySummary:
    preparation_category: str
    target_window_ns: int
    metric: str
    total_repetitions: int
    valid_repetitions: int
    invalid_repetitions: int
    mean: float
    std_dev: float
    median: float
    p95: float
    ci_95_lower: float
    ci_95_upper: float


@dataclass(frozen=True, slots=True)
class InterferenceContrastRow:
    repetition: int
    target_window_ns: int
    treatment_category: str
    control_category: str
    treatment_execution_order_position: int
    control_execution_order_position: int
    treatment_during_median_ns: float | None
    control_during_median_ns: float | None
    paired_median_difference_ns: float | None
    paired_median_ratio: float | None
    treatment_during_p95_ns: float | None
    control_during_p95_ns: float | None
    paired_p95_difference_ns: float | None
    paired_p95_ratio: float | None
    treatment_admission_throughput_fps: float
    control_admission_throughput_fps: float
    paired_admission_throughput_ratio: float
    treatment_completion_throughput_fps: float
    control_completion_throughput_fps: float
    paired_completion_throughput_ratio: float
    treatment_dropped_frames: int
    control_dropped_frames: int
    treatment_queue_occupancy_p95: float
    control_queue_occupancy_p95: float
    within_run_admission_shift_flag: bool
    matched_control_confounded: bool
    matched_control_confounded_reason: str | None


@dataclass(frozen=True, slots=True)
class InterferenceAggregateContrastRow:
    treatment_category: str
    target_window_ns: int
    filter_scope: str  # "all_pairs" vs "non_confounded_pairs"
    total_pair_count: int
    confounded_pair_count: int
    non_confounded_pair_count: int
    included_pair_count: int
    median_paired_p95_ratio: float | None
    p95_paired_p95_ratio: float | None
    bootstrap_ci_95_lower_p95_ratio: float | None
    bootstrap_ci_95_upper_p95_ratio: float | None
    median_paired_admission_throughput_ratio: float
    bootstrap_ci_95_lower_admission_ratio: float
    bootstrap_ci_95_upper_admission_ratio: float
    median_paired_completion_throughput_ratio: float
    bootstrap_ci_95_lower_completion_ratio: float
    bootstrap_ci_95_upper_completion_ratio: float


def calculate_interference_contrasts(
    rows: Sequence[InterferenceSampleRow],
) -> tuple[list[InterferenceContrastRow], list[InterferenceAggregateContrastRow]]:
    """Calculate matched control contrasts pairing treatment categories with no_candidate_preparation."""
    row_map: dict[tuple[str, int, int], InterferenceSampleRow] = {}
    for r in rows:
        row_map[(r.preparation_category, r.target_window_ns, r.repetition)] = r

    treatment_categories = ("sleep_preparation", "cpu_bound_preparation")
    target_windows = sorted({r.target_window_ns for r in rows})
    repetitions = sorted({r.repetition for r in rows})

    contrast_rows: list[InterferenceContrastRow] = []

    for target_ns in target_windows:
        for cat in treatment_categories:
            for rep in repetitions:
                treat_r = row_map.get((cat, target_ns, rep))
                ctrl_r = row_map.get(("no_candidate_preparation", target_ns, rep))
                if treat_r is None or ctrl_r is None:
                    continue

                t_p95 = treat_r.frame_latency_during_p95_ns if treat_r.valid_for_tail_latency else None
                c_p95 = ctrl_r.frame_latency_during_p95_ns if ctrl_r.valid_for_tail_latency else None

                p95_diff = (t_p95 - c_p95) if (t_p95 is not None and c_p95 is not None) else None
                p95_ratio = (t_p95 / c_p95) if (t_p95 is not None and c_p95 is not None and c_p95 > 0) else None

                t_med = treat_r.frame_latency_during_median_ns if treat_r.valid_for_tail_latency else None
                c_med = ctrl_r.frame_latency_during_median_ns if ctrl_r.valid_for_tail_latency else None

                med_diff = (t_med - c_med) if (t_med is not None and c_med is not None) else None
                med_ratio = (t_med / c_med) if (t_med is not None and c_med is not None and c_med > 0) else None

                adm_ratio = (
                    treat_r.admission_throughput_during_fps / ctrl_r.admission_throughput_during_fps
                    if ctrl_r.admission_throughput_during_fps > 0
                    else 1.0
                )
                comp_ratio = (
                    treat_r.completion_throughput_during_fps / ctrl_r.completion_throughput_during_fps
                    if ctrl_r.completion_throughput_during_fps > 0
                    else 1.0
                )

                # Section 7: Matched-control confounding flags
                matched_confounded = False
                reasons: list[str] = []

                if adm_ratio < 0.95 or adm_ratio > 1.05:
                    matched_confounded = True
                    reasons.append(f"Admission throughput ratio vs control ({adm_ratio:.3f}) outside [0.95, 1.05]")

                if treat_r.dropped_frames > 0 and ctrl_r.dropped_frames == 0:
                    matched_confounded = True
                    reasons.append(f"Treatment dropped frames ({treat_r.dropped_frames}) while control dropped 0")

                if treat_r.queue_occupancy_during_p95 > ctrl_r.queue_occupancy_during_p95 + 0.10:
                    matched_confounded = True
                    reasons.append(
                        f"Treatment queue p95 ({treat_r.queue_occupancy_during_p95:.2f}) materially exceeds control ({ctrl_r.queue_occupancy_during_p95:.2f})"
                    )

                if treat_r.admitted_frames_during < 100:
                    matched_confounded = True
                    reasons.append(f"Insufficient completed during-window frames ({treat_r.admitted_frames_during} < 100)")

                within_run_flag = treat_r.within_run_admission_shift_flag or treat_r.latency_interpretation_confounded

                contrast_rows.append(
                    InterferenceContrastRow(
                        repetition=rep,
                        target_window_ns=target_ns,
                        treatment_category=cat,
                        control_category="no_candidate_preparation",
                        treatment_execution_order_position=treat_r.execution_order_position,
                        control_execution_order_position=ctrl_r.execution_order_position,
                        treatment_during_median_ns=t_med,
                        control_during_median_ns=c_med,
                        paired_median_difference_ns=med_diff,
                        paired_median_ratio=med_ratio,
                        treatment_during_p95_ns=t_p95,
                        control_during_p95_ns=c_p95,
                        paired_p95_difference_ns=p95_diff,
                        paired_p95_ratio=p95_ratio,
                        treatment_admission_throughput_fps=treat_r.admission_throughput_during_fps,
                        control_admission_throughput_fps=ctrl_r.admission_throughput_during_fps,
                        paired_admission_throughput_ratio=adm_ratio,
                        treatment_completion_throughput_fps=treat_r.completion_throughput_during_fps,
                        control_completion_throughput_fps=ctrl_r.completion_throughput_during_fps,
                        paired_completion_throughput_ratio=comp_ratio,
                        treatment_dropped_frames=treat_r.dropped_frames,
                        control_dropped_frames=ctrl_r.dropped_frames,
                        treatment_queue_occupancy_p95=treat_r.queue_occupancy_during_p95,
                        control_queue_occupancy_p95=ctrl_r.queue_occupancy_during_p95,
                        within_run_admission_shift_flag=within_run_flag,
                        matched_control_confounded=matched_confounded,
                        matched_control_confounded_reason="; ".join(reasons) if reasons else None,
                    )
                )

    # Aggregate summaries: Both all_pairs and non_confounded_pairs
    by_cat_ns: dict[tuple[str, int], list[InterferenceContrastRow]] = defaultdict(list)
    for c in contrast_rows:
        by_cat_ns[(c.treatment_category, c.target_window_ns)].append(c)

    agg_rows: list[InterferenceAggregateContrastRow] = []

    for (cat, target_ns), c_list in sorted(by_cat_ns.items()):
        total_count = len(c_list)
        conf_count = sum(1 for c in c_list if c.matched_control_confounded)
        non_conf_count = total_count - conf_count

        for scope in ("all_pairs", "non_confounded_pairs"):
            if scope == "non_confounded_pairs":
                sub_list = [c for c in c_list if not c.matched_control_confounded]
            else:
                sub_list = c_list

            inc_count = len(sub_list)
            if inc_count == 0:
                continue

            p95_ratios = [c.paired_p95_ratio for c in sub_list if c.paired_p95_ratio is not None]
            adm_ratios = [c.paired_admission_throughput_ratio for c in sub_list]
            comp_ratios = [c.paired_completion_throughput_ratio for c in sub_list]

            if p95_ratios:
                med_p95_r = float(np.median(p95_ratios))
                p95_p95_r = float(np.percentile(p95_ratios, 95.0))
                rng = np.random.default_rng(42)
                boot = [
                    float(np.median(rng.choice(p95_ratios, size=len(p95_ratios), replace=True)))
                    for _ in range(1000)
                ]
                ci_low_p95, ci_high_p95 = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            else:
                med_p95_r = p95_p95_r = ci_low_p95 = ci_high_p95 = None

            med_adm_r = float(np.median(adm_ratios))
            rng = np.random.default_rng(42)
            boot_adm = [
                float(np.median(rng.choice(adm_ratios, size=len(adm_ratios), replace=True)))
                for _ in range(1000)
            ]
            ci_low_adm, ci_high_adm = float(np.percentile(boot_adm, 2.5)), float(np.percentile(boot_adm, 97.5))

            med_comp_r = float(np.median(comp_ratios))
            boot_comp = [
                float(np.median(rng.choice(comp_ratios, size=len(comp_ratios), replace=True)))
                for _ in range(1000)
            ]
            ci_low_comp, ci_high_comp = float(np.percentile(boot_comp, 2.5)), float(np.percentile(boot_comp, 97.5))

            agg_rows.append(
                InterferenceAggregateContrastRow(
                    treatment_category=cat,
                    target_window_ns=target_ns,
                    filter_scope=scope,
                    total_pair_count=total_count,
                    confounded_pair_count=conf_count,
                    non_confounded_pair_count=non_conf_count,
                    included_pair_count=inc_count,
                    median_paired_p95_ratio=med_p95_r,
                    p95_paired_p95_ratio=p95_p95_r,
                    bootstrap_ci_95_lower_p95_ratio=ci_low_p95,
                    bootstrap_ci_95_upper_p95_ratio=ci_high_p95,
                    median_paired_admission_throughput_ratio=med_adm_r,
                    bootstrap_ci_95_lower_admission_ratio=ci_low_adm,
                    bootstrap_ci_95_upper_admission_ratio=ci_high_adm,
                    median_paired_completion_throughput_ratio=med_comp_r,
                    bootstrap_ci_95_lower_completion_ratio=ci_low_comp,
                    bootstrap_ci_95_upper_completion_ratio=ci_high_comp,
                )
            )

    return contrast_rows, agg_rows


def calculate_interference_summaries(
    rows: Sequence[InterferenceSampleRow],
) -> list[InterferenceCategorySummary]:
    groups: dict[tuple[str, int], list[InterferenceSampleRow]] = defaultdict(list)
    for r in rows:
        groups[(r.preparation_category, r.target_window_ns)].append(r)

    results: list[InterferenceCategorySummary] = []

    metrics_list = [
        "frame_latency_before_median_ns",
        "frame_latency_before_p95_ns",
        "frame_latency_during_median_ns",
        "frame_latency_during_p95_ns",
        "p95_degradation_vs_before_pct",
        "admission_throughput_before_fps",
        "admission_throughput_during_fps",
        "completion_throughput_before_fps",
        "completion_throughput_during_fps",
        "old_plan_frames_completed",
        "queue_occupancy_during_median",
        "dropped_frames",
        "actual_preparation_duration_ns",
    ]

    for (cat, target_ns), r_list in sorted(groups.items()):
        total_reps = len(r_list)
        valid_reps = sum(1 for r in r_list if r.valid_for_tail_latency and r.duration_valid)
        invalid_reps = total_reps - valid_reps

        for metric in metrics_list:
            vals: list[float] = []
            for r in r_list:
                if metric in ("frame_latency_during_p95_ns", "p95_degradation_vs_before_pct"):
                    if not (r.valid_for_tail_latency and r.duration_valid):
                        continue
                v = getattr(r, metric)
                if v is not None:
                    vals.append(float(v))

            arr = np.asarray(vals, dtype=np.float64)
            n = len(arr)
            if n == 0:
                mean_v = std_v = med_v = p95_v = ci_low = ci_high = 0.0
            else:
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

            results.append(
                InterferenceCategorySummary(
                    preparation_category=cat,
                    target_window_ns=target_ns,
                    metric=metric,
                    total_repetitions=total_reps,
                    valid_repetitions=valid_reps,
                    invalid_repetitions=invalid_reps,
                    mean=mean_v,
                    std_dev=std_v,
                    median=med_v,
                    p95=p95_v,
                    ci_95_lower=ci_low,
                    ci_95_upper=ci_high,
                )
            )

    return results


def generate_interference_summary_file(csv_path: str, output_csv_path: str) -> None:
    """Read existing interference CSV file and generate summary and contrast CSVs without running benchmark."""
    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    os.makedirs(output_dir, exist_ok=True)
    rows = read_interference_rows(csv_path)

    # 1. Summary CSV
    summaries = calculate_interference_summaries(rows)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "preparation_category", "target_window_ns", "metric",
            "total_repetitions", "valid_repetitions", "invalid_repetitions",
            "mean", "std_dev", "median", "p95", "ci_95_lower", "ci_95_upper",
        ])
        for s in summaries:
            writer.writerow([
                s.preparation_category, s.target_window_ns, s.metric,
                s.total_repetitions, s.valid_repetitions, s.invalid_repetitions,
                f"{s.mean:.2f}", f"{s.std_dev:.2f}", f"{s.median:.2f}",
                f"{s.p95:.2f}", f"{s.ci_95_lower:.2f}", f"{s.ci_95_upper:.2f}",
            ])

    # 2. Matched Control Contrasts CSVs (Section 6 & 7)
    contrast_rows, agg_rows = calculate_interference_contrasts(rows)

    contrasts_csv_path = os.path.join(output_dir, "interference-contrasts.csv")
    with open(contrasts_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(INTERFERENCE_CONTRAST_HEADERS)
        for c in contrast_rows:
            writer.writerow([
                c.repetition,
                c.target_window_ns,
                c.treatment_category,
                c.control_category,
                c.treatment_execution_order_position,
                c.control_execution_order_position,
                f"{c.treatment_during_median_ns:.2f}" if c.treatment_during_median_ns is not None else "",
                f"{c.control_during_median_ns:.2f}" if c.control_during_median_ns is not None else "",
                f"{c.paired_median_difference_ns:.2f}" if c.paired_median_difference_ns is not None else "",
                f"{c.paired_median_ratio:.4f}" if c.paired_median_ratio is not None else "",
                f"{c.treatment_during_p95_ns:.2f}" if c.treatment_during_p95_ns is not None else "",
                f"{c.control_during_p95_ns:.2f}" if c.control_during_p95_ns is not None else "",
                f"{c.paired_p95_difference_ns:.2f}" if c.paired_p95_difference_ns is not None else "",
                f"{c.paired_p95_ratio:.4f}" if c.paired_p95_ratio is not None else "",
                f"{c.treatment_admission_throughput_fps:.2f}",
                f"{c.control_admission_throughput_fps:.2f}",
                f"{c.paired_admission_throughput_ratio:.4f}",
                f"{c.treatment_completion_throughput_fps:.2f}",
                f"{c.control_completion_throughput_fps:.2f}",
                f"{c.paired_completion_throughput_ratio:.4f}",
                c.treatment_dropped_frames,
                c.control_dropped_frames,
                f"{c.treatment_queue_occupancy_p95:.2f}",
                f"{c.control_queue_occupancy_p95:.2f}",
                str(c.within_run_admission_shift_flag),
                str(c.matched_control_confounded),
                c.matched_control_confounded_reason or "",
            ])

    contrast_summary_path = os.path.join(output_dir, "interference-contrast-summary.csv")
    legacy_agg_path = os.path.join(output_dir, "interference-aggregate-summary.csv")

    for path in (contrast_summary_path, legacy_agg_path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(INTERFERENCE_AGGREGATE_CONTRAST_HEADERS)
            for a in agg_rows:
                writer.writerow([
                    a.treatment_category,
                    a.target_window_ns,
                    a.filter_scope,
                    a.total_pair_count,
                    a.confounded_pair_count,
                    a.non_confounded_pair_count,
                    a.included_pair_count,
                    f"{a.median_paired_p95_ratio:.4f}" if a.median_paired_p95_ratio is not None else "",
                    f"{a.p95_paired_p95_ratio:.4f}" if a.p95_paired_p95_ratio is not None else "",
                    f"{a.bootstrap_ci_95_lower_p95_ratio:.4f}" if a.bootstrap_ci_95_lower_p95_ratio is not None else "",
                    f"{a.bootstrap_ci_95_upper_p95_ratio:.4f}" if a.bootstrap_ci_95_upper_p95_ratio is not None else "",
                    f"{a.median_paired_admission_throughput_ratio:.4f}",
                    f"{a.bootstrap_ci_95_lower_admission_ratio:.4f}",
                    f"{a.bootstrap_ci_95_upper_admission_ratio:.4f}",
                    f"{a.median_paired_completion_throughput_ratio:.4f}",
                    f"{a.bootstrap_ci_95_lower_completion_ratio:.4f}",
                    f"{a.bootstrap_ci_95_upper_completion_ratio:.4f}",
                ])
