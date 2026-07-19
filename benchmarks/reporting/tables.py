"""Table generation for paper outputs (Table 1, Table 2, Table 3)."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

from benchmarks.model import ReconfigurationSampleRow
from benchmarks.reporting.load import BenchmarkArtifactData
from benchmarks.statistics import (
    calculate_paired_stats,
    calculate_summary_stats,
)

__all__ = ["generate_all_tables"]


def generate_all_tables(data: BenchmarkArtifactData, tables_dir: str) -> tuple[str, ...]:
    os.makedirs(tables_dir, exist_ok=True)
    generated_paths: list[str] = []

    p1_csv = os.path.join(tables_dir, "table1_steady_state.csv")
    p1_md = os.path.join(tables_dir, "table1_steady_state.md")
    _generate_table_1(data, p1_csv, p1_md)
    generated_paths.extend([p1_csv, p1_md])

    p2_csv = os.path.join(tables_dir, "table2_reconfiguration.csv")
    p2_md = os.path.join(tables_dir, "table2_reconfiguration.md")
    _generate_table_2(data, p2_csv, p2_md)
    generated_paths.extend([p2_csv, p2_md])

    p3_csv = os.path.join(tables_dir, "table3_conformance.csv")
    p3_md = os.path.join(tables_dir, "table3_conformance.md")
    _generate_table_3(data, p3_csv, p3_md)
    generated_paths.extend([p3_csv, p3_md])

    return tuple(generated_paths)


def _generate_table_1(data: BenchmarkArtifactData, csv_path: str, md_path: str) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in data.steady_state_samples:
        grouped[(sample.topology, sample.workload_id, sample.implementation)].append(
            sample.normalized_ns_per_frame
        )

    headers = [
        "Topology",
        "Workload",
        "Implementation",
        "Median Latency (ns)",
        "p95 (ns)",
        "p99 (ns)",
        "Abs Diff (ns)",
        "Rel Diff (%)",
        "Paired 95% CI (ns)",
    ]

    rows: list[list[str]] = []

    topologies = ("linear_5", "branch_merge_9")
    workloads = ("minimal", "approximately_1_ms", "approximately_5_ms")
    impls = ("hard_coded", "static_compiled", "versioned_compiled")

    for top in topologies:
        for work in workloads:
            hc_vals = grouped.get((top, work, "hard_coded"), [])
            hc_stats = calculate_summary_stats(hc_vals) if hc_vals else None

            for impl in impls:
                vals = grouped.get((top, work, impl), [])
                if not vals:
                    continue
                stats = calculate_summary_stats(vals)

                if impl == "hard_coded" or hc_stats is None or not hc_vals:
                    abs_diff_str = "0.0"
                    rel_diff_str = "0.0%"
                    ci_str = "[0.0, 0.0]"
                else:
                    paired = calculate_paired_stats(hc_vals[: len(vals)], vals[: len(hc_vals)])
                    abs_diff_str = f"{paired.mean_diff:.1f}"
                    rel_diff_str = f"{paired.relative_diff * 100:.2f}%"
                    ci_str = f"[{paired.ci_95_lower:.1f}, {paired.ci_95_upper:.1f}]"

                rows.append([
                    top,
                    work,
                    impl,
                    f"{stats.median:.1f}",
                    f"{stats.p95:.1f}",
                    f"{stats.p99:.1f}",
                    abs_diff_str,
                    rel_diff_str,
                    ci_str,
                ])

    _write_csv_and_md(csv_path, md_path, "Table 1 — Steady-State Execution Overhead", headers, rows)


def _generate_table_2(data: BenchmarkArtifactData, csv_path: str, md_path: str) -> None:
    grouped: dict[tuple[str, str], list[ReconfigurationSampleRow]] = defaultdict(list)
    for sample in data.reconfiguration_samples:
        if sample.graph_size == 10:
            grouped[(sample.edit_type, sample.baseline)].append(sample)

    headers = [
        "Edit Type",
        "Baseline",
        "Req to Effect Median (ns)",
        "Req to Effect p95 (ns)",
        "Max Output Gap Median (ns)",
        "Max Output Gap p95 (ns)",
        "Commit Median (ns)",
        "Processor Reuse Count",
        "Dropped Frames",
        "Duplicated Frames",
    ]

    rows: list[list[str]] = []

    edit_types = (
        "insert_stateless_node",
        "remove_stateless_node",
        "rewire_stateless_edge",
        "compatible_edit_preserving_tracker",
    )
    baselines = ("stop_rebuild_restart", "pause_compile_resume", "prepare_and_commit")

    for edit in edit_types:
        for base in baselines:
            samples = grouped.get((edit, base), [])
            if not samples:
                continue

            eff_vals = [float(s.request_to_effect_ns) for s in samples if s.request_to_effect_ns is not None]
            gap_vals = [float(s.maximum_output_gap_ns) for s in samples if s.maximum_output_gap_ns is not None]
            cmt_vals = [float(s.commit_ns) for s in samples if s.commit_ns is not None]

            eff_stats = calculate_summary_stats(eff_vals) if eff_vals else None
            gap_stats = calculate_summary_stats(gap_vals) if gap_vals else None
            cmt_stats = calculate_summary_stats(cmt_vals) if cmt_vals else None

            reuse_cnt = samples[0].reused_processor_count if samples else 0
            drop_cnt = sum(s.frames_dropped for s in samples)
            dup_cnt = sum(s.frames_duplicated for s in samples)

            rows.append([
                edit,
                base,
                f"{eff_stats.median:.1f}" if eff_stats else "N/A",
                f"{eff_stats.p95:.1f}" if eff_stats else "N/A",
                f"{gap_stats.median:.1f}" if gap_stats else "N/A",
                f"{gap_stats.p95:.1f}" if gap_stats else "N/A",
                f"{cmt_stats.median:.1f}" if cmt_stats else "N/A",
                str(reuse_cnt),
                str(drop_cnt),
                str(dup_cnt),
            ])

    _write_csv_and_md(csv_path, md_path, "Table 2 — Structural Reconfiguration Latency and Continuity", headers, rows)


def _generate_table_3(data: BenchmarkArtifactData, csv_path: str, md_path: str) -> None:
    headers = [
        "Campaign",
        "Frames",
        "Reconfig Requested",
        "Reconfig Committed",
        "Reconfig Rejected",
        "Mixed Plan Frames",
        "Continuity Failures",
        "Active Plan Changes After Failure",
        "Resource Leaks",
        "Terminal Result",
    ]

    rows: list[list[str]] = []

    for c in data.conformance_results:
        leaks = c.candidate_resource_leaks + c.processor_instance_leaks
        rows.append([
            c.campaign_id,
            str(c.frames_completed),
            str(c.reconfigurations_requested),
            str(c.reconfigurations_committed),
            str(c.reconfigurations_rejected),
            str(c.mixed_plan_frames),
            str(c.state_continuity_failures),
            str(c.active_plan_changed_after_failed_candidate),
            str(leaks),
            c.terminal_status,
        ])

    _write_csv_and_md(csv_path, md_path, "Table 3 — Consistency, Failure-Atomicity, and Stateful Conformance", headers, rows)


def _write_csv_and_md(
    csv_path: str, md_path: str, title: str, headers: list[str], rows: list[list[str]]
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(f"# {title}\n\n")
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            file.write("| " + " | ".join(row) + " |\n")
