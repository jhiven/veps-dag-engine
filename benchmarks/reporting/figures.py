"""Figure generation for paper outputs (Figures 1 to 5) using Matplotlib."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.model import ReconfigurationSampleRow
from benchmarks.reporting.load import BenchmarkArtifactData
from benchmarks.statistics import calculate_paired_stats

__all__ = ["generate_all_figures"]


def generate_all_figures(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, ...]:
    os.makedirs(figures_dir, exist_ok=True)
    generated_paths: list[str] = []

    if data.steady_state_samples:
        f1 = _generate_figure_1(data, figures_dir)
        generated_paths.extend(f1)

        f2 = _generate_figure_2(data, figures_dir)
        generated_paths.extend(f2)

    if data.reconfiguration_samples:
        f3 = _generate_figure_3(data, figures_dir)
        generated_paths.extend(f3)

        f4 = _generate_figure_4(data, figures_dir)
        generated_paths.extend(f4)

        f5 = _generate_figure_5(data, figures_dir)
        generated_paths.extend(f5)

    return tuple(generated_paths)


def _save_fig(fig: matplotlib.figure.Figure, base_path: str) -> tuple[str, str]:
    pdf_path = f"{base_path}.pdf"
    png_path = f"{base_path}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def _generate_figure_1(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, str]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in data.steady_state_samples:
        grouped[(s.topology, s.workload_id, s.implementation)].append(s.normalized_ns_per_frame)

    workloads = ("minimal", "approximately_1_ms", "approximately_5_ms")
    workload_labels = ("Minimal", "0.38 ms", "5 ms")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    topologies = ("linear_5", "branch_merge_9")
    titles = ("Linear Topology (5 nodes)", "Branch-Merge Topology (9 nodes)")

    for idx, top in enumerate(topologies):
        ax = axes[idx]
        static_diffs: list[float] = []
        versioned_diffs: list[float] = []

        for work in workloads:
            hc = grouped.get((top, work, "hard_coded"), [])
            st = grouped.get((top, work, "static_compiled"), [])
            ver = grouped.get((top, work, "versioned_compiled"), [])

            if hc and st:
                p_st = calculate_paired_stats(hc[: len(st)], st[: len(hc)])
                static_diffs.append(p_st.mean_diff)
            else:
                static_diffs.append(0.0)

            if hc and ver:
                p_ver = calculate_paired_stats(hc[: len(ver)], ver[: len(hc)])
                versioned_diffs.append(p_ver.mean_diff)
            else:
                versioned_diffs.append(0.0)

        x = np.arange(len(workloads))
        width = 0.35

        ax.bar(x - width / 2, static_diffs, width, label="Static Compiled", color="#2b5c8f")
        ax.bar(x + width / 2, versioned_diffs, width, label="Versioned Compiled", color="#d95f02")

        ax.set_title(titles[idx], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(workload_labels)
        ax.set_ylabel("Absolute Overhead vs Hard-Coded (ns)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()

    base_path = os.path.join(figures_dir, "figure1_absolute_overhead")
    return _save_fig(fig, base_path)


def _generate_figure_2(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, str]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s in data.steady_state_samples:
        grouped[(s.topology, s.workload_id, s.implementation)].append(s.normalized_ns_per_frame)

    workloads = ("minimal", "approximately_1_ms", "approximately_5_ms")
    workload_labels = ("Minimal", "0.38 ms", "5 ms")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    topologies = ("linear_5", "branch_merge_9")
    titles = ("Linear Topology (5 nodes)", "Branch-Merge Topology (9 nodes)")

    for idx, top in enumerate(topologies):
        ax = axes[idx]
        static_rel: list[float] = []
        versioned_rel: list[float] = []

        for work in workloads:
            hc = grouped.get((top, work, "hard_coded"), [])
            st = grouped.get((top, work, "static_compiled"), [])
            ver = grouped.get((top, work, "versioned_compiled"), [])

            if hc and st:
                p_st = calculate_paired_stats(hc[: len(st)], st[: len(hc)])
                static_rel.append(p_st.relative_diff * 100)
            else:
                static_rel.append(0.0)

            if hc and ver:
                p_ver = calculate_paired_stats(hc[: len(ver)], ver[: len(hc)])
                versioned_rel.append(p_ver.relative_diff * 100)
            else:
                versioned_rel.append(0.0)

        x = np.arange(len(workloads))
        width = 0.35

        ax.bar(x - width / 2, static_rel, width, label="Static Compiled", color="#2b5c8f")
        ax.bar(x + width / 2, versioned_rel, width, label="Versioned Compiled", color="#d95f02")

        ax.set_title(titles[idx], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(workload_labels)
        ax.set_ylabel("Relative Overhead vs Hard-Coded (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()

    base_path = os.path.join(figures_dir, "figure2_relative_overhead")
    return _save_fig(fig, base_path)


def _generate_figure_3(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, str]:
    grouped: dict[str, list[ReconfigurationSampleRow]] = defaultdict(list)
    for s in data.reconfiguration_samples:
        if s.baseline == "prepare_and_commit" and s.graph_size == 10:
            grouped[s.edit_type].append(s)

    edit_types = (
        "insert_stateless_node",
        "remove_stateless_node",
        "rewire_stateless_edge",
        "compatible_edit_preserving_tracker",
    )
    edit_labels = ("Insert Node", "Remove Node", "Rewire Edge", "Preserve Tracker")

    validation_means: list[float] = []
    preparation_means: list[float] = []
    boundary_means: list[float] = []
    commit_means: list[float] = []
    retirement_means: list[float] = []

    for edit in edit_types:
        samples = grouped.get(edit, [])
        if samples:
            val = float(np.mean([s.validation_ns or 0 for s in samples]))
            prep = float(np.mean([s.preparation_ns or 0 for s in samples]))
            bound = float(np.mean([s.boundary_wait_ns or 0 for s in samples]))
            cmt = float(np.mean([s.commit_ns or 0 for s in samples]))
            ret = float(np.mean([s.retirement_duration_ns or 0 for s in samples]))
        else:
            val = prep = bound = cmt = ret = 0.0

        validation_means.append(val / 1e3)
        preparation_means.append(prep / 1e3)
        boundary_means.append(bound / 1e3)
        commit_means.append(cmt / 1e3)
        retirement_means.append(ret / 1e3)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(edit_types))
    width = 0.35

    # Critical path stack (Validation, Preparation, Boundary Wait, Commit)
    ax.bar(x - width/2, validation_means, width, label="Validation", color="#7570b3")
    ax.bar(x - width/2, preparation_means, width, bottom=validation_means, label="Preparation", color="#1b9e77")
    bottom_bound = np.add(validation_means, preparation_means)
    ax.bar(x - width/2, boundary_means, width, bottom=bottom_bound, label="Boundary Wait", color="#e6ab02")
    bottom_cmt = np.add(bottom_bound, boundary_means)
    ax.bar(x - width/2, commit_means, width, bottom=bottom_cmt, label="Commit", color="#d95f02")

    # Off-path async retirement (separate bar)
    ax.bar(x + width/2, retirement_means, width, label="Retirement (Off-Path Async)", color="#66a61e", hatch="//")

    ax.set_title("Reconfiguration Latency Breakdown (prepare_and_commit)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(edit_labels, rotation=15)
    ax.set_ylabel("Latency (µs)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    fig.text(
        0.5, 0.01,
        "Note: Retirement duration is reported separately because it executes asynchronously outside the request-to-effect critical path.",
        ha="center", fontsize=8, style="italic"
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    base_path = os.path.join(figures_dir, "figure3_reconfiguration_breakdown")
    return _save_fig(fig, base_path)


def _generate_figure_4(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, str]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in data.reconfiguration_samples:
        if s.graph_size == 10 and s.maximum_output_gap_ns is not None:
            grouped[(s.edit_type, s.baseline)].append(float(s.maximum_output_gap_ns) / 1e6)

    edit_types = (
        "insert_stateless_node",
        "remove_stateless_node",
        "rewire_stateless_edge",
        "compatible_edit_preserving_tracker",
    )
    edit_labels = ("Insert Node", "Remove Node", "Rewire Edge", "Preserve Tracker")
    baselines = ("stop_rebuild_restart", "pause_compile_resume", "prepare_and_commit")
    baseline_labels = ("Stop-Rebuild-Restart", "Pause-Compile-Resume", "Prepare-and-Commit")
    colors = ("#e41a1c", "#377eb8", "#4daf4a")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(edit_types))
    width = 0.25

    for b_idx, base in enumerate(baselines):
        means: list[float] = []
        for edit in edit_types:
            vals = grouped.get((edit, base), [])
            means.append(float(np.mean(vals)) if vals else 0.0)

        offset = (b_idx - 1) * width
        ax.bar(x + offset, means, width, label=baseline_labels[b_idx], color=colors[b_idx])

    ax.set_title("Maximum Output Gap during Reconfiguration", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(edit_labels)
    ax.set_ylabel("Maximum Output Gap (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    base_path = os.path.join(figures_dir, "figure4_maximum_output_gap")
    return _save_fig(fig, base_path)


def _generate_figure_5(data: BenchmarkArtifactData, figures_dir: str) -> tuple[str, str]:
    prep_by_size: dict[int, list[float]] = defaultdict(list)
    effect_by_size: dict[int, list[float]] = defaultdict(list)

    for s in data.reconfiguration_samples:
        if s.baseline == "prepare_and_commit" and s.scenario_id.startswith("graph_size_sensitivity"):
            if s.preparation_ns is not None:
                prep_by_size[s.graph_size].append(float(s.preparation_ns) / 1e3)
            if s.request_to_effect_ns is not None:
                effect_by_size[s.graph_size].append(float(s.request_to_effect_ns) / 1e3)

    sizes = (5, 25, 100)
    prep_means: list[float] = [float(np.mean(prep_by_size.get(sz, [0.0]))) for sz in sizes]
    effect_means: list[float] = [float(np.mean(effect_by_size.get(sz, [0.0]))) for sz in sizes]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(sizes, prep_means, marker="o", linewidth=2, label="Candidate Preparation", color="#1b9e77")
    ax.plot(sizes, effect_means, marker="s", linewidth=2, label="Request-to-Effect", color="#d95f02")

    ax.set_title("Reconfiguration Latency vs Graph Size (prepare_and_commit)", fontsize=11)
    ax.set_xlabel("Graph Size (Number of Nodes)")
    ax.set_ylabel("Latency (µs)")
    ax.set_xticks(sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    base_path = os.path.join(figures_dir, "figure5_graph_size_sensitivity")
    return _save_fig(fig, base_path)
