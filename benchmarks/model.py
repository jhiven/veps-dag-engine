"""Strictly typed benchmark row data models and constants."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SteadyStateSampleRow",
    "ReconfigurationSampleRow",
    "InterferenceSampleRow",
    "ConformanceResultRow",
    "STEADY_STATE_HEADERS",
    "RECONFIGURATION_HEADERS",
    "INTERFERENCE_HEADERS",
    "CONFORMANCE_HEADERS",
    "CriticalPathDecomposition",
    "compute_critical_path_decomposition",
    "validate_reconfiguration_sample",
]


@dataclass(frozen=True, slots=True)
class SteadyStateSampleRow:
    run_id: str
    scenario_id: str
    topology: str
    workload_id: str
    implementation: str
    repetition: int
    execution_order_position: int
    operations: int
    elapsed_ns: int
    normalized_ns_per_frame: float
    frames_completed: int
    plan_version: int


@dataclass(frozen=True, slots=True)
class ReconfigurationSampleRow:
    run_id: str
    scenario_id: str
    baseline: str
    edit_type: str
    graph_size: int
    repetition: int
    old_plan_version: int
    new_plan_version: int
    terminal_status: str
    validation_ns: int | None
    preparation_ns: int | None
    request_to_ready_ns: int | None
    boundary_wait_ns: int | None
    commit_ns: int | None
    request_to_effect_ns: int | None
    retirement_queue_delay_ns: int | None
    retirement_duration_ns: int | None
    commit_to_retirement_complete_ns: int | None
    maximum_output_gap_ns: int | None
    transition_output_gap_ns: int | None
    frames_completed_during_request: int
    old_plan_frames_admitted_after_request_before_commit: int
    frames_dropped: int
    frames_duplicated: int
    reused_processor_count: int
    staged_processor_count: int
    retired_processor_count: int
    state_transition_policy: str
    admission_stop_ns: int | None = None
    executor_teardown_ns: int | None = None
    executor_reconstruction_ns: int | None = None
    publication_ns: int | None = None
    executor_restart_ns: int | None = None
    first_admission_wait_ns: int | None = None
    first_completion_wait_ns: int | None = None
    retirement_ns: int | None = None
    total_synchronous_ns: int | None = None
    phases_may_overlap: bool = False
    sum_of_instrumented_phase_durations_ns: int | None = None
    instrumented_phase_sum_ns: int | None = None
    critical_path_instrumented_ns: int | None = None
    unattributed_critical_path_ns: int | None = None
    unattributed_request_time_ns: int | None = None
    instrumented_duration_overlap_ns: int | None = None
    instrumented_duration_outside_effect_window_ns: int | None = None


@dataclass(frozen=True, slots=True)
class CriticalPathDecomposition:
    sum_of_instrumented_phase_durations_ns: int
    critical_path_instrumented_ns: int
    unattributed_critical_path_ns: int
    instrumented_duration_overlap_ns: int
    instrumented_duration_outside_effect_window_ns: int


def compute_critical_path_decomposition(
    t_request: int,
    t_effect: int,
    phase_intervals: list[tuple[int, int]],
) -> CriticalPathDecomposition:
    """Compute exact critical-path interval union length and phase decomposition.

    Parameters
    ----------
    t_request:
        Timestamp when reconfiguration request was submitted (start of effect observation window).
    t_effect:
        Timestamp when first new-plan frame completed execution (end of effect observation window).
    phase_intervals:
        List of (start_ns, end_ns) timestamp pairs for all instrumented phases.

    Returns
    -------
    CriticalPathDecomposition
        Decomposition containing sum of phase durations, interval union length (critical_path_instrumented_ns),
        unattributed critical path, overlap within effect window, and phase duration outside effect window.
    """
    request_to_effect_ns = max(0, t_effect - t_request)

    unclipped_durations: list[int] = []
    clipped_intervals: list[tuple[int, int]] = []
    clipped_durations: list[int] = []
    outside_durations: list[int] = []

    for s, e in phase_intervals:
        if s > e:
            raise ValueError(f"Phase start {s} must be <= phase end {e}")
        u_dur = e - s
        unclipped_durations.append(u_dur)

        c_start = max(s, t_request)
        c_end = min(e, t_effect)
        if c_start < c_end:
            c_dur = c_end - c_start
            clipped_intervals.append((c_start, c_end))
            clipped_durations.append(c_dur)
            outside_durations.append(u_dur - c_dur)
        else:
            clipped_durations.append(0)
            outside_durations.append(u_dur)

    sum_phase_durations = sum(unclipped_durations)
    sum_intersected_durations = sum(clipped_durations)
    outside_effect_window = sum(outside_durations)

    if not clipped_intervals:
        critical_path_union = 0
    else:
        clipped_intervals.sort(key=lambda x: x[0])
        merged: list[tuple[int, int]] = []
        for start, end in clipped_intervals:
            if not merged:
                merged.append((start, end))
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
        critical_path_union = sum(end - start for start, end in merged)

    critical_path_union = max(0, min(critical_path_union, request_to_effect_ns))
    unattributed_cp = request_to_effect_ns - critical_path_union
    duration_overlap = max(0, sum_intersected_durations - critical_path_union)

    return CriticalPathDecomposition(
        sum_of_instrumented_phase_durations_ns=sum_phase_durations,
        critical_path_instrumented_ns=critical_path_union,
        unattributed_critical_path_ns=unattributed_cp,
        instrumented_duration_overlap_ns=duration_overlap,
        instrumented_duration_outside_effect_window_ns=outside_effect_window,
    )


def validate_reconfiguration_sample(row: ReconfigurationSampleRow, tolerance_ns: int = 10_000_000) -> None:
    durations = (
        row.validation_ns,
        row.preparation_ns,
        row.request_to_ready_ns,
        row.boundary_wait_ns,
        row.commit_ns,
        row.request_to_effect_ns,
        row.retirement_queue_delay_ns,
        row.retirement_duration_ns,
        row.commit_to_retirement_complete_ns,
        row.maximum_output_gap_ns,
        row.transition_output_gap_ns,
        row.admission_stop_ns,
        row.executor_teardown_ns,
        row.executor_reconstruction_ns,
        row.publication_ns,
        row.executor_restart_ns,
        row.first_admission_wait_ns,
        row.first_completion_wait_ns,
        row.retirement_ns,
        row.total_synchronous_ns,
        row.sum_of_instrumented_phase_durations_ns,
        row.instrumented_phase_sum_ns,
        row.critical_path_instrumented_ns,
        row.unattributed_critical_path_ns,
        row.unattributed_request_time_ns,
        row.instrumented_duration_overlap_ns,
        row.instrumented_duration_outside_effect_window_ns,
    )
    for d in durations:
        if d is not None:
            if d < 0:
                raise ValueError(f"Duration must be non-negative, got {d}")

    if row.baseline == "prepare_and_commit":
        if row.admission_stop_ns is not None and row.admission_stop_ns != 0:
            raise ValueError("prepare_and_commit admission_stop_ns must be 0")
        if row.executor_teardown_ns is not None and row.executor_teardown_ns != 0:
            raise ValueError("prepare_and_commit executor_teardown_ns must be 0")
        if row.executor_reconstruction_ns is not None and row.executor_reconstruction_ns != 0:
            raise ValueError("prepare_and_commit executor_reconstruction_ns must be 0")
        if row.executor_restart_ns is not None and row.executor_restart_ns != 0:
            raise ValueError("prepare_and_commit executor_restart_ns must be 0")
        if row.publication_ns is not None and row.commit_ns is not None and row.publication_ns != row.commit_ns:
            raise ValueError("prepare_and_commit publication_ns must equal commit_ns")
        if row.total_synchronous_ns is not None and row.publication_ns is not None and row.total_synchronous_ns != row.publication_ns:
            raise ValueError("prepare_and_commit total_synchronous_ns must equal publication_ns")
        if not row.phases_may_overlap:
            raise ValueError("prepare_and_commit phases_may_overlap must be True")

    elif row.baseline == "pause_compile_resume":
        if row.executor_teardown_ns is not None and row.executor_teardown_ns != 0:
            raise ValueError("pause_compile_resume executor_teardown_ns must be 0")
        if row.executor_reconstruction_ns is not None and row.executor_reconstruction_ns != 0:
            raise ValueError("pause_compile_resume executor_reconstruction_ns must be 0")
        if row.publication_ns is not None and row.commit_ns is not None and row.publication_ns != row.commit_ns:
            raise ValueError("pause_compile_resume publication_ns must equal commit_ns")
        if row.phases_may_overlap:
            raise ValueError("pause_compile_resume phases_may_overlap must be False")

    elif row.baseline == "stop_rebuild_restart":
        if row.phases_may_overlap:
            raise ValueError("stop_rebuild_restart phases_may_overlap must be False")

    if (
        row.request_to_effect_ns is not None
        and row.critical_path_instrumented_ns is not None
    ):
        if not (0 <= row.critical_path_instrumented_ns <= row.request_to_effect_ns):
            raise ValueError(
                f"critical_path_instrumented_ns ({row.critical_path_instrumented_ns}) must be between 0 and request_to_effect_ns ({row.request_to_effect_ns})"
            )

    if (
        row.request_to_effect_ns is not None
        and row.critical_path_instrumented_ns is not None
        and row.unattributed_critical_path_ns is not None
    ):
        expected_unattributed = row.request_to_effect_ns - row.critical_path_instrumented_ns
        if abs(row.unattributed_critical_path_ns - expected_unattributed) > tolerance_ns:
            raise ValueError(
                f"Unattributed critical path mismatch: got {row.unattributed_critical_path_ns}, expected {expected_unattributed}"
            )

    if (
        row.sum_of_instrumented_phase_durations_ns is not None
        and row.critical_path_instrumented_ns is not None
        and row.instrumented_duration_overlap_ns is not None
        and row.instrumented_duration_outside_effect_window_ns is not None
    ):
        expected_sum = (
            row.critical_path_instrumented_ns
            + row.instrumented_duration_overlap_ns
            + row.instrumented_duration_outside_effect_window_ns
        )
        if abs(row.sum_of_instrumented_phase_durations_ns - expected_sum) > tolerance_ns:
            raise ValueError(
                f"Phase duration additive decomposition mismatch: got sum {row.sum_of_instrumented_phase_durations_ns}, expected {expected_sum}"
            )


@dataclass(frozen=True, slots=True)
class InterferenceSampleRow:
    run_id: str
    scenario_id: str
    preparation_category: str
    repetition: int
    frame_latency_before_median_ns: float
    frame_latency_during_median_ns: float
    frame_latency_during_p95_ns: float
    frame_latency_after_median_ns: float
    frames_completed_during_preparation: int
    throughput_before_fps: float
    throughput_during_fps: float
    throughput_after_fps: float


@dataclass(frozen=True, slots=True)
class ConformanceResultRow:
    run_id: str
    campaign_id: str
    scenario_id: str
    scenario_type: str
    frames_submitted: int
    frames_completed: int
    reconfigurations_requested: int
    reconfigurations_committed: int
    reconfigurations_rejected: int
    reconfigurations_failed: int
    mixed_plan_frames: int
    missing_frames: int
    duplicate_frames: int
    retired_plan_executions: int
    invalid_routing_events: int
    state_continuity_failures: int
    unexpected_state_resets: int
    active_plan_changed_after_failed_candidate: int
    candidate_resource_leaks: int
    processor_instance_leaks: int
    terminal_status: str


STEADY_STATE_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "topology",
    "workload_id",
    "implementation",
    "repetition",
    "execution_order_position",
    "operations",
    "elapsed_ns",
    "normalized_ns_per_frame",
    "frames_completed",
    "plan_version",
)

RECONFIGURATION_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "baseline",
    "edit_type",
    "graph_size",
    "repetition",
    "old_plan_version",
    "new_plan_version",
    "terminal_status",
    "validation_ns",
    "preparation_ns",
    "request_to_ready_ns",
    "boundary_wait_ns",
    "commit_ns",
    "request_to_effect_ns",
    "retirement_queue_delay_ns",
    "retirement_duration_ns",
    "commit_to_retirement_complete_ns",
    "maximum_output_gap_ns",
    "transition_output_gap_ns",
    "frames_completed_during_request",
    "old_plan_frames_admitted_after_request_before_commit",
    "frames_dropped",
    "frames_duplicated",
    "reused_processor_count",
    "staged_processor_count",
    "retired_processor_count",
    "state_transition_policy",
    "admission_stop_ns",
    "executor_teardown_ns",
    "executor_reconstruction_ns",
    "publication_ns",
    "executor_restart_ns",
    "first_admission_wait_ns",
    "first_completion_wait_ns",
    "retirement_ns",
    "total_synchronous_ns",
    "phases_may_overlap",
    "sum_of_instrumented_phase_durations_ns",
    "instrumented_phase_sum_ns",
    "critical_path_instrumented_ns",
    "unattributed_critical_path_ns",
    "unattributed_request_time_ns",
    "instrumented_duration_overlap_ns",
    "instrumented_duration_outside_effect_window_ns",
)

INTERFERENCE_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "preparation_category",
    "repetition",
    "frame_latency_before_median_ns",
    "frame_latency_during_median_ns",
    "frame_latency_during_p95_ns",
    "frame_latency_after_median_ns",
    "frames_completed_during_preparation",
    "throughput_before_fps",
    "throughput_during_fps",
    "throughput_after_fps",
)

CONFORMANCE_HEADERS: tuple[str, ...] = (
    "run_id",
    "campaign_id",
    "scenario_id",
    "scenario_type",
    "frames_submitted",
    "frames_completed",
    "reconfigurations_requested",
    "reconfigurations_committed",
    "reconfigurations_rejected",
    "reconfigurations_failed",
    "mixed_plan_frames",
    "missing_frames",
    "duplicate_frames",
    "retired_plan_executions",
    "invalid_routing_events",
    "state_continuity_failures",
    "unexpected_state_resets",
    "active_plan_changed_after_failed_candidate",
    "candidate_resource_leaks",
    "processor_instance_leaks",
    "terminal_status",
)
