"""Strictly typed benchmark row data models and constants."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SteadyStateSampleRow",
    "ReconfigurationSampleRow",
    "AblationSampleRow",
    "MemorySampleRow",
    "InterferenceSampleRow",
    "InterferenceFrameSampleRow",
    "ConformanceResultRow",
    "STEADY_STATE_HEADERS",
    "RECONFIGURATION_HEADERS",
    "ABLATION_HEADERS",
    "MEMORY_HEADERS",
    "INTERFERENCE_HEADERS",
    "INTERFERENCE_FRAME_HEADERS",
    "CONFORMANCE_HEADERS",
    "CriticalPathDecomposition",
    "compute_critical_path_decomposition",
    "validate_reconfiguration_sample",
    "validate_ablation_sample",
    "validate_memory_sample",
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
    grace_period_ns: int | None = None
    handoff_wait_ns: int | None = None
    cleanup_duration_ns: int | None = None
    last_old_frame_completed_ns: int | None = None


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
        # Synchronized admission work is publication plus, when a mutable
        # processor is preserved, the pre-publication drain. The two intervals
        # are disjoint, so neither may be counted inside the other.
        if row.total_synchronous_ns is not None and row.publication_ns is not None:
            expected_synchronous_ns = row.publication_ns + (row.handoff_wait_ns or 0)
            if row.total_synchronous_ns != expected_synchronous_ns:
                raise ValueError(
                    "prepare_and_commit total_synchronous_ns must equal "
                    "publication_ns plus handoff_wait_ns"
                )
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
class AblationSampleRow:
    run_id: str
    scenario_id: str
    block_id: str
    block_seed: int
    variant: str
    variant_name: str
    preparation_placement: str
    retirement_policy: str
    edit_type: str
    repetition: int
    variant_order_position: int
    candidate_graph_fingerprint: str
    preparation_workload_id: str
    old_plan_version: int
    new_plan_version: int
    validation_ns: int | None = None
    synchronous_preparation_ns: int | None = None
    offpath_preparation_ns: int | None = None
    boundary_wait_ns: int | None = None
    publication_ns: int = 0
    synchronous_retirement_ns: int | None = None
    deferred_retirement_ns: int | None = None
    first_effect_wait_ns: int | None = None
    request_to_effect_ns: int = 0
    transition_output_gap_ns: int = 0
    total_synchronous_ns: int = 0
    synchronous_accounting_residual_ns: int = 0
    synchronous_accounting_valid: bool = True
    synchronous_accounting_invalid_reason: str | None = None
    preparation_phase_definition: str = (
        "offpath_preparation_ns contains full off-path candidate preparation operation "
        "including validation, compilation, processor construction, and any synthetic preparation work"
    )
    preparation_accounting_valid: bool = True
    preparation_accounting_invalid_reason: str | None = None
    retirement_duration_ns: int = 0
    old_plan_frames_admitted_after_request_before_commit: int = 0
    peak_live_processors: int = 0
    random_seed: int | None = None
    request_phase_offset_ns: int | None = None


def validate_ablation_sample(row: AblationSampleRow) -> None:
    valid_variants = {
        "variant_a": ("sync_prepare_sync_retire", "synchronous", "synchronous"),
        "variant_b": ("offpath_prepare_sync_retire", "off_path", "synchronous"),
        "variant_c": ("sync_prepare_deferred_retire", "synchronous", "deferred"),
        "variant_d": ("offpath_prepare_deferred_retire", "off_path", "deferred"),
    }
    if row.variant not in valid_variants:
        raise ValueError(f"Unknown ablation variant {row.variant!r}")

    exp_name, exp_prep, exp_ret = valid_variants[row.variant]
    if row.variant_name != exp_name:
        raise ValueError(f"Variant {row.variant} variant_name must be {exp_name!r}, got {row.variant_name!r}")
    if row.preparation_placement != exp_prep:
        raise ValueError(f"Variant {row.variant} preparation_placement must be {exp_prep!r}, got {row.preparation_placement!r}")
    if row.retirement_policy != exp_ret:
        raise ValueError(f"Variant {row.variant} retirement_policy must be {exp_ret!r}, got {row.retirement_policy!r}")

    for name, val in (
        ("request_to_effect_ns", row.request_to_effect_ns),
        ("transition_output_gap_ns", row.transition_output_gap_ns),
        ("total_synchronous_ns", row.total_synchronous_ns),
        ("retirement_duration_ns", row.retirement_duration_ns),
        ("old_plan_frames_admitted_after_request_before_commit", row.old_plan_frames_admitted_after_request_before_commit),
        ("peak_live_processors", row.peak_live_processors),
    ):
        if val < 0:
            raise ValueError(f"Ablation field {name} must be non-negative, got {val}")

    # Synchronous accounting residual check
    if row.variant == "variant_a":
        expected_sync = (row.validation_ns or 0) + (row.synchronous_preparation_ns or 0) + row.publication_ns + (row.synchronous_retirement_ns or 0)
    elif row.variant == "variant_b":
        expected_sync = row.publication_ns + (row.synchronous_retirement_ns or 0)
    elif row.variant == "variant_c":
        expected_sync = (row.validation_ns or 0) + (row.synchronous_preparation_ns or 0) + row.publication_ns
    else:  # variant_d
        expected_sync = row.publication_ns

    residual = row.total_synchronous_ns - expected_sync
    if row.synchronous_accounting_residual_ns != residual:
        raise ValueError(
            f"synchronous_accounting_residual_ns ({row.synchronous_accounting_residual_ns}) does not match total_synchronous_ns - sum ({residual})"
        )


@dataclass(frozen=True, slots=True)
class MemorySampleRow:
    run_id: str
    scenario_id: str
    variant: str
    repetition: int
    rss_before_bytes: int
    rss_after_candidate_prepare_bytes: int
    rss_after_commit_bytes: int
    rss_after_retirement_bytes: int
    observed_peak_rss_bytes: int
    peak_rss_delta_bytes: int
    retained_rss_delta_bytes: int
    peak_live_processors: int


def validate_memory_sample(row: MemorySampleRow) -> None:
    expected_peak = max(
        row.rss_before_bytes,
        row.rss_after_candidate_prepare_bytes,
        row.rss_after_commit_bytes,
        row.rss_after_retirement_bytes,
    )
    if row.observed_peak_rss_bytes != expected_peak:
        raise ValueError(f"observed_peak_rss_bytes ({row.observed_peak_rss_bytes}) must equal max checkpoint ({expected_peak})")
    if row.peak_rss_delta_bytes != (row.observed_peak_rss_bytes - row.rss_before_bytes):
        raise ValueError("peak_rss_delta_bytes mismatch")
    if row.retained_rss_delta_bytes != (row.rss_after_retirement_bytes - row.rss_before_bytes):
        raise ValueError("retained_rss_delta_bytes mismatch")


@dataclass(frozen=True, slots=True)
class InterferenceFrameSampleRow:
    run_id: str
    scenario_id: str
    preparation_category: str
    target_window_ns: int
    repetition: int
    phase: str
    frame_id: int
    plan_version: int
    admission_timestamp_ns: int
    completion_timestamp_ns: int
    frame_latency_ns: int
    queue_occupancy: float
    dropped: bool
    duplicated: bool


@dataclass(frozen=True, slots=True)
class InterferenceSampleRow:
    run_id: str
    scenario_id: str
    preparation_category: str
    target_window_ns: int
    repetition: int
    execution_order_position: int
    random_seed: int
    configured_offered_rate_fps: float
    before_window_start_ns: int
    before_window_end_ns: int
    during_window_start_ns: int
    during_window_end_ns: int
    after_window_start_ns: int
    after_window_end_ns: int
    measured_before_window_duration_ns: int
    measured_during_window_duration_ns: int
    measured_after_window_duration_ns: int
    producer_attempted_frames_before: int
    producer_attempted_frames_during: int
    producer_attempted_frames_after: int
    producer_emitted_frames_before: int
    producer_emitted_frames_during: int
    producer_emitted_frames_after: int
    producer_skipped_deadlines_before: int
    producer_skipped_deadlines_during: int
    producer_skipped_deadlines_after: int
    producer_schedule_lateness_median_ns_before: float
    producer_schedule_lateness_p95_ns_before: float
    producer_schedule_lateness_median_ns_during: float
    producer_schedule_lateness_p95_ns_during: float
    producer_schedule_lateness_median_ns_after: float
    producer_schedule_lateness_p95_ns_after: float
    admitted_frames_before: int
    admitted_frames_during: int
    admitted_frames_after: int
    completed_frames_before: int
    completed_frames_during: int
    completed_frames_after: int
    frame_latency_before_median_ns: float | None
    frame_latency_before_p95_ns: float | None
    frame_latency_during_median_ns: float | None
    frame_latency_during_p95_ns: float | None
    frame_latency_after_median_ns: float | None
    frame_latency_after_p95_ns: float | None
    p95_degradation_vs_before_pct: float | None
    median_degradation_vs_before_pct: float | None
    frames_completed_during_window: int
    frames_completed_while_preparation_active: int | None
    old_plan_frames_completed: int
    admission_throughput_before_fps: float
    admission_throughput_during_fps: float
    admission_throughput_after_fps: float
    completion_throughput_before_fps: float
    completion_throughput_during_fps: float
    completion_throughput_after_fps: float
    admission_rate_ratio_vs_before: float | None
    completion_rate_ratio_vs_before: float | None
    latency_interpretation_confounded: bool
    latency_interpretation_confounded_reason: str | None
    within_run_admission_shift_flag: bool = False
    within_run_admission_shift_reason: str | None = None
    backlog_frames_at_during_start: int = 0
    backlog_frames_at_during_end: int = 0
    backlog_frames_drained_after_window: int = 0
    frames_emitted_during_but_admitted_after: int = 0
    frames_admitted_during_but_emitted_before: int = 0
    flow_accounting_valid: bool = True
    flow_accounting_residual_frames: int = 0
    flow_accounting_invalid_reason: str | None = None
    queue_occupancy_during_median: float = 0.0
    queue_occupancy_during_p95: float = 0.0
    dropped_frames: int = 0
    duplicated_frames: int = 0
    valid_for_tail_latency: bool = True
    invalid_reason: str | None = None
    target_preparation_duration_ns: int | None = None
    actual_preparation_duration_ns: int = 0
    preparation_checksum: int | None = None
    duration_valid: bool = True
    duration_relative_error: float = 0.0
    raw_frame_count_matches_summary: bool = True
    phase_timestamp_bounds_valid: bool = True
    throughput_accounting_valid: bool = True
    throughput_accounting_invalid_reason: str | None = None


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
    grace_period_safety_violations: int
    stateful_handoff_ordering_failures: int
    resource_lifetime_violations: int
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
    "grace_period_ns",
    "handoff_wait_ns",
    "cleanup_duration_ns",
    "last_old_frame_completed_ns",
)

ABLATION_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "block_id",
    "block_seed",
    "variant",
    "variant_name",
    "preparation_placement",
    "retirement_policy",
    "edit_type",
    "repetition",
    "variant_order_position",
    "candidate_graph_fingerprint",
    "preparation_workload_id",
    "old_plan_version",
    "new_plan_version",
    "validation_ns",
    "synchronous_preparation_ns",
    "offpath_preparation_ns",
    "boundary_wait_ns",
    "publication_ns",
    "synchronous_retirement_ns",
    "deferred_retirement_ns",
    "first_effect_wait_ns",
    "request_to_effect_ns",
    "transition_output_gap_ns",
    "total_synchronous_ns",
    "synchronous_accounting_residual_ns",
    "synchronous_accounting_valid",
    "synchronous_accounting_invalid_reason",
    "preparation_phase_definition",
    "preparation_accounting_valid",
    "preparation_accounting_invalid_reason",
    "retirement_duration_ns",
    "old_plan_frames_admitted_after_request_before_commit",
    "peak_live_processors",
    "random_seed",
    "request_phase_offset_ns",
)

ABLATION_CONTRAST_HEADERS: tuple[str, ...] = (
    "scenario_id",
    "contrast",
    "family",
    "metric",
    "pair_count",
    "median_left",
    "median_right",
    "paired_median_difference",
    "paired_median_ratio",
    "paired_log_ratio_mean",
    "paired_bootstrap_ci_95_lower",
    "paired_bootstrap_ci_95_upper",
    "raw_p_value",
    "holm_adjusted_p_value",
)

ABLATION_STRATIFIED_HEADERS: tuple[str, ...] = (
    "scenario_id",
    "frame_group",
    "count",
    "request_to_effect_median_ns",
    "request_to_effect_p95_ns",
    "transition_output_gap_median_ns",
    "transition_output_gap_p95_ns",
    "total_synchronous_median_ns",
    "total_synchronous_p95_ns",
    "offpath_preparation_median_ns",
    "offpath_preparation_p95_ns",
    "deferred_retirement_median_ns",
    "deferred_retirement_p95_ns",
)

MEMORY_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "variant",
    "repetition",
    "rss_before_bytes",
    "rss_after_candidate_prepare_bytes",
    "rss_after_commit_bytes",
    "rss_after_retirement_bytes",
    "observed_peak_rss_bytes",
    "peak_rss_delta_bytes",
    "retained_rss_delta_bytes",
    "peak_live_processors",
)

INTERFERENCE_FRAME_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "preparation_category",
    "target_window_ns",
    "repetition",
    "phase",
    "frame_id",
    "plan_version",
    "admission_timestamp_ns",
    "completion_timestamp_ns",
    "frame_latency_ns",
    "queue_occupancy",
    "dropped",
    "duplicated",
)

INTERFERENCE_HEADERS: tuple[str, ...] = (
    "run_id",
    "scenario_id",
    "preparation_category",
    "target_window_ns",
    "repetition",
    "execution_order_position",
    "random_seed",
    "configured_offered_rate_fps",
    "before_window_start_ns",
    "before_window_end_ns",
    "during_window_start_ns",
    "during_window_end_ns",
    "after_window_start_ns",
    "after_window_end_ns",
    "measured_before_window_duration_ns",
    "measured_during_window_duration_ns",
    "measured_after_window_duration_ns",
    "producer_attempted_frames_before",
    "producer_attempted_frames_during",
    "producer_attempted_frames_after",
    "producer_emitted_frames_before",
    "producer_emitted_frames_during",
    "producer_emitted_frames_after",
    "producer_skipped_deadlines_before",
    "producer_skipped_deadlines_during",
    "producer_skipped_deadlines_after",
    "producer_schedule_lateness_median_ns_before",
    "producer_schedule_lateness_p95_ns_before",
    "producer_schedule_lateness_median_ns_during",
    "producer_schedule_lateness_p95_ns_during",
    "producer_schedule_lateness_median_ns_after",
    "producer_schedule_lateness_p95_ns_after",
    "admitted_frames_before",
    "admitted_frames_during",
    "admitted_frames_after",
    "completed_frames_before",
    "completed_frames_during",
    "completed_frames_after",
    "frame_latency_before_median_ns",
    "frame_latency_before_p95_ns",
    "frame_latency_during_median_ns",
    "frame_latency_during_p95_ns",
    "frame_latency_after_median_ns",
    "frame_latency_after_p95_ns",
    "p95_degradation_vs_before_pct",
    "median_degradation_vs_before_pct",
    "frames_completed_during_window",
    "frames_completed_while_preparation_active",
    "old_plan_frames_completed",
    "admission_throughput_before_fps",
    "admission_throughput_during_fps",
    "admission_throughput_after_fps",
    "completion_throughput_before_fps",
    "completion_throughput_during_fps",
    "completion_throughput_after_fps",
    "admission_rate_ratio_vs_before",
    "completion_rate_ratio_vs_before",
    "latency_interpretation_confounded",
    "latency_interpretation_confounded_reason",
    "within_run_admission_shift_flag",
    "within_run_admission_shift_reason",
    "backlog_frames_at_during_start",
    "backlog_frames_at_during_end",
    "backlog_frames_drained_after_window",
    "frames_emitted_during_but_admitted_after",
    "frames_admitted_during_but_emitted_before",
    "flow_accounting_valid",
    "flow_accounting_residual_frames",
    "flow_accounting_invalid_reason",
    "queue_occupancy_during_median",
    "queue_occupancy_during_p95",
    "dropped_frames",
    "duplicated_frames",
    "valid_for_tail_latency",
    "invalid_reason",
    "target_preparation_duration_ns",
    "actual_preparation_duration_ns",
    "preparation_checksum",
    "duration_valid",
    "duration_relative_error",
    "raw_frame_count_matches_summary",
    "phase_timestamp_bounds_valid",
    "throughput_accounting_valid",
    "throughput_accounting_invalid_reason",
)

INTERFERENCE_CONTRAST_HEADERS: tuple[str, ...] = (
    "repetition",
    "target_window_ns",
    "treatment_category",
    "control_category",
    "treatment_execution_order_position",
    "control_execution_order_position",
    "treatment_during_median_ns",
    "control_during_median_ns",
    "paired_median_difference_ns",
    "paired_median_ratio",
    "treatment_during_p95_ns",
    "control_during_p95_ns",
    "paired_p95_difference_ns",
    "paired_p95_ratio",
    "treatment_admission_throughput_fps",
    "control_admission_throughput_fps",
    "paired_admission_throughput_ratio",
    "treatment_completion_throughput_fps",
    "control_completion_throughput_fps",
    "paired_completion_throughput_ratio",
    "treatment_dropped_frames",
    "control_dropped_frames",
    "treatment_queue_occupancy_p95",
    "control_queue_occupancy_p95",
    "within_run_admission_shift_flag",
    "matched_control_confounded",
    "matched_control_confounded_reason",
)

INTERFERENCE_AGGREGATE_CONTRAST_HEADERS: tuple[str, ...] = (
    "treatment_category",
    "target_window_ns",
    "filter_scope",
    "total_pair_count",
    "confounded_pair_count",
    "non_confounded_pair_count",
    "included_pair_count",
    "median_paired_p95_ratio",
    "p95_paired_p95_ratio",
    "bootstrap_ci_95_lower_p95_ratio",
    "bootstrap_ci_95_upper_p95_ratio",
    "median_paired_admission_throughput_ratio",
    "bootstrap_ci_95_lower_admission_ratio",
    "bootstrap_ci_95_upper_admission_ratio",
    "median_paired_completion_throughput_ratio",
    "bootstrap_ci_95_lower_completion_ratio",
    "bootstrap_ci_95_upper_completion_ratio",
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
    "grace_period_safety_violations",
    "stateful_handoff_ordering_failures",
    "resource_lifetime_violations",
    "terminal_status",
)
