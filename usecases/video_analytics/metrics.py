"""Flow accounting metrics, invariant verification, and live reconfiguration metrics calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from usecases.video_analytics.contracts import DropReason, TerminalStatus

__all__ = [
    "FlowMetricsSummary",
    "ReconfigurationMetricsSummary",
    "calculate_flow_metrics",
    "calculate_reconfiguration_metrics",
]


@dataclass(frozen=True, slots=True)
class FlowMetricsSummary:
    """Aggregate flow accounting summary metrics for a benchmark sub-run."""

    source_frames_received: int
    frames_admitted: int
    frames_completed: int
    ingress_overflow_drop_count: int
    admission_rejection_count: int
    execution_cancelled_count: int
    frames_in_flight_at_window_end: int
    total_dropped_frame_count: int
    drop_rate: float
    duplicated_frame_count: int


@dataclass(frozen=True, slots=True)
class ReconfigurationMetricsSummary:
    """Live reconfiguration transition metrics derived from empirical frame records."""

    first_candidate_output_ns: int | None
    request_to_effect_ns: int | None
    transition_output_gap_ns: int | None
    old_plan_frames_completed_during_prep: int


def calculate_flow_metrics(
    frame_records: Sequence[tuple[int, TerminalStatus, DropReason, bool, bool]],
) -> FlowMetricsSummary:
    """Calculate aggregate flow accounting metrics and enforce the flow accounting invariant.

    Each record in frame_records is a tuple of:
    (frame_id, terminal_status, drop_reason, is_admitted, is_duplicated)
    """
    source_frames_received = len(frame_records)
    frames_admitted = 0
    frames_completed = 0
    ingress_overflow_drops = 0
    admission_rejection_drops = 0
    execution_cancelled_drops = 0
    in_flight_end = 0
    duplicated_count = 0

    for _, status, reason, is_admitted, is_dup in frame_records:
        if is_admitted:
            frames_admitted += 1
        if is_dup:
            duplicated_count += 1

        if status is TerminalStatus.COMPLETED:
            frames_completed += 1
        elif status is TerminalStatus.DROPPED_INGRESS_OVERFLOW or reason is DropReason.INGRESS_OVERFLOW:
            ingress_overflow_drops += 1
        elif status is TerminalStatus.DROPPED_ADMISSION_REJECTED or reason is DropReason.ADMISSION_REJECTED:
            admission_rejection_drops += 1
        elif status is TerminalStatus.CANCELLED_ON_STOP or reason is DropReason.CANCELLED_ON_STOP:
            execution_cancelled_drops += 1
        elif status is TerminalStatus.IN_FLIGHT_AT_WINDOW_END:
            in_flight_end += 1

    total_dropped = ingress_overflow_drops + admission_rejection_drops + execution_cancelled_drops
    drop_rate = (total_dropped / source_frames_received) if source_frames_received > 0 else 0.0

    # Invariant Check: source_frames_received = frames_completed + total_dropped + in_flight_end
    expected_sum = frames_completed + total_dropped + in_flight_end
    if source_frames_received != expected_sum:
        raise ValueError(
            f"Flow accounting invariant violated: source_frames_received ({source_frames_received}) != "
            f"completed ({frames_completed}) + dropped ({total_dropped}) + in_flight ({in_flight_end}) [sum={expected_sum}]"
        )

    return FlowMetricsSummary(
        source_frames_received=source_frames_received,
        frames_admitted=frames_admitted,
        frames_completed=frames_completed,
        ingress_overflow_drop_count=ingress_overflow_drops,
        admission_rejection_count=admission_rejection_drops,
        execution_cancelled_count=execution_cancelled_drops,
        frames_in_flight_at_window_end=in_flight_end,
        total_dropped_frame_count=total_dropped,
        drop_rate=drop_rate,
        duplicated_frame_count=duplicated_count,
    )


def calculate_reconfiguration_metrics(
    request_timestamp_ns: int,
    prep_start_ns: int,
    prep_end_ns: int,
    publication_timestamp_ns: int,
    initial_plan_version: int,
    candidate_plan_version: int,
    completed_frame_records: Sequence[tuple[int, int | None, int | None]],  # (frame_id, plan_version, completion_ns)
) -> ReconfigurationMetricsSummary:
    """Calculate reconfiguration transition metrics from empirical frame completion records.

    each record in completed_frame_records is:
    (frame_id, plan_version, completion_timestamp_ns)
    """
    old_completed_during_prep = 0
    last_old_completion_ns: int | None = None
    first_cand_completion_ns: int | None = None

    for _, plan_ver, comp_ns in completed_frame_records:
        if comp_ns is None:
            continue

        if plan_ver == initial_plan_version:
            if prep_start_ns <= comp_ns <= publication_timestamp_ns:
                old_completed_during_prep += 1
            if last_old_completion_ns is None or comp_ns > last_old_completion_ns:
                last_old_completion_ns = comp_ns
        elif plan_ver == candidate_plan_version:
            if first_cand_completion_ns is None or comp_ns < first_cand_completion_ns:
                first_cand_completion_ns = comp_ns

    request_to_effect_ns: int | None = None
    if first_cand_completion_ns is not None:
        request_to_effect_ns = first_cand_completion_ns - request_timestamp_ns

    gap_ns: int | None = None
    if first_cand_completion_ns is not None and last_old_completion_ns is not None:
        gap_ns = first_cand_completion_ns - last_old_completion_ns

    return ReconfigurationMetricsSummary(
        first_candidate_output_ns=first_cand_completion_ns,
        request_to_effect_ns=request_to_effect_ns,
        transition_output_gap_ns=gap_ns,
        old_plan_frames_completed_during_prep=old_completed_during_prep,
    )
