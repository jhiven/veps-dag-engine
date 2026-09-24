"""Deterministic pilot for the mechanism-independent fixed-window protocol.

These cases exercise `compute_fixed_window_metrics` against synthetic frame
records, so the accounting can be checked without a GPU, an RTSP server, or any
timing luck. They are the offline half of the protocol gate: the endpoint must
not move with a mechanism's behaviour, and outcomes observed during the final
drain must never be backdated into the cutoff classification.
"""

from __future__ import annotations

import pytest

from benchmarks.runners.realworld_video import (
    RealworldVideoFrameSampleRow,
    compute_fixed_window_metrics,
)

BASELINE = 6
TRANSITION = 14
WINDOW = BASELINE + TRANSITION
TRIGGER = BASELINE  # positions 0..5 are pre-request, 6..19 are post-request
CUTOFF_NS = 10_000


def _row(
    position: int,
    *,
    plan_version: int | None = 1,
    admission_ns: int | None = 100,
    completion_ns: int | None = 200,
    dropped: bool = False,
    drop_reason: str = "none",
    terminal_status: str = "completed",
    drop_decision_ns: int | None = None,
    duplicated: bool = False,
) -> RealworldVideoFrameSampleRow:
    return RealworldVideoFrameSampleRow(
        run_id="pilot",
        repetition=1,
        mechanism="VEPS",
        frame_id=position,
        source_timestamp_ns=50,
        admission_timestamp_ns=admission_ns,
        completion_timestamp_ns=completion_ns,
        plan_version=plan_version,
        detector_id="d",
        tracker_instance_id="t",
        queue_occupancy_before_enqueue=0,
        queue_occupancy_after_enqueue=0,
        queue_capacity=4,
        terminal_status=terminal_status,
        drop_reason=drop_reason,
        dropped=dropped,
        duplicated=duplicated,
        inside_measurement_window=True,
        media_pts_ns=0,
        receiver_ingress_timestamp_ns=60,
        enqueue_decision_timestamp_ns=70,
        drop_decision_timestamp_ns=drop_decision_ns,
        media_frame_index=position,
        receiver_position=position,
    )


def _metrics(rows: list[RealworldVideoFrameSampleRow], cutoff_ns: int = CUTOFF_NS) -> dict[str, int | bool | float]:
    return compute_fixed_window_metrics(
        frame_rows=rows,
        request_trigger_receiver_position=TRIGGER,
        cutoff_deadline_ns=cutoff_ns,
        baseline_frame_count=BASELINE,
        transition_frame_count=TRANSITION,
    )


def _all_completed() -> list[RealworldVideoFrameSampleRow]:
    return [
        _row(position, plan_version=1 if position < TRIGGER else 2)
        for position in range(WINDOW)
    ]


def test_a_fully_observed_window_reconciles_exactly() -> None:
    result = _metrics(_all_completed())

    assert result["fixed_window_expected_source_frames"] == WINDOW
    assert result["fixed_window_receiver_observed_frames"] == WINDOW
    assert result["fixed_window_completed_frames"] == WINDOW
    assert result["fixed_window_accounting_residual"] == 0
    assert result["fixed_window_accounting_valid"] is True


def test_the_endpoint_does_not_move_when_the_first_output_arrives_early() -> None:
    """A mechanism that publishes sooner must not shorten its own window."""
    early = [
        _row(position, plan_version=1 if position < TRIGGER else 2)
        for position in range(WINDOW)
    ]
    late = [
        _row(position, plan_version=1 if position < WINDOW - 2 else 2)
        for position in range(WINDOW)
    ]

    early_result = _metrics(early)
    late_result = _metrics(late)

    for key in (
        "fixed_window_start_media_frame_index",
        "fixed_window_end_media_frame_index",
        "fixed_window_expected_source_frames",
        "fixed_window_receiver_observed_frames",
        "fixed_window_completed_frames",
    ):
        assert early_result[key] == late_result[key], key

    # Only the plan attribution differs between the two transition timings.
    assert early_result["fixed_window_new_plan_completions"] == WINDOW - TRIGGER
    assert late_result["fixed_window_new_plan_completions"] == 2


def test_a_completion_after_the_cutoff_is_in_flight_not_completed() -> None:
    """Drain outcomes must never be backdated into the cutoff classification."""
    rows = _all_completed()
    # This frame was admitted before the deadline but finished after it.
    rows[WINDOW - 1] = _row(
        WINDOW - 1,
        plan_version=2,
        admission_ns=CUTOFF_NS - 10,
        completion_ns=CUTOFF_NS + 5_000,
    )

    result = _metrics(rows)

    assert result["fixed_window_completed_frames"] == WINDOW - 1
    assert result["fixed_window_frames_in_flight_at_window_end"] == 1
    assert result["fixed_window_accounting_residual"] == 0
    assert result["fixed_window_accounting_valid"] is True

    # The same records, evaluated with a later deadline, complete instead.
    drained = _metrics(rows, cutoff_ns=CUTOFF_NS + 10_000)
    assert drained["fixed_window_completed_frames"] == WINDOW
    assert drained["fixed_window_frames_in_flight_at_window_end"] == 0


def test_a_frame_not_yet_admitted_at_the_cutoff_counts_as_queued() -> None:
    rows = _all_completed()
    rows[3] = _row(
        3,
        admission_ns=CUTOFF_NS + 1_000,
        completion_ns=CUTOFF_NS + 2_000,
    )

    result = _metrics(rows)

    assert result["fixed_window_queued_frames"] == 1
    assert result["fixed_window_completed_frames"] == WINDOW - 1
    assert result["fixed_window_accounting_residual"] == 0


def test_every_outcome_category_reconciles_against_the_window() -> None:
    """Queued, dropped, rejected, cancelled and completed must sum exactly."""
    rows = _all_completed()
    rows[0] = _row(
        0,
        dropped=True,
        drop_reason="ingress_overflow",
        terminal_status="dropped",
        completion_ns=None,
        admission_ns=None,
        drop_decision_ns=CUTOFF_NS - 100,
    )
    rows[1] = _row(
        1,
        dropped=True,
        drop_reason="admission_rejected",
        terminal_status="rejected",
        completion_ns=None,
        admission_ns=None,
        drop_decision_ns=CUTOFF_NS - 90,
    )
    rows[2] = _row(
        2,
        dropped=True,
        drop_reason="execution_cancelled",
        terminal_status="cancelled",
        completion_ns=None,
        admission_ns=None,
        drop_decision_ns=CUTOFF_NS - 80,
    )
    rows[3] = _row(3, admission_ns=CUTOFF_NS + 10, completion_ns=CUTOFF_NS + 20)
    rows[4] = _row(4, admission_ns=CUTOFF_NS - 10, completion_ns=CUTOFF_NS + 20)

    result = _metrics(rows)

    assert result["fixed_window_ingress_dropped_frames"] == 1
    assert result["fixed_window_admission_rejected_frames"] == 1
    assert result["fixed_window_execution_cancelled_frames"] == 1
    assert result["fixed_window_queued_frames"] == 1
    assert result["fixed_window_frames_in_flight_at_window_end"] == 1
    assert result["fixed_window_completed_frames"] == WINDOW - 5

    accounted = (
        int(result["fixed_window_completed_frames"])
        + int(result["fixed_window_ingress_dropped_frames"])
        + int(result["fixed_window_admission_rejected_frames"])
        + int(result["fixed_window_execution_cancelled_frames"])
        + int(result["fixed_window_queued_frames"])
        + int(result["fixed_window_frames_in_flight_at_window_end"])
    )
    assert accounted == WINDOW
    assert result["fixed_window_accounting_residual"] == 0
    assert result["fixed_window_accounting_valid"] is True


def test_an_underfilled_window_is_reported_as_invalid() -> None:
    """A run that never reached its target may not validate."""
    rows = _all_completed()[:-3]

    result = _metrics(rows)

    assert result["fixed_window_receiver_observed_frames"] == WINDOW - 3
    assert result["fixed_window_accounting_residual"] != 0
    assert result["fixed_window_accounting_valid"] is False


def test_a_duplicated_position_invalidates_the_accounting() -> None:
    rows = _all_completed()
    rows.append(_row(5, duplicated=True))

    result = _metrics(rows)

    assert result["fixed_window_duplicated_frames"] == 1
    assert result["fixed_window_accounting_valid"] is False


def test_rows_outside_the_window_are_ignored() -> None:
    rows = _all_completed()
    rows.append(_row(WINDOW + 5))
    rows.append(_row(-1) if TRIGGER > BASELINE else _row(WINDOW + 6))

    result = _metrics(rows)

    assert result["fixed_window_receiver_observed_frames"] == WINDOW
    assert result["fixed_window_accounting_valid"] is True


def test_a_boundary_without_enough_baseline_positions_is_refused() -> None:
    with pytest.raises(ValueError, match="fixed receiver boundary"):
        compute_fixed_window_metrics(
            frame_rows=_all_completed(),
            request_trigger_receiver_position=BASELINE - 1,
            cutoff_deadline_ns=CUTOFF_NS,
            baseline_frame_count=BASELINE,
            transition_frame_count=TRANSITION,
        )


def test_a_negative_cutoff_is_refused() -> None:
    with pytest.raises(ValueError, match="cutoff_deadline_ns"):
        _metrics(_all_completed(), cutoff_ns=-1)


def test_a_trigger_one_position_late_fails_the_fixed_boundary_protocol() -> None:
    """A late request cannot silently shift the mechanism-independent window."""
    rows = _all_completed()

    on_boundary = _metrics(rows)
    assert on_boundary["fixed_window_start_media_frame_index"] == 0
    assert on_boundary["fixed_window_accounting_residual"] == 0
    assert on_boundary["fixed_window_accounting_valid"] is True

    with pytest.raises(ValueError, match="fixed receiver boundary"):
        compute_fixed_window_metrics(
            frame_rows=rows,
            request_trigger_receiver_position=TRIGGER + 1,
            cutoff_deadline_ns=CUTOFF_NS,
            baseline_frame_count=BASELINE,
            transition_frame_count=TRANSITION,
        )
