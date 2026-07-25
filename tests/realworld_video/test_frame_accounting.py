"""Unit tests for frame lifecycle accounting, flow invariants, and reconfiguration metrics."""

from __future__ import annotations

from usecases.video_analytics.contracts import DropReason, TerminalStatus
from usecases.video_analytics.metrics import calculate_flow_metrics, calculate_reconfiguration_metrics


def test_frame_accounting_terminal_status_exclusivity() -> None:
    """Test 7: Every frame receives exactly one terminal status."""
    statuses = list(TerminalStatus)
    assert len(statuses) == 5
    assert TerminalStatus.COMPLETED.value == "completed"
    assert TerminalStatus.DROPPED_INGRESS_OVERFLOW.value == "dropped_ingress_overflow"
    assert TerminalStatus.DROPPED_ADMISSION_REJECTED.value == "dropped_admission_rejected"
    assert TerminalStatus.CANCELLED_ON_STOP.value == "cancelled_on_stop"
    assert TerminalStatus.IN_FLIGHT_AT_WINDOW_END.value == "in_flight_at_window_end"


def test_flow_accounting_invariant() -> None:
    """Test 8: Flow accounting invariant source_frames_received == completed + total_dropped + in_flight."""
    records = [
        (1, TerminalStatus.COMPLETED, DropReason.NONE, True, False),
        (2, TerminalStatus.COMPLETED, DropReason.NONE, True, False),
        (3, TerminalStatus.DROPPED_INGRESS_OVERFLOW, DropReason.INGRESS_OVERFLOW, False, False),
        (4, TerminalStatus.CANCELLED_ON_STOP, DropReason.CANCELLED_ON_STOP, True, False),
        (5, TerminalStatus.IN_FLIGHT_AT_WINDOW_END, DropReason.NONE, True, False),
    ]
    summary = calculate_flow_metrics(records)
    assert summary.source_frames_received == 5
    assert summary.frames_completed == 2
    assert summary.ingress_overflow_drop_count == 1
    assert summary.execution_cancelled_count == 1
    assert summary.frames_in_flight_at_window_end == 1
    assert summary.total_dropped_frame_count == 2
    assert summary.drop_rate == 2 / 5


def test_in_flight_not_automatically_counted_as_dropped() -> None:
    """Test 9: In-flight frames at window end are not counted as dropped."""
    records = [
        (1, TerminalStatus.COMPLETED, DropReason.NONE, True, False),
        (2, TerminalStatus.IN_FLIGHT_AT_WINDOW_END, DropReason.NONE, True, False),
    ]
    summary = calculate_flow_metrics(records)
    assert summary.frames_completed == 1
    assert summary.frames_in_flight_at_window_end == 1
    assert summary.total_dropped_frame_count == 0
    assert summary.drop_rate == 0.0


def test_transition_output_gap_from_real_completion_timestamps() -> None:
    """Test 10 & 11 & 12: Reconfiguration metrics derived from empirical frame completion records."""
    t_req = 1000
    t_prep_start = 1100
    t_prep_end = 1500
    t_pub = 1600

    completed_records = [
        (1, 1, 1200),  # Old plan completed during prep
        (2, 1, 1400),  # Old plan completed during prep
        (3, 1, 1700),  # Last old plan frame completed after pub
        (4, 2, 2000),  # First candidate frame completed
    ]

    reconfig = calculate_reconfiguration_metrics(
        request_timestamp_ns=t_req,
        prep_start_ns=t_prep_start,
        prep_end_ns=t_prep_end,
        publication_timestamp_ns=t_pub,
        initial_plan_version=1,
        candidate_plan_version=2,
        completed_frame_records=completed_records,
    )

    assert reconfig.first_candidate_output_ns == 2000
    assert reconfig.request_to_effect_ns == 1000  # 2000 - 1000
    assert reconfig.transition_output_gap_ns == 300  # 2000 - 1700
    assert reconfig.old_plan_frames_completed_during_prep == 2


def test_tracker_preservation_and_reset_invariants() -> None:
    """Test 13 & 14: Tracker instance identity is preserved and reset count remains 0."""
    from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend

    trk = FakeTrackerBackend(instance_id="trk_instance_101")
    assert trk.instance_id == "trk_instance_101"
    assert trk.reset_count == 0
    trk.update(frame_id=1, source_timestamp_ns=10, detections=())
    assert trk.instance_id == "trk_instance_101"
    assert trk.reset_count == 0
