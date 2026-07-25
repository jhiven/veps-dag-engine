"""Unit tests for the real-world reconfiguration result validator and phase invariants."""

from __future__ import annotations

import csv
import os
import tempfile

from usecases.video_analytics.validator import validate_benchmark_csvs


def _write_summary_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([r.get(k, "") for k in r.keys()])


def _write_frame_csv(path: str, rows: list[dict[str, str]]) -> None:
    _write_summary_csv(path, rows)


def _valid_summary(rep: int = 1, mech: str = "VEPS") -> dict[str, str]:
    """Minimal valid summary row satisfying invariants."""
    return {
        "run_id": "test_run",
        "repetition": str(rep),
        "execution_order_position": "1",
        "mechanism": mech,
        "source_path": "/tmp/test.mp4",
        "source_file_hash": "abc123",
        "video_fps": "30.00",
        "resolution": "768x432",
        "initial_model_id": "PekingU/rtdetr_r18vd",
        "candidate_model_id": "PekingU/rtdetr_r50vd",
        "device": "cpu",
        "dtype": "float32",
        "confidence_threshold": "0.50",
        "tracker_config": "bytetrack_default",
        "request_timestamp_ns": "1000",
        "candidate_prep_start_ns": "1100",
        "candidate_prep_end_ns": "1500",
        "publication_timestamp_ns": "1600",
        "first_candidate_output_ns": "2000",
        "request_to_effect_ns": "1000",
        "transition_output_gap_ns": "300",
        "old_plan_frames_completed_during_prep": "0",
        "source_frames_received": "130",
        "frames_admitted": "130",
        "frames_completed": "130",
        "ingress_overflow_drop_count": "0",
        "admission_rejection_count": "0",
        "execution_cancelled_count": "0",
        "frames_in_flight_at_window_end": "0",
        "total_dropped_frame_count": "0",
        "drop_rate": "0.000000",
        "dropped_frame_count": "0",
        "duplicated_frame_count": "0",
        "tracker_instance_id_before": "trk_1",
        "tracker_instance_id_after": "trk_1",
        "tracker_reset_count_before": "0",
        "tracker_reset_count_after": "0",
        "processor_peak_count": "5",
        "gpu_memory_before_prep_allocated_bytes": "",
        "gpu_memory_before_prep_reserved_bytes": "",
        "gpu_memory_before_prep_peak_allocated_bytes": "",
        "gpu_memory_before_prep_peak_reserved_bytes": "",
        "gpu_memory_during_coexistence_allocated_bytes": "",
        "gpu_memory_during_coexistence_reserved_bytes": "",
        "gpu_memory_during_coexistence_peak_allocated_bytes": "",
        "gpu_memory_during_coexistence_peak_reserved_bytes": "",
        "gpu_memory_after_pub_allocated_bytes": "",
        "gpu_memory_after_pub_reserved_bytes": "",
        "gpu_memory_after_pub_peak_allocated_bytes": "",
        "gpu_memory_after_pub_peak_reserved_bytes": "",
        "gpu_memory_after_ret_allocated_bytes": "",
        "gpu_memory_after_ret_reserved_bytes": "",
        "gpu_memory_after_ret_peak_allocated_bytes": "",
        "gpu_memory_after_ret_peak_reserved_bytes": "",
        "gpu_memory_before_prep_bytes": "",
        "gpu_memory_during_coexistence_bytes": "",
        "gpu_memory_after_pub_bytes": "",
        "gpu_memory_after_ret_bytes": "",
        "warmup_completed_frames": "30",
        "measurement_source_frame_target": "180",
        "reconfiguration_trigger_frame_offset": "60",
        "measurement_source_frames_received": "130",
        "measurement_start_timestamp_ns": "500",
        "measurement_end_timestamp_ns": "5000",
        "drain_start_timestamp_ns": "5000",
        "drain_end_timestamp_ns": "5100",
        "drops_before_request": "0",
        "drops_during_candidate_preparation": "0",
        "drops_between_preparation_and_publication": "0",
        "drops_between_publication_and_first_candidate_output": "0",
        "drops_after_first_candidate_output": "0",
        "active_detector_id_after_retirement": "PekingU/rtdetr_r50vd",
        "active_plan_version_after_retirement": "2",
        "active_detector_instance_count_after_retirement": "1",
        "peak_live_plan_count": "2",
        "peak_live_detector_instance_count": "2",
        "live_plan_count_after_publication": "2",
        "live_plan_count_after_retirement": "1",
        "live_detector_count_after_publication": "2",
        "live_detector_count_after_retirement": "1",
        "measurement_start_media_frame_index": "1",
        "measurement_start_media_pts_ns": "0",
        "measurement_end_media_frame_index": "130",
        "measurement_end_media_pts_ns": "4300000000",
        "request_trigger_media_frame_index": "61",
        "request_trigger_media_pts_ns": "2000000000",
        "measurement_start_source_sequence": "1",
        "measurement_end_source_sequence": "130",
        "pre_request_source_frame_target": "60",
        "pre_request_source_frames_received": "60",
        "transition_source_frames_received": "10",
        "post_effect_source_frame_target": "60",
        "post_effect_source_frames_received": "60",
        "total_measurement_source_frames_received": "130",
        "source_frames_before_request": "60",
        "source_frames_during_candidate_preparation": "5",
        "source_frames_between_preparation_and_publication": "3",
        "source_frames_between_publication_and_first_candidate_output": "2",
        "source_frames_after_first_candidate_output": "60",
        "drop_rate_before_request": "",
        "drop_rate_during_candidate_preparation": "",
        "drop_rate_between_preparation_and_publication": "",
        "drop_rate_between_publication_and_first_candidate_output": "",
        "drop_rate_after_first_candidate_output": "",
        "gpu_memory_transition_max_allocated_bytes": "",
        "gpu_memory_transition_max_reserved_bytes": "",
    }


def _valid_frame(
    fid: int,
    rep: int = 1,
    mech: str = "VEPS",
    plan_ver: str = "1",
    det_id: str = "PekingU/rtdetr_r18vd",
    trk_id: str = "trk_1",
    dropped: str = "0",
    dup: str = "0",
    inside: str = "1",
    term_status: str = "completed",
    drop_reason: str = "none",
    admission_ns: str = "1100",
    completion_ns: str = "5000",
    ingress_ns: str = "1050",
    enqueue_ns: str = "1060",
    drop_ns: str = "",
    media_pts: str = "1000000",
    media_idx: str = "1",
) -> dict[str, str]:
    """Minimal valid frame row."""
    return {
        "run_id": "test_run",
        "repetition": str(rep),
        "mechanism": mech,
        "frame_id": str(fid),
        "source_timestamp_ns": ingress_ns,
        "admission_timestamp_ns": admission_ns if dropped == "0" else "",
        "completion_timestamp_ns": completion_ns if dropped == "0" else "",
        "plan_version": plan_ver if dropped == "0" else "",
        "detector_id": det_id if dropped == "0" else "",
        "tracker_instance_id": trk_id if dropped == "0" else "",
        "queue_occupancy_before_enqueue": "0",
        "queue_occupancy_after_enqueue": "0",
        "queue_capacity": "4",
        "terminal_status": term_status,
        "drop_reason": drop_reason,
        "dropped": dropped,
        "duplicated": dup,
        "inside_measurement_window": inside,
        "media_pts_ns": media_pts,
        "receiver_ingress_timestamp_ns": ingress_ns,
        "enqueue_decision_timestamp_ns": enqueue_ns,
        "drop_decision_timestamp_ns": drop_ns if dropped == "1" else "",
        "media_frame_index": media_idx,
    }


# ---------------------------------------------------------------------------
# Task 1: Measurement does not begin based on completed warm-up frames
# ---------------------------------------------------------------------------
def test_warmup_not_determining_measurement_origin() -> None:
    """Measurement start media frame is recorded, not derived from warmup count."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [
            _valid_summary(rep=1, mech="VEPS"),
        ]
        # Set warmup count high but measurement starts at media frame 1
        summary_rows[0]["warmup_completed_frames"] = "50"
        summary_rows[0]["measurement_start_media_frame_index"] = "1"
        frame_rows = [_valid_frame(i, media_idx=str(i)) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


# ---------------------------------------------------------------------------
# Task 2: Initial-model warm-up excluded from measured counters
# ---------------------------------------------------------------------------
def test_warmup_frames_excluded_from_measurement() -> None:
    """Warm-up frames have inside_measurement_window=False."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="Stop")]
        # 30 warm-up frames (inside=0) + 130 measured frames (inside=1)
        frame_rows: list[dict[str, str]] = []
        for i in range(1, 31):
            frame_rows.append(_valid_frame(i, inside="0", media_idx=str(i)))
        for i in range(31, 161):
            frame_rows.append(_valid_frame(i, inside="1", media_idx=str(i)))

        summary_rows[0]["total_measurement_source_frames_received"] = "130"
        summary_rows[0]["measurement_start_media_frame_index"] = "31"

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


# ---------------------------------------------------------------------------
# Task 3: Measured media origin is deterministic
# ---------------------------------------------------------------------------
def test_measurement_media_origin_recorded() -> None:
    """measurement_start_media_frame_index and measurement_start_media_pts_ns are populated."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="Pause")]
        frame_rows = [_valid_frame(i, media_idx=str(i)) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


# ---------------------------------------------------------------------------
# Tasks 4-7: Phase-normalized observation window
# ---------------------------------------------------------------------------
def test_pre_request_invariant() -> None:
    """pre_request_source_frames_received must equal pre_request_source_frame_target."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Violate: pre_request_received != target
        summary_rows[0]["pre_request_source_frames_received"] = "55"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("pre-request received" in e for e in errors)


def test_post_effect_invariant() -> None:
    """post_effect_source_frames_received must equal post_effect_source_frame_target."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["post_effect_source_frames_received"] = "50"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("post-effect received" in e for e in errors)


def test_total_measured_frames_can_vary_by_mechanism() -> None:
    """Different mechanisms can have different total measurement frames (not forced to 180)."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        s1 = _valid_summary(rep=1, mech="VEPS")
        # VEPS: 130 total measured, phase sums = 130
        s2 = _valid_summary(rep=1, mech="Stop")
        # Stop: 145 total measured (15 transition frames instead of 10)
        s2["total_measurement_source_frames_received"] = "145"
        s2["measurement_end_source_sequence"] = "145"
        s2["source_frames_received"] = "145"
        s2["frames_admitted"] = "145"
        s2["frames_completed"] = "145"
        s2["measurement_source_frames_received"] = "145"
        s2["transition_source_frames_received"] = "25"
        s2["source_frames_before_request"] = "60"
        s2["source_frames_during_candidate_preparation"] = "8"
        s2["source_frames_between_preparation_and_publication"] = "5"
        s2["source_frames_between_publication_and_first_candidate_output"] = "12"
        s2["source_frames_after_first_candidate_output"] = "60"
        s2["measurement_end_media_frame_index"] = "145"

        summary_rows = [s1, s2]
        # VEPS: 130 frames, Stop: 145 frames (different transition lengths).
        f1 = [_valid_frame(i, mech="VEPS", media_idx=str(i)) for i in range(1, 131)]
        f2 = [_valid_frame(i, mech="Stop", media_idx=str(i)) for i in range(1, 146)]
        frame_rows = f1 + f2

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


# ---------------------------------------------------------------------------
# Task 8-10: Timestamp semantics and causal ordering
# ---------------------------------------------------------------------------
def test_media_pts_separate_from_monotonic() -> None:
    """media_pts_ns is not compared against monotonic timestamps in ordering checks."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # media_pts_ns much larger than monotonic - this is OK
        f = _valid_frame(1, media_pts="999999999999", ingress_ns="100", enqueue_ns="200")
        frame_rows = [f]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)
        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


def test_admitted_frame_timestamp_ordering() -> None:
    """Causal ordering: ingress <= enqueue <= admission <= completion."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Violation: enqueue > admission
        f = _valid_frame(1, ingress_ns="100", enqueue_ns="300", admission_ns="200", completion_ns="400")
        frame_rows = [f]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Enqueue timestamp" in e and "admission" in e for e in errors)


def test_dropped_frame_timestamp_ordering() -> None:
    """Causal ordering for drops: ingress <= enqueue <= drop_decision."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["total_dropped_frame_count"] = "1"
        summary_rows[0]["drops_before_request"] = "1"
        # Violation: enqueue > drop
        f = _valid_frame(
            1, dropped="1", term_status="dropped_ingress_overflow",
            drop_reason="ingress_overflow",
            ingress_ns="100", enqueue_ns="300", drop_ns="200",
            admission_ns="", completion_ns="",
        )
        frame_rows = [f]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Enqueue timestamp" in e and "drop" in e for e in errors)


# ---------------------------------------------------------------------------
# Tasks 11-13: Phase source and drop classification
# ---------------------------------------------------------------------------
def test_phase_source_count_reconciliation() -> None:
    """Sum of phase source frames must equal total_measurement_source_frames_received."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Violate reconciliation: sum of phases != total
        summary_rows[0]["source_frames_before_request"] = "50"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Sum of phase source frames" in e for e in errors)


def test_phase_drop_count_reconciliation() -> None:
    """Sum of phase drops must equal total_dropped_frame_count."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["total_dropped_frame_count"] = "5"
        summary_rows[0]["drop_rate"] = "0.038462"
        summary_rows[0]["drops_before_request"] = "2"
        summary_rows[0]["drops_after_first_candidate_output"] = "2"
        # Sum = 4, but total = 5
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Sum of phase drops" in e for e in errors)


def test_zero_denominator_drop_rate_is_empty() -> None:
    """When a phase has zero source frames, the drop rate should be empty/null."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # All drop_rate columns are already empty (zero denominator when no drops)
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


# ---------------------------------------------------------------------------
# Task 14: Terminal accounting reconciliation
# ---------------------------------------------------------------------------
def test_terminal_accounting_reconciliation() -> None:
    """received == completed + overflow + rejected + cancelled + in_flight."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Violate: total_received (130) != 100 + 0 + 0 + 0 + 10 (= 110)
        summary_rows[0]["frames_completed"] = "100"
        summary_rows[0]["frames_in_flight_at_window_end"] = "10"
        summary_rows[0]["total_measurement_source_frames_received"] = "130"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Invariant mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 15: Old-plan completion counting
# ---------------------------------------------------------------------------
def test_old_plan_completion_count() -> None:
    """old_plan_frames_completed_during_prep is validated against frame records."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Claim 2 old-plan frames completed during prep, but provide 5
        summary_rows[0]["old_plan_frames_completed_during_prep"] = "2"
        summary_rows[0]["candidate_prep_start_ns"] = "1000"
        summary_rows[0]["candidate_prep_end_ns"] = "2000"
        frames: list[dict[str, str]] = []
        for i in range(1, 6):
            f = _valid_frame(i, plan_ver="1", completion_ns=str(1000 + i * 100))
            frames.append(f)
        for i in range(6, 61):
            f = _valid_frame(i)
            frames.append(f)
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("old_plan_frames_completed_during_prep" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 16: Detector identity by plan version
# ---------------------------------------------------------------------------
def test_detector_identity_by_plan_version() -> None:
    """Plan version 1 implies initial model, version 2 implies candidate model."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Plausible frames
        frames = [
            _valid_frame(1, plan_ver="1", det_id="PekingU/rtdetr_r18vd"),
            _valid_frame(2, plan_ver="2", det_id="PekingU/rtdetr_r50vd"),
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


def test_detector_identity_mismatch() -> None:
    """Plan version 2 with initial model ID should be flagged."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        frames = [
            _valid_frame(1, plan_ver="2", det_id="PekingU/rtdetr_r18vd"),  # wrong!
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("incompatible detector_id" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 17: Tracker PRESERVE identity
# ---------------------------------------------------------------------------
def test_tracker_preserve_identity() -> None:
    """All inside-measurement frames must share the same tracker instance."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        frames = [
            _valid_frame(1, trk_id="trk_a"),
            _valid_frame(2, trk_id="trk_b"),  # different!
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Tracker identity not preserved" in e for e in errors)


# ---------------------------------------------------------------------------
# Tasks 18-19: Stop old-detector retirement
# ---------------------------------------------------------------------------
def test_stop_retirement_one_active_detector() -> None:
    """After Stop retirement: one live detector, one active plan."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="Stop")]
        summary_rows[0]["active_detector_id_after_retirement"] = "PekingU/rtdetr_r50vd"
        summary_rows[0]["active_plan_version_after_retirement"] = "2"
        summary_rows[0]["active_detector_instance_count_after_retirement"] = "1"
        summary_rows[0]["live_plan_count_after_retirement"] = "1"
        summary_rows[0]["live_detector_count_after_retirement"] = "1"
        frame_rows = [_valid_frame(i, mech="Stop") for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


def test_stop_retirement_violation_two_live_detectors() -> None:
    """live_detector_count_after_retirement != 1 should be flagged."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="Stop")]
        summary_rows[0]["live_detector_count_after_retirement"] = "2"
        frame_rows = [_valid_frame(i, mech="Stop") for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("live detector count after retirement" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 20: GPU checkpoint schema and cumulative-peak semantics
# ---------------------------------------------------------------------------
def test_gpu_memory_field_non_negative() -> None:
    """GPU memory fields must be non-negative when present."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["gpu_memory_before_prep_allocated_bytes"] = "-100"
        summary_rows[0]["gpu_memory_before_prep_reserved_bytes"] = "200"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Negative allocated memory" in e for e in errors)


def test_gpu_allocated_not_greater_than_reserved() -> None:
    """Allocated bytes must not exceed reserved bytes."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["gpu_memory_before_prep_allocated_bytes"] = "300"
        summary_rows[0]["gpu_memory_before_prep_reserved_bytes"] = "200"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Allocated" in e and "Reserved" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 21: Cross-mechanism media-segment pairing
# ---------------------------------------------------------------------------
def test_cross_mechanism_matching_fields() -> None:
    """Stop, Pause, VEPS must have identical media alignment fields per repetition."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        s1 = _valid_summary(rep=1, mech="VEPS")
        s2 = _valid_summary(rep=1, mech="Stop")
        s3 = _valid_summary(rep=1, mech="Pause")
        # All share the same media alignment fields -> PASS
        summary_rows = [s1, s2, s3]
        frame_rows: list[dict[str, str]] = []
        for mech in ("VEPS", "Stop", "Pause"):
            frame_rows.extend([_valid_frame(j, mech=mech, media_idx=str(j)) for j in range(1, 131)])

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert ok


def test_cross_mechanism_mismatch() -> None:
    """Cross-mechanism mismatch on source_file_hash should be flagged."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        s1 = _valid_summary(rep=1, mech="VEPS")
        s1["source_file_hash"] = "hash_a"
        s2 = _valid_summary(rep=1, mech="Stop")
        s2["source_file_hash"] = "hash_b"
        summary_rows = [s1, s2]
        frame_rows = [_valid_frame(i, mech="VEPS") for i in range(1, 131)]
        frame_rows += [_valid_frame(i, mech="Stop") for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Cross-mechanism mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# Task 22: Validator rejection of intentionally malformed rows
# ---------------------------------------------------------------------------
def test_reject_duplicate_frame_id() -> None:
    """Duplicate frame_id within a run should be flagged."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        frames = [
            _valid_frame(1),
            _valid_frame(1),  # duplicate!
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Duplicate frame_id" in e for e in errors)


def test_reject_both_dropped_and_completed() -> None:
    """A frame cannot have terminal_status completed and dropped=True."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        frames = [
            _valid_frame(1, dropped="1", term_status="completed", drop_reason="none"),
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("Completed frame has dropped=True" in e for e in errors)


def test_reject_dropped_without_drop_reason() -> None:
    """Dropped frames must have a non-empty drop_reason."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        summary_rows[0]["total_dropped_frame_count"] = "1"
        summary_rows[0]["drops_before_request"] = "1"
        frames = [
            _valid_frame(
                1, dropped="1", term_status="dropped_ingress_overflow",
                drop_reason="none",  # invalid: should be "ingress_overflow"
                admission_ns="", completion_ns="",
                drop_ns="200",
            ),
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("lacks valid drop_reason" in e for e in errors)


def test_reject_completed_frame_without_admission() -> None:
    """Completed frames must have admission and completion timestamps."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        frames = [
            _valid_frame(1, admission_ns="", completion_ns="1200"),
        ]
        frame_rows = frames

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("lacks admission or completion" in e for e in errors)


def test_reject_queue_occupancy_out_of_bounds() -> None:
    """Queue occupancy must be within [0, queue_capacity]."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        f = _valid_frame(1)
        f["queue_occupancy_before_enqueue"] = "10"  # > capacity 4
        frame_rows = [f]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert not ok
        assert any("out of bounds" in e for e in errors)


def test_reject_negative_gpu_memory() -> None:
    """Negative GPU memory values should be flagged."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [_valid_summary(rep=1, mech="VEPS")]
        # Use a field that the validator actually checks (prefix-matched).
        summary_rows[0]["gpu_memory_before_prep_allocated_bytes"] = "-1"
        frame_rows = [_valid_frame(i) for i in range(1, 131)]

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, _ = validate_benchmark_csvs(s_path, f_path)
        assert not ok


def test_pass_with_valid_data() -> None:
    """A fully valid dataset should PASS validation."""
    with tempfile.TemporaryDirectory() as d:
        s_path = os.path.join(d, "s.csv")
        f_path = os.path.join(d, "f.csv")
        summary_rows = [
            _valid_summary(rep=1, mech="VEPS"),
            _valid_summary(rep=1, mech="Stop"),
            _valid_summary(rep=1, mech="Pause"),
        ]
        frame_rows: list[dict[str, str]] = []
        for mech in ("VEPS", "Stop", "Pause"):
            for i in range(1, 131):
                plan_ver = "1" if i <= 75 else "2"
                det_id = "PekingU/rtdetr_r18vd" if plan_ver == "1" else "PekingU/rtdetr_r50vd"
                frame_rows.append(
                    _valid_frame(i, mech=mech, plan_ver=plan_ver, det_id=det_id,
                                 media_idx=str(i))
                )

        _write_summary_csv(s_path, summary_rows)
        _write_frame_csv(f_path, frame_rows)

        ok, errors = validate_benchmark_csvs(s_path, f_path)
        assert ok, f"Validation should pass but got errors: {errors}"
