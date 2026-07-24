"""Unit tests for E6 active-path interference campaign (Sections 6, 7, 8, 9, 10, 12)."""

from __future__ import annotations

import os
import tempfile

from benchmarks.model import (
    InterferenceFrameSampleRow,
    InterferenceSampleRow,
)
from benchmarks.reporting.interference import (
    calculate_interference_contrasts,
    generate_interference_summary_file,
)
from benchmarks.runners.interference import run_cpu_bound_workload
from benchmarks.storage import (
    append_interference_frame_rows,
    append_interference_rows,
    read_interference_frame_rows,
    read_interference_rows,
    write_interference_frame_header,
    write_interference_header,
)


def test_cpu_bound_workload_execution() -> None:
    """Verify CPU-bound preparation workload runs until deadline and produces checksum."""
    target_ns = 5_000_000  # 5 ms test duration
    elapsed_ns, checksum = run_cpu_bound_workload(target_ns)

    assert elapsed_ns >= target_ns
    assert isinstance(checksum, int)
    assert checksum != 0


def test_producer_pacing_and_lateness() -> None:
    """6, 12D. Verify producer pacing, deadline skipping, and schedule lateness recording."""
    row = InterferenceSampleRow(
        run_id="run_pacing",
        scenario_id="scen_pacing",
        preparation_category="sleep_preparation",
        target_window_ns=200_000_000,
        repetition=1,
        execution_order_position=1,
        random_seed=42,
        configured_offered_rate_fps=1000.0,
        before_window_start_ns=1_000_000_000,
        before_window_end_ns=2_000_000_000,
        during_window_start_ns=2_000_000_000,
        during_window_end_ns=2_200_000_000,
        after_window_start_ns=2_200_000_000,
        after_window_end_ns=3_200_000_000,
        measured_before_window_duration_ns=1_000_000_000,
        measured_during_window_duration_ns=200_000_000,
        measured_after_window_duration_ns=1_000_000_000,
        producer_attempted_frames_before=1000,
        producer_attempted_frames_during=200,
        producer_attempted_frames_after=1000,
        producer_emitted_frames_before=1000,
        producer_emitted_frames_during=200,
        producer_emitted_frames_after=1000,
        producer_skipped_deadlines_before=0,
        producer_skipped_deadlines_during=0,
        producer_skipped_deadlines_after=0,
        producer_schedule_lateness_median_ns_before=500.0,
        producer_schedule_lateness_p95_ns_before=1200.0,
        producer_schedule_lateness_median_ns_during=600.0,
        producer_schedule_lateness_p95_ns_during=1500.0,
        producer_schedule_lateness_median_ns_after=500.0,
        producer_schedule_lateness_p95_ns_after=1200.0,
        admitted_frames_before=1000,
        admitted_frames_during=200,
        admitted_frames_after=1000,
        completed_frames_before=1000,
        completed_frames_during=200,
        completed_frames_after=1000,
        frame_latency_before_median_ns=50000.0,
        frame_latency_before_p95_ns=80000.0,
        frame_latency_during_median_ns=60000.0,
        frame_latency_during_p95_ns=100000.0,
        frame_latency_after_median_ns=50000.0,
        frame_latency_after_p95_ns=80000.0,
        p95_degradation_vs_before_pct=25.0,
        median_degradation_vs_before_pct=20.0,
        frames_completed_during_window=200,
        frames_completed_while_preparation_active=200,
        old_plan_frames_completed=200,
        admission_throughput_before_fps=1000.0,
        admission_throughput_during_fps=1000.0,
        admission_throughput_after_fps=1000.0,
        completion_throughput_before_fps=1000.0,
        completion_throughput_during_fps=1000.0,
        completion_throughput_after_fps=1000.0,
        admission_rate_ratio_vs_before=1.0,
        completion_rate_ratio_vs_before=1.0,
        latency_interpretation_confounded=False,
        latency_interpretation_confounded_reason=None,
        queue_occupancy_during_median=0.0,
        queue_occupancy_during_p95=0.0,
        dropped_frames=0,
        duplicated_frames=0,
        valid_for_tail_latency=True,
        invalid_reason=None,
        target_preparation_duration_ns=200_000_000,
        actual_preparation_duration_ns=205_000_000,
        preparation_checksum=None,
        duration_valid=True,
        duration_relative_error=0.025,
        raw_frame_count_matches_summary=True,
        phase_timestamp_bounds_valid=True,
        throughput_accounting_valid=True,
        throughput_accounting_invalid_reason=None,
    )

    assert row.producer_emitted_frames_before == 1000
    assert row.producer_skipped_deadlines_during == 0
    assert row.throughput_accounting_valid is True


def test_matched_control_contrasts() -> None:
    """9, 12G. Verify matched no-candidate control contrasts and confounding flag calculation."""
    rows: list[InterferenceSampleRow] = []

    # Control row: no_candidate_preparation
    ctrl_row = InterferenceSampleRow(
        run_id="run_ctrl",
        scenario_id="interference_no_candidate_200ms",
        preparation_category="no_candidate_preparation",
        target_window_ns=200_000_000,
        repetition=1,
        execution_order_position=1,
        random_seed=42,
        configured_offered_rate_fps=1000.0,
        before_window_start_ns=1_000_000_000,
        before_window_end_ns=2_000_000_000,
        during_window_start_ns=2_000_000_000,
        during_window_end_ns=2_200_000_000,
        after_window_start_ns=2_200_000_000,
        after_window_end_ns=3_200_000_000,
        measured_before_window_duration_ns=1_000_000_000,
        measured_during_window_duration_ns=200_000_000,
        measured_after_window_duration_ns=1_000_000_000,
        producer_attempted_frames_before=1000,
        producer_attempted_frames_during=200,
        producer_attempted_frames_after=1000,
        producer_emitted_frames_before=1000,
        producer_emitted_frames_during=200,
        producer_emitted_frames_after=1000,
        producer_skipped_deadlines_before=0,
        producer_skipped_deadlines_during=0,
        producer_skipped_deadlines_after=0,
        producer_schedule_lateness_median_ns_before=500.0,
        producer_schedule_lateness_p95_ns_before=1200.0,
        producer_schedule_lateness_median_ns_during=500.0,
        producer_schedule_lateness_p95_ns_during=1200.0,
        producer_schedule_lateness_median_ns_after=500.0,
        producer_schedule_lateness_p95_ns_after=1200.0,
        admitted_frames_before=1000,
        admitted_frames_during=200,
        admitted_frames_after=1000,
        completed_frames_before=1000,
        completed_frames_during=200,
        completed_frames_after=1000,
        frame_latency_before_median_ns=50000.0,
        frame_latency_before_p95_ns=80000.0,
        frame_latency_during_median_ns=50000.0,
        frame_latency_during_p95_ns=80000.0,
        frame_latency_after_median_ns=50000.0,
        frame_latency_after_p95_ns=80000.0,
        p95_degradation_vs_before_pct=0.0,
        median_degradation_vs_before_pct=0.0,
        frames_completed_during_window=200,
        frames_completed_while_preparation_active=None,  # None for no-candidate
        old_plan_frames_completed=200,
        admission_throughput_before_fps=1000.0,
        admission_throughput_during_fps=1000.0,
        admission_throughput_after_fps=1000.0,
        completion_throughput_before_fps=1000.0,
        completion_throughput_during_fps=1000.0,
        completion_throughput_after_fps=1000.0,
        admission_rate_ratio_vs_before=1.0,
        completion_rate_ratio_vs_before=1.0,
        latency_interpretation_confounded=False,
        latency_interpretation_confounded_reason=None,
        queue_occupancy_during_median=0.0,
        queue_occupancy_during_p95=0.0,
        dropped_frames=0,
        duplicated_frames=0,
        valid_for_tail_latency=True,
        invalid_reason=None,
        target_preparation_duration_ns=None,  # None for no-candidate
        actual_preparation_duration_ns=0,
        preparation_checksum=None,
        duration_valid=True,
        duration_relative_error=0.0,
    )

    # Treatment row: sleep_preparation
    treat_row = InterferenceSampleRow(
        run_id="run_ctrl",
        scenario_id="interference_sleep_200ms",
        preparation_category="sleep_preparation",
        target_window_ns=200_000_000,
        repetition=1,
        execution_order_position=2,
        random_seed=42,
        configured_offered_rate_fps=1000.0,
        before_window_start_ns=1_000_000_000,
        before_window_end_ns=2_000_000_000,
        during_window_start_ns=2_000_000_000,
        during_window_end_ns=2_200_000_000,
        after_window_start_ns=2_200_000_000,
        after_window_end_ns=3_200_000_000,
        measured_before_window_duration_ns=1_000_000_000,
        measured_during_window_duration_ns=200_000_000,
        measured_after_window_duration_ns=1_000_000_000,
        producer_attempted_frames_before=1000,
        producer_attempted_frames_during=200,
        producer_attempted_frames_after=1000,
        producer_emitted_frames_before=1000,
        producer_emitted_frames_during=200,
        producer_emitted_frames_after=1000,
        producer_skipped_deadlines_before=0,
        producer_skipped_deadlines_during=0,
        producer_skipped_deadlines_after=0,
        producer_schedule_lateness_median_ns_before=500.0,
        producer_schedule_lateness_p95_ns_before=1200.0,
        producer_schedule_lateness_median_ns_during=600.0,
        producer_schedule_lateness_p95_ns_during=1500.0,
        producer_schedule_lateness_median_ns_after=500.0,
        producer_schedule_lateness_p95_ns_after=1200.0,
        admitted_frames_before=1000,
        admitted_frames_during=200,
        admitted_frames_after=1000,
        completed_frames_before=1000,
        completed_frames_during=200,
        completed_frames_after=1000,
        frame_latency_before_median_ns=50000.0,
        frame_latency_before_p95_ns=80000.0,
        frame_latency_during_median_ns=60000.0,
        frame_latency_during_p95_ns=100000.0,
        frame_latency_after_median_ns=50000.0,
        frame_latency_after_p95_ns=80000.0,
        p95_degradation_vs_before_pct=25.0,
        median_degradation_vs_before_pct=20.0,
        frames_completed_during_window=200,
        frames_completed_while_preparation_active=200,
        old_plan_frames_completed=200,
        admission_throughput_before_fps=1000.0,
        admission_throughput_during_fps=1000.0,
        admission_throughput_after_fps=1000.0,
        completion_throughput_before_fps=1000.0,
        completion_throughput_during_fps=1000.0,
        completion_throughput_after_fps=1000.0,
        admission_rate_ratio_vs_before=1.0,
        completion_rate_ratio_vs_before=1.0,
        latency_interpretation_confounded=False,
        latency_interpretation_confounded_reason=None,
        queue_occupancy_during_median=0.0,
        queue_occupancy_during_p95=0.0,
        dropped_frames=0,
        duplicated_frames=0,
        valid_for_tail_latency=True,
        invalid_reason=None,
        target_preparation_duration_ns=200_000_000,
        actual_preparation_duration_ns=205_000_000,
        preparation_checksum=None,
        duration_valid=True,
        duration_relative_error=0.025,
    )

    rows.extend([ctrl_row, treat_row])

    contrast_rows, agg_rows = calculate_interference_contrasts(rows)

    assert len(contrast_rows) == 1
    c = contrast_rows[0]
    assert c.treatment_category == "sleep_preparation"
    assert c.control_category == "no_candidate_preparation"
    assert c.treatment_during_p95_ns == 100000.0
    assert c.control_during_p95_ns == 80000.0
    assert c.paired_p95_difference_ns == 20000.0
    assert c.matched_control_confounded is False
    assert c.within_run_admission_shift_flag is False

    assert len(agg_rows) == 2  # all_pairs and non_confounded_pairs
    agg = agg_rows[0]
    assert agg.treatment_category == "sleep_preparation"
    assert agg.total_pair_count == 1
    assert agg.median_paired_p95_ratio == 1.25


def test_interference_csv_roundtrip() -> None:
    """Roundtrip CSV storage for raw frame samples and repetition samples."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_csv = f"{tmp_dir}/interference-frame-samples.csv"
        rep_csv = f"{tmp_dir}/interference-samples.csv"

        write_interference_frame_header(frame_csv)
        write_interference_header(rep_csv)

        frame_sample = InterferenceFrameSampleRow(
            run_id="run_f",
            scenario_id="scen_f",
            preparation_category="cpu_bound_preparation",
            target_window_ns=500_000_000,
            repetition=1,
            phase="during",
            frame_id=42,
            plan_version=1,
            admission_timestamp_ns=1000,
            completion_timestamp_ns=2000,
            frame_latency_ns=1000,
            queue_occupancy=0.05,
            dropped=False,
            duplicated=False,
        )
        append_interference_frame_rows(frame_csv, [frame_sample])
        read_frames = read_interference_frame_rows(frame_csv)
        assert len(read_frames) == 1
        assert read_frames[0].frame_id == 42
        assert read_frames[0].phase == "during"

        rep_sample = InterferenceSampleRow(
            run_id="run_f",
            scenario_id="scen_f",
            preparation_category="cpu_bound_preparation",
            target_window_ns=500_000_000,
            repetition=1,
            execution_order_position=2,
            random_seed=42,
            configured_offered_rate_fps=1000.0,
            before_window_start_ns=100,
            before_window_end_ns=200,
            during_window_start_ns=200,
            during_window_end_ns=300,
            after_window_start_ns=300,
            after_window_end_ns=400,
            measured_before_window_duration_ns=1_000_000_000,
            measured_during_window_duration_ns=500_000_000,
            measured_after_window_duration_ns=1_000_000_000,
            producer_attempted_frames_before=1000,
            producer_attempted_frames_during=500,
            producer_attempted_frames_after=1000,
            producer_emitted_frames_before=1000,
            producer_emitted_frames_during=500,
            producer_emitted_frames_after=1000,
            producer_skipped_deadlines_before=0,
            producer_skipped_deadlines_during=0,
            producer_skipped_deadlines_after=0,
            producer_schedule_lateness_median_ns_before=10.0,
            producer_schedule_lateness_p95_ns_before=20.0,
            producer_schedule_lateness_median_ns_during=10.0,
            producer_schedule_lateness_p95_ns_during=20.0,
            producer_schedule_lateness_median_ns_after=10.0,
            producer_schedule_lateness_p95_ns_after=20.0,
            admitted_frames_before=1000,
            admitted_frames_during=500,
            admitted_frames_after=1000,
            completed_frames_before=1000,
            completed_frames_during=500,
            completed_frames_after=1000,
            frame_latency_before_median_ns=50000.0,
            frame_latency_before_p95_ns=80000.0,
            frame_latency_during_median_ns=55000.0,
            frame_latency_during_p95_ns=90000.0,
            frame_latency_after_median_ns=50000.0,
            frame_latency_after_p95_ns=80000.0,
            p95_degradation_vs_before_pct=12.5,
            median_degradation_vs_before_pct=10.0,
            frames_completed_during_window=500,
            frames_completed_while_preparation_active=500,
            old_plan_frames_completed=500,
            admission_throughput_before_fps=1000.0,
            admission_throughput_during_fps=1000.0,
            admission_throughput_after_fps=1000.0,
            completion_throughput_before_fps=1000.0,
            completion_throughput_during_fps=1000.0,
            completion_throughput_after_fps=1000.0,
            admission_rate_ratio_vs_before=1.0,
            completion_rate_ratio_vs_before=1.0,
            latency_interpretation_confounded=False,
            latency_interpretation_confounded_reason=None,
            queue_occupancy_during_median=0.0,
            queue_occupancy_during_p95=0.0,
            dropped_frames=0,
            duplicated_frames=0,
            valid_for_tail_latency=True,
            invalid_reason=None,
            target_preparation_duration_ns=500_000_000,
            actual_preparation_duration_ns=505_000_000,
            preparation_checksum=123456789,
            duration_valid=True,
            duration_relative_error=0.010,
        )
        append_interference_rows(rep_csv, [rep_sample])
        read_reps = read_interference_rows(rep_csv)
        assert len(read_reps) == 1
        assert read_reps[0].preparation_checksum == 123456789
        assert read_reps[0].target_window_ns == 500_000_000

        summary_csv = f"{tmp_dir}/interference-summary.csv"
        generate_interference_summary_file(rep_csv, summary_csv)
        assert os.path.exists(summary_csv)
        assert os.path.exists(f"{tmp_dir}/interference-contrasts.csv")
        assert os.path.exists(f"{tmp_dir}/interference-contrast-summary.csv")


def test_target_window_matching_isolation() -> None:
    """12C. Verify 200ms treatments match only 200ms controls, and 500ms treatments match only 500ms controls."""
    rows: list[InterferenceSampleRow] = []

    for target_ns in (200_000_000, 500_000_000):
        for cat in ("no_candidate_preparation", "sleep_preparation"):
            rows.append(
                InterferenceSampleRow(
                    run_id="run_m",
                    scenario_id=f"scen_{cat}_{target_ns}",
                    preparation_category=cat,
                    target_window_ns=target_ns,
                    repetition=1,
                    execution_order_position=1,
                    random_seed=42,
                    configured_offered_rate_fps=1000.0,
                    before_window_start_ns=100,
                    before_window_end_ns=200,
                    during_window_start_ns=200,
                    during_window_end_ns=200 + target_ns,
                    after_window_start_ns=200 + target_ns,
                    after_window_end_ns=300 + target_ns,
                    measured_before_window_duration_ns=100,
                    measured_during_window_duration_ns=target_ns,
                    measured_after_window_duration_ns=100,
                    producer_attempted_frames_before=10,
                    producer_attempted_frames_during=20,
                    producer_attempted_frames_after=10,
                    producer_emitted_frames_before=10,
                    producer_emitted_frames_during=20,
                    producer_emitted_frames_after=10,
                    producer_skipped_deadlines_before=0,
                    producer_skipped_deadlines_during=0,
                    producer_skipped_deadlines_after=0,
                    producer_schedule_lateness_median_ns_before=0.0,
                    producer_schedule_lateness_p95_ns_before=0.0,
                    producer_schedule_lateness_median_ns_during=0.0,
                    producer_schedule_lateness_p95_ns_during=0.0,
                    producer_schedule_lateness_median_ns_after=0.0,
                    producer_schedule_lateness_p95_ns_after=0.0,
                    admitted_frames_before=10,
                    admitted_frames_during=20,
                    admitted_frames_after=10,
                    completed_frames_before=10,
                    completed_frames_during=20,
                    completed_frames_after=10,
                    frame_latency_before_median_ns=1000.0,
                    frame_latency_before_p95_ns=2000.0,
                    frame_latency_during_median_ns=1000.0 if cat == "no_candidate_preparation" else 1500.0,
                    frame_latency_during_p95_ns=2000.0 if cat == "no_candidate_preparation" else 3000.0,
                    frame_latency_after_median_ns=1000.0,
                    frame_latency_after_p95_ns=2000.0,
                    p95_degradation_vs_before_pct=0.0,
                    median_degradation_vs_before_pct=0.0,
                    frames_completed_during_window=20,
                    frames_completed_while_preparation_active=20 if cat != "no_candidate_preparation" else None,
                    old_plan_frames_completed=20,
                    admission_throughput_before_fps=1000.0,
                    admission_throughput_during_fps=1000.0,
                    admission_throughput_after_fps=1000.0,
                    completion_throughput_before_fps=1000.0,
                    completion_throughput_during_fps=1000.0,
                    completion_throughput_after_fps=1000.0,
                    admission_rate_ratio_vs_before=1.0,
                    completion_rate_ratio_vs_before=1.0,
                    latency_interpretation_confounded=False,
                    latency_interpretation_confounded_reason=None,
                    queue_occupancy_during_median=0.0,
                    queue_occupancy_during_p95=0.0,
                    dropped_frames=0,
                    duplicated_frames=0,
                    valid_for_tail_latency=True,
                    invalid_reason=None,
                    target_preparation_duration_ns=target_ns if cat != "no_candidate_preparation" else None,
                    actual_preparation_duration_ns=target_ns,
                    preparation_checksum=None,
                    duration_valid=True,
                    duration_relative_error=0.0,
                )
            )

    contrast_rows, _ = calculate_interference_contrasts(rows)
    assert len(contrast_rows) == 2
    pair_200 = next(c for c in contrast_rows if c.target_window_ns == 200_000_000)
    pair_500 = next(c for c in contrast_rows if c.target_window_ns == 500_000_000)

    assert pair_200.target_window_ns == 200_000_000
    assert pair_500.target_window_ns == 500_000_000
    assert pair_200.paired_p95_ratio == 1.5
    assert pair_500.paired_p95_ratio == 1.5


def test_backlog_and_flow_accounting_invariant() -> None:
    """12E. Verify backlog tracking and flow accounting invariant."""
    row = InterferenceSampleRow(
        run_id="run_b",
        scenario_id="scen_backlog",
        preparation_category="cpu_bound_preparation",
        target_window_ns=200_000_000,
        repetition=1,
        execution_order_position=1,
        random_seed=42,
        configured_offered_rate_fps=1000.0,
        before_window_start_ns=100,
        before_window_end_ns=200,
        during_window_start_ns=200,
        during_window_end_ns=400,
        after_window_start_ns=400,
        after_window_end_ns=500,
        measured_before_window_duration_ns=100,
        measured_during_window_duration_ns=200,
        measured_after_window_duration_ns=100,
        producer_attempted_frames_before=100,
        producer_attempted_frames_during=200,
        producer_attempted_frames_after=100,
        producer_emitted_frames_before=100,
        producer_emitted_frames_during=200,
        producer_emitted_frames_after=100,
        producer_skipped_deadlines_before=0,
        producer_skipped_deadlines_during=0,
        producer_skipped_deadlines_after=0,
        producer_schedule_lateness_median_ns_before=0.0,
        producer_schedule_lateness_p95_ns_before=0.0,
        producer_schedule_lateness_median_ns_during=0.0,
        producer_schedule_lateness_p95_ns_during=0.0,
        producer_schedule_lateness_median_ns_after=0.0,
        producer_schedule_lateness_p95_ns_after=0.0,
        admitted_frames_before=100,
        admitted_frames_during=180,
        admitted_frames_after=120,
        completed_frames_before=100,
        completed_frames_during=180,
        completed_frames_after=120,
        frame_latency_before_median_ns=1000.0,
        frame_latency_before_p95_ns=2000.0,
        frame_latency_during_median_ns=1200.0,
        frame_latency_during_p95_ns=2500.0,
        frame_latency_after_median_ns=1000.0,
        frame_latency_after_p95_ns=2000.0,
        p95_degradation_vs_before_pct=25.0,
        median_degradation_vs_before_pct=20.0,
        frames_completed_during_window=180,
        frames_completed_while_preparation_active=180,
        old_plan_frames_completed=180,
        admission_throughput_before_fps=1000.0,
        admission_throughput_during_fps=900.0,
        admission_throughput_after_fps=1200.0,
        completion_throughput_before_fps=1000.0,
        completion_throughput_during_fps=900.0,
        completion_throughput_after_fps=1200.0,
        admission_rate_ratio_vs_before=0.9,
        completion_rate_ratio_vs_before=0.9,
        latency_interpretation_confounded=True,
        latency_interpretation_confounded_reason="Admission rate shifted",
        within_run_admission_shift_flag=True,
        within_run_admission_shift_reason="Admission rate shifted",
        backlog_frames_at_during_start=0,
        backlog_frames_at_during_end=20,
        backlog_frames_drained_after_window=20,
        frames_emitted_during_but_admitted_after=20,
        frames_admitted_during_but_emitted_before=0,
        flow_accounting_valid=True,
        flow_accounting_residual_frames=0,
        flow_accounting_invalid_reason=None,
    )

    # flow invariant: initial_backlog(0) + emitted(200) == completed(180) + dropped(0) + final_backlog(20)
    assert row.backlog_frames_at_during_start + row.producer_emitted_frames_during == (
        row.completed_frames_during + row.dropped_frames + row.backlog_frames_at_during_end
    )
    assert row.flow_accounting_valid is True
    assert row.flow_accounting_residual_frames == 0

