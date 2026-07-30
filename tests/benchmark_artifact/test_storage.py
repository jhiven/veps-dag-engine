"""Tests for typed CSV storage and serialization."""

from __future__ import annotations

import tempfile
from benchmarks.model import (
    ConformanceResultRow,
    ReconfigurationSampleRow,
    SteadyStateSampleRow,
)
from benchmarks.storage import (
    append_conformance_rows,
    append_reconfiguration_rows,
    append_steady_state_rows,
    read_conformance_rows,
    read_reconfiguration_rows,
    read_steady_state_rows,
    write_conformance_header,
    write_reconfiguration_header,
    write_steady_state_header,
)


def test_steady_state_storage_roundtrip() -> None:
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        path = f.name

    write_steady_state_header(path)

    row1 = SteadyStateSampleRow(
        run_id="run_1",
        scenario_id="steady_linear_5_minimal",
        topology="linear_5",
        workload_id="minimal",
        implementation="hard_coded",
        repetition=1,
        execution_order_position=1,
        operations=100,
        elapsed_ns=10000,
        normalized_ns_per_frame=100.0,
        frames_completed=100,
        plan_version=1,
    )
    row2 = SteadyStateSampleRow(
        run_id="run_1",
        scenario_id="steady_linear_5_minimal",
        topology="linear_5",
        workload_id="minimal",
        implementation="versioned_compiled",
        repetition=1,
        execution_order_position=2,
        operations=100,
        elapsed_ns=12000,
        normalized_ns_per_frame=120.0,
        frames_completed=100,
        plan_version=1,
    )

    append_steady_state_rows(path, [row1])
    append_steady_state_rows(path, [row2])

    read_rows = read_steady_state_rows(path)
    assert len(read_rows) == 2
    assert read_rows[0] == row1
    assert read_rows[1] == row2


def test_reconfiguration_storage_roundtrip() -> None:
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        path = f.name

    write_reconfiguration_header(path)

    row = ReconfigurationSampleRow(
        run_id="run_1",
        scenario_id="reconfig_insert",
        baseline="prepare_and_commit",
        edit_type="insert_stateless_node",
        graph_size=10,
        repetition=1,
        old_plan_version=1,
        new_plan_version=2,
        terminal_status="COMMITTED",
        validation_ns=1000,
        preparation_ns=5000,
        request_to_ready_ns=6000,
        boundary_wait_ns=500,
        commit_ns=200,
        request_to_effect_ns=7000,
        retirement_queue_delay_ns=100,
        retirement_duration_ns=200,
        commit_to_retirement_complete_ns=300,
        maximum_output_gap_ns=5000,
        transition_output_gap_ns=1200,
        frames_completed_during_request=5,
        old_plan_frames_admitted_after_request_before_commit=5,
        frames_dropped=0,
        frames_duplicated=0,
        reused_processor_count=9,
        staged_processor_count=1,
        retired_processor_count=0,
        state_transition_policy="AUTO",
        admission_stop_ns=10,
        executor_teardown_ns=20,
        executor_reconstruction_ns=30,
        publication_ns=40,
        executor_restart_ns=50,
        first_admission_wait_ns=60,
        first_completion_wait_ns=70,
        retirement_ns=80,
        total_synchronous_ns=90,
        instrumented_phase_sum_ns=360,
        unattributed_request_time_ns=5,
    )

    append_reconfiguration_rows(path, [row])
    read_rows = read_reconfiguration_rows(path)
    assert len(read_rows) == 1
    assert read_rows[0] == row


def test_conformance_storage_roundtrip() -> None:
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        path = f.name

    write_conformance_header(path)

    row = ConformanceResultRow(
        run_id="run_1",
        campaign_id="frame_consistency",
        scenario_id="conformance_stress",
        scenario_type="stress",
        frames_submitted=100,
        frames_completed=100,
        reconfigurations_requested=10,
        reconfigurations_committed=10,
        reconfigurations_rejected=0,
        reconfigurations_failed=0,
        mixed_plan_frames=0,
        missing_frames=0,
        duplicate_frames=0,
        retired_plan_executions=0,
        invalid_routing_events=0,
        state_continuity_failures=0,
        unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0,
        processor_instance_leaks=0,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
        terminal_status="PASS",
    )

    append_conformance_rows(path, [row])
    read_rows = read_conformance_rows(path)
    assert len(read_rows) == 1
    assert read_rows[0] == row
