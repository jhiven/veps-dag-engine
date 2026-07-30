"""Tests for report generation and artifact verification."""

from __future__ import annotations

import os
import tempfile
import pytest

from benchmarks.cli import verify_cmd
from benchmarks.model import (
    ConformanceResultRow,
    ReconfigurationSampleRow,
    SteadyStateSampleRow,
)
from benchmarks.reporting.figures import generate_all_figures
from benchmarks.reporting.load import load_benchmark_artifact
from benchmarks.reporting.summarize import generate_summary_csv
from benchmarks.reporting.tables import generate_all_tables
from benchmarks.storage import (
    append_conformance_rows,
    append_reconfiguration_rows,
    append_steady_state_rows,
    write_conformance_header,
    write_reconfiguration_header,
    write_steady_state_header,
)


def test_reporting_from_fixture_csvs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        steady_path = os.path.join(tmp_dir, "steady-state-samples.csv")
        reconfig_path = os.path.join(tmp_dir, "reconfiguration-samples.csv")
        conform_path = os.path.join(tmp_dir, "conformance-results.csv")

        write_steady_state_header(steady_path)
        write_reconfiguration_header(reconfig_path)
        write_conformance_header(conform_path)

        append_steady_state_rows(
            steady_path,
            [
                SteadyStateSampleRow(
                    run_id="r1",
                    scenario_id="s1",
                    topology="linear_5",
                    workload_id="minimal",
                    implementation="hard_coded",
                    repetition=1,
                    execution_order_position=1,
                    operations=10,
                    elapsed_ns=1000,
                    normalized_ns_per_frame=100.0,
                    frames_completed=10,
                    plan_version=1,
                )
            ],
        )

        append_reconfiguration_rows(
            reconfig_path,
            [
                ReconfigurationSampleRow(
                    run_id="r1",
                    scenario_id="sc1",
                    baseline="prepare_and_commit",
                    edit_type="insert_stateless_node",
                    graph_size=10,
                    repetition=1,
                    old_plan_version=1,
                    new_plan_version=2,
                    terminal_status="COMMITTED",
                    validation_ns=100,
                    preparation_ns=500,
                    request_to_ready_ns=600,
                    boundary_wait_ns=50,
                    commit_ns=20,
                    request_to_effect_ns=700,
                    retirement_queue_delay_ns=10,
                    retirement_duration_ns=20,
                    commit_to_retirement_complete_ns=30,
                    maximum_output_gap_ns=2000,
                    transition_output_gap_ns=200,
                    frames_completed_during_request=2,
                    old_plan_frames_admitted_after_request_before_commit=2,
                    frames_dropped=0,
                    frames_duplicated=0,
                    reused_processor_count=9,
                    staged_processor_count=1,
                    retired_processor_count=0,
                    state_transition_policy="AUTO",
                )
            ],
        )

        append_conformance_rows(
            conform_path,
            [
                ConformanceResultRow(
                    run_id="r1",
                    campaign_id="stress",
                    scenario_id="stress_sc",
                    scenario_type="stress",
                    frames_submitted=10,
                    frames_completed=10,
                    reconfigurations_requested=1,
                    reconfigurations_committed=1,
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
            ],
        )

        data = load_benchmark_artifact(tmp_dir)

        summary_csv = os.path.join(tmp_dir, "summary.csv")
        tables_dir = os.path.join(tmp_dir, "tables")
        figures_dir = os.path.join(tmp_dir, "figures")

        generate_summary_csv(data, summary_csv)
        t_paths = generate_all_tables(data, tables_dir)
        f_paths = generate_all_figures(data, figures_dir)

        assert os.path.exists(summary_csv)
        assert len(t_paths) == 6  # 3 tables x 2 formats (csv + md)
        assert len(f_paths) == 10  # 5 figures x 2 formats (pdf + png)


def test_verify_invalid_artifact_rejection() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Empty directory should fail verification
        with pytest.raises(ValueError, match="Missing run.json"):
            verify_cmd(tmp_dir)
