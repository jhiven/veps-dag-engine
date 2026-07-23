"""Typed append-only CSV serialization and deserialization for benchmark rows."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable 
from typing import TextIO

from benchmarks.model import (
    CONFORMANCE_HEADERS,
    INTERFERENCE_HEADERS,
    RECONFIGURATION_HEADERS,
    STEADY_STATE_HEADERS,
    ConformanceResultRow,
    InterferenceSampleRow,
    ReconfigurationSampleRow,
    SteadyStateSampleRow,
)

__all__ = [
    "write_steady_state_header",
    "append_steady_state_rows",
    "read_steady_state_rows",
    "write_reconfiguration_header",
    "append_reconfiguration_rows",
    "read_reconfiguration_rows",
    "write_conformance_header",
    "append_conformance_rows",
    "read_conformance_rows",
    "write_interference_header",
    "append_interference_rows",
    "read_interference_rows",
]


def _format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else ""


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    v = value.strip()
    return int(v) if v else None


def write_steady_state_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(STEADY_STATE_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(STEADY_STATE_HEADERS)
        file_or_path.flush()


def append_steady_state_rows(
    file_or_path: str | TextIO, rows: Iterable[SteadyStateSampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.topology,
            row.workload_id,
            row.implementation,
            str(row.repetition),
            str(row.execution_order_position),
            str(row.operations),
            str(row.elapsed_ns),
            str(row.normalized_ns_per_frame),
            str(row.frames_completed),
            str(row.plan_version),
        ]
        for row in rows
    ]

    if isinstance(file_or_path, str):
        with open(file_or_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(formatted_rows)
            file.flush()
    else:
        writer = csv.writer(file_or_path)
        writer.writerows(formatted_rows)
        file_or_path.flush()


def read_steady_state_rows(path: str) -> tuple[SteadyStateSampleRow, ...]:
    rows: list[SteadyStateSampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                SteadyStateSampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    topology=d["topology"],
                    workload_id=d["workload_id"],
                    implementation=d["implementation"],
                    repetition=int(d["repetition"]),
                    execution_order_position=int(d["execution_order_position"]),
                    operations=int(d["operations"]),
                    elapsed_ns=int(d["elapsed_ns"]),
                    normalized_ns_per_frame=float(d["normalized_ns_per_frame"]),
                    frames_completed=int(d["frames_completed"]),
                    plan_version=int(d["plan_version"]),
                )
            )
    return tuple(rows)


def write_reconfiguration_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(RECONFIGURATION_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(RECONFIGURATION_HEADERS)
        file_or_path.flush()


def append_reconfiguration_rows(
    file_or_path: str | TextIO, rows: Iterable[ReconfigurationSampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.baseline,
            row.edit_type,
            str(row.graph_size),
            str(row.repetition),
            str(row.old_plan_version),
            str(row.new_plan_version),
            row.terminal_status,
            _format_optional_int(row.validation_ns),
            _format_optional_int(row.preparation_ns),
            _format_optional_int(row.request_to_ready_ns),
            _format_optional_int(row.boundary_wait_ns),
            _format_optional_int(row.commit_ns),
            _format_optional_int(row.request_to_effect_ns),
            _format_optional_int(row.retirement_queue_delay_ns),
            _format_optional_int(row.retirement_duration_ns),
            _format_optional_int(row.commit_to_retirement_complete_ns),
            _format_optional_int(row.maximum_output_gap_ns),
            _format_optional_int(row.transition_output_gap_ns),
            str(row.frames_completed_during_request),
            str(row.old_plan_frames_admitted_after_request_before_commit),
            str(row.frames_dropped),
            str(row.frames_duplicated),
            str(row.reused_processor_count),
            str(row.staged_processor_count),
            str(row.retired_processor_count),
            row.state_transition_policy,
            _format_optional_int(row.admission_stop_ns),
            _format_optional_int(row.executor_teardown_ns),
            _format_optional_int(row.executor_reconstruction_ns),
            _format_optional_int(row.publication_ns),
            _format_optional_int(row.executor_restart_ns),
            _format_optional_int(row.first_admission_wait_ns),
            _format_optional_int(row.first_completion_wait_ns),
            _format_optional_int(row.retirement_ns),
            _format_optional_int(row.total_synchronous_ns),
            str(row.phases_may_overlap),
            _format_optional_int(row.sum_of_instrumented_phase_durations_ns),
            _format_optional_int(row.instrumented_phase_sum_ns),
            _format_optional_int(row.critical_path_instrumented_ns),
            _format_optional_int(row.unattributed_critical_path_ns),
            _format_optional_int(row.unattributed_request_time_ns),
            _format_optional_int(row.instrumented_duration_overlap_ns),
            _format_optional_int(row.instrumented_duration_outside_effect_window_ns),
        ]
        for row in rows
    ]

    if isinstance(file_or_path, str):
        with open(file_or_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(formatted_rows)
            file.flush()
    else:
        writer = csv.writer(file_or_path)
        writer.writerows(formatted_rows)
        file_or_path.flush()


def read_reconfiguration_rows(path: str) -> tuple[ReconfigurationSampleRow, ...]:
    rows: list[ReconfigurationSampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                ReconfigurationSampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    baseline=d["baseline"],
                    edit_type=d["edit_type"],
                    graph_size=int(d["graph_size"]),
                    repetition=int(d["repetition"]),
                    old_plan_version=int(d["old_plan_version"]),
                    new_plan_version=int(d["new_plan_version"]),
                    terminal_status=d["terminal_status"],
                    validation_ns=_parse_optional_int(d.get("validation_ns")),
                    preparation_ns=_parse_optional_int(d.get("preparation_ns")),
                    request_to_ready_ns=_parse_optional_int(d.get("request_to_ready_ns")),
                    boundary_wait_ns=_parse_optional_int(d.get("boundary_wait_ns")),
                    commit_ns=_parse_optional_int(d.get("commit_ns")),
                    request_to_effect_ns=_parse_optional_int(d.get("request_to_effect_ns")),
                    retirement_queue_delay_ns=_parse_optional_int(d.get("retirement_queue_delay_ns")),
                    retirement_duration_ns=_parse_optional_int(d.get("retirement_duration_ns")),
                    commit_to_retirement_complete_ns=_parse_optional_int(d.get("commit_to_retirement_complete_ns")),
                    maximum_output_gap_ns=_parse_optional_int(d.get("maximum_output_gap_ns")),
                    transition_output_gap_ns=_parse_optional_int(d.get("transition_output_gap_ns")),
                    frames_completed_during_request=int(d["frames_completed_during_request"]),
                    old_plan_frames_admitted_after_request_before_commit=int(d.get("old_plan_frames_admitted_after_request_before_commit", 0)),
                    frames_dropped=int(d["frames_dropped"]),
                    frames_duplicated=int(d["frames_duplicated"]),
                    reused_processor_count=int(d["reused_processor_count"]),
                    staged_processor_count=int(d["staged_processor_count"]),
                    retired_processor_count=int(d["retired_processor_count"]),
                    state_transition_policy=d["state_transition_policy"],
                    admission_stop_ns=_parse_optional_int(d.get("admission_stop_ns")),
                    executor_teardown_ns=_parse_optional_int(d.get("executor_teardown_ns")),
                    executor_reconstruction_ns=_parse_optional_int(d.get("executor_reconstruction_ns")),
                    publication_ns=_parse_optional_int(d.get("publication_ns")),
                    executor_restart_ns=_parse_optional_int(d.get("executor_restart_ns")),
                    first_admission_wait_ns=_parse_optional_int(d.get("first_admission_wait_ns")),
                    first_completion_wait_ns=_parse_optional_int(d.get("first_completion_wait_ns")),
                    retirement_ns=_parse_optional_int(d.get("retirement_ns")),
                    total_synchronous_ns=_parse_optional_int(d.get("total_synchronous_ns")),
                    phases_may_overlap=d.get("phases_may_overlap", "False").lower() == "true",
                    sum_of_instrumented_phase_durations_ns=_parse_optional_int(d.get("sum_of_instrumented_phase_durations_ns")),
                    instrumented_phase_sum_ns=_parse_optional_int(d.get("instrumented_phase_sum_ns")),
                    critical_path_instrumented_ns=_parse_optional_int(d.get("critical_path_instrumented_ns")),
                    unattributed_critical_path_ns=_parse_optional_int(d.get("unattributed_critical_path_ns")),
                    unattributed_request_time_ns=_parse_optional_int(d.get("unattributed_request_time_ns")),
                    instrumented_duration_overlap_ns=_parse_optional_int(d.get("instrumented_duration_overlap_ns")),
                    instrumented_duration_outside_effect_window_ns=_parse_optional_int(d.get("instrumented_duration_outside_effect_window_ns")),
                )
            )
    return tuple(rows)


def write_conformance_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CONFORMANCE_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(CONFORMANCE_HEADERS)
        file_or_path.flush()


def append_conformance_rows(
    file_or_path: str | TextIO, rows: Iterable[ConformanceResultRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.campaign_id,
            row.scenario_id,
            row.scenario_type,
            str(row.frames_submitted),
            str(row.frames_completed),
            str(row.reconfigurations_requested),
            str(row.reconfigurations_committed),
            str(row.reconfigurations_rejected),
            str(row.reconfigurations_failed),
            str(row.mixed_plan_frames),
            str(row.missing_frames),
            str(row.duplicate_frames),
            str(row.retired_plan_executions),
            str(row.invalid_routing_events),
            str(row.state_continuity_failures),
            str(row.unexpected_state_resets),
            str(row.active_plan_changed_after_failed_candidate),
            str(row.candidate_resource_leaks),
            str(row.processor_instance_leaks),
            row.terminal_status,
        ]
        for row in rows
    ]

    if isinstance(file_or_path, str):
        with open(file_or_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(formatted_rows)
            file.flush()
    else:
        writer = csv.writer(file_or_path)
        writer.writerows(formatted_rows)
        file_or_path.flush()


def read_conformance_rows(path: str) -> tuple[ConformanceResultRow, ...]:
    rows: list[ConformanceResultRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                ConformanceResultRow(
                    run_id=d["run_id"],
                    campaign_id=d["campaign_id"],
                    scenario_id=d["scenario_id"],
                    scenario_type=d["scenario_type"],
                    frames_submitted=int(d["frames_submitted"]),
                    frames_completed=int(d["frames_completed"]),
                    reconfigurations_requested=int(d["reconfigurations_requested"]),
                    reconfigurations_committed=int(d["reconfigurations_committed"]),
                    reconfigurations_rejected=int(d["reconfigurations_rejected"]),
                    reconfigurations_failed=int(d["reconfigurations_failed"]),
                    mixed_plan_frames=int(d["mixed_plan_frames"]),
                    missing_frames=int(d["missing_frames"]),
                    duplicate_frames=int(d["duplicate_frames"]),
                    retired_plan_executions=int(d["retired_plan_executions"]),
                    invalid_routing_events=int(d["invalid_routing_events"]),
                    state_continuity_failures=int(d["state_continuity_failures"]),
                    unexpected_state_resets=int(d["unexpected_state_resets"]),
                    active_plan_changed_after_failed_candidate=int(
                        d["active_plan_changed_after_failed_candidate"]
                    ),
                    candidate_resource_leaks=int(d["candidate_resource_leaks"]),
                    processor_instance_leaks=int(d["processor_instance_leaks"]),
                    terminal_status=d["terminal_status"],
                )
            )
    return tuple(rows)


def write_interference_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(INTERFERENCE_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(INTERFERENCE_HEADERS)
        file_or_path.flush()


def append_interference_rows(
    file_or_path: str | TextIO, rows: Iterable[InterferenceSampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.preparation_category,
            str(row.repetition),
            f"{row.frame_latency_before_median_ns:.2f}",
            f"{row.frame_latency_during_median_ns:.2f}",
            f"{row.frame_latency_during_p95_ns:.2f}",
            f"{row.frame_latency_after_median_ns:.2f}",
            str(row.frames_completed_during_preparation),
            f"{row.throughput_before_fps:.2f}",
            f"{row.throughput_during_fps:.2f}",
            f"{row.throughput_after_fps:.2f}",
        ]
        for row in rows
    ]

    if isinstance(file_or_path, str):
        with open(file_or_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(formatted_rows)
            file.flush()
    else:
        writer = csv.writer(file_or_path)
        writer.writerows(formatted_rows)
        file_or_path.flush()


def read_interference_rows(path: str) -> tuple[InterferenceSampleRow, ...]:
    rows: list[InterferenceSampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                InterferenceSampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    preparation_category=d["preparation_category"],
                    repetition=int(d["repetition"]),
                    frame_latency_before_median_ns=float(d["frame_latency_before_median_ns"]),
                    frame_latency_during_median_ns=float(d["frame_latency_during_median_ns"]),
                    frame_latency_during_p95_ns=float(d["frame_latency_during_p95_ns"]),
                    frame_latency_after_median_ns=float(d["frame_latency_after_median_ns"]),
                    frames_completed_during_preparation=int(d["frames_completed_during_preparation"]),
                    throughput_before_fps=float(d["throughput_before_fps"]),
                    throughput_during_fps=float(d["throughput_during_fps"]),
                    throughput_after_fps=float(d["throughput_after_fps"]),
                )
            )
    return tuple(rows)
