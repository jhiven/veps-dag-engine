"""Typed append-only CSV serialization and deserialization for benchmark rows."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable 
from typing import TextIO

from benchmarks.model import (
    ABLATION_HEADERS,
    CONFORMANCE_HEADERS,
    INTERFERENCE_FRAME_HEADERS,
    INTERFERENCE_HEADERS,
    MEMORY_HEADERS,
    RECONFIGURATION_HEADERS,
    STEADY_STATE_HEADERS,
    AblationSampleRow,
    ConformanceResultRow,
    InterferenceFrameSampleRow,
    InterferenceSampleRow,
    MemorySampleRow,
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
    "write_ablation_header",
    "append_ablation_rows",
    "read_ablation_rows",
    "write_memory_header",
    "append_memory_rows",
    "read_memory_rows",
    "write_conformance_header",
    "append_conformance_rows",
    "read_conformance_rows",
    "write_interference_frame_header",
    "append_interference_frame_rows",
    "read_interference_frame_rows",
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


def _format_optional_float(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else ""


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    return float(v) if v else None


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


def write_ablation_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(ABLATION_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(ABLATION_HEADERS)
        file_or_path.flush()


def append_ablation_rows(
    file_or_path: str | TextIO, rows: Iterable[AblationSampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.block_id,
            str(row.block_seed),
            row.variant,
            row.variant_name,
            row.preparation_placement,
            row.retirement_policy,
            row.edit_type,
            str(row.repetition),
            str(row.variant_order_position),
            row.candidate_graph_fingerprint,
            row.preparation_workload_id,
            str(row.old_plan_version),
            str(row.new_plan_version),
            _format_optional_int(row.validation_ns),
            _format_optional_int(row.synchronous_preparation_ns),
            _format_optional_int(row.offpath_preparation_ns),
            _format_optional_int(row.boundary_wait_ns),
            str(row.publication_ns),
            _format_optional_int(row.synchronous_retirement_ns),
            _format_optional_int(row.deferred_retirement_ns),
            _format_optional_int(row.first_effect_wait_ns),
            str(row.request_to_effect_ns),
            str(row.transition_output_gap_ns),
            str(row.total_synchronous_ns),
            str(row.synchronous_accounting_residual_ns),
            str(row.synchronous_accounting_valid),
            row.synchronous_accounting_invalid_reason or "",
            row.preparation_phase_definition,
            str(row.preparation_accounting_valid),
            row.preparation_accounting_invalid_reason or "",
            str(row.retirement_duration_ns),
            str(row.old_plan_frames_admitted_after_request_before_commit),
            str(row.peak_live_processors),
            _format_optional_int(row.random_seed),
            _format_optional_int(row.request_phase_offset_ns),
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


def read_ablation_rows(path: str) -> tuple[AblationSampleRow, ...]:
    rows: list[AblationSampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            variant = d["variant"]
            variant_name = d.get("variant_name")
            if not variant_name:
                names = {
                    "variant_a": "sync_prepare_sync_retire",
                    "variant_b": "offpath_prepare_sync_retire",
                    "variant_c": "sync_prepare_deferred_retire",
                    "variant_d": "offpath_prepare_deferred_retire",
                }
                variant_name = names.get(variant, variant)
            prep_placement = d.get("preparation_placement")
            if not prep_placement:
                prep_placement = "off_path" if variant in ("variant_b", "variant_d") else "synchronous"
            ret_policy = d.get("retirement_policy")
            if not ret_policy:
                ret_policy = "deferred" if variant in ("variant_c", "variant_d") else "synchronous"

            rows.append(
                AblationSampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    block_id=d.get("block_id", f"{d['scenario_id']}_{d.get('edit_type', 'edit')}_block_{d['repetition']}"),
                    block_seed=int(d.get("block_seed", d.get("random_seed", 0) or 0)),
                    variant=variant,
                    variant_name=variant_name,
                    preparation_placement=prep_placement,
                    retirement_policy=ret_policy,
                    edit_type=d["edit_type"],
                    repetition=int(d["repetition"]),
                    variant_order_position=int(d.get("variant_order_position", 1)),
                    candidate_graph_fingerprint=d.get("candidate_graph_fingerprint", "fp_default"),
                    preparation_workload_id=d.get("preparation_workload_id", "wl_default"),
                    old_plan_version=int(d["old_plan_version"]),
                    new_plan_version=int(d["new_plan_version"]),
                    validation_ns=_parse_optional_int(d.get("validation_ns")),
                    synchronous_preparation_ns=_parse_optional_int(d.get("synchronous_preparation_ns")),
                    offpath_preparation_ns=_parse_optional_int(d.get("offpath_preparation_ns")),
                    boundary_wait_ns=_parse_optional_int(d.get("boundary_wait_ns")),
                    publication_ns=int(d.get("publication_ns", 0)),
                    synchronous_retirement_ns=_parse_optional_int(d.get("synchronous_retirement_ns")),
                    deferred_retirement_ns=_parse_optional_int(d.get("deferred_retirement_ns")),
                    first_effect_wait_ns=_parse_optional_int(d.get("first_effect_wait_ns")),
                    request_to_effect_ns=int(d.get("request_to_effect_ns", 0)),
                    transition_output_gap_ns=int(d.get("transition_output_gap_ns", 0)),
                    total_synchronous_ns=int(d.get("total_synchronous_ns", 0)),
                    synchronous_accounting_residual_ns=int(d.get("synchronous_accounting_residual_ns", 0)),
                    synchronous_accounting_valid=d.get("synchronous_accounting_valid", "True").lower() == "true",
                    synchronous_accounting_invalid_reason=d.get("synchronous_accounting_invalid_reason") or None,
                    preparation_phase_definition=d.get(
                        "preparation_phase_definition",
                        "offpath_preparation_ns contains full off-path candidate preparation operation "
                        "including validation, compilation, processor construction, and any synthetic preparation work",
                    ),
                    preparation_accounting_valid=d.get("preparation_accounting_valid", "True").lower() == "true",
                    preparation_accounting_invalid_reason=d.get("preparation_accounting_invalid_reason") or None,
                    retirement_duration_ns=int(d.get("retirement_duration_ns", 0)),
                    old_plan_frames_admitted_after_request_before_commit=int(
                        d.get("old_plan_frames_admitted_after_request_before_commit", 0)
                    ),
                    peak_live_processors=int(d.get("peak_live_processors", 0)),
                    random_seed=_parse_optional_int(d.get("random_seed")),
                    request_phase_offset_ns=_parse_optional_int(d.get("request_phase_offset_ns")),
                )
            )
    return tuple(rows)


def write_memory_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(MEMORY_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(MEMORY_HEADERS)
        file_or_path.flush()


def append_memory_rows(
    file_or_path: str | TextIO, rows: Iterable[MemorySampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.variant,
            str(row.repetition),
            str(row.rss_before_bytes),
            str(row.rss_after_candidate_prepare_bytes),
            str(row.rss_after_commit_bytes),
            str(row.rss_after_retirement_bytes),
            str(row.observed_peak_rss_bytes),
            str(row.peak_rss_delta_bytes),
            str(row.retained_rss_delta_bytes),
            str(row.peak_live_processors),
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


def read_memory_rows(path: str) -> tuple[MemorySampleRow, ...]:
    rows: list[MemorySampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                MemorySampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    variant=d["variant"],
                    repetition=int(d["repetition"]),
                    rss_before_bytes=int(d["rss_before_bytes"]),
                    rss_after_candidate_prepare_bytes=int(d["rss_after_candidate_prepare_bytes"]),
                    rss_after_commit_bytes=int(d["rss_after_commit_bytes"]),
                    rss_after_retirement_bytes=int(d["rss_after_retirement_bytes"]),
                    observed_peak_rss_bytes=int(d["observed_peak_rss_bytes"]),
                    peak_rss_delta_bytes=int(d["peak_rss_delta_bytes"]),
                    retained_rss_delta_bytes=int(d["retained_rss_delta_bytes"]),
                    peak_live_processors=int(d["peak_live_processors"]),
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
                    active_plan_changed_after_failed_candidate=int(d["active_plan_changed_after_failed_candidate"]),
                    candidate_resource_leaks=int(d["candidate_resource_leaks"]),
                    processor_instance_leaks=int(d["processor_instance_leaks"]),
                    terminal_status=d["terminal_status"],
                )
            )
    return tuple(rows)


def write_interference_frame_header(file_or_path: str | TextIO) -> None:
    if isinstance(file_or_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(file_or_path)), exist_ok=True)
        with open(file_or_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(INTERFERENCE_FRAME_HEADERS)
    else:
        writer = csv.writer(file_or_path)
        writer.writerow(INTERFERENCE_FRAME_HEADERS)
        file_or_path.flush()


def append_interference_frame_rows(
    file_or_path: str | TextIO, rows: Iterable[InterferenceFrameSampleRow]
) -> None:
    formatted_rows = [
        [
            row.run_id,
            row.scenario_id,
            row.preparation_category,
            str(row.target_window_ns),
            str(row.repetition),
            row.phase,
            str(row.frame_id),
            str(row.plan_version),
            str(row.admission_timestamp_ns),
            str(row.completion_timestamp_ns),
            str(row.frame_latency_ns),
            f"{row.queue_occupancy:.2f}",
            str(row.dropped),
            str(row.duplicated),
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


def read_interference_frame_rows(path: str) -> tuple[InterferenceFrameSampleRow, ...]:
    rows: list[InterferenceFrameSampleRow] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for d in reader:
            rows.append(
                InterferenceFrameSampleRow(
                    run_id=d["run_id"],
                    scenario_id=d["scenario_id"],
                    preparation_category=d["preparation_category"],
                    target_window_ns=int(d["target_window_ns"]),
                    repetition=int(d["repetition"]),
                    phase=d["phase"],
                    frame_id=int(d["frame_id"]),
                    plan_version=int(d["plan_version"]),
                    admission_timestamp_ns=int(d["admission_timestamp_ns"]),
                    completion_timestamp_ns=int(d["completion_timestamp_ns"]),
                    frame_latency_ns=int(d["frame_latency_ns"]),
                    queue_occupancy=float(d["queue_occupancy"]),
                    dropped=d["dropped"].lower() == "true",
                    duplicated=d["duplicated"].lower() == "true",
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
            str(row.target_window_ns),
            str(row.repetition),
            str(row.execution_order_position),
            str(row.random_seed),
            f"{row.configured_offered_rate_fps:.2f}",
            str(row.before_window_start_ns),
            str(row.before_window_end_ns),
            str(row.during_window_start_ns),
            str(row.during_window_end_ns),
            str(row.after_window_start_ns),
            str(row.after_window_end_ns),
            str(row.measured_before_window_duration_ns),
            str(row.measured_during_window_duration_ns),
            str(row.measured_after_window_duration_ns),
            str(row.producer_attempted_frames_before),
            str(row.producer_attempted_frames_during),
            str(row.producer_attempted_frames_after),
            str(row.producer_emitted_frames_before),
            str(row.producer_emitted_frames_during),
            str(row.producer_emitted_frames_after),
            str(row.producer_skipped_deadlines_before),
            str(row.producer_skipped_deadlines_during),
            str(row.producer_skipped_deadlines_after),
            f"{row.producer_schedule_lateness_median_ns_before:.2f}",
            f"{row.producer_schedule_lateness_p95_ns_before:.2f}",
            f"{row.producer_schedule_lateness_median_ns_during:.2f}",
            f"{row.producer_schedule_lateness_p95_ns_during:.2f}",
            f"{row.producer_schedule_lateness_median_ns_after:.2f}",
            f"{row.producer_schedule_lateness_p95_ns_after:.2f}",
            str(row.admitted_frames_before),
            str(row.admitted_frames_during),
            str(row.admitted_frames_after),
            str(row.completed_frames_before),
            str(row.completed_frames_during),
            str(row.completed_frames_after),
            _format_optional_float(row.frame_latency_before_median_ns),
            _format_optional_float(row.frame_latency_before_p95_ns),
            _format_optional_float(row.frame_latency_during_median_ns),
            _format_optional_float(row.frame_latency_during_p95_ns),
            _format_optional_float(row.frame_latency_after_median_ns),
            _format_optional_float(row.frame_latency_after_p95_ns),
            _format_optional_float(row.p95_degradation_vs_before_pct),
            _format_optional_float(row.median_degradation_vs_before_pct),
            str(row.frames_completed_during_window),
            _format_optional_int(row.frames_completed_while_preparation_active),
            str(row.old_plan_frames_completed),
            f"{row.admission_throughput_before_fps:.2f}",
            f"{row.admission_throughput_during_fps:.2f}",
            f"{row.admission_throughput_after_fps:.2f}",
            f"{row.completion_throughput_before_fps:.2f}",
            f"{row.completion_throughput_during_fps:.2f}",
            f"{row.completion_throughput_after_fps:.2f}",
            _format_optional_float(row.admission_rate_ratio_vs_before),
            _format_optional_float(row.completion_rate_ratio_vs_before),
            str(row.latency_interpretation_confounded),
            row.latency_interpretation_confounded_reason or "",
            str(row.within_run_admission_shift_flag),
            row.within_run_admission_shift_reason or "",
            str(row.backlog_frames_at_during_start),
            str(row.backlog_frames_at_during_end),
            str(row.backlog_frames_drained_after_window),
            str(row.frames_emitted_during_but_admitted_after),
            str(row.frames_admitted_during_but_emitted_before),
            str(row.flow_accounting_valid),
            str(row.flow_accounting_residual_frames),
            row.flow_accounting_invalid_reason or "",
            f"{row.queue_occupancy_during_median:.2f}",
            f"{row.queue_occupancy_during_p95:.2f}",
            str(row.dropped_frames),
            str(row.duplicated_frames),
            str(row.valid_for_tail_latency),
            row.invalid_reason or "",
            _format_optional_int(row.target_preparation_duration_ns),
            str(row.actual_preparation_duration_ns),
            _format_optional_int(row.preparation_checksum),
            str(row.duration_valid),
            f"{row.duration_relative_error:.4f}",
            str(row.raw_frame_count_matches_summary),
            str(row.phase_timestamp_bounds_valid),
            str(row.throughput_accounting_valid),
            row.throughput_accounting_invalid_reason or "",
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
                    target_window_ns=int(d.get("target_window_ns", 200000000)),
                    repetition=int(d["repetition"]),
                    execution_order_position=int(d.get("execution_order_position", 1)),
                    random_seed=int(d.get("random_seed", 0)),
                    configured_offered_rate_fps=float(d.get("configured_offered_rate_fps", 1000.0)),
                    before_window_start_ns=int(d.get("before_window_start_ns", 0)),
                    before_window_end_ns=int(d.get("before_window_end_ns", 0)),
                    during_window_start_ns=int(d.get("during_window_start_ns", 0)),
                    during_window_end_ns=int(d.get("during_window_end_ns", 0)),
                    after_window_start_ns=int(d.get("after_window_start_ns", 0)),
                    after_window_end_ns=int(d.get("after_window_end_ns", 0)),
                    measured_before_window_duration_ns=int(d.get("measured_before_window_duration_ns", 1000000000)),
                    measured_during_window_duration_ns=int(d.get("measured_during_window_duration_ns", 200000000)),
                    measured_after_window_duration_ns=int(d.get("measured_after_window_duration_ns", 1000000000)),
                    producer_attempted_frames_before=int(d.get("producer_attempted_frames_before", 0)),
                    producer_attempted_frames_during=int(d.get("producer_attempted_frames_during", 0)),
                    producer_attempted_frames_after=int(d.get("producer_attempted_frames_after", 0)),
                    producer_emitted_frames_before=int(d.get("producer_emitted_frames_before", 0)),
                    producer_emitted_frames_during=int(d.get("producer_emitted_frames_during", 0)),
                    producer_emitted_frames_after=int(d.get("producer_emitted_frames_after", 0)),
                    producer_skipped_deadlines_before=int(d.get("producer_skipped_deadlines_before", 0)),
                    producer_skipped_deadlines_during=int(d.get("producer_skipped_deadlines_during", 0)),
                    producer_skipped_deadlines_after=int(d.get("producer_skipped_deadlines_after", 0)),
                    producer_schedule_lateness_median_ns_before=float(d.get("producer_schedule_lateness_median_ns_before", 0.0)),
                    producer_schedule_lateness_p95_ns_before=float(d.get("producer_schedule_lateness_p95_ns_before", 0.0)),
                    producer_schedule_lateness_median_ns_during=float(d.get("producer_schedule_lateness_median_ns_during", 0.0)),
                    producer_schedule_lateness_p95_ns_during=float(d.get("producer_schedule_lateness_p95_ns_during", 0.0)),
                    producer_schedule_lateness_median_ns_after=float(d.get("producer_schedule_lateness_median_ns_after", 0.0)),
                    producer_schedule_lateness_p95_ns_after=float(d.get("producer_schedule_lateness_p95_ns_after", 0.0)),
                    admitted_frames_before=int(d.get("admitted_frames_before", d.get("frame_count_before", 0))),
                    admitted_frames_during=int(d.get("admitted_frames_during", d.get("frame_count_during", 0))),
                    admitted_frames_after=int(d.get("admitted_frames_after", d.get("frame_count_after", 0))),
                    completed_frames_before=int(d.get("completed_frames_before", d.get("frame_count_before", 0))),
                    completed_frames_during=int(d.get("completed_frames_during", d.get("frame_count_during", 0))),
                    completed_frames_after=int(d.get("completed_frames_after", d.get("frame_count_after", 0))),
                    frame_latency_before_median_ns=_parse_optional_float(d.get("frame_latency_before_median_ns")),
                    frame_latency_before_p95_ns=_parse_optional_float(d.get("frame_latency_before_p95_ns")),
                    frame_latency_during_median_ns=_parse_optional_float(d.get("frame_latency_during_median_ns")),
                    frame_latency_during_p95_ns=_parse_optional_float(d.get("frame_latency_during_p95_ns")),
                    frame_latency_after_median_ns=_parse_optional_float(d.get("frame_latency_after_median_ns")),
                    frame_latency_after_p95_ns=_parse_optional_float(d.get("frame_latency_after_p95_ns")),
                    p95_degradation_vs_before_pct=_parse_optional_float(d.get("p95_degradation_vs_before_pct", d.get("p95_degradation_pct"))),
                    median_degradation_vs_before_pct=_parse_optional_float(d.get("median_degradation_vs_before_pct")),
                    frames_completed_during_window=int(d.get("frames_completed_during_window", d.get("frames_completed_during_preparation", 0))),
                    frames_completed_while_preparation_active=_parse_optional_int(d.get("frames_completed_while_preparation_active")),
                    old_plan_frames_completed=int(d.get("old_plan_frames_completed", 0)),
                    admission_throughput_before_fps=float(d.get("admission_throughput_before_fps", d.get("throughput_before_fps", 0.0))),
                    admission_throughput_during_fps=float(d.get("admission_throughput_during_fps", d.get("throughput_during_fps", 0.0))),
                    admission_throughput_after_fps=float(d.get("admission_throughput_after_fps", d.get("throughput_after_fps", 0.0))),
                    completion_throughput_before_fps=float(d.get("completion_throughput_before_fps", d.get("throughput_before_fps", 0.0))),
                    completion_throughput_during_fps=float(d.get("completion_throughput_during_fps", d.get("throughput_during_fps", 0.0))),
                    completion_throughput_after_fps=float(d.get("completion_throughput_after_fps", d.get("throughput_after_fps", 0.0))),
                    admission_rate_ratio_vs_before=_parse_optional_float(d.get("admission_rate_ratio_vs_before")),
                    completion_rate_ratio_vs_before=_parse_optional_float(d.get("completion_rate_ratio_vs_before")),
                    latency_interpretation_confounded=d.get("latency_interpretation_confounded", "False").lower() == "true",
                    latency_interpretation_confounded_reason=d.get("latency_interpretation_confounded_reason") or None,
                    within_run_admission_shift_flag=d.get("within_run_admission_shift_flag", d.get("latency_interpretation_confounded", "False")).lower() == "true",
                    within_run_admission_shift_reason=d.get("within_run_admission_shift_reason", d.get("latency_interpretation_confounded_reason")) or None,
                    backlog_frames_at_during_start=int(d.get("backlog_frames_at_during_start", 0)),
                    backlog_frames_at_during_end=int(d.get("backlog_frames_at_during_end", 0)),
                    backlog_frames_drained_after_window=int(d.get("backlog_frames_drained_after_window", 0)),
                    frames_emitted_during_but_admitted_after=int(d.get("frames_emitted_during_but_admitted_after", 0)),
                    frames_admitted_during_but_emitted_before=int(d.get("frames_admitted_during_but_emitted_before", 0)),
                    flow_accounting_valid=d.get("flow_accounting_valid", "True").lower() == "true",
                    flow_accounting_residual_frames=int(d.get("flow_accounting_residual_frames", 0)),
                    flow_accounting_invalid_reason=d.get("flow_accounting_invalid_reason") or None,
                    queue_occupancy_during_median=float(d.get("queue_occupancy_during_median", d.get("queue_occupancy_median", 0.0))),
                    queue_occupancy_during_p95=float(d.get("queue_occupancy_during_p95", 0.0)),
                    dropped_frames=int(d.get("dropped_frames", 0)),
                    duplicated_frames=int(d.get("duplicated_frames", 0)),
                    valid_for_tail_latency=d.get("valid_for_tail_latency", "True").lower() == "true",
                    invalid_reason=d.get("invalid_reason") or None,
                    target_preparation_duration_ns=_parse_optional_int(d.get("target_preparation_duration_ns")),
                    actual_preparation_duration_ns=int(d.get("actual_preparation_duration_ns", 0)),
                    preparation_checksum=_parse_optional_int(d.get("preparation_checksum")),
                    duration_valid=d.get("duration_valid", "True").lower() == "true",
                    duration_relative_error=float(d.get("duration_relative_error", 0.0)),
                    raw_frame_count_matches_summary=d.get("raw_frame_count_matches_summary", "True").lower() == "true",
                    phase_timestamp_bounds_valid=d.get("phase_timestamp_bounds_valid", "True").lower() == "true",
                    throughput_accounting_valid=d.get("throughput_accounting_valid", "True").lower() == "true",
                    throughput_accounting_invalid_reason=d.get("throughput_accounting_invalid_reason") or None,
                )
            )
    return tuple(rows)
