"""Strictly typed benchmark row data models and constants."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SteadyStateSampleRow",
    "ReconfigurationSampleRow",
    "ConformanceResultRow",
    "STEADY_STATE_HEADERS",
    "RECONFIGURATION_HEADERS",
    "CONFORMANCE_HEADERS",
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
    frames_completed_during_request: int
    old_plan_frames_completed_during_preparation: int
    frames_dropped: int
    frames_duplicated: int
    reused_processor_count: int
    staged_processor_count: int
    retired_processor_count: int
    state_transition_policy: str


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
    "frames_completed_during_request",
    "old_plan_frames_completed_during_preparation",
    "frames_dropped",
    "frames_duplicated",
    "reused_processor_count",
    "staged_processor_count",
    "retired_processor_count",
    "state_transition_policy",
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
    "terminal_status",
)
