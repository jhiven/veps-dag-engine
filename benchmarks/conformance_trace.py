"""Schema-2 raw conformance traces and offline replay validation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

from nedo_vision_dag_engine.instrumentation import JsonValue

SCHEMA_VERSION = "2.0.0"
_REQUEST_EDGES: dict[str, frozenset[str]] = {
    "request_received": frozenset({"request_validated", "request_stale", "request_aborted", "request_failed"}),
    "request_validated": frozenset({"request_prepared", "request_rejected", "request_failed", "request_aborted"}),
    "request_prepared": frozenset({"request_ready", "request_rejected", "request_failed", "request_aborted"}),
    "request_ready": frozenset({"request_committed", "request_stale", "request_failed", "request_aborted"}),
}
_REQUEST_KINDS = frozenset(_REQUEST_EDGES) | frozenset({
    "request_committed", "request_rejected", "request_failed",
    "request_aborted", "request_stale",
})

__all__ = [
    "SCHEMA_VERSION",
    "ConformanceTrace",
    "ReplayReport",
    "replay_trace",
    "validate_conformance_directory",
]


class ConformanceTrace:
    """Thread-safe append-only writer for one campaign and seed."""

    __slots__ = ("_campaign_id", "_file", "_lock", "_run_id", "_seed", "_sequence")

    def __init__(self, path: str, run_id: str, campaign_id: str, seed: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "x", encoding="utf-8")
        self._run_id = run_id
        self._campaign_id = campaign_id
        self._seed = seed
        self._sequence = 1
        self._lock = Lock()

    def emit(self, kind: str, fields: Mapping[str, JsonValue]) -> None:
        with self._lock:
            event: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self._run_id,
                "campaign_id": self._campaign_id,
                "seed": self._seed,
                "event_sequence": self._sequence,
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "event_kind": kind,
            }
            event.update(fields)
            self._file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            self._file.flush()
            self._sequence += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> ConformanceTrace:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ReplayReport:
    path: str
    campaign_id: str
    seed: int
    event_count: int
    counts: dict[str, int]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def replay_trace(path: str) -> ReplayReport:
    """Derive counts and ordering invariants exclusively from a JSONL trace."""
    errors: list[str] = []
    campaign_id = ""
    seed = -1
    run_id = ""
    event_count = 0
    # Randomized traces are much larger than controlled schedules. Retain
    # only their derived state, not hundreds of megabytes of JSON objects.
    schedule_events: list[dict[str, Any]] = []
    previous_timestamp = -1
    counts: dict[str, int] = {}
    offered: dict[int, int] = {}
    admitted: dict[int, int] = {}
    terminal: dict[int, str] = {}
    terminal_index: dict[int, int] = {}
    frame_plans: dict[int, int] = {}
    leases: set[tuple[int, int]] = set()
    node_entries: set[tuple[int, str, str]] = set()
    plan_processors: dict[int, set[str]] = {}
    # Each plan's declared node bindings, from staging, reuse and reset events.
    # A node execution must use the instance its frame's leased plan binds.
    plan_bindings: dict[int, dict[str, set[str]]] = {}
    candidate_processors: dict[int, set[str]] = {}
    cleanup_attempts: set[str] = set()
    cleanup_terminals: set[str] = set()
    request_received: set[str] = set()
    request_terminal: set[str] = set()
    request_state: dict[str, str] = {}
    campaign_completed = False
    campaign_started_count = 0
    campaign_completed_count = 0
    declared_frames: int | None = None
    declared_requests: int | None = None

    with open(path, "r", encoding="utf-8") as file:
      for index, line in enumerate(file, 1):
        event_count = index
        try:
            raw: Any = json.loads(line)
        except Exception as error:
            errors.append(f"line {index}: invalid JSON: {error}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"line {index}: event must be a JSON object")
            continue
        event = cast(dict[str, Any], raw)
        if index == 1:
            campaign_id = str(event.get("campaign_id", ""))
            run_id = str(event.get("run_id", ""))
            seed_raw = event.get("seed")
            seed = seed_raw if isinstance(seed_raw, int) else -1
        if campaign_id != "randomized_consistency":
            schedule_events.append(event)
        if event.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"event {index}: unsupported schema_version")
        if event.get("event_sequence") != index:
            errors.append(f"event {index}: missing or non-contiguous event_sequence")
        timestamp = event.get("monotonic_timestamp_ns")
        if not isinstance(timestamp, int) or timestamp < previous_timestamp:
            errors.append(f"event {index}: non-monotonic timestamp")
        else:
            previous_timestamp = timestamp
        if (event.get("campaign_id") != campaign_id or event.get("seed") != seed
                or event.get("run_id") != run_id):
            errors.append(f"event {index}: run/campaign/seed envelope changed")

        kind = str(event.get("event_kind", ""))
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "campaign_started":
            campaign_started_count += 1
            if index != 1:
                errors.append("campaign_started is not the first event")
            frame_target = event.get("target_frames")
            request_target = event.get("target_requests")
            if isinstance(frame_target, int):
                declared_frames = frame_target
            if isinstance(request_target, int):
                declared_requests = request_target
        frame_raw = event.get("frame_id")
        frame_id = frame_raw if isinstance(frame_raw, int) else None
        plan_raw = event.get("plan_id")
        plan_id = plan_raw if isinstance(plan_raw, int) else None
        request_id = event.get("request_id")
        if kind in _REQUEST_KINDS and isinstance(request_id, str):
            prior = request_state.get(request_id)
            if kind == "request_received":
                if prior is not None:
                    errors.append(f"request {request_id}: duplicate receipt")
            elif prior is None or kind not in _REQUEST_EDGES.get(prior, frozenset()):
                errors.append(f"request {request_id}: malformed transition {prior!r} -> {kind!r}")
            request_state[request_id] = kind
        if kind == "request_received" and isinstance(request_id, str):
            if request_id in request_received:
                errors.append(f"request {request_id}: duplicate receipt")
            request_received.add(request_id)
        elif kind in {"request_committed", "request_rejected", "request_failed",
                      "request_aborted", "request_stale"} and isinstance(request_id, str):
            if request_id not in request_received:
                errors.append(f"request {request_id}: terminal outcome without receipt")
            if request_id in request_terminal:
                errors.append(f"request {request_id}: duplicate terminal outcome")
            request_terminal.add(request_id)
            if kind != "request_committed":
                candidate_processors.clear()
        processor_id = event.get("processor_instance_id")
        node_raw = event.get("node_id")
        if (kind in {"processor_staged", "processor_reused", "processor_reset"}
                and plan_id is not None and isinstance(processor_id, str)
                and isinstance(node_raw, str)):
            plan_bindings.setdefault(plan_id, {}).setdefault(node_raw, set()).add(processor_id)
        if kind in {"processor_staged", "processor_reused"} and plan_id is not None and isinstance(processor_id, str):
            target = plan_processors if plan_id == 1 else candidate_processors
            target.setdefault(plan_id, set()).add(processor_id)
        elif kind == "plan_published" and plan_id is not None:
            plan_processors[plan_id] = candidate_processors.pop(plan_id, set())
        elif kind == "processor_cleanup_attempted" and isinstance(processor_id, str):
            if processor_id in cleanup_attempts:
                errors.append(f"processor {processor_id}: duplicate cleanup attempt")
            cleanup_attempts.add(processor_id)
            for leased_frame, leased_plan in leases:
                if processor_id in plan_processors.get(leased_plan, set()):
                    errors.append(
                        f"processor {processor_id}: cleanup while frame {leased_frame} leases plan {leased_plan}"
                    )
        elif kind in {"processor_cleanup_completed", "processor_cleanup_failed"} and isinstance(processor_id, str):
            if processor_id not in cleanup_attempts:
                errors.append(f"processor {processor_id}: cleanup terminal without attempt")
            if processor_id in cleanup_terminals:
                errors.append(f"processor {processor_id}: duplicate cleanup terminal")
            cleanup_terminals.add(processor_id)

        if kind == "frame_offered" and frame_id is not None:
            if frame_id in offered:
                errors.append(f"frame {frame_id}: duplicate offer")
            offered[frame_id] = index
        elif kind == "frame_admitted" and frame_id is not None:
            if frame_id in admitted:
                errors.append(f"frame {frame_id}: duplicate admission")
            admitted[frame_id] = index
            if plan_id is not None:
                previous_plan = frame_plans.setdefault(frame_id, plan_id)
                if previous_plan != plan_id:
                    errors.append(f"frame {frame_id}: admission changed its leased plan")
        elif kind in {"frame_completed", "frame_failed", "frame_cancelled"} and frame_id is not None:
            if frame_id in terminal:
                errors.append(f"frame {frame_id}: duplicate terminal outcome")
            terminal[frame_id] = kind
            terminal_index[frame_id] = index
            if plan_id is not None and frame_plans.get(frame_id) != plan_id:
                errors.append(f"frame {frame_id}: terminal event changed its plan")
        elif kind == "lease_acquired" and frame_id is not None and plan_id is not None:
            key = (frame_id, plan_id)
            if key in leases:
                errors.append(f"frame {frame_id}: duplicate lease acquisition")
            leases.add(key)
            previous_plan = frame_plans.setdefault(frame_id, plan_id)
            if previous_plan != plan_id:
                errors.append(f"frame {frame_id}: acquired leases on multiple plans")
        elif kind == "lease_released" and frame_id is not None and plan_id is not None:
            key = (frame_id, plan_id)
            if key not in leases:
                errors.append(f"frame {frame_id}: lease release without acquisition")
            else:
                leases.remove(key)
        elif kind == "node_entered" and frame_id is not None:
            bound = plan_bindings.get(plan_id, {}).get(str(node_raw)) if plan_id is not None else None
            if not bound or processor_id not in bound:
                errors.append(
                    f"frame {frame_id}: node {node_raw!r} executed instance {processor_id!r} "
                    f"that plan {plan_id} does not bind"
                )
            if plan_id is not None and isinstance(processor_id, str):
                plan_processors.setdefault(plan_id, set()).add(processor_id)
            if frame_plans.get(frame_id) != plan_id:
                errors.append(f"frame {frame_id}: node entered under a different plan")
            key = (frame_id, str(event.get("node_id", "")), str(event.get("processor_instance_id", "")))
            if key in node_entries:
                errors.append(f"frame {frame_id}: duplicate node entry {key[1]!r}")
            node_entries.add(key)
        elif kind == "node_exited" and frame_id is not None:
            key = (frame_id, str(event.get("node_id", "")), str(event.get("processor_instance_id", "")))
            if key not in node_entries:
                errors.append(f"frame {frame_id}: node exit without entry {key[1]!r}")
            else:
                node_entries.remove(key)
        elif kind == "campaign_completed":
            campaign_completed_count += 1
            campaign_completed = True
            if event.get("outcome") != "passed":
                errors.append("campaign terminal outcome is not passed")

    for frame_id, admission_index in admitted.items():
        offer_index = offered.get(frame_id)
        if offer_index is None or offer_index > admission_index:
            errors.append(f"frame {frame_id}: admission is not preceded by an offer")
        if frame_id not in terminal:
            errors.append(f"frame {frame_id}: admitted without terminal outcome")
        elif terminal_index[frame_id] <= admission_index:
            errors.append(f"frame {frame_id}: terminal outcome preceded admission")
    for frame_id, outcome in terminal.items():
        offer_index = offered.get(frame_id)
        if offer_index is None or terminal_index[frame_id] <= offer_index:
            errors.append(f"frame {frame_id}: terminal {outcome} is not preceded by an offer")
    if set(offered) != set(terminal):
        errors.append(f"offered frames without terminal outcomes: {sorted(set(offered) - set(terminal))!r}")
    if leases:
        errors.append(f"unreleased leases: {sorted(leases)!r}")
    if node_entries:
        errors.append(f"unclosed node executions: {sorted(node_entries)!r}")
    if not campaign_completed:
        errors.append("campaign_completed event is missing")
    if campaign_started_count != 1 or campaign_completed_count != 1:
        errors.append("campaign must have exactly one start and one completion event")
    if counts.get("plan_published", 0) != counts.get("request_committed", 0):
        errors.append("publication and committed-request counts differ")
    if request_received != request_terminal:
        errors.append(f"requests without terminal outcomes: {sorted(request_received - request_terminal)!r}")
    if cleanup_attempts != cleanup_terminals:
        errors.append(f"cleanup attempts without outcomes: {sorted(cleanup_attempts - cleanup_terminals)!r}")
    if campaign_id == "randomized_consistency":
        if declared_frames is None or declared_requests is None:
            errors.append("randomized target counts are not declared")
        else:
            if len(offered) != declared_frames or counts.get("frame_completed", 0) != declared_frames:
                errors.append("randomized frame target was not reached")
            if len(request_received) != declared_requests:
                errors.append("randomized request target was not reached")
    else:
        errors.extend(_validate_controlled_schedule(campaign_id, schedule_events))

    return ReplayReport(path, campaign_id, seed, event_count, counts, tuple(errors))


def _validate_controlled_schedule(campaign_id: str, events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    def positions(kind: str, **fields: object) -> list[int]:
        return [
            index for index, event in enumerate(events, 1)
            if event.get("event_kind") == kind
            and all(event.get(key) == value for key, value in fields.items())
        ]

    def require_order(label: str, *groups: list[int]) -> None:
        if any(not group for group in groups) or not all(
            left[0] < right[0] for left, right in zip(groups, groups[1:])
        ):
            errors.append(f"{campaign_id}: required {label} ordering was not exercised")

    if campaign_id == "stateless_publication":
        require_order(
            "old lease, publication, lease release, cleanup",
            positions("lease_acquired", plan_id=1), positions("plan_published", plan_id=2),
            positions("lease_released", plan_id=1), positions("processor_cleanup_attempted"),
        )
    elif campaign_id == "mutable_handoff":
        require_order(
            "old lease, gate close, later admission attempt, release, drain, publication, new lease",
            positions("lease_acquired", plan_id=1), positions("handoff_gate_closed"),
            positions("frame_admission_attempted", frame_id=2),
            positions("lease_released", plan_id=1), positions("handoff_drain_completed"),
            positions("plan_published", plan_id=2), positions("lease_acquired", plan_id=2),
        )
    elif campaign_id == "failure_atomicity":
        for request_id, terminal in (("fail_cycle", "request_rejected"),
                                     ("fail_type", "request_rejected"),
                                     ("fail_setup", "request_failed")):
            if len(positions(terminal, request_id=request_id)) != 1:
                errors.append(f"{campaign_id}: {request_id} has wrong terminal outcome")
        if positions("plan_published"):
            errors.append(f"{campaign_id}: a failed candidate published a plan")
        if len(positions("processor_cleanup_completed", outcome="staging_rollback")) < 2:
            errors.append(f"{campaign_id}: staged setup-failure processors were not rolled back")
        cycle_rejection = next((event for event in events if event.get("request_id") == "fail_cycle"
                                and event.get("event_kind") == "request_rejected"), None)
        if cycle_rejection is None or "cycle" not in str(cycle_rejection.get("error_message", "")).lower():
            errors.append(f"{campaign_id}: cycle request did not fail cycle validation")
    elif campaign_id == "candidate_failure":
        require_order(
            "staged rollback, failed request, valid publication, new frame",
            positions("processor_staged", node_id="n1", plan_id=2),
            positions("processor_cleanup_completed", node_id="n1", outcome="staging_rollback"),
            positions("request_failed", request_id="mem-fail"),
            positions("request_committed", request_id="mem-fail-2"),
            positions("frame_completed", plan_id=2),
        )
        if len(positions("plan_published")) != 1:
            errors.append(f"{campaign_id}: expected one valid publication")
    elif campaign_id == "cleanup_failure":
        require_order(
            "publication, cleanup failure, replacement-frame completion",
            positions("plan_published"), positions("processor_cleanup_failed"),
            positions("frame_completed", plan_id=2),
        )
        if len(positions("processor_cleanup_attempted")) != 1:
            errors.append(f"{campaign_id}: expected exactly one cleanup attempt")
    elif campaign_id == "frame_exception":
        require_order(
            "frame failure with lease release",
            positions("lease_acquired"), positions("lease_released"), positions("frame_failed"),
        )
    elif campaign_id == "state_policy":
        if len(positions("processor_reset")) != 1 or len(positions("request_committed")) != 2:
            errors.append(f"{campaign_id}: preserve/reset policy schedule was not exercised")
    elif campaign_id == "successive_generation_ownership":
        if len(positions("plan_published")) != 3 or len(positions("processor_cleanup_completed")) != 2:
            errors.append(f"{campaign_id}: successive generation ownership schedule was not exercised")
    elif campaign_id:
        errors.append(f"unknown controlled campaign {campaign_id!r}")
    return errors


def validate_conformance_directory(
    path: str,
    required_campaigns: frozenset[str],
    randomized_seeds: tuple[int, ...],
) -> tuple[ReplayReport, ...]:
    trace_paths = sorted(
        os.path.join(path, name) for name in os.listdir(path) if name.endswith(".jsonl")
    )
    reports = tuple(replay_trace(trace_path) for trace_path in trace_paths)
    failures = [report for report in reports if not report.passed]
    if failures:
        details = "; ".join(f"{r.path}: {r.errors}" for r in failures)
        raise ValueError(f"conformance replay failed: {details}")
    present = {report.campaign_id for report in reports}
    missing = required_campaigns.difference(present)
    if missing:
        raise ValueError(f"required conformance campaigns are missing: {sorted(missing)!r}")
    observed_random_seeds = {
        report.seed for report in reports if report.campaign_id == "randomized_consistency"
    }
    if observed_random_seeds != set(randomized_seeds):
        raise ValueError(
            "randomized conformance seeds mismatch: "
            f"expected {list(randomized_seeds)!r}, got {sorted(observed_random_seeds)!r}"
        )
    return reports
