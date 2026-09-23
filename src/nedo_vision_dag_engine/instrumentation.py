"""Immutable runtime events, append-only logging, and derived measurements."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

from nedo_vision_dag_engine.processor import StateTransitionEvent
from nedo_vision_dag_engine.type_system import StateTransitionPolicy

__all__ = [
    "NanosecondClock",
    "FrameStatus",
    "ReconfigurationStatus",
    "RetirementStatus",
    "RuntimeEventKind",
    "FrameExecutionEvent",
    "ReconfigurationEvent",
    "ReconfigurationMeasurement",
    "ReconfigurationTimings",
    "StateReuseEvent",
    "RuntimeEvent",
    "RuntimeEventEnvelope",
    "RuntimeMetricsSnapshot",
    "RuntimeInstrumentation",
    "RuntimeEvidenceSink",
    "emit_runtime_evidence",
    "install_runtime_evidence_sink",
    "runtime_evidence_sink",
]


type NanosecondClock = Callable[[], int]
type JsonValue = None | bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue]
type RuntimeEvidenceSink = Callable[[str, Mapping[str, JsonValue]], None]

_evidence_lock = Lock()
_evidence_sink: RuntimeEvidenceSink | None = None


def install_runtime_evidence_sink(
    sink: RuntimeEvidenceSink | None,
) -> RuntimeEvidenceSink | None:
    """Install a process-wide evidence sink and return the previous sink.

    Conformance campaigns are serialized.  A process-wide sink intentionally
    captures events from their worker and retirement threads as well.

    The lock serializes installers against each other so the returned previous
    sink is coherent. Readers do not take it: a campaign installs its sink
    before it starts and removes it after it ends, so a reader never needs to
    observe an installation that is still in progress.
    """
    global _evidence_sink
    with _evidence_lock:
        previous = _evidence_sink
        _evidence_sink = sink
        return previous


def runtime_evidence_sink() -> RuntimeEvidenceSink | None:
    """Return the installed sink, or None.

    Reading the global is a single atomic load under both the default and the
    free-threaded builds, so this takes no lock. Callers on a per-frame path
    read it once and branch on the result rather than calling
    :func:`emit_runtime_evidence` per node: building the keyword payload costs
    far more than the event is worth when nothing is listening.
    """
    return _evidence_sink


def emit_runtime_evidence(kind: str, **fields: JsonValue) -> None:
    """Emit one evidence event when a sink is installed.

    This is for paths that run once per request or per transition. Per-node and
    per-frame paths must use :func:`runtime_evidence_sink` instead so that an
    idle runtime pays nothing.
    """
    sink = _evidence_sink
    if sink is not None:
        sink(kind, fields)


class FrameStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ReconfigurationStatus(Enum):
    RECEIVED = "received"
    VALIDATING = "validating"
    PREPARING = "preparing"
    READY = "ready"
    COMMITTED = "committed"
    REJECTED = "rejected"
    FAILED = "failed"
    STALE = "stale"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ReconfigurationStatus.COMMITTED,
            ReconfigurationStatus.REJECTED,
            ReconfigurationStatus.FAILED,
            ReconfigurationStatus.STALE,
            ReconfigurationStatus.ABORTED,
        }


class RetirementStatus(Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"


class RuntimeEventKind(Enum):
    FRAME_EXECUTION = "frame_execution"
    RECONFIGURATION_TRANSITION = "reconfiguration_transition"
    RECONFIGURATION_MEASUREMENT = "reconfiguration_measurement"
    STATE_TRANSITION = "state_transition"
    STATE_REUSE = "state_reuse"


@dataclass(frozen=True, slots=True)
class FrameExecutionEvent:
    frame_id: int
    plan_version: int
    admission_timestamp_ns: int
    completion_timestamp_ns: int
    executed_node_ids: tuple[str, ...]
    skipped_node_ids: tuple[str, ...]
    status: FrameStatus
    error: str | None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative.")
        if self.plan_version < 1:
            raise ValueError("plan_version must be positive.")
        _validate_timestamp(self.admission_timestamp_ns, "admission_timestamp_ns")
        _validate_timestamp(self.completion_timestamp_ns, "completion_timestamp_ns")
        if self.completion_timestamp_ns < self.admission_timestamp_ns:
            raise ValueError("completion_timestamp_ns must not precede admission_timestamp_ns.")
        if self.status is FrameStatus.COMPLETED and self.error is not None:
            raise ValueError("a completed frame must not carry an error.")
        if self.status is FrameStatus.FAILED and self.error is None:
            raise ValueError("a failed frame must carry an error.")

    @property
    def latency_ns(self) -> int:
        return self.completion_timestamp_ns - self.admission_timestamp_ns


@dataclass(frozen=True, slots=True)
class ReconfigurationEvent:
    request_id: str
    status: ReconfigurationStatus
    occurred_at_ns: int
    reason: str | None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        _validate_timestamp(self.occurred_at_ns, "occurred_at_ns")


@dataclass(frozen=True, slots=True)
class ReconfigurationMeasurement:
    request_id: str
    base_version: int
    candidate_version: int | None
    specification_hash: str
    submitted_at_ns: int
    request_received_ns: int
    validation_started_ns: int | None
    validation_completed_ns: int | None
    preparation_started_ns: int | None
    preparation_completed_ns: int | None
    ready_ns: int | None
    commit_started_ns: int | None
    committed_at_ns: int | None
    first_new_frame_admitted_ns: int | None
    first_new_frame_completed_ns: int | None
    retirement_started_ns: int | None
    retirement_completed_ns: int | None
    terminal_status: ReconfigurationStatus
    failure_reason: str | None
    candidate_cleanup_failure_count: int
    retirement_failure_count: int
    old_plan_frames_admitted_after_request_before_commit: int = 0
    retirement_status: RetirementStatus = RetirementStatus.NOT_REQUIRED
    retirement_failure_reason: str | None = None
    grace_period_start_ns: int | None = None
    grace_period_complete_ns: int | None = None
    last_old_frame_completed_ns: int | None = None
    handoff_wait_start_ns: int | None = None
    handoff_wait_complete_ns: int | None = None
    cleanup_start_ns: int | None = None
    cleanup_end_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        if self.base_version < 1:
            raise ValueError("base_version must be positive.")
        if self.candidate_version is not None and self.candidate_version < 1:
            raise ValueError("candidate_version must be positive when present.")
        if not self.specification_hash:
            raise ValueError("specification_hash must be non-empty.")
        for name, value in self._timestamps():
            if value is not None:
                _validate_timestamp(value, name)
        if self.candidate_cleanup_failure_count < 0:
            raise ValueError("candidate_cleanup_failure_count must be non-negative.")
        if self.retirement_failure_count < 0:
            raise ValueError("retirement_failure_count must be non-negative.")
        if self.old_plan_frames_admitted_after_request_before_commit < 0:
            raise ValueError("old_plan_frames_admitted_after_request_before_commit must be non-negative.")
        _validate_ordered_pair(
            self.validation_started_ns,
            self.validation_completed_ns,
            "validation",
        )
        _validate_ordered_pair(
            self.preparation_started_ns,
            self.preparation_completed_ns,
            "preparation",
        )
        _validate_ordered_pair(self.ready_ns, self.commit_started_ns, "boundary wait")
        _validate_ordered_pair(self.commit_started_ns, self.committed_at_ns, "commit")
        _validate_ordered_pair(
            self.committed_at_ns,
            self.retirement_started_ns,
            "retirement queue",
        )
        _validate_ordered_pair(
            self.retirement_started_ns,
            self.retirement_completed_ns,
            "retirement execution",
        )
        _validate_ordered_pair(
            self.grace_period_start_ns,
            self.grace_period_complete_ns,
            "grace period",
        )
        _validate_ordered_pair(
            self.handoff_wait_start_ns,
            self.handoff_wait_complete_ns,
            "handoff wait",
        )
        _validate_ordered_pair(
            self.cleanup_start_ns,
            self.cleanup_end_ns,
            "cleanup",
        )

    def _timestamps(self) -> tuple[tuple[str, int | None], ...]:
        return (
            ("submitted_at_ns", self.submitted_at_ns),
            ("request_received_ns", self.request_received_ns),
            ("validation_started_ns", self.validation_started_ns),
            ("validation_completed_ns", self.validation_completed_ns),
            ("preparation_started_ns", self.preparation_started_ns),
            ("preparation_completed_ns", self.preparation_completed_ns),
            ("ready_ns", self.ready_ns),
            ("commit_started_ns", self.commit_started_ns),
            ("committed_at_ns", self.committed_at_ns),
            ("first_new_frame_admitted_ns", self.first_new_frame_admitted_ns),
            ("first_new_frame_completed_ns", self.first_new_frame_completed_ns),
            ("retirement_started_ns", self.retirement_started_ns),
            ("retirement_completed_ns", self.retirement_completed_ns),
            ("grace_period_start_ns", self.grace_period_start_ns),
            ("grace_period_complete_ns", self.grace_period_complete_ns),
            ("last_old_frame_completed_ns", self.last_old_frame_completed_ns),
            ("handoff_wait_start_ns", self.handoff_wait_start_ns),
            ("handoff_wait_complete_ns", self.handoff_wait_complete_ns),
            ("cleanup_start_ns", self.cleanup_start_ns),
            ("cleanup_end_ns", self.cleanup_end_ns),
        )

    @property
    def timings(self) -> ReconfigurationTimings:
        return ReconfigurationTimings(
            validation_ns=_duration(self.validation_started_ns, self.validation_completed_ns),
            preparation_ns=_duration(self.preparation_started_ns, self.preparation_completed_ns),
            request_to_ready_ns=_duration(self.request_received_ns, self.ready_ns),
            boundary_wait_ns=_duration(self.ready_ns, self.commit_started_ns),
            commit_ns=_duration(self.commit_started_ns, self.committed_at_ns),
            request_to_effect_ns=_duration(
                self.request_received_ns,
                self.first_new_frame_completed_ns,
            ),
            retirement_queue_delay_ns=_duration(self.committed_at_ns, self.retirement_started_ns),
            retirement_duration_ns=_duration(self.retirement_started_ns, self.retirement_completed_ns),
            commit_to_retirement_complete_ns=_duration(self.committed_at_ns, self.retirement_completed_ns),
            grace_period_ns=_duration(self.grace_period_start_ns, self.grace_period_complete_ns),
            handoff_wait_ns=_duration(self.handoff_wait_start_ns, self.handoff_wait_complete_ns),
            cleanup_duration_ns=_duration(self.cleanup_start_ns, self.cleanup_end_ns),
        )


@dataclass(frozen=True, slots=True)
class ReconfigurationTimings:
    validation_ns: int | None
    preparation_ns: int | None
    request_to_ready_ns: int | None
    boundary_wait_ns: int | None
    commit_ns: int | None
    request_to_effect_ns: int | None
    retirement_queue_delay_ns: int | None
    retirement_duration_ns: int | None
    commit_to_retirement_complete_ns: int | None
    grace_period_ns: int | None = None
    handoff_wait_ns: int | None = None
    cleanup_duration_ns: int | None = None


@dataclass(frozen=True, slots=True)
class StateReuseEvent:
    node_id: str
    old_plan_version: int
    new_plan_version: int
    committed_at_ns: int

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty.")
        if self.old_plan_version < 1 or self.new_plan_version < 1:
            raise ValueError("plan versions must be positive.")
        if self.new_plan_version <= self.old_plan_version:
            raise ValueError("new_plan_version must be greater than old_plan_version.")
        _validate_timestamp(self.committed_at_ns, "committed_at_ns")


type RuntimeEvent = (
    FrameExecutionEvent
    | ReconfigurationEvent
    | ReconfigurationMeasurement
    | StateTransitionEvent
    | StateReuseEvent
)


@dataclass(frozen=True, slots=True)
class RuntimeEventEnvelope:
    sequence_number: int
    recorded_at_ns: int
    kind: RuntimeEventKind
    event: RuntimeEvent

    def __post_init__(self) -> None:
        if self.sequence_number < 1:
            raise ValueError("sequence_number must be positive.")
        _validate_timestamp(self.recorded_at_ns, "recorded_at_ns")


@dataclass(frozen=True, slots=True)
class RuntimeMetricsSnapshot:
    frame_count: int
    completed_frame_count: int
    failed_frame_count: int
    duplicated_frame_count: int
    mixed_version_frame_count: int
    maximum_output_gap_ns: int | None
    maximum_frame_latency_ns: int | None
    reconfiguration_count: int
    committed_reconfiguration_count: int
    rejected_reconfiguration_count: int
    failed_reconfiguration_count: int
    stale_reconfiguration_count: int
    aborted_reconfiguration_count: int
    state_reset_count: int
    state_reuse_count: int


class RuntimeInstrumentation:
    """Thread-safe append-only event store shared by one pipeline runtime."""

    __slots__ = (
        "_clock",
        "_events",
        "_latest_reconfiguration_measurements",
        "_lock",
        "_next_sequence_number",
    )

    def __init__(self, clock: NanosecondClock = time.monotonic_ns) -> None:
        self._clock = clock
        self._lock = Lock()
        self._events: list[RuntimeEventEnvelope] = []
        self._latest_reconfiguration_measurements: dict[str, ReconfigurationMeasurement] = {}
        self._next_sequence_number = 1

    def record_frame(self, event: FrameExecutionEvent) -> RuntimeEventEnvelope:
        return self._append(RuntimeEventKind.FRAME_EXECUTION, event)

    def record_reconfiguration_event(
        self,
        event: ReconfigurationEvent,
    ) -> RuntimeEventEnvelope:
        event_kind = {
            ReconfigurationStatus.RECEIVED: "request_received",
            ReconfigurationStatus.VALIDATING: "request_validated",
            ReconfigurationStatus.PREPARING: "request_prepared",
            ReconfigurationStatus.READY: "request_ready",
            ReconfigurationStatus.COMMITTED: "request_committed",
            ReconfigurationStatus.REJECTED: "request_rejected",
            ReconfigurationStatus.FAILED: "request_failed",
            ReconfigurationStatus.STALE: "request_rejected",
            ReconfigurationStatus.ABORTED: "request_aborted",
        }[event.status]
        emit_runtime_evidence(
            event_kind,
            request_id=event.request_id,
            outcome=event.status.value,
            error_message=event.reason,
        )
        return self._append(RuntimeEventKind.RECONFIGURATION_TRANSITION, event)

    def record_reconfiguration_measurement(
        self,
        event: ReconfigurationMeasurement,
    ) -> RuntimeEventEnvelope:
        with self._lock:
            self._latest_reconfiguration_measurements[event.request_id] = event
            return self._append_locked(RuntimeEventKind.RECONFIGURATION_MEASUREMENT, event)

    def record_state_transition(
        self,
        event: StateTransitionEvent,
    ) -> RuntimeEventEnvelope:
        emit_runtime_evidence(
            "processor_reset",
            node_id=event.node_id,
            plan_id=event.new_plan_version,
            outcome=event.policy.value,
        )
        return self._append(RuntimeEventKind.STATE_TRANSITION, event)

    def record_state_reuse(self, event: StateReuseEvent) -> RuntimeEventEnvelope:
        emit_runtime_evidence(
            "processor_reused",
            node_id=event.node_id,
            plan_id=event.new_plan_version,
            outcome="reused",
        )
        return self._append(RuntimeEventKind.STATE_REUSE, event)

    def event_log(self) -> tuple[RuntimeEventEnvelope, ...]:
        with self._lock:
            return tuple(self._events)

    def frame_events(self) -> tuple[FrameExecutionEvent, ...]:
        return tuple(
            envelope.event
            for envelope in self.event_log()
            if isinstance(envelope.event, FrameExecutionEvent)
        )

    def reconfiguration_events(self) -> tuple[ReconfigurationEvent, ...]:
        return tuple(
            envelope.event
            for envelope in self.event_log()
            if isinstance(envelope.event, ReconfigurationEvent)
        )

    def state_transition_events(self) -> tuple[StateTransitionEvent, ...]:
        return tuple(
            envelope.event
            for envelope in self.event_log()
            if isinstance(envelope.event, StateTransitionEvent)
        )

    def state_reuse_events(self) -> tuple[StateReuseEvent, ...]:
        return tuple(
            envelope.event
            for envelope in self.event_log()
            if isinstance(envelope.event, StateReuseEvent)
        )

    def reconfiguration_measurement(
        self,
        request_id: str,
    ) -> ReconfigurationMeasurement:
        with self._lock:
            measurement = self._latest_reconfiguration_measurements.get(request_id)
            if measurement is None:
                raise KeyError(f"unknown reconfiguration request id {request_id!r}.")
            return measurement

    def reconfiguration_measurements(self) -> tuple[ReconfigurationMeasurement, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._latest_reconfiguration_measurements.values(),
                    key=lambda measurement: (
                        measurement.request_received_ns,
                        measurement.request_id,
                    ),
                )
            )

    def metrics_snapshot(self) -> RuntimeMetricsSnapshot:
        with self._lock:
            events = tuple(self._events)
            measurements = tuple(
                sorted(
                    self._latest_reconfiguration_measurements.values(),
                    key=lambda measurement: (
                        measurement.request_received_ns,
                        measurement.request_id,
                    ),
                )
            )
        frame_events = tuple(
            envelope.event
            for envelope in events
            if isinstance(envelope.event, FrameExecutionEvent)
        )
        transition_events = tuple(
            envelope.event
            for envelope in events
            if isinstance(envelope.event, StateTransitionEvent)
        )
        reuse_events = tuple(
            envelope.event
            for envelope in events
            if isinstance(envelope.event, StateReuseEvent)
        )

        frame_occurrences: dict[int, int] = {}
        frame_versions: dict[int, set[int]] = {}
        for event in frame_events:
            frame_occurrences[event.frame_id] = frame_occurrences.get(event.frame_id, 0) + 1
            frame_versions.setdefault(event.frame_id, set()).add(event.plan_version)

        completion_timestamps = tuple(event.completion_timestamp_ns for event in frame_events)
        output_gaps = tuple(
            current - previous
            for previous, current in zip(
                completion_timestamps,
                completion_timestamps[1:],
                strict=False,
            )
            if current >= previous
        )
        frame_latencies = tuple(event.latency_ns for event in frame_events)

        return RuntimeMetricsSnapshot(
            frame_count=len(frame_events),
            completed_frame_count=sum(
                event.status is FrameStatus.COMPLETED for event in frame_events
            ),
            failed_frame_count=sum(event.status is FrameStatus.FAILED for event in frame_events),
            duplicated_frame_count=sum(
                occurrence_count - 1
                for occurrence_count in frame_occurrences.values()
                if occurrence_count > 1
            ),
            mixed_version_frame_count=sum(
                len(versions) > 1 for versions in frame_versions.values()
            ),
            maximum_output_gap_ns=max(output_gaps, default=None),
            maximum_frame_latency_ns=max(frame_latencies, default=None),
            reconfiguration_count=len(measurements),
            committed_reconfiguration_count=_count_status(
                measurements,
                ReconfigurationStatus.COMMITTED,
            ),
            rejected_reconfiguration_count=_count_status(
                measurements,
                ReconfigurationStatus.REJECTED,
            ),
            failed_reconfiguration_count=_count_status(
                measurements,
                ReconfigurationStatus.FAILED,
            ),
            stale_reconfiguration_count=_count_status(
                measurements,
                ReconfigurationStatus.STALE,
            ),
            aborted_reconfiguration_count=_count_status(
                measurements,
                ReconfigurationStatus.ABORTED,
            ),
            state_reset_count=sum(
                event.policy is StateTransitionPolicy.RESET for event in transition_events
            ),
            state_reuse_count=len(reuse_events),
        )

    def json_lines(self) -> tuple[str, ...]:
        return tuple(
            json.dumps(
                _envelope_to_json(envelope),
                sort_keys=True,
                separators=(",", ":"),
            )
            for envelope in self.event_log()
        )

    def write_json_lines(self, path: str | Path) -> None:
        destination = Path(path)
        lines = self.json_lines()
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
                stream.write("\n")

    def _append(
        self,
        kind: RuntimeEventKind,
        event: RuntimeEvent,
    ) -> RuntimeEventEnvelope:
        with self._lock:
            return self._append_locked(kind, event)

    def _append_locked(
        self,
        kind: RuntimeEventKind,
        event: RuntimeEvent,
    ) -> RuntimeEventEnvelope:
        envelope = RuntimeEventEnvelope(
            sequence_number=self._next_sequence_number,
            recorded_at_ns=self._clock(),
            kind=kind,
            event=event,
        )
        self._events.append(envelope)
        self._next_sequence_number += 1
        return envelope


def _validate_timestamp(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_ordered_pair(start: int | None, end: int | None, label: str) -> None:
    if start is not None and end is not None and end < start:
        raise ValueError(f"{label} end timestamp must not precede its start timestamp.")


def _duration(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return end - start


def _count_status(
    measurements: tuple[ReconfigurationMeasurement, ...],
    status: ReconfigurationStatus,
) -> int:
    return sum(measurement.terminal_status is status for measurement in measurements)


def _optional_int(value: int | None) -> JsonValue:
    return value


def _optional_string(value: str | None) -> JsonValue:
    return value


def _envelope_to_json(envelope: RuntimeEventEnvelope) -> dict[str, JsonValue]:
    return {
        "sequence_number": envelope.sequence_number,
        "recorded_at_ns": envelope.recorded_at_ns,
        "kind": envelope.kind.value,
        "event": _event_to_json(envelope.event),
    }


def _event_to_json(event: RuntimeEvent) -> dict[str, JsonValue]:
    if isinstance(event, FrameExecutionEvent):
        return {
            "frame_id": event.frame_id,
            "plan_version": event.plan_version,
            "admission_timestamp_ns": event.admission_timestamp_ns,
            "completion_timestamp_ns": event.completion_timestamp_ns,
            "executed_node_ids": event.executed_node_ids,
            "skipped_node_ids": event.skipped_node_ids,
            "status": event.status.value,
            "error": _optional_string(event.error),
        }
    if isinstance(event, ReconfigurationEvent):
        return {
            "request_id": event.request_id,
            "status": event.status.value,
            "occurred_at_ns": event.occurred_at_ns,
            "reason": _optional_string(event.reason),
        }
    if isinstance(event, ReconfigurationMeasurement):
        return {
            "request_id": event.request_id,
            "base_version": event.base_version,
            "candidate_version": _optional_int(event.candidate_version),
            "specification_hash": event.specification_hash,
            "submitted_at_ns": event.submitted_at_ns,
            "request_received_ns": event.request_received_ns,
            "validation_started_ns": _optional_int(event.validation_started_ns),
            "validation_completed_ns": _optional_int(event.validation_completed_ns),
            "preparation_started_ns": _optional_int(event.preparation_started_ns),
            "preparation_completed_ns": _optional_int(event.preparation_completed_ns),
            "ready_ns": _optional_int(event.ready_ns),
            "commit_started_ns": _optional_int(event.commit_started_ns),
            "committed_at_ns": _optional_int(event.committed_at_ns),
            "first_new_frame_admitted_ns": _optional_int(event.first_new_frame_admitted_ns),
            "first_new_frame_completed_ns": _optional_int(event.first_new_frame_completed_ns),
            "retirement_started_ns": _optional_int(event.retirement_started_ns),
            "retirement_completed_ns": _optional_int(event.retirement_completed_ns),
            "terminal_status": event.terminal_status.value,
            "failure_reason": _optional_string(event.failure_reason),
            "candidate_cleanup_failure_count": event.candidate_cleanup_failure_count,
            "retirement_failure_count": event.retirement_failure_count,
            "old_plan_frames_admitted_after_request_before_commit": event.old_plan_frames_admitted_after_request_before_commit,
            "retirement_status": event.retirement_status.value,
            "retirement_failure_reason": _optional_string(event.retirement_failure_reason),
            "grace_period_start_ns": _optional_int(event.grace_period_start_ns),
            "grace_period_complete_ns": _optional_int(event.grace_period_complete_ns),
            "last_old_frame_completed_ns": _optional_int(event.last_old_frame_completed_ns),
            "handoff_wait_start_ns": _optional_int(event.handoff_wait_start_ns),
            "handoff_wait_complete_ns": _optional_int(event.handoff_wait_complete_ns),
            "cleanup_start_ns": _optional_int(event.cleanup_start_ns),
            "cleanup_end_ns": _optional_int(event.cleanup_end_ns),
        }
    if isinstance(event, StateTransitionEvent):
        return {
            "node_id": event.node_id,
            "old_plan_version": event.old_plan_version,
            "new_plan_version": event.new_plan_version,
            "policy": event.policy.value,
            "reason": event.reason,
            "committed_at_ns": event.committed_at_ns,
        }
    return {
        "node_id": event.node_id,
        "old_plan_version": event.old_plan_version,
        "new_plan_version": event.new_plan_version,
        "committed_at_ns": event.committed_at_ns,
    }
