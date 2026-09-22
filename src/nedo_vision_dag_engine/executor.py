"""Static frame execution, frame instrumentation, and atomic plan publication."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock

from nedo_vision_dag_engine.instrumentation import (
    FrameExecutionEvent,
    FrameStatus,
    NanosecondClock,
    RuntimeInstrumentation,
    emit_runtime_evidence,
    runtime_evidence_sink,
)
from nedo_vision_dag_engine.lifecycle import PlanLifecycleState
from nedo_vision_dag_engine.plan import ExecutionPlan, ExecutionStep, construct_inputs
from nedo_vision_dag_engine.processor import FrameContext
from nedo_vision_dag_engine.type_system import MISSING
from nedo_vision_dag_engine.workspace import Workspace, WorkspacePool

__all__ = [
    "FrameStatus",
    "FrameResult",
    "PlanSwap",
    "PublicationResult",
    "execute_frame",
    "PipelineExecutor",
]

# PublicationResult is a alias of PlanSwap for plan publication
type PublicationResult = PlanSwap


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_id: int
    plan_version: int
    status: FrameStatus
    executed_node_ids: tuple[str, ...]
    skipped_node_ids: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PlanSwap:
    previous_plan: ExecutionPlan
    active_plan: ExecutionPlan


@dataclass(frozen=True, slots=True)
class FrameAdmission:
    """Internal coordination record for one controller-admitted frame.

    The controller obtains this record while its admission gate is closed and
    must pass it exactly once to :meth:`execute_admitted_frame`.
    """

    admission_id: int
    plan: ExecutionPlan
    frame_id: int
    admitted_at_ns: int


def _materialize_inputs(step: ExecutionStep, raw_inputs: dict[str, object]) -> object:
    input_schema = step.processor_ref.descriptor.input_schema
    if input_schema is object:
        return raw_inputs
    try:
        return input_schema(**raw_inputs)
    except TypeError as error:
        raise ValueError(
            f"failed to materialize inputs for node {step.node_id!r} into {input_schema!r}: {error}"
        ) from error


def execute_frame(
    plan: ExecutionPlan,
    frame_id: int,
    admitted_at_ns: int,
    workspace: Workspace,
) -> FrameResult:
    context = FrameContext(frame_id=frame_id, plan_version=plan.version, admitted_at_ns=admitted_at_ns)
    executed: list[str] = []
    skipped: list[str] = []
    # Read once per frame, not once per node: with no sink installed the whole
    # node-level evidence path must cost nothing, because steady-state
    # measurements run with evidence disabled.
    evidence = runtime_evidence_sink()

    for step in plan.steps:
        if not step.readiness_rule.is_ready(workspace):
            workspace.set(step.output_index, MISSING)
            skipped.append(step.node_id)
            continue

        try:
            if evidence is not None:
                evidence(
                    "node_entered",
                    {
                        "frame_id": frame_id,
                        "plan_id": plan.version,
                        "node_id": step.node_id,
                        "processor_instance_id": f"0x{id(step.processor_ref):x}",
                    },
                )
            raw_inputs = construct_inputs(step.input_bindings, workspace)
            materialized_inputs = _materialize_inputs(step, raw_inputs)
            output = step.processor_ref.process(materialized_inputs, context)
        except Exception as error:
            if evidence is not None:
                evidence(
                    "node_exited",
                    {
                        "frame_id": frame_id,
                        "plan_id": plan.version,
                        "node_id": step.node_id,
                        "processor_instance_id": f"0x{id(step.processor_ref):x}",
                        "outcome": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
            return FrameResult(
                frame_id=frame_id,
                plan_version=plan.version,
                status=FrameStatus.FAILED,
                executed_node_ids=tuple(executed),
                skipped_node_ids=tuple(skipped),
                error=repr(error),
            )

        workspace.set(step.output_index, output)
        executed.append(step.node_id)
        if evidence is not None:
            evidence(
                "node_exited",
                {
                    "frame_id": frame_id,
                    "plan_id": plan.version,
                    "node_id": step.node_id,
                    "processor_instance_id": f"0x{id(step.processor_ref):x}",
                    "outcome": "completed",
                },
            )

    return FrameResult(
        frame_id=frame_id,
        plan_version=plan.version,
        status=FrameStatus.COMPLETED,
        executed_node_ids=tuple(executed),
        skipped_node_ids=tuple(skipped),
        error=None,
    )


class PipelineExecutor:
    """Orders frame admission with publication and protects leased execution.

    Two locks cooperate to separate plan-reference access from frame
    execution:

    - ``_plan_lock`` protects reads and writes of the ``_active_plan``
      reference.  Any caller that only needs to read the current plan
      (e.g. a reconfiguration controller capturing a snapshot for
      background compilation) acquires this lock alone and releases it
      in microseconds, even while a frame is mid-execution.

    - ``_admission_lock`` serializes the admission linearization point with
      plan publication.  Admission selects one immutable plan and increments
      its lease while this lock is held, then releases it before execution.

    - ``_frame_execution_lock`` preserves the executor's single-frame
      processor and workspace-pool contract.  Publication never acquires this
      lock, so it may complete while an already-admitted old-plan frame is
      still executing.

    Lock ordering is always ``_admission_lock`` then ``_plan_lock`` then
    ``_in_flight_lock``.  Frame execution holds none of those locks.
    """

    __slots__ = (
        "_active_plan",
        "_admission_lock",
        "_clock",
        "_frame_execution_lock",
        "_in_flight_lock",
        "_instrumentation",
        "_manager_token",
        "_next_admission_id",
        "_next_frame_id",
        "_pending_admissions",
        "_plan_inflight",
        "_plan_last_completion_ns",
        "_plan_lifecycle",
        "_plan_lock",
        "_plan_quiescent_events",
        "_workspace_pool",
    )

    def __init__(
        self,
        initial_plan: ExecutionPlan,
        workspace_pool: WorkspacePool | None = None,
        instrumentation: RuntimeInstrumentation | None = None,
        clock: NanosecondClock = time.monotonic_ns,
    ) -> None:
        self._active_plan = initial_plan
        self._workspace_pool = workspace_pool if workspace_pool is not None else WorkspacePool()
        self._next_admission_id = 0
        self._next_frame_id = 0
        self._plan_lock = Lock()
        self._admission_lock = Lock()
        self._frame_execution_lock = Lock()
        self._in_flight_lock = Lock()
        self._manager_token: object | None = None
        self._clock = clock
        self._instrumentation = (
            instrumentation if instrumentation is not None else RuntimeInstrumentation(clock=clock)
        )
        self._plan_inflight: dict[int, int] = {}
        self._plan_last_completion_ns: dict[int, int] = {}
        self._plan_quiescent_events: dict[int, Event] = {}
        self._pending_admissions: dict[int, FrameAdmission] = {}
        self._plan_lifecycle: dict[int, PlanLifecycleState] = {
            initial_plan.version: PlanLifecycleState.ACTIVE,
        }

    @property
    def active_plan(self) -> ExecutionPlan:
        with self._plan_lock:
            return self._active_plan

    def active_plan_snapshot(self) -> ExecutionPlan:
        """Return the current active plan without holding the execution lock.

        This is safe for a reconfiguration controller to call while a
        frame is executing: the plan reference itself is immutable and
        the read is protected by ``_plan_lock`` alone.
        """
        with self._plan_lock:
            return self._active_plan

    def plan_lifecycle_state(self, plan_version: int) -> PlanLifecycleState | None:
        """Return the lifecycle state for a plan version, or None if unknown."""
        with self._plan_lock:
            return self._plan_lifecycle.get(plan_version)

    def wait_for_plan_quiescent(self, plan_version: int, timeout: float | None = None) -> bool:
        """Block until no frames are in-flight on the given plan version.

        Returns True if the plan became quiescent, False on timeout.
        Safe to call from any thread; does not acquire ``_admission_lock``.
        """
        event: Event | None = None
        with self._in_flight_lock:
            count = self._plan_inflight.get(plan_version, 0)
            if count == 0:
                return True
            event = self._plan_quiescent_events.setdefault(plan_version, Event())

        # Wait outside any lock so frames can complete
        assert event is not None
        result = event.wait(timeout=timeout)
        return result

    def _increment_inflight(self, plan_version: int) -> None:
        """Increment the in-flight counter for a plan version.

        Caller may hold ``_admission_lock`` or ``_plan_lock``; this method
        acquires ``_in_flight_lock`` internally.
        """
        with self._in_flight_lock:
            self._plan_inflight[plan_version] = self._plan_inflight.get(plan_version, 0) + 1

    def _decrement_inflight(self, plan_version: int, completed_at_ns: int | None) -> None:
        """Decrement the in-flight counter and signal waiters if it reaches zero.

        Acquires ``_in_flight_lock`` internally.  Safe to call while holding
        ``_admission_lock``.
        """
        with self._in_flight_lock:
            if completed_at_ns is not None:
                previous_completion_ns = self._plan_last_completion_ns.get(plan_version)
                if previous_completion_ns is None or completed_at_ns > previous_completion_ns:
                    self._plan_last_completion_ns[plan_version] = completed_at_ns
            current = self._plan_inflight.get(plan_version, 0)
            if current <= 0:
                raise RuntimeError(
                    f"plan version {plan_version} has no in-flight lease to release"
                )
            new_count = current - 1
            if new_count == 0:
                del self._plan_inflight[plan_version]
                event = self._plan_quiescent_events.pop(plan_version, None)
                if event is not None:
                    event.set()
            else:
                self._plan_inflight[plan_version] = new_count

    def last_frame_completion_ns(self, plan_version: int) -> int | None:
        """Return the latest observed lease-release boundary for a plan."""
        with self._in_flight_lock:
            return self._plan_last_completion_ns.get(plan_version)

    def update_plan_lifecycle(self, plan_version: int, state: PlanLifecycleState) -> None:
        """Update the lifecycle state for a plan version."""
        with self._plan_lock:
            self._plan_lifecycle[plan_version] = state

    def claim_management(self, token: object) -> None:
        """Claim exclusive management of this executor."""
        with self._admission_lock:
            with self._plan_lock:
                if self._manager_token is not None:
                    raise RuntimeError("executor is already managed")
                self._manager_token = token

    def release_management(self, token: object) -> None:
        """Release exclusive management of this executor."""
        with self._admission_lock:
            with self._plan_lock:
                if self._manager_token is not token:
                    raise RuntimeError("invalid management token for release")
                self._manager_token = None

    def _require_unmanaged(self) -> None:
        if self._manager_token is not None:
            raise RuntimeError(
                "this executor is managed by a ReconfigurationController; "
                "use the controller's admit_frame() method instead."
            )

    @property
    def instrumentation(self) -> RuntimeInstrumentation:
        return self._instrumentation

    @property
    def clock(self) -> NanosecondClock:
        """Return the executor's monotonic clock.

        Managed controllers must use this exact clock so every runtime
        timestamp belongs to one comparable domain.
        """
        return self._clock

    def commit(self, new_plan: ExecutionPlan) -> PlanSwap:
        """Public commit. Raises if managed by a controller."""
        with self._admission_lock:
            with self._plan_lock:
                self._require_unmanaged()
                return self._commit_plan_locked(new_plan)

    def commit_managed(self, token: object, new_plan: ExecutionPlan) -> PlanSwap:
        with self._admission_lock:
            with self._plan_lock:
                if token is not self._manager_token:
                    raise RuntimeError("invalid executor management token")
                return self._commit_plan_locked(new_plan)

    def commit_if_version(self, expected_version: int, new_plan: ExecutionPlan) -> PlanSwap | None:
        """Public conditional commit. Raises if managed by a controller."""
        with self._admission_lock:
            with self._plan_lock:
                self._require_unmanaged()
                if self._active_plan.version != expected_version:
                    return None
                return self._commit_plan_locked(new_plan)

    def commit_if_version_managed(
        self, token: object, expected_version: int, new_plan: ExecutionPlan
    ) -> PlanSwap | None:
        with self._admission_lock:
            with self._plan_lock:
                if token is not self._manager_token:
                    raise RuntimeError("invalid executor management token")
                if self._active_plan.version != expected_version:
                    return None
                return self._commit_plan_locked(new_plan)

    def admit_frame(self, admitted_at_ns: int, frame_id: int | None = None) -> FrameResult:
        """Public admission path. Raises if managed by a controller."""
        admission = self._reserve_frame(
            token=None,
            is_managed=False,
            admitted_at_ns=admitted_at_ns,
            frame_id=frame_id,
        )
        return self.execute_admitted_frame(admission)

    def admit_frame_managed(
        self, token: object, admitted_at_ns: int, frame_id: int | None = None
    ) -> FrameResult:
        """Execute a frame when managed by a controller."""
        admission = self.reserve_frame_managed(token, admitted_at_ns, frame_id)
        return self.execute_admitted_frame(admission)

    def reserve_frame_managed(
        self,
        token: object,
        admitted_at_ns: int,
        frame_id: int | None = None,
    ) -> FrameAdmission:
        """Acquire a managed frame lease without executing processor code.

        This is the controller-facing half of managed admission.  The caller
        must pass the returned record exactly once to
        :meth:`execute_admitted_frame` so the lease is released.
        """
        return self._reserve_frame(
            token=token,
            is_managed=True,
            admitted_at_ns=admitted_at_ns,
            frame_id=frame_id,
        )

    def cancel_reserved_frame_managed(
        self,
        token: object,
        admission: FrameAdmission,
    ) -> None:
        """Cancel an unexecuted managed reservation and release its lease.

        A reservation has exactly one terminal action: execution or
        cancellation.  Foreign and already-consumed records are rejected
        without changing lease accounting.
        """
        with self._admission_lock:
            with self._plan_lock:
                if token is not self._manager_token:
                    raise RuntimeError("invalid executor management token")
            self._consume_admission_locked(admission)
        emit_runtime_evidence(
            "frame_cancelled",
            frame_id=admission.frame_id,
            plan_id=admission.plan.version,
            outcome="reservation_cancelled",
        )
        self._release_admission_lease(admission.plan.version, admission.frame_id)

    def _reserve_frame(
        self, token: object | None, is_managed: bool, admitted_at_ns: int, frame_id: int | None = None
    ) -> FrameAdmission:
        """Select a plan and acquire its lease at the admission boundary."""
        if admitted_at_ns < 0:
            emit_runtime_evidence("frame_offered", frame_id=frame_id)
            emit_runtime_evidence(
                "frame_failed",
                frame_id=frame_id,
                outcome="admission_rejected",
                error_type="ValueError",
                error_message="admitted_at_ns must be non-negative",
            )
            raise ValueError("admitted_at_ns must be non-negative.")
        if frame_id is not None and frame_id < 0:
            emit_runtime_evidence("frame_offered", frame_id=frame_id)
            emit_runtime_evidence(
                "frame_failed",
                frame_id=frame_id,
                outcome="admission_rejected",
                error_type="ValueError",
                error_message="frame_id must be non-negative",
            )
            raise ValueError("frame_id must be non-negative.")

        with self._admission_lock:
            with self._plan_lock:
                if is_managed:
                    if token is not self._manager_token:
                        raise RuntimeError("invalid executor management token")
                else:
                    self._require_unmanaged()
                plan = self._active_plan
                self._increment_inflight(plan.version)
            resolved_frame_id = self._next_frame_id if frame_id is None else frame_id
            evidence = runtime_evidence_sink()
            if evidence is not None:
                evidence("frame_offered", {"frame_id": resolved_frame_id})
            self._next_frame_id = max(self._next_frame_id, resolved_frame_id + 1)
            admission_id = self._next_admission_id
            self._next_admission_id += 1
            admission = FrameAdmission(
                admission_id=admission_id,
                plan=plan,
                frame_id=resolved_frame_id,
                admitted_at_ns=admitted_at_ns,
            )
            self._pending_admissions[admission_id] = admission
            if evidence is not None:
                evidence(
                    "lease_acquired",
                    {"frame_id": resolved_frame_id, "plan_id": plan.version},
                )
                evidence(
                    "frame_admitted",
                    {"frame_id": resolved_frame_id, "plan_id": plan.version},
                )
            return admission

    def execute_admitted_frame(self, admission: FrameAdmission) -> FrameResult:
        """Execute one admitted frame and release its lease on every exit."""
        plan = admission.plan
        with self._admission_lock:
            self._consume_admission_locked(admission)

        try:
            with self._frame_execution_lock:
                workspace = self._workspace_pool.acquire(plan.output_slot_count)
                try:
                    result = execute_frame(
                        plan,
                        admission.frame_id,
                        admission.admitted_at_ns,
                        workspace,
                    )
                finally:
                    self._workspace_pool.release(workspace)
        finally:
            completed_at_ns = self._release_admission_lease(plan.version, admission.frame_id)

        assert completed_at_ns is not None
        evidence = runtime_evidence_sink()
        if evidence is not None:
            evidence(
                "frame_completed" if result.status is FrameStatus.COMPLETED else "frame_failed",
                {
                    "frame_id": result.frame_id,
                    "plan_id": result.plan_version,
                    "outcome": result.status.value,
                    "error_message": result.error,
                },
            )
        self._instrumentation.record_frame(
            FrameExecutionEvent(
                frame_id=result.frame_id,
                plan_version=result.plan_version,
                admission_timestamp_ns=admission.admitted_at_ns,
                completion_timestamp_ns=completed_at_ns,
                executed_node_ids=result.executed_node_ids,
                skipped_node_ids=result.skipped_node_ids,
                status=result.status,
                error=result.error,
            )
        )
        return result

    def _consume_admission_locked(self, admission: FrameAdmission) -> None:
        pending = self._pending_admissions.get(admission.admission_id)
        if pending is not admission:
            raise RuntimeError("frame admission is foreign or has already been consumed")
        del self._pending_admissions[admission.admission_id]

    def _release_admission_lease(self, plan_version: int, frame_id: int) -> int | None:
        """Release one lease even when reading the completion clock fails."""
        completed_at_ns: int | None = None
        try:
            completed_at_ns = self._clock()
            return completed_at_ns
        finally:
            self._decrement_inflight(plan_version, completed_at_ns)
            evidence = runtime_evidence_sink()
            if evidence is not None:
                evidence(
                    "lease_released",
                    {
                        "frame_id": frame_id,
                        "plan_id": plan_version,
                        "outcome": "released",
                    },
                )

    def snapshot_active_plan(self) -> ExecutionPlan:
        """Centralized accessor for taking a thread-safe snapshot of the active plan.

        Protected by ``_plan_lock``. Safe to call without holding ``_admission_lock``.
        """
        with self._plan_lock:
            return self._active_plan

    def publish_candidate_plan(self, new_plan: ExecutionPlan) -> PlanSwap:
        """Centralized publication linearization point for candidate plans.

        Linearization point:
            The plan reference atomic assignment ``self._active_plan = new_plan``
            is performed while holding ``_plan_lock``.
            Caller MUST hold ``_admission_lock`` before calling this method to prevent
            concurrent plan commits or admission races. No external callbacks or
            processor code runs while holding these locks.
        """
        previous_plan = self._active_plan
        if new_plan.version <= previous_plan.version:
            raise ValueError(
                f"new plan version {new_plan.version} must be greater than active plan "
                f"version {previous_plan.version}."
            )
        self._active_plan = new_plan
        emit_runtime_evidence(
            "plan_published",
            plan_id=new_plan.version,
            previous_plan_id=previous_plan.version,
            outcome="committed",
        )

        # Update plan lifecycle states
        self._plan_lifecycle[previous_plan.version] = PlanLifecycleState.SUPERSEDED
        self._plan_lifecycle[new_plan.version] = PlanLifecycleState.ACTIVE

        return PlanSwap(previous_plan=previous_plan, active_plan=new_plan)

    def _commit_plan_locked(self, new_plan: ExecutionPlan) -> PlanSwap:
        """Swap plan while holding ``_plan_lock`` and ``_admission_lock``."""
        return self.publish_candidate_plan(new_plan)
