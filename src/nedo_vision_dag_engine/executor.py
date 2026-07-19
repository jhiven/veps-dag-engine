"""Static frame execution, frame instrumentation, and atomic plan publication."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

from nedo_vision_dag_engine.instrumentation import (
    FrameExecutionEvent,
    FrameStatus,
    NanosecondClock,
    RuntimeInstrumentation,
)
from nedo_vision_dag_engine.plan import ExecutionPlan, ExecutionStep, construct_inputs
from nedo_vision_dag_engine.processor import FrameContext
from nedo_vision_dag_engine.type_system import MISSING
from nedo_vision_dag_engine.workspace import Workspace, WorkspacePool

__all__ = [
    "FrameStatus",
    "FrameResult",
    "PlanSwap",
    "execute_frame",
    "PipelineExecutor",
]


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

    for step in plan.steps:
        if not step.readiness_rule.is_ready(workspace):
            workspace.set(step.output_index, MISSING)
            skipped.append(step.node_id)
            continue

        try:
            raw_inputs = construct_inputs(step.input_bindings, workspace)
            materialized_inputs = _materialize_inputs(step, raw_inputs)
            output = step.processor_ref.process(materialized_inputs, context)
        except Exception as error:
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

    return FrameResult(
        frame_id=frame_id,
        plan_version=plan.version,
        status=FrameStatus.COMPLETED,
        executed_node_ids=tuple(executed),
        skipped_node_ids=tuple(skipped),
        error=None,
    )


class PipelineExecutor:
    """Serializes frame admission and plan publication at one boundary lock."""

    __slots__ = (
        "_active_plan",
        "_boundary_lock",
        "_clock",
        "_instrumentation",
        "_next_frame_id",
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
        self._next_frame_id = 0
        self._boundary_lock = Lock()
        self._clock = clock
        self._instrumentation = (
            instrumentation if instrumentation is not None else RuntimeInstrumentation(clock=clock)
        )

    @property
    def active_plan(self) -> ExecutionPlan:
        with self._boundary_lock:
            return self._active_plan

    @property
    def instrumentation(self) -> RuntimeInstrumentation:
        return self._instrumentation

    def commit(self, new_plan: ExecutionPlan) -> PlanSwap:
        with self._boundary_lock:
            return self._commit_locked(new_plan)

    def commit_if_version(self, expected_version: int, new_plan: ExecutionPlan) -> PlanSwap | None:
        with self._boundary_lock:
            if self._active_plan.version != expected_version:
                return None
            return self._commit_locked(new_plan)

    def admit_frame(self, admitted_at_ns: int, frame_id: int | None = None) -> FrameResult:
        if admitted_at_ns < 0:
            raise ValueError("admitted_at_ns must be non-negative.")

        with self._boundary_lock:
            plan = self._active_plan
            resolved_frame_id = self._next_frame_id if frame_id is None else frame_id
            if resolved_frame_id < 0:
                raise ValueError("frame_id must be non-negative.")
            self._next_frame_id = max(self._next_frame_id, resolved_frame_id + 1)

            workspace = self._workspace_pool.acquire(plan.output_slot_count)
            try:
                result = execute_frame(plan, resolved_frame_id, admitted_at_ns, workspace)
            finally:
                self._workspace_pool.release(workspace)

            completed_at_ns = self._clock()
            self._instrumentation.record_frame(
                FrameExecutionEvent(
                    frame_id=result.frame_id,
                    plan_version=result.plan_version,
                    admission_timestamp_ns=admitted_at_ns,
                    completion_timestamp_ns=completed_at_ns,
                    executed_node_ids=result.executed_node_ids,
                    skipped_node_ids=result.skipped_node_ids,
                    status=result.status,
                    error=result.error,
                )
            )
            return result

    def _commit_locked(self, new_plan: ExecutionPlan) -> PlanSwap:
        previous_plan = self._active_plan
        if new_plan.version <= previous_plan.version:
            raise ValueError(
                f"new plan version {new_plan.version} must be greater than active plan "
                f"version {previous_plan.version}."
            )
        self._active_plan = new_plan
        return PlanSwap(previous_plan=previous_plan, active_plan=new_plan)