"""Processor ownership, rollback, retirement, and shutdown semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nedo_vision_dag_engine.plan import ExecutionPlan
from nedo_vision_dag_engine.processor import Processor

__all__ = [
    "CleanupReason",
    "PlanLifecycleState",
    "ProcessorCleanupFailure",
    "CleanupReport",
    "ProcessorStagingArea",
    "cleanup_candidate_processors",
    "retire_superseded_processors",
    "shutdown_plan_processors",
]


class CleanupReason(Enum):
    STAGING_ROLLBACK = "staging_rollback"
    CANDIDATE_DISCARDED = "candidate_discarded"
    PLAN_RETIREMENT = "plan_retirement"
    RUNTIME_SHUTDOWN = "runtime_shutdown"


class PlanLifecycleState(Enum):
    """Per-plan-version lifecycle states tracked by the executor.

    A plan transitions through these states as it is published, superseded,
    quiesced (all in-flight frames completed), and eventually retired.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    QUIESCENT = "quiescent"
    RETIRED = "retired"
    RETIREMENT_FAILED = "retirement_failed"


@dataclass(frozen=True, slots=True)
class ProcessorCleanupFailure:
    node_id: str
    processor_type: str
    reason: CleanupReason
    error: str


@dataclass(frozen=True, slots=True)
class CleanupReport:
    reason: CleanupReason
    attempted_node_ids: tuple[str, ...]
    cleaned_node_ids: tuple[str, ...]
    failures: tuple[ProcessorCleanupFailure, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class _ProcessorTarget:
    node_id: str
    processor_ref: Processor


class _StagingState(Enum):
    OPEN = "open"
    RELEASED = "released"
    ROLLED_BACK = "rolled_back"


class ProcessorStagingArea:
    """Owns newly constructed processors until a complete plan is built.

    A processor is registered immediately after construction, before setup
    begins. If any later preparation step fails, rollback cleans every staged
    processor in reverse construction order. Release transfers ownership to
    the candidate plan and permanently disables rollback.
    """

    __slots__ = (
        "_node_ids",
        "_processor_identities",
        "_rollback_report",
        "_state",
        "_targets",
    )

    def __init__(self) -> None:
        self._targets: list[_ProcessorTarget] = []
        self._node_ids: set[str] = set()
        self._processor_identities: set[int] = set()
        self._state = _StagingState.OPEN
        self._rollback_report: CleanupReport | None = None

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(self._node_ids)

    def register(self, node_id: str, processor_ref: Processor) -> None:
        self._require_open("register a processor")
        if not node_id:
            raise ValueError("staged processor node_id must be non-empty.")
        if node_id in self._node_ids:
            raise ValueError(f"node {node_id!r} already owns a staged processor.")

        processor_identity = id(processor_ref)
        if processor_identity in self._processor_identities:
            raise ValueError(
                f"processor factory returned the same instance for more than one node; "
                f"node {node_id!r} cannot share a staged processor instance."
            )

        self._targets.append(_ProcessorTarget(node_id=node_id, processor_ref=processor_ref))
        self._node_ids.add(node_id)
        self._processor_identities.add(processor_identity)

    def release(self) -> frozenset[str]:
        self._require_open("release staged processors")
        self._state = _StagingState.RELEASED
        return self.node_ids

    def rollback(self) -> CleanupReport:
        if self._state is _StagingState.RELEASED:
            raise RuntimeError("cannot roll back processors after ownership was released to a candidate plan.")
        if self._state is _StagingState.ROLLED_BACK:
            assert self._rollback_report is not None
            return self._rollback_report

        report = _cleanup_targets(
            tuple(reversed(self._targets)),
            CleanupReason.STAGING_ROLLBACK,
        )
        self._state = _StagingState.ROLLED_BACK
        self._rollback_report = report
        return report

    def _require_open(self, action: str) -> None:
        if self._state is not _StagingState.OPEN:
            raise RuntimeError(f"cannot {action}; staging area is {self._state.value}.")


def cleanup_candidate_processors(
    candidate_plan: ExecutionPlan,
    staged_node_ids: frozenset[str],
) -> CleanupReport:
    """Clean candidate-exclusive processors after stale, rejected, or aborted publication."""
    targets = _select_plan_targets(candidate_plan, staged_node_ids)
    return _cleanup_targets(tuple(reversed(targets)), CleanupReason.CANDIDATE_DISCARDED)


def retire_superseded_processors(
    previous_plan: ExecutionPlan,
    active_plan: ExecutionPlan,
) -> CleanupReport:
    """Clean processors no longer referenced after a successful plan commit."""
    active_processor_identities = frozenset(id(step.processor_ref) for step in active_plan.steps)
    targets = tuple(
        _ProcessorTarget(node_id=step.node_id, processor_ref=step.processor_ref)
        for step in reversed(previous_plan.steps)
        if id(step.processor_ref) not in active_processor_identities
    )
    return _cleanup_targets(targets, CleanupReason.PLAN_RETIREMENT)


def shutdown_plan_processors(plan: ExecutionPlan) -> CleanupReport:
    """Clean every processor owned by a plan during runtime shutdown."""
    targets = tuple(
        _ProcessorTarget(node_id=step.node_id, processor_ref=step.processor_ref)
        for step in reversed(plan.steps)
    )
    return _cleanup_targets(targets, CleanupReason.RUNTIME_SHUTDOWN)


def _select_plan_targets(
    plan: ExecutionPlan,
    node_ids: frozenset[str],
) -> tuple[_ProcessorTarget, ...]:
    steps_by_node_id = {step.node_id: step for step in plan.steps}
    unknown_node_ids = node_ids.difference(steps_by_node_id)
    if unknown_node_ids:
        raise ValueError(
            f"cannot clean processors absent from plan version {plan.version}: "
            f"{sorted(unknown_node_ids)!r}."
        )

    return tuple(
        _ProcessorTarget(node_id=step.node_id, processor_ref=step.processor_ref)
        for step in plan.steps
        if step.node_id in node_ids
    )


def _cleanup_targets(
    targets: tuple[_ProcessorTarget, ...],
    reason: CleanupReason,
) -> CleanupReport:
    attempted_node_ids: list[str] = []
    cleaned_node_ids: list[str] = []
    failures: list[ProcessorCleanupFailure] = []
    seen_processor_identities: set[int] = set()

    for target in targets:
        processor_identity = id(target.processor_ref)
        if processor_identity in seen_processor_identities:
            continue
        seen_processor_identities.add(processor_identity)
        attempted_node_ids.append(target.node_id)

        try:
            target.processor_ref.cleanup()
        except Exception as error:
            failures.append(
                ProcessorCleanupFailure(
                    node_id=target.node_id,
                    processor_type=target.processor_ref.descriptor.type_name,
                    reason=reason,
                    error=repr(error),
                )
            )
        else:
            cleaned_node_ids.append(target.node_id)

    return CleanupReport(
        reason=reason,
        attempted_node_ids=tuple(attempted_node_ids),
        cleaned_node_ids=tuple(cleaned_node_ids),
        failures=tuple(failures),
    )