"""Serialized background reconfiguration with frame-boundary publication."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from queue import Empty, Queue
from threading import Condition, Lock, RLock, Thread, current_thread
from types import TracebackType

from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    CompilationFailure,
    CompilationFailureKind,
    CycleDetected,
    StateDirective,
    WorkflowCompiler,
    topological_order,
)
from nedo_vision_dag_engine.executor import FrameResult, PipelineExecutor
from nedo_vision_dag_engine.instrumentation import (
    NanosecondClock,
    ReconfigurationEvent,
    ReconfigurationMeasurement,
    ReconfigurationStatus,
    RuntimeInstrumentation,
    StateReuseEvent,
)
from nedo_vision_dag_engine.lifecycle import (
    CleanupReport,
    cleanup_candidate_processors,
    retire_superseded_processors,
)
from nedo_vision_dag_engine.plan import ExecutionPlan
from nedo_vision_dag_engine.processor import StateTransitionEvent
from nedo_vision_dag_engine.registry import RegistrySnapshot
from nedo_vision_dag_engine.specification import WorkflowSpecification, specification_hash
from nedo_vision_dag_engine.type_system import StatePolicy
from nedo_vision_dag_engine.validation import ValidationError, validate_workflow

__all__ = [
    "NanosecondClock",
    "ReconfigurationStatus",
    "ReconfigurationRequest",
    "ReconfigurationEvent",
    "ReconfigurationRecord",
    "BoundaryCommitResult",
    "ReconfigurationError",
    "DuplicateReconfigurationRequest",
    "ReconfigurationInProgress",
    "ReconfigurationControllerClosed",
    "ReconfigurationController",
]


_ALLOWED_TRANSITIONS: dict[ReconfigurationStatus, frozenset[ReconfigurationStatus]] = {
    ReconfigurationStatus.RECEIVED: frozenset(
        {
            ReconfigurationStatus.VALIDATING,
            ReconfigurationStatus.STALE,
            ReconfigurationStatus.ABORTED,
            ReconfigurationStatus.FAILED,
        }
    ),
    ReconfigurationStatus.VALIDATING: frozenset(
        {
            ReconfigurationStatus.PREPARING,
            ReconfigurationStatus.REJECTED,
            ReconfigurationStatus.FAILED,
            ReconfigurationStatus.ABORTED,
        }
    ),
    ReconfigurationStatus.PREPARING: frozenset(
        {
            ReconfigurationStatus.READY,
            ReconfigurationStatus.REJECTED,
            ReconfigurationStatus.FAILED,
            ReconfigurationStatus.ABORTED,
        }
    ),
    ReconfigurationStatus.READY: frozenset(
        {
            ReconfigurationStatus.COMMITTED,
            ReconfigurationStatus.STALE,
            ReconfigurationStatus.ABORTED,
        }
    ),
    ReconfigurationStatus.COMMITTED: frozenset(),
    ReconfigurationStatus.REJECTED: frozenset(),
    ReconfigurationStatus.FAILED: frozenset(),
    ReconfigurationStatus.STALE: frozenset(),
    ReconfigurationStatus.ABORTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ReconfigurationRequest:
    request_id: str
    base_version: int
    target_specification: WorkflowSpecification
    state_directive: StateDirective
    submitted_at_ns: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        if self.base_version < 1:
            raise ValueError("base_version must be positive.")
        if self.submitted_at_ns < 0:
            raise ValueError("submitted_at_ns must be non-negative.")


@dataclass(frozen=True, slots=True)
class ReconfigurationRecord:
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
    commit_ns: int | None
    first_new_frame_admitted_ns: int | None
    first_new_frame_completed_ns: int | None
    retirement_completed_ns: int | None
    status: ReconfigurationStatus
    failure_reason: str | None
    candidate_cleanup_report: CleanupReport | None
    retirement_report: CleanupReport | None
    reused_processor_count: int = 0
    staged_processor_count: int = 0
    retired_processor_count: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@dataclass(frozen=True, slots=True)
class BoundaryCommitResult:
    request_id: str
    status: ReconfigurationStatus
    observed_active_version: int
    candidate_version: int
    committed_at_ns: int | None
    state_transition_events: tuple[StateTransitionEvent, ...]
    cleanup_report: CleanupReport

    def __post_init__(self) -> None:
        if self.status not in {ReconfigurationStatus.COMMITTED, ReconfigurationStatus.STALE}:
            raise ValueError("boundary commit result status must be COMMITTED or STALE.")


class ReconfigurationError(Exception):
    pass


class DuplicateReconfigurationRequest(ReconfigurationError):
    pass


class ReconfigurationInProgress(ReconfigurationError):
    pass


class ReconfigurationControllerClosed(ReconfigurationError):
    pass


@dataclass(frozen=True, slots=True)
class _CompilationJob:
    request: ReconfigurationRequest
    previous_plan: ExecutionPlan


@dataclass(frozen=True, slots=True)
class _ReadyCandidate:
    request: ReconfigurationRequest
    candidate: CompiledCandidate


@dataclass(frozen=True, slots=True)
class _StopCommand:
    pass


type _WorkItem = _CompilationJob | _StopCommand

_STOP = _StopCommand()


class ReconfigurationController:
    """Owns one serialized reconfiguration transaction at a time.

    Candidate compilation runs on a dedicated worker. The worker may read the
    previous plan and may reference reusable processors, but only the executor
    can publish a candidate. Publication uses an atomic base-version check and
    shares the executor's frame-boundary lock with frame admission.
    """

    __slots__ = (
        "_abort_requested_ids",
        "_active_request_id",
        "_boundary_lock",
        "_clock",
        "_closed",
        "_committed_request_by_plan_version",
        "_compiler",
        "_compiler_lock",
        "_condition",
        "_executor",
        "_instrumentation",
        "_ready_candidates",
        "_ready_queue",
        "_records",
        "_registry",
        "_state_lock",
        "_work_queue",
        "_worker",
    )

    def __init__(
        self,
        executor: PipelineExecutor,
        compiler: WorkflowCompiler,
        registry: RegistrySnapshot,
        clock: NanosecondClock = time.monotonic_ns,
        worker_name: str = "dag-reconfiguration",
    ) -> None:
        if not worker_name:
            raise ValueError("worker_name must be non-empty.")

        self._executor = executor
        self._compiler = compiler
        self._registry = registry
        self._clock = clock
        self._state_lock = RLock()
        self._condition = Condition(self._state_lock)
        self._boundary_lock = RLock()
        self._compiler_lock = Lock()
        self._work_queue: Queue[_WorkItem] = Queue()
        self._ready_queue: Queue[str] = Queue()
        self._records: dict[str, ReconfigurationRecord] = {}
        self._instrumentation: RuntimeInstrumentation = executor.instrumentation
        self._ready_candidates: dict[str, _ReadyCandidate] = {}
        self._abort_requested_ids: set[str] = set()
        self._committed_request_by_plan_version: dict[int, str] = {}
        self._active_request_id: str | None = None
        self._closed = False
        self._worker = Thread(target=self._worker_main, name=worker_name, daemon=True)
        self._worker.start()

    @property
    def active_plan(self) -> ExecutionPlan:
        return self._executor.active_plan

    @property
    def instrumentation(self) -> RuntimeInstrumentation:
        return self._instrumentation

    def submit(self, request: ReconfigurationRequest) -> ReconfigurationRecord:
        received_at_ns = self._clock()
        previous_plan = self._executor.active_plan
        target_hash = specification_hash(request.target_specification)

        with self._condition:
            self._require_open_locked()
            if request.request_id in self._records:
                raise DuplicateReconfigurationRequest(
                    f"request id {request.request_id!r} has already been submitted."
                )
            if self._active_request_id is not None:
                raise ReconfigurationInProgress(
                    f"request {self._active_request_id!r} is still active; concurrent "
                    "reconfiguration transactions are not supported."
                )

            record = ReconfigurationRecord(
                request_id=request.request_id,
                base_version=request.base_version,
                candidate_version=None,
                specification_hash=target_hash,
                submitted_at_ns=request.submitted_at_ns,
                request_received_ns=received_at_ns,
                validation_started_ns=None,
                validation_completed_ns=None,
                preparation_started_ns=None,
                preparation_completed_ns=None,
                ready_ns=None,
                commit_started_ns=None,
                commit_ns=None,
                first_new_frame_admitted_ns=None,
                first_new_frame_completed_ns=None,
                retirement_completed_ns=None,
                status=ReconfigurationStatus.RECEIVED,
                failure_reason=None,
                candidate_cleanup_report=None,
                retirement_report=None,
            )
            self._records[request.request_id] = record
            self._instrumentation.record_reconfiguration_event(
                ReconfigurationEvent(
                    request_id=request.request_id,
                    status=ReconfigurationStatus.RECEIVED,
                    occurred_at_ns=received_at_ns,
                    reason=None,
                )
            )
            self._record_measurement(record)

            if request.base_version != previous_plan.version:
                stale_at_ns = self._clock()
                reason = (
                    f"request base version {request.base_version} does not match active plan "
                    f"version {previous_plan.version}."
                )
                stale_record = replace(
                    record,
                    status=ReconfigurationStatus.STALE,
                    failure_reason=reason,
                )
                self._publish_transition_locked(stale_record, stale_at_ns, reason)
                return stale_record

            self._active_request_id = request.request_id
            self._work_queue.put(_CompilationJob(request=request, previous_plan=previous_plan))
            self._condition.notify_all()
            return record

    def abort(self, request_id: str, reason: str = "aborted by caller") -> ReconfigurationRecord:
        if not reason:
            raise ValueError("abort reason must be non-empty.")

        with self._boundary_lock:
            ready_candidate: _ReadyCandidate | None = None
            with self._condition:
                record = self._record_or_raise_locked(request_id)
                if record.is_terminal:
                    return record

                if record.status is ReconfigurationStatus.RECEIVED:
                    aborted_at_ns = self._clock()
                    aborted_record = replace(
                        record,
                        status=ReconfigurationStatus.ABORTED,
                        failure_reason=reason,
                    )
                    self._abort_requested_ids.discard(request_id)
                    self._finish_terminal_locked(aborted_record, aborted_at_ns, reason)
                    return aborted_record

                if record.status is ReconfigurationStatus.READY:
                    ready_candidate = self._ready_candidates.pop(request_id)
                else:
                    self._abort_requested_ids.add(request_id)
                    return record

            assert ready_candidate is not None
            cleanup_report, cleanup_error = self._discard_candidate(ready_candidate.candidate)
            aborted_at_ns = self._clock()
            combined_reason = _combine_reasons(reason, cleanup_error)

            with self._condition:
                current = self._record_or_raise_locked(request_id)
                aborted_record = replace(
                    current,
                    status=ReconfigurationStatus.ABORTED,
                    failure_reason=combined_reason,
                    candidate_cleanup_report=cleanup_report,
                )
                self._abort_requested_ids.discard(request_id)
                self._finish_terminal_locked(aborted_record, aborted_at_ns, combined_reason)
                return aborted_record

    def commit_ready(self) -> BoundaryCommitResult | None:
        with self._boundary_lock:
            with self._condition:
                self._require_open_locked()
            return self._commit_ready_unlocked()

    def admit_frame(self, admitted_at_ns: int, frame_id: int | None = None) -> FrameResult:
        with self._boundary_lock:
            with self._condition:
                self._require_open_locked()
            self._commit_ready_unlocked()
            result = self._executor.admit_frame(admitted_at_ns=admitted_at_ns, frame_id=frame_id)
            completed_at_ns = self._clock()
            self._record_first_frame(result, admitted_at_ns, completed_at_ns)
            return result

    def record(self, request_id: str) -> ReconfigurationRecord:
        with self._condition:
            return self._record_or_raise_locked(request_id)

    def records(self) -> tuple[ReconfigurationRecord, ...]:
        with self._condition:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda record: (record.request_received_ns, record.request_id),
                )
            )

    def event_log(self) -> tuple[ReconfigurationEvent, ...]:
        return self._instrumentation.reconfiguration_events()

    def state_transition_events(self) -> tuple[StateTransitionEvent, ...]:
        return self._instrumentation.state_transition_events()

    def state_reuse_events(self) -> tuple[StateReuseEvent, ...]:
        return self._instrumentation.state_reuse_events()

    def wait_for_status(
        self,
        request_id: str,
        statuses: frozenset[ReconfigurationStatus],
        timeout_seconds: float | None = None,
    ) -> ReconfigurationRecord | None:
        if not statuses:
            raise ValueError("statuses must contain at least one status.")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative.")

        with self._condition:
            self._record_or_raise_locked(request_id)
            reached = self._condition.wait_for(
                lambda: self._records[request_id].status in statuses,
                timeout=timeout_seconds,
            )
            if not reached:
                return None
            return self._records[request_id]

    def wait_for_terminal(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> ReconfigurationRecord | None:
        return self.wait_for_status(
            request_id=request_id,
            statuses=frozenset(
                status for status in ReconfigurationStatus if status.is_terminal
            ),
            timeout_seconds=timeout_seconds,
        )

    def wait_for_effect(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> ReconfigurationRecord | None:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative.")

        with self._condition:
            self._record_or_raise_locked(request_id)
            reached = self._condition.wait_for(
                lambda: (
                    self._records[request_id].is_terminal
                    and (
                        self._records[request_id].status is not ReconfigurationStatus.COMMITTED
                        or self._records[request_id].first_new_frame_completed_ns is not None
                    )
                ),
                timeout=timeout_seconds,
            )
            if not reached:
                return None
            return self._records[request_id]

    def close(self) -> None:
        if current_thread() is self._worker:
            raise RuntimeError("the reconfiguration worker cannot close its own controller.")

        ready_candidate: _ReadyCandidate | None = None
        ready_request_id: str | None = None
        with self._boundary_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                active_request_id = self._active_request_id
                if active_request_id is not None:
                    active_record = self._records[active_request_id]
                    if active_record.status is ReconfigurationStatus.READY:
                        ready_request_id = active_request_id
                        ready_candidate = self._ready_candidates.pop(active_request_id)
                    elif not active_record.is_terminal:
                        self._abort_requested_ids.add(active_request_id)
                self._work_queue.put(_STOP)
                self._condition.notify_all()

            if ready_candidate is not None and ready_request_id is not None:
                cleanup_report, cleanup_error = self._discard_candidate(ready_candidate.candidate)
                aborted_at_ns = self._clock()
                reason = _combine_reasons("controller closed before candidate commit", cleanup_error)
                with self._condition:
                    current = self._record_or_raise_locked(ready_request_id)
                    aborted_record = replace(
                        current,
                        status=ReconfigurationStatus.ABORTED,
                        failure_reason=reason,
                        candidate_cleanup_report=cleanup_report,
                    )
                    self._abort_requested_ids.discard(ready_request_id)
                    self._finish_terminal_locked(aborted_record, aborted_at_ns, reason)

        self._worker.join()

    def __enter__(self) -> ReconfigurationController:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _worker_main(self) -> None:
        while True:
            item = self._work_queue.get()
            try:
                if isinstance(item, _StopCommand):
                    return
                try:
                    self._prepare_job(item)
                except Exception as error:
                    self._fail_unexpected_worker_error(item.request.request_id, error)
            finally:
                self._work_queue.task_done()

    def _prepare_job(self, job: _CompilationJob) -> None:
        request = job.request
        validating_at_ns = self._clock()
        with self._condition:
            record = self._record_or_raise_locked(request.request_id)
            if record.is_terminal:
                return
            if request.request_id in self._abort_requested_ids:
                aborted_record = replace(
                    record,
                    status=ReconfigurationStatus.ABORTED,
                    failure_reason="aborted before validation started",
                )
                self._abort_requested_ids.discard(request.request_id)
                self._finish_terminal_locked(
                    aborted_record,
                    validating_at_ns,
                    aborted_record.failure_reason,
                )
                return
            validating_record = replace(
                record,
                status=ReconfigurationStatus.VALIDATING,
                validation_started_ns=validating_at_ns,
            )
            self._publish_transition_locked(validating_record, validating_at_ns, None)

        try:
            validation_result, _resolver = validate_workflow(
                request.target_specification,
                self._registry,
            )
        except Exception as error:
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=ReconfigurationStatus.FAILED,
                reason=f"validation raised unexpectedly: {error!r}",
                validation_completed_at_ns=self._clock(),
                preparation_completed_at_ns=None,
            )
            return

        validation_completed_at_ns = self._clock()
        if not validation_result.is_valid:
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=ReconfigurationStatus.REJECTED,
                reason=_format_validation_errors(validation_result.errors),
                validation_completed_at_ns=validation_completed_at_ns,
                preparation_completed_at_ns=None,
            )
            return

        try:
            topological_order(
                request.target_specification.nodes,
                request.target_specification.edges,
            )
        except CycleDetected as error:
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=ReconfigurationStatus.REJECTED,
                reason=str(error),
                validation_completed_at_ns=validation_completed_at_ns,
                preparation_completed_at_ns=None,
            )
            return
        except Exception as error:
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=ReconfigurationStatus.FAILED,
                reason=f"cycle detection raised unexpectedly: {error!r}",
                validation_completed_at_ns=validation_completed_at_ns,
                preparation_completed_at_ns=None,
            )
            return

        preparing_at_ns = self._clock()
        with self._condition:
            record = self._record_or_raise_locked(request.request_id)
            if request.request_id in self._abort_requested_ids:
                aborted_record = replace(
                    record,
                    status=ReconfigurationStatus.ABORTED,
                    validation_completed_ns=validation_completed_at_ns,
                    failure_reason="aborted after validation completed",
                )
                self._abort_requested_ids.discard(request.request_id)
                self._finish_terminal_locked(
                    aborted_record,
                    preparing_at_ns,
                    aborted_record.failure_reason,
                )
                return
            preparing_record = replace(
                record,
                status=ReconfigurationStatus.PREPARING,
                validation_completed_ns=validation_completed_at_ns,
                preparation_started_ns=preparing_at_ns,
            )
            self._publish_transition_locked(preparing_record, preparing_at_ns, None)

        try:
            with self._compiler_lock:
                compilation_result = self._compiler.compile(
                    specification=request.target_specification,
                    registry=self._registry,
                    previous_plan=job.previous_plan,
                    state_directive=request.state_directive,
                )
        except Exception as error:
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=ReconfigurationStatus.FAILED,
                reason=f"compiler raised unexpectedly: {error!r}",
                validation_completed_at_ns=validation_completed_at_ns,
                preparation_completed_at_ns=self._clock(),
            )
            return

        preparation_completed_at_ns = self._clock()
        if isinstance(compilation_result, CompilationFailure):
            failure_status = (
                ReconfigurationStatus.REJECTED
                if compilation_result.kind is CompilationFailureKind.REJECTED
                else ReconfigurationStatus.FAILED
            )
            self._finish_preparation_failure(
                request_id=request.request_id,
                status=failure_status,
                reason=compilation_result.reason,
                validation_completed_at_ns=validation_completed_at_ns,
                preparation_completed_at_ns=preparation_completed_at_ns,
            )
            return

        with self._condition:
            abort_requested = request.request_id in self._abort_requested_ids

        if abort_requested:
            cleanup_report, cleanup_error = self._discard_candidate(compilation_result)
            aborted_at_ns = self._clock()
            reason = _combine_reasons("aborted while candidate was being prepared", cleanup_error)
            with self._condition:
                record = self._record_or_raise_locked(request.request_id)
                aborted_record = replace(
                    record,
                    candidate_version=compilation_result.plan.version,
                    preparation_completed_ns=preparation_completed_at_ns,
                    status=ReconfigurationStatus.ABORTED,
                    failure_reason=reason,
                    candidate_cleanup_report=cleanup_report,
                )
                self._abort_requested_ids.discard(request.request_id)
                self._finish_terminal_locked(aborted_record, aborted_at_ns, reason)
            return

        ready_at_ns = self._clock()
        ready_candidate = _ReadyCandidate(request=request, candidate=compilation_result)
        with self._condition:
            record = self._record_or_raise_locked(request.request_id)
            self._ready_candidates[request.request_id] = ready_candidate
            ready_record = replace(
                record,
                candidate_version=compilation_result.plan.version,
                preparation_completed_ns=preparation_completed_at_ns,
                ready_ns=ready_at_ns,
                status=ReconfigurationStatus.READY,
                reused_processor_count=len(compilation_result.reused_node_ids),
                staged_processor_count=len(compilation_result.staged_node_ids),
                retired_processor_count=len(job.previous_plan.steps) - len(compilation_result.reused_node_ids),
            )
            self._publish_transition_locked(ready_record, ready_at_ns, None)
            self._ready_queue.put(request.request_id)

    def _finish_preparation_failure(
        self,
        request_id: str,
        status: ReconfigurationStatus,
        reason: str,
        validation_completed_at_ns: int,
        preparation_completed_at_ns: int | None,
    ) -> None:
        terminal_at_ns = (
            preparation_completed_at_ns
            if preparation_completed_at_ns is not None
            else validation_completed_at_ns
        )
        with self._condition:
            record = self._record_or_raise_locked(request_id)
            if request_id in self._abort_requested_ids:
                status = ReconfigurationStatus.ABORTED
                reason = _combine_reasons("aborted during candidate preparation", reason)
                self._abort_requested_ids.discard(request_id)
            terminal_record = replace(
                record,
                validation_completed_ns=validation_completed_at_ns,
                preparation_completed_ns=preparation_completed_at_ns,
                status=status,
                failure_reason=reason,
            )
            self._finish_terminal_locked(terminal_record, terminal_at_ns, reason)

    def _fail_unexpected_worker_error(self, request_id: str, error: Exception) -> None:
        failed_at_ns = self._clock()
        reason = f"reconfiguration worker failed unexpectedly: {error!r}"
        with self._condition:
            record = self._records.get(request_id)
            if record is None or record.is_terminal:
                return
            failed_record = replace(
                record,
                status=ReconfigurationStatus.FAILED,
                failure_reason=reason,
            )
            self._abort_requested_ids.discard(request_id)
            self._finish_terminal_locked(failed_record, failed_at_ns, reason)

    def _commit_ready_unlocked(self) -> BoundaryCommitResult | None:
        ready_candidate = self._take_ready_candidate()
        if ready_candidate is None:
            return None

        request = ready_candidate.request
        candidate = ready_candidate.candidate
        commit_started_at_ns = self._clock()
        plan_swap = self._executor.commit_if_version(
            expected_version=request.base_version,
            new_plan=candidate.plan,
        )

        if plan_swap is None:
            observed_active_version = self._executor.active_plan.version
            cleanup_report, cleanup_error = self._discard_candidate(candidate)
            stale_at_ns = self._clock()
            reason = _combine_reasons(
                (
                    f"candidate was prepared from version {request.base_version}, but active "
                    f"plan version is {observed_active_version}."
                ),
                cleanup_error,
            )
            with self._condition:
                record = self._record_or_raise_locked(request.request_id)
                stale_record = replace(
                    record,
                    commit_started_ns=commit_started_at_ns,
                    status=ReconfigurationStatus.STALE,
                    failure_reason=reason,
                    candidate_cleanup_report=cleanup_report,
                )
                self._finish_terminal_locked(stale_record, stale_at_ns, reason)
            return BoundaryCommitResult(
                request_id=request.request_id,
                status=ReconfigurationStatus.STALE,
                observed_active_version=observed_active_version,
                candidate_version=candidate.plan.version,
                committed_at_ns=None,
                state_transition_events=(),
                cleanup_report=cleanup_report,
            )

        committed_at_ns = self._clock()
        transition_events = tuple(
            StateTransitionEvent(
                node_id=transition.node_id,
                old_plan_version=plan_swap.previous_plan.version,
                new_plan_version=plan_swap.active_plan.version,
                policy=transition.policy,
                reason=transition.reason,
                committed_at_ns=committed_at_ns,
            )
            for transition in candidate.pending_state_transitions
        )
        reuse_events = tuple(
            StateReuseEvent(
                node_id=step.node_id,
                old_plan_version=plan_swap.previous_plan.version,
                new_plan_version=plan_swap.active_plan.version,
                committed_at_ns=committed_at_ns,
            )
            for step in plan_swap.active_plan.steps
            if step.node_id in candidate.reused_node_ids
            and step.processor_ref.descriptor.state_policy
            in {StatePolicy.PRESERVABLE, StatePolicy.RESETTABLE}
        )

        retirement_report = retire_superseded_processors(
            previous_plan=plan_swap.previous_plan,
            active_plan=plan_swap.active_plan,
        )
        retirement_completed_at_ns = self._clock()
        retirement_reason = _cleanup_failure_reason("processor retirement", retirement_report)

        with self._compiler_lock:
            self._compiler.forget_version(plan_swap.previous_plan.version)

        with self._condition:
            record = self._record_or_raise_locked(request.request_id)
            committed_record = replace(
                record,
                status=ReconfigurationStatus.COMMITTED,
                commit_started_ns=commit_started_at_ns,
                commit_ns=committed_at_ns,
                retirement_completed_ns=retirement_completed_at_ns,
                failure_reason=retirement_reason,
                retirement_report=retirement_report,
                reused_processor_count=len(candidate.reused_node_ids),
                staged_processor_count=len(candidate.staged_node_ids),
                retired_processor_count=len(retirement_report.cleaned_node_ids),
            )
            for transition_event in transition_events:
                self._instrumentation.record_state_transition(transition_event)
            for reuse_event in reuse_events:
                self._instrumentation.record_state_reuse(reuse_event)
            self._committed_request_by_plan_version[plan_swap.active_plan.version] = request.request_id
            self._finish_terminal_locked(
                committed_record,
                committed_at_ns,
                retirement_reason,
            )

        return BoundaryCommitResult(
            request_id=request.request_id,
            status=ReconfigurationStatus.COMMITTED,
            observed_active_version=plan_swap.active_plan.version,
            candidate_version=candidate.plan.version,
            committed_at_ns=committed_at_ns,
            state_transition_events=transition_events,
            cleanup_report=retirement_report,
        )

    def _take_ready_candidate(self) -> _ReadyCandidate | None:
        while True:
            try:
                request_id = self._ready_queue.get_nowait()
            except Empty:
                return None

            with self._condition:
                ready_candidate = self._ready_candidates.pop(request_id, None)
            if ready_candidate is not None:
                return ready_candidate

    def _discard_candidate(
        self,
        candidate: CompiledCandidate,
    ) -> tuple[CleanupReport, str | None]:
        cleanup_report = cleanup_candidate_processors(
            candidate_plan=candidate.plan,
            staged_node_ids=candidate.staged_node_ids,
        )
        with self._compiler_lock:
            self._compiler.forget_version(candidate.plan.version)
        return cleanup_report, _cleanup_failure_reason("candidate cleanup", cleanup_report)

    def _record_first_frame(
        self,
        result: FrameResult,
        admitted_at_ns: int,
        completed_at_ns: int,
    ) -> None:
        with self._condition:
            request_id = self._committed_request_by_plan_version.get(result.plan_version)
            if request_id is None:
                return
            record = self._records[request_id]
            if record.first_new_frame_admitted_ns is not None:
                return
            updated_record = replace(
                record,
                first_new_frame_admitted_ns=admitted_at_ns,
                first_new_frame_completed_ns=completed_at_ns,
            )
            self._records[request_id] = updated_record
            self._record_measurement(updated_record)
            self._committed_request_by_plan_version.pop(result.plan_version, None)
            self._condition.notify_all()

    def _record_measurement(self, record: ReconfigurationRecord) -> None:
        candidate_cleanup_failure_count = (
            len(record.candidate_cleanup_report.failures)
            if record.candidate_cleanup_report is not None
            else 0
        )
        retirement_failure_count = (
            len(record.retirement_report.failures)
            if record.retirement_report is not None
            else 0
        )
        self._instrumentation.record_reconfiguration_measurement(
            ReconfigurationMeasurement(
                request_id=record.request_id,
                base_version=record.base_version,
                candidate_version=record.candidate_version,
                specification_hash=record.specification_hash,
                submitted_at_ns=record.submitted_at_ns,
                request_received_ns=record.request_received_ns,
                validation_started_ns=record.validation_started_ns,
                validation_completed_ns=record.validation_completed_ns,
                preparation_started_ns=record.preparation_started_ns,
                preparation_completed_ns=record.preparation_completed_ns,
                ready_ns=record.ready_ns,
                commit_started_ns=record.commit_started_ns,
                committed_at_ns=record.commit_ns,
                first_new_frame_admitted_ns=record.first_new_frame_admitted_ns,
                first_new_frame_completed_ns=record.first_new_frame_completed_ns,
                retirement_completed_ns=record.retirement_completed_ns,
                terminal_status=record.status,
                failure_reason=record.failure_reason,
                candidate_cleanup_failure_count=candidate_cleanup_failure_count,
                retirement_failure_count=retirement_failure_count,
            )
        )

    def _publish_transition_locked(
        self,
        record: ReconfigurationRecord,
        occurred_at_ns: int,
        reason: str | None,
    ) -> None:
        previous = self._records[record.request_id]
        if record.status is previous.status:
            raise RuntimeError(
                f"request {record.request_id!r} attempted to publish duplicate status "
                f"{record.status.value!r}."
            )
        if record.status not in _ALLOWED_TRANSITIONS[previous.status]:
            raise RuntimeError(
                f"invalid reconfiguration transition for request {record.request_id!r}: "
                f"{previous.status.value!r} -> {record.status.value!r}."
            )
        self._records[record.request_id] = record
        self._instrumentation.record_reconfiguration_event(
            ReconfigurationEvent(
                request_id=record.request_id,
                status=record.status,
                occurred_at_ns=occurred_at_ns,
                reason=reason,
            )
        )
        self._record_measurement(record)
        self._condition.notify_all()

    def _finish_terminal_locked(
        self,
        record: ReconfigurationRecord,
        occurred_at_ns: int,
        reason: str | None,
    ) -> None:
        if not record.status.is_terminal:
            raise ValueError("terminal completion requires a terminal status.")
        self._publish_transition_locked(record, occurred_at_ns, reason)
        if self._active_request_id == record.request_id:
            self._active_request_id = None
        self._condition.notify_all()

    def _record_or_raise_locked(self, request_id: str) -> ReconfigurationRecord:
        record = self._records.get(request_id)
        if record is None:
            raise KeyError(f"unknown reconfiguration request id {request_id!r}.")
        return record

    def _require_open_locked(self) -> None:
        if self._closed:
            raise ReconfigurationControllerClosed("reconfiguration controller is closed.")


def _format_validation_errors(errors: tuple[ValidationError, ...]) -> str:
    return "; ".join(f"{error.code.value}: {error.message}" for error in errors)


def _cleanup_failure_reason(operation: str, report: CleanupReport) -> str | None:
    if report.succeeded:
        return None
    details = "; ".join(
        f"{failure.node_id} ({failure.processor_type}): {failure.error}"
        for failure in report.failures
    )
    return f"{operation} reported failure(s): {details}"


def _combine_reasons(primary: str, secondary: str | None) -> str:
    if secondary is None:
        return primary
    return f"{primary} {secondary}"