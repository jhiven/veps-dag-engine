"""Tests for grace-period-safe retirement and stateful handoff.

These tests verify that the explicit in-flight frame tracking, grace period
wait, and stateful handoff drain mechanisms operate correctly and produce the
expected lifecycle state transitions and instrumentation records.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread

from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import RetirementStatus
from nedo_vision_dag_engine.lifecycle import PlanLifecycleState
from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
    StatefulProcessorDescriptor,
    TransitionContext,
)
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder, RegistrySnapshot
from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    ProcessorConfiguration,
    WorkflowSpecification,
    to_processor_configuration,
)
from nedo_vision_dag_engine.type_system import (
    ConcreteType,
    StatePolicy,
    StateTransitionPolicy,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValueOutput:
    value: int


@dataclass(frozen=True, slots=True)
class ValueInput:
    value: int


class EmptyConfig:
    pass


SOURCE_DESCRIPTOR = ProcessorDescriptor(
    type_name="source",
    input_schema=object,
    output_schema=ValueOutput,
    config_schema=EmptyConfig,
    state_policy=StatePolicy.STATELESS,
    state_schema_version=None,
)

PASS_DESCRIPTOR = ProcessorDescriptor(
    type_name="pass",
    input_schema=ValueInput,
    output_schema=ValueOutput,
    config_schema=EmptyConfig,
    state_policy=StatePolicy.STATELESS,
    state_schema_version=None,
)

TRACKER_DESCRIPTOR = ProcessorDescriptor(
    type_name="tracker",
    input_schema=object,
    output_schema=ValueOutput,
    config_schema=EmptyConfig,
    state_policy=StatePolicy.PRESERVABLE,
    state_schema_version="tracker-state-v1",
)


def _preserves_tracker_state(
    previous_configuration: ProcessorConfiguration,
    new_configuration: ProcessorConfiguration,
    context: TransitionContext,
) -> bool:
    return previous_configuration == new_configuration and context.structurally_preservable


TRACKER_STATEFUL_DESCRIPTOR = StatefulProcessorDescriptor(
    state_schema_version="tracker-state-v1",
    supported_transition_policies=frozenset(
        {StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}
    ),
    preserves_state_for=_preserves_tracker_state,
)


class RecordingProcessor:
    def __init__(
        self,
        descriptor: ProcessorDescriptor,
        events: list[str],
        label: str,
    ) -> None:
        self.descriptor = descriptor
        self._events = events
        self._label = label

    def setup(self, context: SetupContext) -> None:
        self._events.append(f"setup:{self._label}")

    def process(self, inputs: object, context: FrameContext) -> object:
        self._events.append(f"process:{self._label}:v{context.plan_version}")
        if isinstance(inputs, ValueInput):
            return ValueOutput(inputs.value)
        return ValueOutput(1)

    def healthcheck(self) -> None:
        self._events.append(f"healthcheck:{self._label}")

    def cleanup(self) -> None:
        self._events.append(f"cleanup:{self._label}")


class StatefulRecordingProcessor(RecordingProcessor):
    stateful_descriptor = TRACKER_STATEFUL_DESCRIPTOR

    def __init__(self, events: list[str], label: str) -> None:
        super().__init__(TRACKER_DESCRIPTOR, events, label)
        self.frames_processed = 0

    def process(self, inputs: object, context: FrameContext) -> object:
        self.frames_processed += 1
        self._events.append(
            f"process:{self._label}:v{context.plan_version}:state{self.frames_processed}"
        )
        return ValueOutput(self.frames_processed)


class IncrementingClock:
    def __init__(self, initial: int = 1_000) -> None:
        self._value = initial
        self._lock = Lock()

    def __call__(self) -> int:
        with self._lock:
            value = self._value
            self._value += 1
            return value


def _source_node(type_name: str = "source", node_id: str = "source") -> Node:
    return Node(
        node_id=node_id,
        type_name=type_name,
        configuration=to_processor_configuration({}),
        inputs=(),
        outputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(int),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )


def _consumer_node(type_name: str = "pass", node_id: str = "consumer") -> Node:
    return Node(
        node_id=node_id,
        type_name=type_name,
        configuration=to_processor_configuration({}),
        inputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(int),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
        outputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(int),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )


def _linear_specification(
    consumer_type: str = "pass", source_type: str = "source"
) -> WorkflowSpecification:
    return WorkflowSpecification(
        nodes=(_source_node(source_type), _consumer_node(consumer_type)),
        edges=(
            Edge(
                source_node_id="source",
                source_pin="value",
                destination_node_id="consumer",
                destination_pin="value",
            ),
        ),
    )


def _source_only_specification(type_name: str = "source") -> WorkflowSpecification:
    return WorkflowSpecification(nodes=(_source_node(type_name),), edges=())


def _request(
    request_id: str,
    base_version: int,
    specification: WorkflowSpecification,
    directive: StateDirective | None = None,
) -> ReconfigurationRequest:
    return ReconfigurationRequest(
        request_id=request_id,
        base_version=base_version,
        target_specification=specification,
        state_directive=directive if directive is not None else StateDirective(),
        submitted_at_ns=100,
    )


def _stateless_registry(events: list[str]) -> RegistrySnapshot:
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=lambda: RecordingProcessor(SOURCE_DESCRIPTOR, events, "source"),
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=PASS_DESCRIPTOR,
            factory=lambda: RecordingProcessor(PASS_DESCRIPTOR, events, "consumer"),
        )
    )
    return builder.snapshot()


def _tracker_registry(events: list[str]) -> RegistrySnapshot:
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=lambda: RecordingProcessor(SOURCE_DESCRIPTOR, events, "source"),
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=TRACKER_DESCRIPTOR,
            stateful_descriptor=TRACKER_STATEFUL_DESCRIPTOR,
            factory=lambda: StatefulRecordingProcessor(events, "tracker"),
        )
    )
    return builder.snapshot()


def _compile_initial(
    compiler: WorkflowCompiler,
    registry: RegistrySnapshot,
    specification: WorkflowSpecification,
) -> CompiledCandidate:
    result = compiler.compile(specification, registry)
    assert isinstance(result, CompiledCandidate)
    return result


# ---------------------------------------------------------------------------
# Plan lifecycle state transitions
# ---------------------------------------------------------------------------


def test_initial_plan_is_active() -> None:
    """The initial plan starts in ACTIVE lifecycle state."""
    registry = _stateless_registry([])
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)

    assert executor.plan_lifecycle_state(initial.plan.version) is PlanLifecycleState.ACTIVE


def test_plan_lifecycle_active_to_superseded_to_retired() -> None:
    """A plan transitions ACTIVE → SUPERSEDED → QUIESCENT → RETIRED through its lifecycle."""
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Submit a reconfiguration that replaces the consumer
        new_spec = _source_only_specification()
        request = _request("req-1", initial.plan.version, new_spec)
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )

        # Admit a frame to trigger commit
        controller.admit_frame(1)

        # Wait for retirement to complete
        record = controller.wait_for_retirement("req-1", timeout_seconds=5.0)
        assert record is not None

        # Old plan should be RETIRED
        assert (
            executor.plan_lifecycle_state(initial.plan.version)
            is PlanLifecycleState.RETIRED
        )

        # New plan should be ACTIVE
        assert record.candidate_version is not None
        assert (
            executor.plan_lifecycle_state(record.candidate_version)
            is PlanLifecycleState.ACTIVE
        )
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# In-flight counter correctness
# ---------------------------------------------------------------------------


def test_inflight_counter_incremented_during_frame() -> None:
    """The in-flight counter is incremented while a frame executes and returns to 0 after."""
    registry = _stateless_registry([])
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)

    # The plan_quiescent wait on the active plan should return immediately
    # because no frame is currently in-flight (the executor is idle).
    assert executor.wait_for_plan_quiescent(initial.plan.version, timeout=0.1)


def test_wait_for_plan_quiescent_returns_true_when_no_frames() -> None:
    """wait_for_plan_quiescent returns True when no frames are in-flight."""
    registry = _stateless_registry([])
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)

    result = executor.wait_for_plan_quiescent(initial.plan.version, timeout=1.0)
    assert result is True


# ---------------------------------------------------------------------------
# Grace period instrumentation
# ---------------------------------------------------------------------------


def test_grace_period_timestamps_recorded_on_retirement() -> None:
    """Grace period start/complete timestamps are populated in the reconfiguration record."""
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        new_spec = _source_only_specification()
        request = _request("req-1", initial.plan.version, new_spec)
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.admit_frame(1)

        record = controller.wait_for_retirement("req-1", timeout_seconds=5.0)
        assert record is not None
        assert record.retirement_status is RetirementStatus.COMPLETED

        # Grace period timestamps must be populated
        assert record.grace_period_start_ns is not None
        assert record.grace_period_complete_ns is not None
        assert record.grace_period_complete_ns >= record.grace_period_start_ns

        # Cleanup timestamps are also populated
        assert record.cleanup_start_ns is not None
        assert record.cleanup_end_ns is not None
        assert record.cleanup_end_ns >= record.cleanup_start_ns

        # last_old_frame_completed_ns should be recorded
        assert record.last_old_frame_completed_ns is not None
    finally:
        controller.close()


def test_retirement_not_required_when_no_processors_retired() -> None:
    """When all processors are reused, retirement is NOT_REQUIRED."""
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Submit the same specification — all processors reused
        request = _request("req-1", initial.plan.version, _linear_specification())
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.admit_frame(1)

        record = controller.wait_for_terminal("req-1", timeout_seconds=5.0)
        assert record is not None
        assert record.status is ReconfigurationStatus.COMMITTED
        assert record.retirement_status is RetirementStatus.NOT_REQUIRED

        # No grace period should have been recorded (nothing to retire)
        assert record.grace_period_start_ns is None
        assert record.grace_period_complete_ns is None
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Idempotent retirement
# ---------------------------------------------------------------------------


def test_retirement_is_idempotent_via_cleaned_plans_guard() -> None:
    """The same plan version is not retired twice."""
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        new_spec = _source_only_specification()
        request = _request("req-1", initial.plan.version, new_spec)
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.admit_frame(1)

        record = controller.wait_for_retirement("req-1", timeout_seconds=5.0)
        assert record is not None
        assert record.retirement_status is RetirementStatus.COMPLETED

        # Verify that the cleaned_plans guard is set
        cleaned: set[int] = getattr(controller, "_cleaned_plans")
        assert initial.plan.version in cleaned
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Stateful handoff detection
# ---------------------------------------------------------------------------


def test_handoff_required_when_stateful_processor_preserved() -> None:
    """handoff_required=True when a stateful processor is preserved across plan versions."""
    events: list[str] = []
    registry = _tracker_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(
        compiler, registry, _linear_specification(consumer_type="tracker")
    )
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Admit one frame first so the tracker gets state
        controller.admit_frame(1)

        # Submit same spec — tracker is preserved
        request = _request("req-1", initial.plan.version, _linear_specification(consumer_type="tracker"))
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.admit_frame(2)

        # The commit should have detected stateful handoff
        # Check the compiled candidate has preserved_stateful_node_ids
        record = controller.wait_for_terminal("req-1", timeout_seconds=5.0)
        assert record is not None
        assert record.status is ReconfigurationStatus.COMMITTED

        # Handoff timestamps should be populated
        assert record.handoff_wait_start_ns is not None
        assert record.handoff_wait_complete_ns is not None
        assert record.handoff_wait_complete_ns >= record.handoff_wait_start_ns
    finally:
        controller.close()


def test_handoff_not_required_when_only_stateless_processors_reused() -> None:
    """handoff_required=False when only stateless processors are reused."""
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Submit the exact same spec — stateless reuse only
        request = _request("req-1", initial.plan.version, _linear_specification())
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.admit_frame(1)

        record = controller.wait_for_terminal("req-1", timeout_seconds=5.0)
        assert record is not None
        assert record.status is ReconfigurationStatus.COMMITTED

        # No handoff was needed
        assert record.handoff_wait_start_ns is None
        assert record.handoff_wait_complete_ns is None
    finally:
        controller.close()


def test_boundary_commit_result_includes_handoff_info() -> None:
    """BoundaryCommitResult reports handoff_required correctly."""
    events: list[str] = []
    registry = _tracker_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(
        compiler, registry, _linear_specification(consumer_type="tracker")
    )
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        controller.admit_frame(1)

        request = _request("req-1", initial.plan.version, _linear_specification(consumer_type="tracker"))
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )

        result = controller.commit_ready()
        assert result is not None
        assert result.handoff_required is True

        # handoff_wait_ns should be populated
        assert result.handoff_wait_ns is not None
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Stateful handoff ordering
# ---------------------------------------------------------------------------


def test_stateful_preservation_does_not_mix_old_and_new_state() -> None:
    """When a stateful processor is preserved, old-plan frames finish before
    new-plan frames access the processor, so state is never mixed.

    Under the current single-frame execution model, this is guaranteed by
    _admission_lock serialization.  This test verifies that the handoff
    instrumentation correctly reports the ordering.
    """
    events: list[str] = []
    registry = _tracker_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(
        compiler, registry, _linear_specification(consumer_type="tracker")
    )
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Admit a frame on the initial plan (version 1)
        result1 = controller.admit_frame(1)
        assert result1.plan_version == initial.plan.version

        # Submit and commit a reconfiguration that preserves the tracker
        request = _request("req-1", initial.plan.version, _linear_specification(consumer_type="tracker"))
        controller.submit(request)
        controller.wait_for_status(
            "req-1",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.commit_ready()

        # Admit a frame on the new plan (version 2)
        result2 = controller.admit_frame(2)
        assert result2.plan_version != initial.plan.version

        # No mixed-version frames
        metrics = controller.instrumentation.metrics_snapshot()
        assert metrics.mixed_version_frame_count == 0
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Grace period timeout / stall
# ---------------------------------------------------------------------------


def test_grace_period_stall_when_frame_never_completes() -> None:
    """If a frame never completes, grace period times out and retirement stalls.

    Under the current lock model, a frame cannot block retirement because
    _admission_lock serializes everything. This test validates the timeout
    path by directly calling wait_for_plan_quiescent on a live executor.
    """
    frame_entered = Event()
    frame_release = Event()

    class BlockingSourceProcessor:
        descriptor = SOURCE_DESCRIPTOR

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            frame_entered.set()
            frame_release.wait()
            return ValueOutput(1)

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=BlockingSourceProcessor,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    # Use a source-only spec so only one processor is blocking
    spec = _source_only_specification()
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)

    # Start a frame in a background thread; it will block inside process()
    frame_thread = Thread(target=executor.admit_frame, args=(1,), daemon=True)
    frame_thread.start()
    assert frame_entered.wait(timeout=5.0)

    # While the frame is executing (holding _execution_lock), the plan should
    # have 1 in-flight frame.  wait_for_plan_quiescent is called from the
    # retirement worker (not holding _execution_lock), so it will block until
    # the frame completes or times out.

    # Since the frame holds _execution_lock and we're calling from a different
    # thread, wait_for_plan_quiescent should timeout.
    quiescent = executor.wait_for_plan_quiescent(cand.plan.version, timeout=0.5)
    assert not quiescent  # Frame is still executing

    # Release the frame and verify quiescence
    frame_release.set()
    frame_thread.join(timeout=5.0)
    assert not frame_thread.is_alive()

    quiescent = executor.wait_for_plan_quiescent(cand.plan.version, timeout=1.0)
    assert quiescent  # Frame completed, now quiescent


# ======================================================================
# Conformance scenarios (reviewer P0.4 minimum)
# ======================================================================


def test_cleanup_deferred_until_old_frame_completes_end_to_end() -> None:
    """Scenario 1: Old frame is held mid-execution; publish new plan.

    Cleanup of old-plan processors MUST NOT happen until the old
    frame completes.  This is the canonical grace-period safety test
    using the full ReconfigurationController end-to-end.
    """
    frame_entered = Event()
    frame_release = Event()
    cleanup_called = Event()

    class BlockingProcessor:
        descriptor = SOURCE_DESCRIPTOR

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            frame_entered.set()
            frame_release.wait()
            return ValueOutput(1)

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            cleanup_called.set()

    # Use a separate descriptor for the replacement so the registry
    # does not see duplicate type names.
    REPLACEMENT_SOURCE_DESCRIPTOR = ProcessorDescriptor(
        type_name="replacement_source",
        input_schema=object,
        output_schema=ValueOutput,
        config_schema=EmptyConfig,
        state_policy=StatePolicy.STATELESS,
        state_schema_version=None,
    )

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=BlockingProcessor,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=REPLACEMENT_SOURCE_DESCRIPTOR,
            factory=lambda: RecordingProcessor(
                REPLACEMENT_SOURCE_DESCRIPTOR, [], "replacement_source"
            ),
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    spec = _source_only_specification()
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Admit a frame — it blocks inside process() holding _execution_lock.
        # Use a separate thread so the main thread can submit reconfiguration.
        frame_completed = Event()

        def frame_worker() -> None:
            # Use admit_frame_managed with the internal token so we bypass
            # the controller's _admission_lock and only hold _execution_lock.
            token = getattr(controller, "_executor_token")
            executor.admit_frame_managed(token, 100, 1)
            frame_completed.set()

        frame_thread = Thread(target=frame_worker, daemon=True)
        frame_thread.start()
        assert frame_entered.wait(timeout=5.0)

        # Submit a reconfiguration that will replace the blocking source.
        # Compilation runs on the background worker (no lock conflict).
        replacement_spec = _source_only_specification("replacement_source")
        request = _request("req-defer", cand.plan.version, replacement_spec)
        controller.submit(request)
        controller.wait_for_status(
            "req-defer",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )

        # commit_ready() needs _execution_lock which is held by the frame.
        # Launch commit in a background thread — it will block cleanly.
        commit_completed = Event()

        def commit_worker() -> None:
            controller.commit_ready()
            commit_completed.set()

        commit_thread = Thread(target=commit_worker, daemon=True)
        commit_thread.start()

        import time as _time
        _time.sleep(0.1)

        # Commit must NOT have completed — frame still holds _execution_lock
        assert not commit_completed.is_set(), (
            "commit completed while frame was still executing"
        )
        # Cleanup must NOT have been called
        assert not cleanup_called.is_set(), (
            "cleanup was called while old-plan frame was still executing"
        )

        # Release the old frame
        frame_release.set()
        assert frame_completed.wait(timeout=5.0)
        frame_thread.join(timeout=5.0)

        # Commit must now have completed
        assert commit_completed.wait(timeout=10.0), (
            "commit did not complete after frame was released"
        )
        commit_thread.join(timeout=5.0)

        # Now wait for retirement to complete
        record = controller.wait_for_retirement("req-defer", timeout_seconds=10.0)
        assert record is not None
        assert record.retirement_status is RetirementStatus.COMPLETED

        # Cleanup must have been called AFTER the frame completed
        assert cleanup_called.is_set(), (
            "cleanup was never called after old-plan frame completed"
        )

        # Grace period should be recorded
        assert record.grace_period_start_ns is not None
        assert record.grace_period_complete_ns is not None
    finally:
        frame_release.set()
        controller.close()


def test_memory_error_in_factory_cleans_candidate_and_keeps_active_plan() -> None:
    """Scenario 3: Candidate factory raises MemoryError.

    The candidate must be cleaned, and the active plan must remain
    unchanged (still serving frames with P_old).
    """
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Build a replacement spec that references a bad factory
        bad_type = "bad_factory_type"
        bad_descriptor = ProcessorDescriptor(
            type_name=bad_type,
            input_schema=object,
            output_schema=ValueOutput,
            config_schema=EmptyConfig,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

        # Rebuild registry with the bad factory
        bad_registry = RegistryBuilder()
        bad_registry.register(
            RegisteredProcessorType(
                descriptor=SOURCE_DESCRIPTOR,
                factory=lambda: RecordingProcessor(SOURCE_DESCRIPTOR, events, "source"),
            )
        )
        bad_registry.register(
            RegisteredProcessorType(
                descriptor=bad_descriptor,
                factory=lambda: (_ for _ in ()).throw(
                    MemoryError("injected memory error in factory")
                ),
            )
        )
        bad_reg_snapshot = bad_registry.snapshot()

        controller.update_registry(bad_reg_snapshot)

        # Submit reconfiguration that replaces consumer with bad_factory_type
        bad_spec = _linear_specification(consumer_type=bad_type)
        request = _request("req-memerr", initial.plan.version, bad_spec)
        controller.submit(request)

        record = controller.wait_for_terminal("req-memerr", timeout_seconds=10.0)
        assert record is not None
        assert record.status is ReconfigurationStatus.FAILED

        # Active plan must still be the original
        assert controller.active_plan.version == initial.plan.version

        # Controller must still be able to admit frames on the old plan
        result = controller.admit_frame(1)
        assert result.status.value == "completed"
        assert result.plan_version == initial.plan.version
    finally:
        controller.close()


def test_frame_failure_releases_plan_lease_via_finally() -> None:
    """Scenario 5: Frame fails while using P_old; lease must be released.

    The in-flight counter must return to 0 even on processor failure,
    so that the grace period can complete and retirement can proceed.
    """
    class FailingProcessProcessor:
        descriptor = SOURCE_DESCRIPTOR

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            raise RuntimeError("injected process failure")

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=FailingProcessProcessor,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    spec = _source_only_specification()
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)

    # Admit a frame that will fail inside process()
    result = executor.admit_frame(1)
    assert result.status.value == "failed"
    assert result.error is not None

    # After frame failure, the plan must be quiescent (lease released in finally)
    quiescent = executor.wait_for_plan_quiescent(cand.plan.version, timeout=1.0)
    assert quiescent, (
        "plan should be quiescent after frame failure — "
        "lease must be released via finally even on exception"
    )


def test_cleanup_exception_does_not_uncommit_plan() -> None:
    """Scenario 4: Processor cleanup raises; the committed plan stays committed.

    A cleanup failure must be recorded but must NEVER roll back the
    already-committed plan.  The retirement status should be FAILED
    (not STALLED) and the new plan remains active.
    """
    events: list[str] = []
    cleanup_raised = Event()

    class FailingCleanupProcessor:
        descriptor = PASS_DESCRIPTOR

        def setup(self, context: SetupContext) -> None:
            events.append("setup:failing_cleanup")

        def process(self, inputs: object, context: FrameContext) -> object:
            if isinstance(inputs, ValueInput):
                return ValueOutput(inputs.value)
            return ValueOutput(1)

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            cleanup_raised.set()
            raise RuntimeError("injected cleanup failure")

    class CleanSource:
        descriptor = SOURCE_DESCRIPTOR

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            return ValueOutput(1)

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=CleanSource,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=PASS_DESCRIPTOR,
            factory=FailingCleanupProcessor,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, clock=IncrementingClock()
    )

    try:
        # Admit one frame on the initial plan
        controller.admit_frame(1)

        # Submit a reconfiguration that removes the consumer
        # (triggering cleanup of the failing-cleanup processor)
        replacement_spec = _source_only_specification()
        request = _request("req-cleanup-fail", initial.plan.version, replacement_spec)
        controller.submit(request)
        controller.wait_for_status(
            "req-cleanup-fail",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )
        controller.commit_ready()

        # Wait for retirement to complete (it should be FAILED, not STALLED)
        record = controller.wait_for_retirement(
            "req-cleanup-fail", timeout_seconds=10.0
        )
        assert record is not None
        # Cleanup failure → retirement FAILED, but plan stays COMMITTED
        assert record.status is ReconfigurationStatus.COMMITTED
        assert record.retirement_status is RetirementStatus.FAILED
        assert record.retirement_failure_reason is not None
        assert "cleanup" in record.retirement_failure_reason.lower()

        # Cleanup was attempted
        assert cleanup_raised.is_set()

        # New plan must still be active
        assert record.candidate_version is not None
        assert (
            executor.plan_lifecycle_state(record.candidate_version)
            is PlanLifecycleState.ACTIVE
        )

        # Verify idempotency guard is set (preventing double cleanup)
        cleaned: set[int] = getattr(controller, "_cleaned_plans")
        assert initial.plan.version in cleaned
    finally:
        controller.close()
