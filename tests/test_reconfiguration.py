from __future__ import annotations

import time
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread

import pytest

from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import FrameResult, PipelineExecutor, PlanSwap
from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
    StatefulProcessorDescriptor,
    TransitionContext,
)
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationInProgress,
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

FAIL_DESCRIPTOR = ProcessorDescriptor(
    type_name="fail",
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
        fail_setup: bool = False,
    ) -> None:
        self.descriptor = descriptor
        self._events = events
        self._label = label
        self._fail_setup = fail_setup

    def setup(self, context: SetupContext) -> None:
        self._events.append(f"setup:{self._label}")
        if self._fail_setup:
            raise RuntimeError(f"setup failed for {self._label}")

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


class BlockingProcessor(RecordingProcessor):
    def __init__(self, entered: Event, release: Event) -> None:
        super().__init__(SOURCE_DESCRIPTOR, [], "blocking")
        self._entered = entered
        self._release = release

    def process(self, inputs: object, context: FrameContext) -> object:
        self._entered.set()
        if not self._release.wait(timeout=2.0):
            raise RuntimeError("test did not release blocking processor")
        return ValueOutput(1)


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


def _consumer_node(type_name: str = "pass") -> Node:
    return Node(
        node_id="consumer",
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


def _source_only_specification(type_name: str = "source") -> WorkflowSpecification:
    return WorkflowSpecification(nodes=(_source_node(type_name),), edges=())


def _linear_specification(consumer_type: str = "pass", source_type: str = "source") -> WorkflowSpecification:
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


def _invalid_consumer_only_specification() -> WorkflowSpecification:
    return WorkflowSpecification(nodes=(_consumer_node(),), edges=())


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


def _stateless_registry(events: list[str], fail_consumer_setup: bool = False) -> RegistrySnapshot:
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
    builder.register(
        RegisteredProcessorType(
            descriptor=FAIL_DESCRIPTOR,
            factory=lambda: RecordingProcessor(
                FAIL_DESCRIPTOR,
                events,
                "failing-consumer",
                fail_setup=fail_consumer_setup,
            ),
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


def test_ready_candidate_commits_before_next_frame_and_records_effect() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("add-consumer", 1, _linear_specification()))
        ready = controller.wait_for_status(
            "add-consumer",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None
        assert executor.active_plan.version == 1

        frame_result = controller.admit_frame(admitted_at_ns=10_000, frame_id=7)
        controller.wait_for_retirement("add-consumer")
        record = controller.record("add-consumer")

        assert frame_result.plan_version == 2
        assert frame_result.executed_node_ids == ("source", "consumer")
        assert record.status is ReconfigurationStatus.COMMITTED
        assert record.commit_ns is not None
        assert record.first_new_frame_admitted_ns == 10_000
        assert record.first_new_frame_completed_ns is not None
        assert record.retirement_report is not None
        assert record.retirement_report.attempted_node_ids == ()
        assert tuple(event.status for event in controller.event_log()) == (
            ReconfigurationStatus.RECEIVED,
            ReconfigurationStatus.VALIDATING,
            ReconfigurationStatus.PREPARING,
            ReconfigurationStatus.READY,
            ReconfigurationStatus.COMMITTED,
        )
    finally:
        controller.close()


def test_stale_candidate_is_cleaned_without_replacing_active_plan() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("stale", 1, _linear_specification()))
        ready = controller.wait_for_status(
            "stale",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None

        external_plan = replace(initial.plan, version=3)
        token = getattr(executor, "_manager_token")
        executor.commit_managed(token, external_plan)
        result = controller.commit_ready()
        record = controller.record("stale")

        assert result is not None
        assert result.status is ReconfigurationStatus.STALE
        assert result.observed_active_version == 3
        assert executor.active_plan is external_plan
        assert record.status is ReconfigurationStatus.STALE
        assert record.candidate_cleanup_report is not None
        assert record.candidate_cleanup_report.cleaned_node_ids == ("consumer",)
        assert "cleanup:consumer" in events
        assert "cleanup:source" not in events
    finally:
        controller.close()


def test_invalid_workflow_is_rejected_without_staging_processors() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    initial_event_count = len(events)
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("invalid", 1, _invalid_consumer_only_specification()))
        terminal = controller.wait_for_terminal("invalid", timeout_seconds=2.0)

        assert terminal is not None
        assert terminal.status is ReconfigurationStatus.REJECTED
        assert terminal.failure_reason is not None
        assert "missing_required_input" in terminal.failure_reason
        assert executor.active_plan is initial.plan
        assert len(events) == initial_event_count
    finally:
        controller.close()


def test_setup_failure_is_failure_atomic_and_rolls_back_staging() -> None:
    events: list[str] = []
    registry = _stateless_registry(events, fail_consumer_setup=True)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("setup-failure", 1, _linear_specification("fail")))
        terminal = controller.wait_for_terminal("setup-failure", timeout_seconds=2.0)

        assert terminal is not None
        assert terminal.status is ReconfigurationStatus.FAILED
        assert executor.active_plan is initial.plan
        assert events[-2:] == ["setup:failing-consumer", "cleanup:failing-consumer"]
        assert "cleanup:source" not in events
    finally:
        controller.close()


def test_compatible_stateful_processor_is_reused_across_commit() -> None:
    events: list[str] = []
    tracker_instances: list[StatefulRecordingProcessor] = []
    builder = RegistryBuilder()

    def create_tracker() -> StatefulRecordingProcessor:
        processor = StatefulRecordingProcessor(events, f"tracker-{len(tracker_instances)}")
        tracker_instances.append(processor)
        return processor

    builder.register(
        RegisteredProcessorType(
            descriptor=TRACKER_DESCRIPTOR,
            factory=create_tracker,
            stateful_descriptor=TRACKER_STATEFUL_DESCRIPTOR,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=PASS_DESCRIPTOR,
            factory=lambda: RecordingProcessor(PASS_DESCRIPTOR, events, "consumer"),
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification("tracker"))
    initial_tracker = initial.plan.steps[0].processor_ref
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        first = controller.admit_frame(admitted_at_ns=1)
        assert first.plan_version == 1

        controller.submit(
            _request(
                "preserve-tracker",
                1,
                _linear_specification(source_type="tracker"),
            )
        )
        ready = controller.wait_for_status(
            "preserve-tracker",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None
        second = controller.admit_frame(admitted_at_ns=2)

        active_tracker = executor.active_plan.steps[0].processor_ref
        assert second.plan_version == 2
        assert active_tracker is initial_tracker
        assert len(tracker_instances) == 1
        assert controller.state_transition_events() == ()
        assert "cleanup:tracker-0" not in events
    finally:
        controller.close()


def test_explicit_reset_uses_new_stateful_instance_and_emits_event() -> None:
    events: list[str] = []
    tracker_instances: list[StatefulRecordingProcessor] = []
    builder = RegistryBuilder()

    def create_tracker() -> StatefulRecordingProcessor:
        processor = StatefulRecordingProcessor(events, f"tracker-{len(tracker_instances)}")
        tracker_instances.append(processor)
        return processor

    builder.register(
        RegisteredProcessorType(
            descriptor=TRACKER_DESCRIPTOR,
            factory=create_tracker,
            stateful_descriptor=TRACKER_STATEFUL_DESCRIPTOR,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    initial_specification = _source_only_specification("tracker")
    initial = _compile_initial(compiler, registry, initial_specification)
    old_tracker = initial.plan.steps[0].processor_ref
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.admit_frame(admitted_at_ns=1)
        controller.submit(
            _request(
                "reset-tracker",
                1,
                initial_specification,
                StateDirective(reset_node_ids=frozenset({"source"})),
            )
        )
        ready = controller.wait_for_status(
            "reset-tracker",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None
        result = controller.commit_ready()

        assert result is not None
        assert result.status is ReconfigurationStatus.COMMITTED
        assert executor.active_plan.steps[0].processor_ref is not old_tracker
        assert len(tracker_instances) == 2
        assert len(result.state_transition_events) == 1
        transition = result.state_transition_events[0]
        assert transition.node_id == "source"
        assert transition.old_plan_version == 1
        assert transition.new_plan_version == 2
        assert transition.policy is StateTransitionPolicy.RESET
        controller.admit_frame(1, 1)
        controller.wait_for_retirement("reset-tracker")
        assert "cleanup:tracker-0" in events
        assert "cleanup:tracker-1" not in events
    finally:
        controller.close()


def test_abort_ready_candidate_cleans_staged_resources() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("abort-ready", 1, _linear_specification()))
        ready = controller.wait_for_status(
            "abort-ready",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None

        aborted = controller.abort("abort-ready")

        assert aborted.status is ReconfigurationStatus.ABORTED
        assert aborted.candidate_cleanup_report is not None
        assert aborted.candidate_cleanup_report.cleaned_node_ids == ("consumer",)
        assert executor.active_plan is initial.plan
        assert controller.commit_ready() is None
    finally:
        controller.close()


def test_second_request_is_rejected_while_first_is_ready() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    controller = ReconfigurationController(
        PipelineExecutor(initial.plan),
        compiler,
        registry,
        clock=IncrementingClock(),
    )

    try:
        controller.submit(_request("first", 1, _linear_specification()))
        ready = controller.wait_for_status(
            "first",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None

        with pytest.raises(ReconfigurationInProgress):
            controller.submit(_request("second", 1, _linear_specification()))
    finally:
        controller.close()


def test_executor_plan_swap_waits_for_in_flight_frame() -> None:
    entered = Event()
    release = Event()
    processor = BlockingProcessor(entered, release)
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=lambda: processor,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    successor = replace(initial.plan, version=2)
    executor = PipelineExecutor(initial.plan)
    frame_results: list[FrameResult] = []
    swap_results: list[PlanSwap | None] = []
    swap_finished = Event()

    def execute() -> None:
        frame_results.append(executor.admit_frame(admitted_at_ns=1, frame_id=1))

    def commit() -> None:
        swap_results.append(executor.commit_if_version(1, successor))
        swap_finished.set()

    frame_thread = Thread(target=execute)
    commit_thread = Thread(target=commit)
    frame_thread.start()
    assert entered.wait(timeout=1.0)
    commit_thread.start()
    time.sleep(0.05)

    assert not swap_finished.is_set()
    release.set()
    frame_thread.join(timeout=1.0)
    commit_thread.join(timeout=1.0)

    assert frame_results[0].plan_version == 1
    assert swap_results[0] is not None
    assert executor.active_plan.version == 2

def test_submit_does_not_block_during_frame_execution() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        # We need a blocking frame to simulate an in-flight execution.
        # If we just acquire `_execution_lock`, `submit` should still succeed immediately
        # because `submit` only acquires `_plan_lock` indirectly via `active_plan_snapshot`.
        execution_lock = getattr(executor, "_execution_lock")
        with execution_lock:
            # We are currently executing a frame
            controller.submit(_request("nonblocking", 1, _linear_specification()))
            # If submit blocked on _execution_lock, it would deadlock here.
            # Thus, the test will hang if it's broken.
    finally:
        controller.close()

def test_managed_executor_rejects_direct_admit_frame() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        with pytest.raises(RuntimeError, match="managed by a ReconfigurationController"):
            executor.admit_frame(1, 1)
    finally:
        controller.close()

def test_managed_executor_rejects_direct_commit() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _source_only_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        with pytest.raises(RuntimeError, match="managed by a ReconfigurationController"):
            executor.commit(initial.plan)
    finally:
        controller.close()

def test_retirement_runs_after_first_new_frame() -> None:
    events: list[str] = []
    registry = _stateless_registry(events)
    compiler = WorkflowCompiler("test")
    initial = _compile_initial(compiler, registry, _linear_specification())
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

    try:
        controller.submit(_request("req1", 1, _source_only_specification()))
        controller.wait_for_status("req1", frozenset({ReconfigurationStatus.READY}), timeout_seconds=2.0)
        
        # Commit it
        res = controller.commit_ready()
        assert res is not None
        assert res.retirement_deferred is True
        assert res.cleanup_report is None

        # Now admit a frame
        controller.admit_frame(1, 1)

        # Now wait for retirement
        controller.wait_for_retirement("req1")

        # Now the retirement should have happened
        record = controller.record("req1")
        assert record.retirement_report is not None
        assert "cleanup:consumer" in events
    finally:
        controller.close()