from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock

from nedo_vision_dag_engine.compiler import CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import (
    FrameExecutionEvent,
    FrameStatus,
    ReconfigurationMeasurement,
    ReconfigurationStatus,
    RuntimeEventKind,
    RuntimeInstrumentation,
    StateReuseEvent,
)
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
)
from nedo_vision_dag_engine.registry import (
    RegisteredProcessorType,
    RegistryBuilder,
    RegistrySnapshot,
)
from nedo_vision_dag_engine.specification import (
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
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


class EmptyConfig:
    @classmethod
    def validate_configuration(cls, configuration: object) -> tuple[str, ...]:
        return ()


SOURCE_DESCRIPTOR = ProcessorDescriptor(
    type_name="source",
    input_schema=object,
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
    state_schema_version="1",
)


def _preserves_state(
    previous_configuration: object,
    new_configuration: object,
    context: TransitionContext,
) -> bool:
    return context.structurally_preservable


TRACKER_STATEFUL_DESCRIPTOR = StatefulProcessorDescriptor(
    state_schema_version="1",
    supported_transition_policies=frozenset(
        {
            StateTransitionPolicy.PRESERVE,
            StateTransitionPolicy.RESET,
            StateTransitionPolicy.REJECT,
        }
    ),
    preserves_state_for=_preserves_state,
)


class SourceProcessor:
    descriptor = SOURCE_DESCRIPTOR

    def setup(self, context: SetupContext) -> None:
        return None

    def process(self, inputs: object, context: FrameContext) -> object:
        return ValueOutput(context.frame_id)

    def healthcheck(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


class TrackerProcessor(SourceProcessor):
    descriptor = TRACKER_DESCRIPTOR
    stateful_descriptor = TRACKER_STATEFUL_DESCRIPTOR


class IncrementingClock:
    def __init__(self, initial: int = 1_000) -> None:
        self._value = initial
        self._lock = Lock()

    def __call__(self) -> int:
        with self._lock:
            value = self._value
            self._value += 1
            return value


def _node(type_name: str) -> Node:
    return Node(
        node_id="source",
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


def _specification(type_name: str) -> WorkflowSpecification:
    return WorkflowSpecification(nodes=(_node(type_name),), edges=())


def _registry() -> RegistrySnapshot:
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SOURCE_DESCRIPTOR,
            factory=SourceProcessor,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=TRACKER_DESCRIPTOR,
            factory=TrackerProcessor,
            stateful_descriptor=TRACKER_STATEFUL_DESCRIPTOR,
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


def test_executor_records_immutable_frame_event() -> None:
    clock = IncrementingClock(10)
    instrumentation = RuntimeInstrumentation(clock=clock)
    compiler = WorkflowCompiler("test")
    registry = _registry()
    initial = _compile_initial(compiler, registry, _specification("source"))
    executor = PipelineExecutor(
        initial.plan,
        instrumentation=instrumentation,
        clock=clock,
    )

    result = executor.admit_frame(admitted_at_ns=5, frame_id=7)
    events = instrumentation.frame_events()

    assert result.status is FrameStatus.COMPLETED
    assert len(events) == 1
    assert events[0] == FrameExecutionEvent(
        frame_id=7,
        plan_version=1,
        admission_timestamp_ns=5,
        completion_timestamp_ns=10,
        executed_node_ids=("source",),
        skipped_node_ids=(),
        status=FrameStatus.COMPLETED,
        error=None,
    )
    assert instrumentation.event_log()[0].kind is RuntimeEventKind.FRAME_EXECUTION


def test_metrics_detect_duplicate_and_mixed_version_frame_ids() -> None:
    instrumentation = RuntimeInstrumentation(clock=IncrementingClock())
    for frame_id, plan_version, admitted, completed in (
        (1, 1, 10, 20),
        (1, 2, 21, 31),
        (2, 2, 32, 50),
    ):
        instrumentation.record_frame(
            FrameExecutionEvent(
                frame_id=frame_id,
                plan_version=plan_version,
                admission_timestamp_ns=admitted,
                completion_timestamp_ns=completed,
                executed_node_ids=("source",),
                skipped_node_ids=(),
                status=FrameStatus.COMPLETED,
                error=None,
            )
        )

    snapshot = instrumentation.metrics_snapshot()

    assert snapshot.frame_count == 3
    assert snapshot.duplicated_frame_count == 1
    assert snapshot.mixed_version_frame_count == 1
    assert snapshot.maximum_output_gap_ns == 19
    assert snapshot.maximum_frame_latency_ns == 18


def test_reconfiguration_measurement_exposes_all_intervals() -> None:
    measurement = ReconfigurationMeasurement(
        request_id="r1",
        base_version=1,
        candidate_version=2,
        specification_hash="hash",
        submitted_at_ns=5,
        request_received_ns=10,
        validation_started_ns=11,
        validation_completed_ns=20,
        preparation_started_ns=21,
        preparation_completed_ns=40,
        ready_ns=41,
        commit_started_ns=50,
        committed_at_ns=53,
        first_new_frame_admitted_ns=54,
        first_new_frame_completed_ns=60,
        retirement_started_ns=55,
        retirement_completed_ns=58,
        terminal_status=ReconfigurationStatus.COMMITTED,
        failure_reason=None,
        candidate_cleanup_failure_count=0,
        retirement_failure_count=0,
    )

    timings = measurement.timings

    assert timings.validation_ns == 9
    assert timings.preparation_ns == 19
    assert timings.request_to_ready_ns == 31
    assert timings.boundary_wait_ns == 9
    assert timings.commit_ns == 3
    assert timings.request_to_effect_ns == 50
    assert timings.retirement_queue_delay_ns == 2
    assert timings.retirement_duration_ns == 3
    assert timings.commit_to_retirement_complete_ns == 5


def test_controller_records_latest_measurement_and_state_reuse() -> None:
    clock = IncrementingClock(1_000)
    instrumentation = RuntimeInstrumentation(clock=clock)
    registry = _registry()
    compiler = WorkflowCompiler("test")
    specification = _specification("tracker")
    initial = _compile_initial(compiler, registry, specification)
    executor = PipelineExecutor(
        initial.plan,
        instrumentation=instrumentation,
        clock=clock,
    )
    controller = ReconfigurationController(executor, compiler, registry)

    try:
        request = ReconfigurationRequest(
            request_id="preserve",
            base_version=1,
            target_specification=specification,
            state_directive=StateDirective(),
            submitted_at_ns=900,
        )
        controller.submit(request)
        ready = controller.wait_for_status(
            "preserve",
            frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=2.0,
        )
        assert ready is not None

        controller.admit_frame(admitted_at_ns=1_010, frame_id=1)
        measurement = instrumentation.reconfiguration_measurement("preserve")
        reuse_events = instrumentation.state_reuse_events()

        assert measurement.terminal_status is ReconfigurationStatus.COMMITTED
        assert measurement.validation_started_ns is not None
        assert measurement.preparation_started_ns is not None
        assert measurement.commit_started_ns is not None
        assert measurement.committed_at_ns is not None
        assert measurement.first_new_frame_completed_ns is not None
        assert measurement.timings.commit_ns is not None
        assert len(reuse_events) == 1
        assert reuse_events[0] == StateReuseEvent(
            node_id="source",
            old_plan_version=1,
            new_plan_version=2,
            committed_at_ns=measurement.committed_at_ns,
        )
        assert instrumentation.metrics_snapshot().state_reuse_count == 1
    finally:
        controller.close()


def test_json_lines_preserve_sequence_and_event_kind() -> None:
    instrumentation = RuntimeInstrumentation(clock=IncrementingClock())
    instrumentation.record_frame(
        FrameExecutionEvent(
            frame_id=1,
            plan_version=1,
            admission_timestamp_ns=10,
            completion_timestamp_ns=20,
            executed_node_ids=("source",),
            skipped_node_ids=(),
            status=FrameStatus.COMPLETED,
            error=None,
        )
    )

    lines = instrumentation.json_lines()
    parsed = json.loads(lines[0])

    assert parsed["sequence_number"] == 1
    assert parsed["kind"] == "frame_execution"
    assert parsed["event"]["frame_id"] == 1