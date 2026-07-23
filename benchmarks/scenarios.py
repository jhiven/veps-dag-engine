"""Processor workloads, topologies, calibration, and graph generators for benchmarks."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
)
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder, RegistrySnapshot
from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    WorkflowSpecification,
    to_processor_configuration,
)
from nedo_vision_dag_engine.type_system import ConcreteType, StatePolicy
from tests.support.processors import PassOutput
from tests.support.tracker import (
    SYNTHETIC_TRACKER_DESCRIPTOR,
    SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    make_synthetic_tracker,
)

__all__ = [
    "WorkloadProcessor",
    "StagingDelayedWorkloadProcessor",
    "WorkloadCalibrationDetail",
    "calibrate_workload_iterations",
    "create_workload_registry",
    "build_linear_5_spec",
    "build_branch_merge_9_spec",
    "execute_hard_coded_linear_5",
    "execute_hard_coded_branch_merge_9",
    "generate_layered_dag",
    "create_reconfiguration_registry",
    "create_reconfiguration_stress_registry",
    "make_reconfiguration_base_spec",
    "apply_reconfiguration_edit",
]


@dataclass(frozen=True, slots=True)
class WorkloadCalibrationDetail:
    workload_id: str
    calibrated_iterations: int
    median_ns: float
    p95_ns: float
    calibration_repetitions: int
    target_description: str
    workload_target_ns: int = 0
    workload_calibration_parameter: int = 0
    pilot_measured_processor_ns: float = 0.0
    expected_topology_frame_ns: float = 0.0
    calibration_runtime_id: str = ""


def do_cpu_work(iterations: int) -> int:
    acc = 0
    for i in range(iterations):
        acc = (acc + i * 3 + 7) % 1000003
    return acc


def _measure_iterations(iterations: int, repetitions: int = 30) -> tuple[float, float]:
    times_ns: list[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        do_cpu_work(iterations)
        times_ns.append(float(time.perf_counter_ns() - t0))
    return float(np.median(times_ns)), float(np.percentile(times_ns, 95))


def calibrate_workload_iterations(calibration_repetitions: int = 30) -> dict[str, WorkloadCalibrationDetail]:
    """Calibrate CPU work iterations per processor invocation before benchmark runs."""
    test_iters = 10000
    start = time.perf_counter_ns()
    do_cpu_work(test_iters)
    elapsed = time.perf_counter_ns() - start

    ns_per_iter = elapsed / test_iters if elapsed > 0 else 1.0

    iters_1ms = max(1, int(1_000_000 / ns_per_iter))
    iters_5ms = max(1, int(5_000_000 / ns_per_iter))

    for _ in range(3):
        t0 = time.perf_counter_ns()
        do_cpu_work(iters_1ms)
        dt_1ms = time.perf_counter_ns() - t0
        if dt_1ms > 0:
            iters_1ms = max(1, int(iters_1ms * (1_000_000 / dt_1ms)))

        t0 = time.perf_counter_ns()
        do_cpu_work(iters_5ms)
        dt_5ms = time.perf_counter_ns() - t0
        if dt_5ms > 0:
            iters_5ms = max(1, int(iters_5ms * (5_000_000 / dt_5ms)))

    med_1ms, p95_1ms = _measure_iterations(iters_1ms, calibration_repetitions)
    med_5ms, p95_5ms = _measure_iterations(iters_5ms, calibration_repetitions)
    runtime_id = f"{sys.implementation.name}-{sys.version.split()[0]}"

    return {
        "minimal": WorkloadCalibrationDetail(
            workload_id="minimal",
            calibrated_iterations=0,
            median_ns=0.0,
            p95_ns=0.0,
            calibration_repetitions=calibration_repetitions,
            target_description="minimal zero-cost baseline",
            workload_target_ns=0,
            workload_calibration_parameter=0,
            pilot_measured_processor_ns=0.0,
            expected_topology_frame_ns=0.0,
            calibration_runtime_id=runtime_id,
        ),
        "approximately_1_ms": WorkloadCalibrationDetail(
            workload_id="approximately_1_ms",
            calibrated_iterations=iters_1ms,
            median_ns=med_1ms,
            p95_ns=p95_1ms,
            calibration_repetitions=calibration_repetitions,
            target_description="approximately 1 ms per processor invocation",
            workload_target_ns=1_000_000,
            workload_calibration_parameter=iters_1ms,
            pilot_measured_processor_ns=med_1ms,
            expected_topology_frame_ns=med_1ms * 5,
            calibration_runtime_id=runtime_id,
        ),
        "approximately_5_ms": WorkloadCalibrationDetail(
            workload_id="approximately_5_ms",
            calibrated_iterations=iters_5ms,
            median_ns=med_5ms,
            p95_ns=p95_5ms,
            calibration_repetitions=calibration_repetitions,
            target_description="approximately 5 ms per processor invocation",
            workload_target_ns=5_000_000,
            workload_calibration_parameter=iters_5ms,
            pilot_measured_processor_ns=med_5ms,
            expected_topology_frame_ns=med_5ms * 5,
            calibration_runtime_id=runtime_id,
        ),
    }


class WorkloadProcessor:
    """A typed processor executing deterministic CPU work."""

    __slots__ = ("_iterations", "descriptor")

    def __init__(self, type_name: str, iterations: int) -> None:
        self.descriptor = ProcessorDescriptor(
            type_name=type_name,
            input_schema=object,
            output_schema=PassOutput,
            config_schema=object,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )
        self._iterations = iterations

    def setup(self, context: SetupContext) -> None:
        pass

    def process(self, inputs: object, context: FrameContext) -> object:
        if self._iterations > 0:
            do_cpu_work(self._iterations)
        return PassOutput(value=context.frame_id)

    def healthcheck(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class StagingDelayedWorkloadProcessor(WorkloadProcessor):
    """A WorkloadProcessor whose ``setup()`` sleeps for a configurable
    duration, simulating non-trivial model loading or resource staging.

    The delay is applied **only during initial setup**, not on every frame.
    This is the key property that the ``prepare_and_commit`` baseline
    exploits: staging happens off-path on the background worker while the
    old plan continues processing frames.
    """

    __slots__ = ("_staging_delay_s",)

    def __init__(self, type_name: str, iterations: int, staging_delay_s: float) -> None:
        super().__init__(type_name, iterations)
        self._staging_delay_s = staging_delay_s

    def setup(self, context: SetupContext) -> None:
        if self._staging_delay_s > 0:
            time.sleep(self._staging_delay_s)


def create_reconfiguration_stress_registry(
    staging_delay_s: float = 0.0,
    extra_tracker: bool = True,
) -> RegistrySnapshot:
    """Registry whose ``workload_node_99`` (the type used for ``n_extra``
    in insert/rewire edits) carries a staging delay, simulating model-load
    latency during candidate preparation.
    """
    builder = RegistryBuilder()
    for i in range(150):
        type_name = f"workload_node_{i}"
        descriptor = ProcessorDescriptor(
            type_name=type_name,
            input_schema=object,
            output_schema=PassOutput,
            config_schema=object,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )
        if i == 99 and staging_delay_s > 0:
            t_name = type_name
            delay = staging_delay_s
            builder.register(
                RegisteredProcessorType(
                    descriptor=descriptor,
                    factory=lambda t=t_name, d=delay: StagingDelayedWorkloadProcessor(t, 0, d),
                )
            )
        else:
            t_name = type_name
            builder.register(
                RegisteredProcessorType(
                    descriptor=descriptor,
                    factory=lambda t=t_name: WorkloadProcessor(t, 0),
                )
            )
    if extra_tracker:
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=make_synthetic_tracker,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
    return builder.snapshot()


def create_workload_registry(iterations: int) -> RegistrySnapshot:
    builder = RegistryBuilder()
    for i in range(150):
        type_name = f"workload_node_{i}"
        descriptor = ProcessorDescriptor(
            type_name=type_name,
            input_schema=object,
            output_schema=PassOutput,
            config_schema=object,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )
        t_name = type_name
        builder.register(
            RegisteredProcessorType(
                descriptor=descriptor,
                factory=lambda t=t_name: WorkloadProcessor(t, iterations),
            )
        )
    return builder.snapshot()


def _make_workload_node(node_id: str, type_index: int, input_pins: tuple[str, ...] = ("value",)) -> Node:
    inputs = tuple(
        Pin(
            name=pin_name,
            payload_type=ConcreteType(object),
            cardinality=PinCardinality.SINGLE,
            requirement=PinRequirement.REQUIRED,
        )
        for pin_name in input_pins
    )
    outputs = (
        Pin(
            name="value",
            payload_type=ConcreteType(object),
            cardinality=PinCardinality.SINGLE,
            requirement=PinRequirement.REQUIRED,
        ),
    )
    return Node(
        node_id=node_id,
        type_name=f"workload_node_{type_index}",
        configuration=to_processor_configuration({}),
        inputs=inputs,
        outputs=outputs,
    )


def build_linear_5_spec() -> WorkflowSpecification:
    nodes = tuple(_make_workload_node(f"node_{i}", i, () if i == 0 else ("value",)) for i in range(5))
    edges = tuple(
        Edge(
            source_node_id=f"node_{i}",
            source_pin="value",
            destination_node_id=f"node_{i+1}",
            destination_pin="value",
        )
        for i in range(4)
    )
    return WorkflowSpecification(nodes=nodes, edges=edges)


def execute_hard_coded_linear_5(
    processors: tuple[WorkloadProcessor, ...], context: FrameContext
) -> PassOutput:
    val: object = None
    for proc in processors:
        val = proc.process(val, context)
    return val if isinstance(val, PassOutput) else PassOutput(value=context.frame_id)


def build_branch_merge_9_spec() -> WorkflowSpecification:
    nodes = (
        _make_workload_node("node_0", 0, ()),
        _make_workload_node("b1_1", 1, ("value",)),
        _make_workload_node("b1_2", 2, ("value",)),
        _make_workload_node("b1_3", 3, ("value",)),
        _make_workload_node("b2_1", 4, ("value",)),
        _make_workload_node("b2_2", 5, ("value",)),
        _make_workload_node("b2_3", 6, ("value",)),
        _make_workload_node("merge", 7, ("in_b1", "in_b2")),
        _make_workload_node("sink", 8, ("value",)),
    )
    edges = (
        Edge("node_0", "value", "b1_1", "value"),
        Edge("b1_1", "value", "b1_2", "value"),
        Edge("b1_2", "value", "b1_3", "value"),
        Edge("node_0", "value", "b2_1", "value"),
        Edge("b2_1", "value", "b2_2", "value"),
        Edge("b2_2", "value", "b2_3", "value"),
        Edge("b1_3", "value", "merge", "in_b1"),
        Edge("b2_3", "value", "merge", "in_b2"),
        Edge("merge", "value", "sink", "value"),
    )
    return WorkflowSpecification(nodes=nodes, edges=edges)


def execute_hard_coded_branch_merge_9(
    processors_by_id: dict[str, WorkloadProcessor], context: FrameContext
) -> PassOutput:
    y0 = processors_by_id["node_0"].process(None, context)
    y1_1 = processors_by_id["b1_1"].process({"value": y0}, context)
    y1_2 = processors_by_id["b1_2"].process({"value": y1_1}, context)
    y1_3 = processors_by_id["b1_3"].process({"value": y1_2}, context)

    y2_1 = processors_by_id["b2_1"].process({"value": y0}, context)
    y2_2 = processors_by_id["b2_2"].process({"value": y2_1}, context)
    y2_3 = processors_by_id["b2_3"].process({"value": y2_2}, context)

    y_merge = processors_by_id["merge"].process({"in_b1": y1_3, "in_b2": y2_3}, context)
    y_sink = processors_by_id["sink"].process({"value": y_merge}, context)
    return y_sink if isinstance(y_sink, PassOutput) else PassOutput(value=context.frame_id)


def generate_layered_dag(num_nodes: int, node_prefix: str = "n") -> WorkflowSpecification:
    """Generate a deterministic sparse layered DAG with num_nodes."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")

    nodes: list[Node] = []
    edges: list[Edge] = []

    nodes.append(_make_workload_node(f"{node_prefix}0", 0, ()))

    for i in range(1, num_nodes - 1):
        if i >= 2 and i % 2 == 1:
            in_pin = f"{node_prefix}{i-2}"
            nodes.append(_make_workload_node(f"{node_prefix}{i}", i, ("value",)))
            edges.append(Edge(in_pin, "value", f"{node_prefix}{i}", "value"))
        else:
            in_pin = f"{node_prefix}{i-1}"
            nodes.append(_make_workload_node(f"{node_prefix}{i}", i, ("value",)))
            edges.append(Edge(in_pin, "value", f"{node_prefix}{i}", "value"))

    last_idx = num_nodes - 1
    if num_nodes > 3:
        nodes.append(_make_workload_node(f"{node_prefix}{last_idx}", last_idx, ("in_a", "in_b")))
        edges.append(Edge(f"{node_prefix}{last_idx-1}", "value", f"{node_prefix}{last_idx}", "in_a"))
        edges.append(Edge(f"{node_prefix}{last_idx-2}", "value", f"{node_prefix}{last_idx}", "in_b"))
    else:
        nodes.append(_make_workload_node(f"{node_prefix}{last_idx}", last_idx, ("value",)))
        edges.append(Edge(f"{node_prefix}{last_idx-1}", "value", f"{node_prefix}{last_idx}", "value"))

    return WorkflowSpecification(nodes=tuple(nodes), edges=tuple(edges))


def create_reconfiguration_registry(extra_tracker: bool = True) -> RegistrySnapshot:
    builder = RegistryBuilder()
    for i in range(150):
        type_name = f"workload_node_{i}"
        descriptor = ProcessorDescriptor(
            type_name=type_name,
            input_schema=object,
            output_schema=PassOutput,
            config_schema=object,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )
        t_name = type_name
        builder.register(
            RegisteredProcessorType(
                descriptor=descriptor,
                factory=lambda t=t_name: WorkloadProcessor(t, 0),
            )
        )
    if extra_tracker:
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=make_synthetic_tracker,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
    return builder.snapshot()


def make_reconfiguration_base_spec(with_tracker: bool = False) -> WorkflowSpecification:
    n1_node = (
        Node(
            node_id="tracker",
            type_name=SYNTHETIC_TRACKER_DESCRIPTOR.type_name,
            configuration=to_processor_configuration({}),
            inputs=(
                Pin(
                    name="value",
                    payload_type=ConcreteType(object),
                    cardinality=PinCardinality.SINGLE,
                    requirement=PinRequirement.REQUIRED,
                ),
            ),
            outputs=(
                Pin(
                    name="value",
                    payload_type=ConcreteType(object),
                    cardinality=PinCardinality.SINGLE,
                    requirement=PinRequirement.REQUIRED,
                ),
            ),
        )
        if with_tracker
        else _make_workload_node("n1", 1, ("value",))
    )
    nodes = (
        _make_workload_node("n0", 0, ()),
        n1_node,
        _make_workload_node("n2", 2, ("value",)),
        _make_workload_node("n3", 3, ("value",)),
        _make_workload_node("n4", 4, ("value",)),
        _make_workload_node("n5", 5, ("value",)),
        _make_workload_node("n6", 6, ("value",)),
        _make_workload_node("n7", 7, ("value",)),
        _make_workload_node("n8", 8, ("in_branch1", "in_branch2")),
        _make_workload_node("n9", 9, ("value",)),
    )
    edges = (
        Edge("n0", "value", "tracker" if with_tracker else "n1", "value"),
        Edge("tracker" if with_tracker else "n1", "value", "n2", "value"),
        Edge("n2", "value", "n3", "value"),
        Edge("n3", "value", "n4", "value"),
        Edge("n4", "value", "n5", "value"),
        Edge("n5", "value", "n8", "in_branch1"),
        Edge("n2", "value", "n6", "value"),
        Edge("n6", "value", "n7", "value"),
        Edge("n7", "value", "n8", "in_branch2"),
        Edge("n8", "value", "n9", "value"),
    )
    return WorkflowSpecification(nodes=nodes, edges=edges)


def apply_reconfiguration_edit(
    base_spec: WorkflowSpecification, edit_type: str
) -> WorkflowSpecification:
    if edit_type == "insert_stateless_node":
        extra_node = _make_workload_node("n_extra", 99, ("value",))
        new_nodes = base_spec.nodes + (extra_node,)
        new_edges_list: list[Edge] = []
        for e in base_spec.edges:
            if e.source_node_id == "n3" and e.destination_node_id == "n4":
                new_edges_list.append(Edge("n3", "value", "n_extra", "value"))
                new_edges_list.append(Edge("n_extra", "value", "n4", e.destination_pin))
            else:
                new_edges_list.append(e)
        return WorkflowSpecification(nodes=new_nodes, edges=tuple(new_edges_list))

    elif edit_type == "remove_stateless_node":
        new_nodes = tuple(n for n in base_spec.nodes if n.node_id != "n4")
        new_edges_list: list[Edge] = []
        for e in base_spec.edges:
            if e.destination_node_id == "n4":
                continue
            if e.source_node_id == "n4":
                new_edges_list.append(Edge("n3", "value", e.destination_node_id, e.destination_pin))
            else:
                new_edges_list.append(e)
        return WorkflowSpecification(nodes=new_nodes, edges=tuple(new_edges_list))

    elif edit_type == "rewire_stateless_edge":
        new_edges = tuple(
            Edge("n1", "value", "n6", "value") if e.source_node_id == "n2" and e.destination_node_id == "n6"
            else e
            for e in base_spec.edges
        )
        return WorkflowSpecification(nodes=base_spec.nodes, edges=new_edges)

    elif edit_type == "compatible_edit_preserving_tracker":
        extra_node = _make_workload_node("n_extra", 99, ("value",))
        has_tracker = any(n.node_id == "tracker" for n in base_spec.nodes)
        if has_tracker:
            nodes = base_spec.nodes + (extra_node,)
            edges_list: list[Edge] = []
            for e in base_spec.edges:
                if e.source_node_id == "n5" and e.destination_node_id == "n8":
                    edges_list.append(Edge("n5", "value", "n_extra", "value"))
                    edges_list.append(Edge("n_extra", "value", "n8", "in_branch1"))
                else:
                    edges_list.append(e)
            return WorkflowSpecification(nodes=nodes, edges=tuple(edges_list))

        tracker_node = Node(
            node_id="tracker",
            type_name=SYNTHETIC_TRACKER_DESCRIPTOR.type_name,
            configuration=to_processor_configuration({}),
            inputs=(),
            outputs=(
                Pin(
                    name="value",
                    payload_type=ConcreteType(object),
                    cardinality=PinCardinality.SINGLE,
                    requirement=PinRequirement.REQUIRED,
                ),
            ),
        )
        nodes = tuple(n for n in base_spec.nodes if n.node_id != "n1") + (tracker_node, extra_node)

        edges_list = []
        for e in base_spec.edges:
            if e.source_node_id == "n1":
                edges_list.append(Edge("tracker", "value", e.destination_node_id, e.destination_pin))
            elif e.destination_node_id == "n1":
                continue
            elif e.source_node_id == "n5" and e.destination_node_id == "n8":
                edges_list.append(Edge("n5", "value", "n_extra", "value"))
                edges_list.append(Edge("n_extra", "value", "n8", "in_branch1"))
            else:
                edges_list.append(e)
        return WorkflowSpecification(nodes=nodes, edges=tuple(edges_list))

    else:
        raise ValueError(f"Unknown edit_type {edit_type!r}")
