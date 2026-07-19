from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any

from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    CompilationFailure,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.lifecycle import (
    CleanupReason,
    ProcessorStagingArea,
    cleanup_candidate_processors,
    retire_superseded_processors,
)
from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
)
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
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


class RecordingProcessor:
    def __init__(
        self,
        descriptor: ProcessorDescriptor,
        events: list[str],
        label: str,
        fail_setup: bool = False,
        fail_cleanup: bool = False,
    ) -> None:
        self.descriptor = descriptor
        self._events = events
        self._label = label
        self._fail_setup = fail_setup
        self._fail_cleanup = fail_cleanup

    def setup(self, context: SetupContext) -> None:
        self._events.append(f"setup:{self._label}")
        if self._fail_setup:
            raise RuntimeError(f"setup failed for {self._label}")

    def process(self, inputs: object, context: FrameContext) -> object:
        return ValueOutput(value=1)

    def healthcheck(self) -> None:
        self._events.append(f"healthcheck:{self._label}")

    def cleanup(self) -> None:
        self._events.append(f"cleanup:{self._label}")
        if self._fail_cleanup:
            raise RuntimeError(f"cleanup failed for {self._label}")


def _source_node() -> Node:
    # Extracted to satisfy Pyright strict (avoids dict[Unknown, Unknown] and tuple[Unknown, ...])
    empty_config: dict[str, Any] = {}
    empty_inputs: tuple[Pin, ...] = ()

    return Node(
        node_id="source",
        type_name="source",
        configuration=to_processor_configuration(empty_config),
        inputs=empty_inputs,
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
    empty_config: dict[str, Any] = {}

    return Node(
        node_id="consumer",
        type_name=type_name,
        configuration=to_processor_configuration(empty_config),
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


def _linear_specification(type_name: str = "pass") -> WorkflowSpecification:
    return WorkflowSpecification(
        nodes=(_source_node(), _consumer_node(type_name)),
        edges=(
            Edge(
                source_node_id="source",
                source_pin="value",
                destination_node_id="consumer",
                destination_pin="value",
            ),
        ),
    )


def _source_only_specification() -> WorkflowSpecification:
    empty_edges: tuple[Edge, ...] = ()
    return WorkflowSpecification(nodes=(_source_node(),), edges=empty_edges)


# --- Pytest Fixtures ---


@pytest.fixture
def events() -> list[str]:
    """Provides a fresh events list for each test."""
    return []


@pytest.fixture
def registry_builder(events: list[str]) -> RegistryBuilder:
    """Provides a pre-populated registry builder that captures test events."""
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
                FAIL_DESCRIPTOR, events, "consumer", fail_setup=True
            ),
        )
    )
    return builder


# --- Tests ---


def test_staging_rollback_is_reverse_order_and_idempotent(events: list[str]) -> None:
    first = RecordingProcessor(SOURCE_DESCRIPTOR, events, "first")
    second = RecordingProcessor(PASS_DESCRIPTOR, events, "second")
    staging = ProcessorStagingArea()
    staging.register("first", first)
    staging.register("second", second)

    first_report = staging.rollback()
    second_report = staging.rollback()

    assert first_report is second_report
    assert first_report.reason is CleanupReason.STAGING_ROLLBACK
    assert first_report.cleaned_node_ids == ("second", "first")
    assert events == ["cleanup:second", "cleanup:first"]


def test_staging_rollback_continues_after_cleanup_failure(events: list[str]) -> None:
    first = RecordingProcessor(SOURCE_DESCRIPTOR, events, "first")
    second = RecordingProcessor(PASS_DESCRIPTOR, events, "second", fail_cleanup=True)
    staging = ProcessorStagingArea()
    staging.register("first", first)
    staging.register("second", second)

    report = staging.rollback()

    assert report.cleaned_node_ids == ("first",)
    assert tuple(failure.node_id for failure in report.failures) == ("second",)
    assert events == ["cleanup:second", "cleanup:first"]


def test_compiler_rolls_back_all_staged_processors_when_later_setup_fails(
    events: list[str], registry_builder: RegistryBuilder
) -> None:
    result = WorkflowCompiler("test").compile(
        _linear_specification("fail"), registry_builder.snapshot()
    )

    assert isinstance(result, CompilationFailure)
    assert events == [
        "setup:source",
        "healthcheck:source",
        "setup:consumer",
        "cleanup:consumer",
        "cleanup:source",
    ]


def test_successful_compilation_releases_staged_processors_without_cleanup(
    events: list[str], registry_builder: RegistryBuilder
) -> None:
    result = WorkflowCompiler("test").compile(
        _linear_specification(), registry_builder.snapshot()
    )

    assert isinstance(result, CompiledCandidate)
    assert result.staged_node_ids == frozenset({"source", "consumer"})
    assert not any(event.startswith("cleanup:") for event in events)


def test_candidate_cleanup_does_not_touch_reused_processors(
    events: list[str], registry_builder: RegistryBuilder
) -> None:
    compiler = WorkflowCompiler("test")
    initial = compiler.compile(
        _source_only_specification(), registry_builder.snapshot()
    )
    assert isinstance(initial, CompiledCandidate)

    candidate = compiler.compile(
        _linear_specification(),
        registry_builder.snapshot(),
        previous_plan=initial.plan,
    )
    assert isinstance(candidate, CompiledCandidate)

    report = cleanup_candidate_processors(candidate.plan, candidate.staged_node_ids)

    assert candidate.reused_node_ids == frozenset({"source"})
    assert candidate.staged_node_ids == frozenset({"consumer"})
    assert report.cleaned_node_ids == ("consumer",)
    assert events[-1] == "cleanup:consumer"
    assert "cleanup:source" not in events


def test_retirement_cleans_replaced_or_removed_processors_only(
    events: list[str], registry_builder: RegistryBuilder
) -> None:
    compiler = WorkflowCompiler("test")
    initial = compiler.compile(_linear_specification(), registry_builder.snapshot())
    assert isinstance(initial, CompiledCandidate)

    candidate = compiler.compile(
        _source_only_specification(),
        registry_builder.snapshot(),
        previous_plan=initial.plan,
    )
    assert isinstance(candidate, CompiledCandidate)

    report = retire_superseded_processors(initial.plan, candidate.plan)

    assert report.cleaned_node_ids == ("consumer",)
    assert events[-1] == "cleanup:consumer"
    assert "cleanup:source" not in events