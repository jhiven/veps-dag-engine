"""Regression gates for fail-closed ownership and admission repairs."""

from __future__ import annotations

import pytest

from nedo_vision_dag_engine.compiler import (
    CompilationFailure,
    CompiledCandidate,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
from tests.support.processors import LifecycleLog
from tests.support.registries import compile_initial, stateless_registry, tracker_registry
from tests.support.tracker import (
    SYNTHETIC_TRACKER_DESCRIPTOR,
    SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    SyntheticTracker,
    make_synthetic_tracker,
)
from tests.support.workflows import source_only, tracker_only


def test_cancelled_reservation_releases_lease_exactly_once() -> None:
    compiler = WorkflowCompiler("test")
    registry = stateless_registry(LifecycleLog())
    initial = compile_initial(compiler, registry, source_only())
    executor = PipelineExecutor(initial.plan)
    token = object()
    executor.claim_management(token)
    admission = executor.reserve_frame_managed(token, admitted_at_ns=1, frame_id=7)

    executor.cancel_reserved_frame_managed(token, admission)

    assert executor.wait_for_plan_quiescent(initial.plan.version, timeout=0.0)
    with pytest.raises(RuntimeError, match="foreign or has already been consumed"):
        executor.cancel_reserved_frame_managed(token, admission)
    with pytest.raises(RuntimeError, match="foreign or has already been consumed"):
        executor.execute_admitted_frame(admission)
    executor.release_management(token)


def test_foreign_reservation_is_rejected_without_releasing_either_lease() -> None:
    compiler_a = WorkflowCompiler("test")
    compiler_b = WorkflowCompiler("test")
    registry = stateless_registry(LifecycleLog())
    plan_a = compile_initial(compiler_a, registry, source_only()).plan
    plan_b = compile_initial(compiler_b, registry, source_only()).plan
    executor_a = PipelineExecutor(plan_a)
    executor_b = PipelineExecutor(plan_b)
    token_a, token_b = object(), object()
    executor_a.claim_management(token_a)
    executor_b.claim_management(token_b)
    admission_a = executor_a.reserve_frame_managed(token_a, 1, 1)

    with pytest.raises(RuntimeError, match="foreign or has already been consumed"):
        executor_b.cancel_reserved_frame_managed(token_b, admission_a)
    assert not executor_a.wait_for_plan_quiescent(plan_a.version, timeout=0.0)
    executor_a.cancel_reserved_frame_managed(token_a, admission_a)
    assert executor_a.wait_for_plan_quiescent(plan_a.version, timeout=0.0)
    executor_a.release_management(token_a)
    executor_b.release_management(token_b)


def test_clock_failure_during_cancellation_still_releases_lease() -> None:
    class FailingClock:
        def __call__(self) -> int:
            raise RuntimeError("clock failed")

    compiler = WorkflowCompiler("test")
    registry = stateless_registry(LifecycleLog())
    initial = compile_initial(compiler, registry, source_only())
    executor = PipelineExecutor(initial.plan, clock=FailingClock())
    token = object()
    executor.claim_management(token)
    admission = executor.reserve_frame_managed(token, 1, 1)

    with pytest.raises(RuntimeError, match="clock failed"):
        executor.cancel_reserved_frame_managed(token, admission)
    assert executor.wait_for_plan_quiescent(initial.plan.version, timeout=0.0)
    executor.release_management(token)


@pytest.mark.parametrize(
    ("directive", "expected"),
    [
        (StateDirective(frozenset({"absent"})), "absent from the target graph"),
        (StateDirective(frozenset({"source"})), "stateless node"),
    ],
)
def test_invalid_state_directives_are_rejected(
    directive: StateDirective,
    expected: str,
) -> None:
    compiler = WorkflowCompiler("test")
    registry = stateless_registry(LifecycleLog())
    initial = compile_initial(compiler, registry, source_only())

    result = compiler.compile(source_only(), registry, initial.plan, directive)

    assert isinstance(result, CompilationFailure)
    assert expected in result.reason


def test_reset_of_newly_added_stateful_node_is_rejected() -> None:
    compiler = WorkflowCompiler("test")
    initial_registry = stateless_registry(LifecycleLog())
    initial = compile_initial(compiler, initial_registry, source_only())
    target_registry = tracker_registry()

    result = compiler.compile(
        tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name),
        target_registry,
        initial.plan,
        StateDirective(frozenset({"tracker"})),
    )

    assert isinstance(result, CompilationFailure)
    assert "no previous state to reset" in result.reason


def test_factory_alias_of_active_processor_is_never_setup_or_cleaned() -> None:
    created: list[SyntheticTracker] = []

    def factory() -> SyntheticTracker:
        if created:
            return created[0]
        tracker = make_synthetic_tracker()
        created.append(tracker)
        return tracker

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
            factory=factory,
            stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
        )
    )
    registry = builder.snapshot()
    compiler = WorkflowCompiler("test")
    specification = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
    initial = compiler.compile(specification, registry)
    assert isinstance(initial, CompiledCandidate)
    before = created[0].snapshot_state()

    result = compiler.compile(
        specification,
        registry,
        initial.plan,
        StateDirective(frozenset({"tracker"})),
    )

    assert isinstance(result, CompilationFailure)
    assert "still owned by the previous active plan" in result.reason
    after = created[0].snapshot_state()
    assert after.setup_count == before.setup_count == 1
    assert after.cleanup_count == before.cleanup_count == 0
