"""Registry and compiler helpers shared across the test suite.

These helpers eliminate repetitive registry-building boilerplate while
remaining fully typed. Each function returns a ``RegistrySnapshot``
ready for use with ``WorkflowCompiler``.
"""

from __future__ import annotations

from nedo_vision_dag_engine.compiler import CompiledCandidate, WorkflowCompiler
from nedo_vision_dag_engine.processor import Processor
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder, RegistrySnapshot
from nedo_vision_dag_engine.specification import WorkflowSpecification
from tests.support.processors import (
    STATELESS_PASS_DESCRIPTOR,
    STATELESS_SOURCE_DESCRIPTOR,
    InjectedFailurePoint,
    LifecycleLog,
    make_pass_processor,
    make_source_processor,
)
from tests.support.tracker import (
    SYNTHETIC_TRACKER_DESCRIPTOR,
    SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    SYNTHETIC_TRACKER_V2_DESCRIPTOR,
    SYNTHETIC_TRACKER_V2_STATEFUL_DESCRIPTOR,
    TRACKER_RAISING_DESCRIPTOR,
    TRACKER_RAISING_PREDICATE_STATEFUL_DESCRIPTOR,
    make_raising_predicate_tracker,
    make_synthetic_tracker,
    make_synthetic_tracker_v2,
)


def stateless_registry(
    log: LifecycleLog,
    source_failure: InjectedFailurePoint = InjectedFailurePoint.NONE,
    pass_failure: InjectedFailurePoint = InjectedFailurePoint.NONE,
    fail_source_factory: bool = False,
    fail_pass_factory: bool = False,
) -> RegistrySnapshot:
    """Build a registry containing the two stateless test processor types.

    ``source_failure`` and ``pass_failure`` inject the given failure into
    each type's processors. ``fail_*_factory`` makes the factory itself
    raise instead of the lifecycle methods.
    """
    builder = RegistryBuilder()

    def source_factory() -> Processor:
        if fail_source_factory:
            raise RuntimeError("injected factory failure for source")
        return make_source_processor(log, source_failure, "source")

    def pass_factory() -> Processor:
        if fail_pass_factory:
            raise RuntimeError("injected factory failure for pass")
        return make_pass_processor(log, pass_failure, "pass")

    builder.register(
        RegisteredProcessorType(
            descriptor=STATELESS_SOURCE_DESCRIPTOR,
            factory=source_factory,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=STATELESS_PASS_DESCRIPTOR,
            factory=pass_factory,
        )
    )
    return builder.snapshot()


def tracker_registry(
    extra_stateless: bool = False,
    log: LifecycleLog | None = None,
) -> RegistrySnapshot:
    """Build a registry containing the synthetic tracker type.

    When ``extra_stateless`` is True, the stateless pass processor is also
    included so that tests can add a downstream stateless node to a
    tracker-only plan.
    """
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
            factory=make_synthetic_tracker,
            stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
        )
    )
    if extra_stateless:
        _log = log if log is not None else LifecycleLog()
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=lambda: make_pass_processor(_log, label="downstream"),
            )
        )
    return builder.snapshot()


def tracker_and_v2_registry() -> RegistrySnapshot:
    """Registry containing both tracker v1 and v2 for type-change rejection tests."""
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
            factory=make_synthetic_tracker,
            stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=SYNTHETIC_TRACKER_V2_DESCRIPTOR,
            factory=make_synthetic_tracker_v2,
            stateful_descriptor=SYNTHETIC_TRACKER_V2_STATEFUL_DESCRIPTOR,
        )
    )
    return builder.snapshot()


def tracker_raising_predicate_registry() -> RegistrySnapshot:
    """Registry for a tracker whose preservation predicate always raises."""
    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=TRACKER_RAISING_DESCRIPTOR,
            factory=make_raising_predicate_tracker,
            stateful_descriptor=TRACKER_RAISING_PREDICATE_STATEFUL_DESCRIPTOR,
        )
    )
    return builder.snapshot()


def compile_initial(
    compiler: WorkflowCompiler,
    registry: RegistrySnapshot,
    specification: WorkflowSpecification,
) -> CompiledCandidate:
    """Compile an initial plan and assert that it succeeded."""
    result = compiler.compile(specification, registry)
    assert isinstance(result, CompiledCandidate), (
        f"initial compilation failed unexpectedly: {result!r}"
    )
    return result
