"""Typed failure-injection processor utilities.

Every processor here is strongly typed and implements the ``Processor``
protocol. No "universal fake" with arbitrary dictionary behavior exists.

Design notes
------------
- ``InjectedFailurePoint`` enumerates every lifecycle phase at which a
  failure can be injected without covering warm-up separately, because
  the engine invokes setup then healthcheck in a single combined call;
  warm-up is not a separately observable phase in the current engine.
- ``LifecycleObservation`` is an immutable record of a single lifecycle
  event, stored in a thread-safe ``LifecycleLog``.
- ``FailingProcessor`` is the concrete typed processor; callers pick the
  failure point via the enum rather than through a raw callback.
- Processors record typed lifecycle observations including concurrency
  counters so tests can verify that at most one frame executes a given
  processor instance at a time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
)
from nedo_vision_dag_engine.specification import ProcessorConfiguration
from nedo_vision_dag_engine.type_system import StatePolicy


class InjectedFailurePoint(Enum):
    """Which processor lifecycle phase raises an injected exception."""

    NONE = "none"
    FACTORY = "factory"
    SETUP = "setup"
    HEALTHCHECK = "healthcheck"
    PROCESS = "process"
    CLEANUP = "cleanup"
    PREDICATE = "predicate"


@dataclass(frozen=True, slots=True)
class LifecycleObservation:
    """One recorded lifecycle event for a single processor instance."""

    instance_id: int
    event: str
    frame_id: int | None = None
    plan_version: int | None = None


class LifecycleLog:
    """Thread-safe append-only log of lifecycle observations.

    Multiple processor instances may share one log so that tests can
    observe the interleaved sequence across a whole pipeline.
    """

    __slots__ = ("_lock", "_observations")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observations: list[LifecycleObservation] = []

    def append(self, observation: LifecycleObservation) -> None:
        with self._lock:
            self._observations.append(observation)

    def observations(self) -> tuple[LifecycleObservation, ...]:
        with self._lock:
            return tuple(self._observations)

    def events_for(self, instance_id: int) -> tuple[str, ...]:
        return tuple(o.event for o in self.observations() if o.instance_id == instance_id)

    def all_events(self) -> tuple[str, ...]:
        return tuple(o.event for o in self.observations())

    def count_event(self, event: str) -> int:
        return sum(1 for o in self.observations() if o.event == event)

    def count_event_for(self, event: str, instance_id: int) -> int:
        return sum(
            1 for o in self.observations() if o.event == event and o.instance_id == instance_id
        )


@dataclass(slots=True)
class InstanceMetrics:
    """Per-instance lifecycle counters, updated under a lock.

    ``active_invocation_count`` and ``max_concurrent_invocations`` allow
    tests to verify that a preserved mutable processor is never invoked
    concurrently by two plan versions.
    """

    setup_count: int = 0
    healthcheck_count: int = 0
    process_count: int = 0
    cleanup_count: int = 0
    cleanup_succeeded: bool = False
    active_invocation_count: int = 0
    max_concurrent_invocations: int = 0
    processed_frame_ids: list[int] = field(default_factory=list[int])
    observed_plan_versions: list[int] = field(default_factory=list[int])


_METRIC_LOCK: Final = threading.Lock()


class FailingProcessor:
    """A typed test processor that injects failures at a chosen lifecycle phase.

    Observations are appended to a shared ``LifecycleLog`` and per-instance
    counters are maintained in ``metrics``.  The instance identifier used for
    observation keys is ``id(self)`` so that test code can correlate log
    entries with specific object instances.
    """

    def __init__(
        self,
        descriptor: ProcessorDescriptor,
        log: LifecycleLog,
        failure_point: InjectedFailurePoint = InjectedFailurePoint.NONE,
        label: str = "",
    ) -> None:
        self.descriptor = descriptor
        self._log = log
        self._failure_point = failure_point
        self._label = label or descriptor.type_name
        self._metrics_lock = threading.Lock()
        self.metrics = InstanceMetrics()

    @property
    def instance_id(self) -> int:
        return id(self)

    def setup(self, context: SetupContext) -> None:
        with self._metrics_lock:
            self.metrics.setup_count += 1
        self._log.append(LifecycleObservation(instance_id=self.instance_id, event="setup"))
        if self._failure_point is InjectedFailurePoint.SETUP:
            raise RuntimeError(f"injected setup failure for {self._label!r}")

    def process(self, inputs: object, context: FrameContext) -> object:
        with self._metrics_lock:
            self.metrics.active_invocation_count += 1
            if self.metrics.active_invocation_count > self.metrics.max_concurrent_invocations:
                self.metrics.max_concurrent_invocations = self.metrics.active_invocation_count

        try:
            self._log.append(
                LifecycleObservation(
                    instance_id=self.instance_id,
                    event="process",
                    frame_id=context.frame_id,
                    plan_version=context.plan_version,
                )
            )
            if self._failure_point is InjectedFailurePoint.PROCESS:
                raise RuntimeError(f"injected process failure for {self._label!r}")

            with self._metrics_lock:
                self.metrics.process_count += 1
                self.metrics.processed_frame_ids.append(context.frame_id)
                self.metrics.observed_plan_versions.append(context.plan_version)
            return PassOutput(value=context.frame_id)
        finally:
            with self._metrics_lock:
                self.metrics.active_invocation_count -= 1

    def healthcheck(self) -> None:
        with self._metrics_lock:
            self.metrics.healthcheck_count += 1
        self._log.append(LifecycleObservation(instance_id=self.instance_id, event="healthcheck"))
        if self._failure_point is InjectedFailurePoint.HEALTHCHECK:
            raise RuntimeError(f"injected healthcheck failure for {self._label!r}")

    def cleanup(self) -> None:
        with self._metrics_lock:
            self.metrics.cleanup_count += 1
        self._log.append(LifecycleObservation(instance_id=self.instance_id, event="cleanup"))
        if self._failure_point is InjectedFailurePoint.CLEANUP:
            raise RuntimeError(f"injected cleanup failure for {self._label!r}")
        with self._metrics_lock:
            self.metrics.cleanup_succeeded = True


@dataclass(frozen=True, slots=True)
class PassOutput:
    """Output produced by simple pass-through processors in the test suite."""

    value: int


@dataclass(frozen=True, slots=True)
class PassInput:
    """Input consumed by simple pass-through processors in the test suite."""

    value: int


STATELESS_SOURCE_DESCRIPTOR: Final = ProcessorDescriptor(
    type_name="stateless_source",
    input_schema=object,
    output_schema=PassOutput,
    config_schema=object,
    state_policy=StatePolicy.STATELESS,
    state_schema_version=None,
)

STATELESS_PASS_DESCRIPTOR: Final = ProcessorDescriptor(
    type_name="stateless_pass",
    input_schema=PassInput,
    output_schema=PassOutput,
    config_schema=object,
    state_policy=StatePolicy.STATELESS,
    state_schema_version=None,
)


def make_source_processor(
    log: LifecycleLog,
    failure_point: InjectedFailurePoint = InjectedFailurePoint.NONE,
    label: str = "source",
) -> FailingProcessor:
    """Return a stateless source processor with optional failure injection."""
    return FailingProcessor(STATELESS_SOURCE_DESCRIPTOR, log, failure_point, label)


def make_pass_processor(
    log: LifecycleLog,
    failure_point: InjectedFailurePoint = InjectedFailurePoint.NONE,
    label: str = "pass",
) -> FailingProcessor:
    """Return a stateless pass-through processor with optional failure injection."""
    return FailingProcessor(STATELESS_PASS_DESCRIPTOR, log, failure_point, label)


def make_failing_factory(
    descriptor: ProcessorDescriptor,
    log: LifecycleLog,
    label: str = "bad_factory",
) -> FailingProcessor:
    """Raise immediately when called, simulating a constructor failure.

    This function raises so that it can be used directly as a
    ``ProcessorFactory`` that always fails; the test registers it in the
    registry and the compiler's factory-call path will catch the exception.
    """
    raise RuntimeError(f"injected factory failure for {label!r}")


def make_blocking_source_processor(
    log: LifecycleLog,
    entered: threading.Event,
    release: threading.Event,
    label: str = "blocking_source",
) -> "BlockingSourceProcessor":
    """Return a source processor that blocks inside process until released."""
    return BlockingSourceProcessor(log, entered, release, label)


class BlockingSourceProcessor:
    """A source processor that signals when it enters process and blocks until released.

    Used to test frame-boundary concurrency: a test can start a frame,
    then attempt a plan commit from another thread while process is
    blocked, and verify that the commit does not happen until the frame
    exits.
    """

    __slots__ = (
        "_entered",
        "_label",
        "_log",
        "_metrics_lock",
        "_release",
        "descriptor",
        "metrics",
    )

    def __init__(
        self,
        log: LifecycleLog,
        entered: threading.Event,
        release: threading.Event,
        label: str = "blocking_source",
    ) -> None:
        self.descriptor = STATELESS_SOURCE_DESCRIPTOR
        self._log = log
        self._entered = entered
        self._release = release
        self._label = label
        self._metrics_lock = threading.Lock()
        self.metrics = InstanceMetrics()

    @property
    def instance_id(self) -> int:
        return id(self)

    def setup(self, context: SetupContext) -> None:
        with self._metrics_lock:
            self.metrics.setup_count += 1
        self._log.append(LifecycleObservation(instance_id=self.instance_id, event="setup"))

    def process(self, inputs: object, context: FrameContext) -> object:
        with self._metrics_lock:
            self.metrics.active_invocation_count += 1
            if self.metrics.active_invocation_count > self.metrics.max_concurrent_invocations:
                self.metrics.max_concurrent_invocations = self.metrics.active_invocation_count

        try:
            self._log.append(
                LifecycleObservation(
                    instance_id=self.instance_id,
                    event="process_enter",
                    frame_id=context.frame_id,
                    plan_version=context.plan_version,
                )
            )
            self._entered.set()
            if not self._release.wait(timeout=5.0):
                raise RuntimeError("test did not release blocking processor within timeout")
            self._log.append(
                LifecycleObservation(
                    instance_id=self.instance_id,
                    event="process_exit",
                    frame_id=context.frame_id,
                    plan_version=context.plan_version,
                )
            )
            with self._metrics_lock:
                self.metrics.process_count += 1
                self.metrics.processed_frame_ids.append(context.frame_id)
                self.metrics.observed_plan_versions.append(context.plan_version)
            return PassOutput(value=context.frame_id)
        finally:
            with self._metrics_lock:
                self.metrics.active_invocation_count -= 1

    def healthcheck(self) -> None:
        with self._metrics_lock:
            self.metrics.healthcheck_count += 1

    def cleanup(self) -> None:
        with self._metrics_lock:
            self.metrics.cleanup_count += 1
        self._log.append(LifecycleObservation(instance_id=self.instance_id, event="cleanup"))
        with self._metrics_lock:
            self.metrics.cleanup_succeeded = True


def configuration_from_config(config: ProcessorConfiguration) -> ProcessorConfiguration:
    """Identity function that makes the type explicit for Pyright."""
    return config
