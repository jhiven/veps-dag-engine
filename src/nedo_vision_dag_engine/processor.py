"""The processor contract.

This module defines what every processor implementation must expose to
the compiler, the registry, and the executor:

- ``Processor``, the structural interface every node's runtime instance
  implements;
- ``ProcessorDescriptor``, the immutable metadata every processor type
  publishes (schemas, state policy);
- ``StatefulProcessor`` / ``StatefulProcessorDescriptor``, the additional
  contract a processor with cross-frame mutable state must satisfy so the
  reconfiguration controller (built later, in ``reconfiguration.py`` and
  ``lifecycle.py``) can decide whether that state may be preserved, must
  be reset, or makes the request unreconcilable;
- ``SetupContext`` / ``FrameContext``, the immutable context objects
  passed into ``setup`` and ``process`` respectively;
- ``StateTransitionEvent``, the audit record emitted whenever a stateful
  node's state is reset or otherwise transitioned across a plan version
  boundary.

This module intentionally does not implement the reuse/preservation
decision algorithms themselves: those depend on a concrete
reconfiguration request and the previous/candidate plans, which are
defined later. Here we only fix the shape of the predicate
(``preserves_state_for``) and the context it is evaluated with.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nedo_vision_dag_engine.specification import ProcessorConfiguration
from nedo_vision_dag_engine.type_system import PayloadType, StatePolicy, StateTransitionPolicy

__all__ = [
    "ProcessorDescriptor",
    "StatefulProcessorDescriptor",
    "TransitionContext",
    "SetupContext",
    "FrameContext",
    "Processor",
    "StatefulProcessor",
    "StateTransitionEvent",
]


@dataclass(frozen=True, slots=True)
class ProcessorDescriptor:
    """Immutable metadata published by every registered processor type.

    ``input_schema``, ``output_schema``, and ``config_schema`` are typed
    as ``type[object]`` rather than a bare ``type``, so that a schema
    class is always a concrete Python type rather than an implicit
    ``Any``.
    """

    type_name: str
    input_schema: type[object]
    output_schema: type[object]
    config_schema: type[object]
    state_policy: StatePolicy
    state_schema_version: str | None

    def __post_init__(self) -> None:
        if self.state_policy is StatePolicy.MIGRATABLE:
            raise ValueError(
                "StatePolicy.MIGRATABLE is reserved for future general "
                "state-transformation support; the initial implementation "
                "supports only STATELESS, PRESERVABLE, and RESETTABLE."
            )
        if self.state_policy is StatePolicy.STATELESS and self.state_schema_version is not None:
            raise ValueError(
                f"processor {self.type_name!r} is STATELESS and must not declare a "
                "state_schema_version, since it has no cross-frame state to version."
            )
        if self.state_policy is not StatePolicy.STATELESS and self.state_schema_version is None:
            raise ValueError(
                f"processor {self.type_name!r} has state policy {self.state_policy!r} and "
                "must declare a state_schema_version."
            )


@dataclass(frozen=True, slots=True)
class TransitionContext:
    """The information available to a stateful processor's
    ``preserves_state_for`` predicate when deciding whether a
    reconfiguration transition preserves state semantics.

    Each field corresponds directly to one condition a preservation
    decision needs to check: the caller (the reconfiguration controller,
    built later) is responsible for establishing these facts by comparing
    the previous and candidate node specifications before invoking the
    predicate; the predicate only combines them with any
    processor-specific compatibility logic of its own.
    """

    node_id: str
    previous_processor_type: str
    new_processor_type: str
    previous_state_schema_version: str
    new_state_schema_version: str
    previous_input_payload_types: dict[str, PayloadType]
    new_input_payload_types: dict[str, PayloadType]
    upstream_semantic_coordinate_system_unchanged: bool
    label_taxonomy_compatible: bool
    source_identity_unchanged: bool
    explicit_reset_requested: bool

    @property
    def processor_type_unchanged(self) -> bool:
        return self.previous_processor_type == self.new_processor_type

    @property
    def state_schema_version_unchanged(self) -> bool:
        return self.previous_state_schema_version == self.new_state_schema_version

    @property
    def input_payload_types_unchanged(self) -> bool:
        return self.previous_input_payload_types == self.new_input_payload_types

    @property
    def structurally_preservable(self) -> bool:
        """The node-identity-independent structural conditions common to
        every stateful processor, excluding any processor-specific
        compatibility logic that only ``preserves_state_for`` itself can
        evaluate."""
        return (
            self.processor_type_unchanged
            and self.state_schema_version_unchanged
            and self.input_payload_types_unchanged
            and self.upstream_semantic_coordinate_system_unchanged
            and self.label_taxonomy_compatible
            and self.source_identity_unchanged
            and not self.explicit_reset_requested
        )


@dataclass(frozen=True, slots=True)
class StatefulProcessorDescriptor:
    """Additional descriptor a stateful processor publishes alongside its
    ``ProcessorDescriptor``.

    ``preserves_state_for`` receives the previous configuration, the new
    configuration, and a ``TransitionContext`` describing the surrounding
    circumstances of the transition, and returns whether the processor
    declares this specific transition state-compatible.
    """

    state_schema_version: str
    supported_transition_policies: frozenset[StateTransitionPolicy]
    preserves_state_for: Callable[[ProcessorConfiguration, ProcessorConfiguration, TransitionContext], bool]

    def __post_init__(self) -> None:
        if not self.state_schema_version:
            raise ValueError("state_schema_version must be a non-empty string.")
        if not self.supported_transition_policies:
            raise ValueError("a stateful processor must support at least one transition policy.")
        if StateTransitionPolicy.MIGRATE in self.supported_transition_policies:
            raise ValueError(
                "StateTransitionPolicy.MIGRATE is reserved for a future implementation; "
                "general state transformation is out of scope for now."
            )


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Immutable context passed to ``Processor.setup`` during candidate
    preparation.

    ``candidate_plan_version`` identifies the plan version being staged so
    that setup-time logging and diagnostics can be correlated with a
    specific reconfiguration attempt, even though the processor is not yet
    visible to any executor at this point.
    """

    node_id: str
    configuration: ProcessorConfiguration
    registry_snapshot_id: str
    candidate_plan_version: int
    warm_up_requested: bool


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Immutable per-frame context passed to ``Processor.process``."""

    frame_id: int
    plan_version: int
    admitted_at_ns: int


@runtime_checkable
class Processor(Protocol):
    """The structural interface every processor's runtime instance
    implements.

    ``runtime_checkable`` allows ``isinstance`` checks for the presence of
    these members, but per Python's typing semantics this only verifies
    that the attributes and methods exist, not that their signatures or
    the ``descriptor`` value's contents conform; the compiler and
    validation layers remain responsible for full contract enforcement.
    """

    descriptor: ProcessorDescriptor

    def setup(self, context: SetupContext) -> None: ...

    def process(self, inputs: object, context: FrameContext) -> object: ...

    def healthcheck(self) -> None: ...

    def cleanup(self) -> None: ...


@runtime_checkable
class StatefulProcessor(Processor, Protocol):
    """A ``Processor`` that additionally publishes a
    ``StatefulProcessorDescriptor``.

    Only processors declaring ``state_policy`` of ``PRESERVABLE`` or
    ``RESETTABLE`` on their ``ProcessorDescriptor`` are expected to satisfy
    this extended protocol.
    """

    stateful_descriptor: StatefulProcessorDescriptor


@dataclass(frozen=True, slots=True)
class StateTransitionEvent:
    """Audit record emitted whenever a stateful node's state is reset
    across a plan-version boundary. State reset must never happen
    silently; this record is what makes it visible.
    """

    node_id: str
    old_plan_version: int
    new_plan_version: int
    policy: StateTransitionPolicy
    reason: str
    committed_at_ns: int