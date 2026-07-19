"""The immutable execution plan.

This module defines the data produced by a successful compilation and
consumed, read-only, by the frame-execution loop in ``executor.py``:

- ``InputBinding`` -- one resolved edge, in workspace-slot-index form
  rather than node-id form. String node identifiers are resolved into
  integer output-slot indices during compilation, so the executor never
  has to do a name lookup while processing a frame.
- ``ReadinessRule`` -- the explicit per-step readiness predicate that
  replaces "unconditionally skip every transitive descendant of a gate"
  with something that can express fork, merge, optional-input, and
  bypass structures correctly.
- ``ExecutionStep`` -- one compiled node invocation.
- ``ExecutionPlan`` -- the full immutable plan: version, topological
  order (the ``steps`` tuple order itself), compiled steps, provenance
  metadata, and workspace size.

Implementation note on multi-output nodes: a node's declared output pins
may be more than one, but ``ExecutionStep`` carries exactly one
``output_index``. This module therefore treats each step's single
workspace slot as holding *all* of that node's produced outputs bundled
in one object (e.g. a dataclass or mapping whose fields/keys are the
output pin names), and ``InputBinding`` uses ``source_pin`` to extract
the specific named output at read time. This keeps the one-slot-per-step
shape while still supporting nodes with more than one declared output
pin.

Implementation note on multi-producer pins: a pin declared with
``PinCardinality.MULTIPLE`` may be fed by more than one edge, so more
than one ``InputBinding`` may legitimately share the same
``destination_pin``. ``construct_inputs`` aggregates every binding for a
given destination pin into one value: a single scalar when there is
exactly one binding, or a tuple of values when there is more than one.

Every ``ExecutionPlan`` this module accepts already has its steps in
topological order and every input binding wired to an earlier step; this
is verified structurally at construction time, so a topologically
invalid plan simply cannot be represented.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from nedo_vision_dag_engine.processor import Processor
from nedo_vision_dag_engine.type_system import MISSING

__all__ = [
    "WorkspaceView",
    "BranchPredicate",
    "ActivationCondition",
    "InputFactory",
    "InputBinding",
    "ReadinessRule",
    "ExecutionStep",
    "ExecutionPlan",
    "construct_inputs",
]

type WorkspaceView = Sequence[object]
"""A read-only view of one frame's workspace: index i holds the output of
the step whose ``output_index`` equals i, or ``MISSING`` if that step has
not produced a value for this frame (skipped, or not yet executed)."""

type BranchPredicate = Callable[[WorkspaceView], bool]
type ActivationCondition = Callable[[WorkspaceView], bool]
type InputFactory = Callable[["tuple[InputBinding, ...]", WorkspaceView], "dict[str, object]"]


@dataclass(frozen=True, slots=True)
class InputBinding:
    """One resolved data dependency feeding a single destination pin.

    More than one ``InputBinding`` may share the same ``destination_pin``
    when that pin declares ``PinCardinality.MULTIPLE``; each binding then
    represents one of the pin's several producers.
    """

    destination_pin: str
    source_index: int
    source_pin: str
    required: bool


@dataclass(frozen=True, slots=True)
class ReadinessRule:
    """The explicit readiness predicate for one step.

    A step executes only if every required source slot holds a value,
    its optional branch predicate (when present) evaluates to true, and
    its optional activation condition (when present) also evaluates to
    true. This permits safe fork, merge, optional-input, and bypass
    structures without ever unconditionally skipping a merge node that
    remains reachable through another branch.

    ``activation_condition`` is also where a multi-producer required pin
    (``PinCardinality.MULTIPLE``) gets its "at least one of these sources
    present" check: that condition is an OR across several source
    indices, which does not fit ``required_source_indices`` (an AND
    across all of them), so the compiler builds a small closure for it
    and attaches it here instead.
    """

    required_source_indices: tuple[int, ...]
    branch_predicate: BranchPredicate | None = None
    activation_condition: ActivationCondition | None = None

    def is_ready(self, workspace: WorkspaceView) -> bool:
        for index in self.required_source_indices:
            if workspace[index] is MISSING:
                return False
        if self.branch_predicate is not None and not self.branch_predicate(workspace):
            return False
        if self.activation_condition is not None and not self.activation_condition(workspace):
            return False
        return True


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """One compiled node invocation."""

    node_id: str
    output_index: int
    processor_ref: Processor
    input_bindings: tuple[InputBinding, ...]
    readiness_rule: ReadinessRule
    output_schema: type[object]

    def __post_init__(self) -> None:
        if self.output_index < 0:
            raise ValueError(f"step {self.node_id!r} has a negative output_index ({self.output_index}).")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The immutable execution plan for one compiled workflow version.

    ``steps`` is stored already in topological order; that order is the
    plan's execution order, so no separate ordering field is kept.
    Construction verifies, once, that this order is actually consistent
    (every input binding's source slot is produced by a step that
    precedes the consuming step) so that a topologically invalid plan can
    never exist as an ``ExecutionPlan`` instance.
    """

    version: int
    specification_hash: str
    registry_snapshot_id: str
    compiler_version: str
    steps: tuple[ExecutionStep, ...]
    output_slot_count: int
    processor_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("plan version must be a positive, monotonically increasing integer.")

        if self.output_slot_count != len(self.steps):
            raise ValueError(
                f"output_slot_count ({self.output_slot_count}) must equal the number of "
                f"steps ({len(self.steps)}); each step owns exactly one output slot."
            )

        output_indices = [step.output_index for step in self.steps]
        if set(output_indices) != set(range(self.output_slot_count)):
            raise ValueError(
                "step output_index values must form a dense range 0..output_slot_count-1 "
                "with no gaps or duplicates."
            )

        expected_processor_ids = frozenset(step.node_id for step in self.steps)
        if self.processor_ids != expected_processor_ids:
            raise ValueError("processor_ids must equal exactly the set of node_id values present in steps.")

        position_by_output_index: dict[int, int] = {
            step.output_index: position for position, step in enumerate(self.steps)
        }
        for position, step in enumerate(self.steps):
            for binding in step.input_bindings:
                if binding.source_index not in position_by_output_index:
                    raise ValueError(
                        f"step {step.node_id!r} input binding references unknown output slot "
                        f"{binding.source_index}."
                    )
                source_position = position_by_output_index[binding.source_index]
                if source_position >= position:
                    raise ValueError(
                        f"step {step.node_id!r} at position {position} depends on output slot "
                        f"{binding.source_index}, produced at position {source_position}, which "
                        "does not precede it; steps must be stored in topological order."
                    )
            for index in step.readiness_rule.required_source_indices:
                if index not in position_by_output_index:
                    raise ValueError(
                        f"step {step.node_id!r} readiness rule references unknown output slot {index}."
                    )

    def step_by_node_id(self, node_id: str) -> ExecutionStep | None:
        return next((step for step in self.steps if step.node_id == node_id), None)


def _extract_named_output(value: object, pin_name: str) -> object:
    """Extract one named output from a step's bundled output value.

    Supports both a mapping-shaped output (``value[pin_name]``) and an
    attribute-shaped output such as a dataclass instance
    (``getattr(value, pin_name)``), so a processor may return whichever
    representation suits its ``output_schema``.
    """
    if isinstance(value, Mapping):
        return value[pin_name] # type: ignore
    return getattr(value, pin_name)


def construct_inputs(input_bindings: tuple[InputBinding, ...], workspace: WorkspaceView) -> dict[str, object]:
    """Default, non-specialized input construction.

    Returns a plain keyword-style mapping from destination pin name to
    value; the executor is responsible for materializing this into
    whichever concrete ``input_schema`` type the target processor
    expects.

    Bindings are grouped by ``destination_pin`` first. A pin fed by
    exactly one binding gets a plain scalar value (or ``MISSING`` if
    that single, optional binding's source has not produced anything
    yet). A pin fed by more than one binding -- a multi-producer pin --
    gets a tuple of whatever values its producers have actually made
    available so far, skipping any that are still ``MISSING``.

    A compiler may substitute a specialized, generated equivalent behind
    the ``InputFactory`` type alias; this function is the reference
    implementation such a specialization must remain equivalent to.
    """
    bindings_by_destination: dict[str, list[InputBinding]] = {}
    for binding in input_bindings:
        bindings_by_destination.setdefault(binding.destination_pin, []).append(binding)

    constructed: dict[str, object] = {}
    for destination_pin, bindings in bindings_by_destination.items():
        if len(bindings) == 1:
            binding = bindings[0]
            producer_output = workspace[binding.source_index]
            if producer_output is MISSING:
                if binding.required:
                    raise ValueError(
                        f"required input binding for destination pin {destination_pin!r} has no "
                        f"value at output slot {binding.source_index}; the readiness rule should "
                        "have prevented this step from executing."
                    )
                constructed[destination_pin] = MISSING
                continue
            constructed[destination_pin] = _extract_named_output(producer_output, binding.source_pin)
            continue

        values: list[object] = []
        for binding in bindings:
            producer_output = workspace[binding.source_index]
            if producer_output is MISSING:
                continue
            values.append(_extract_named_output(producer_output, binding.source_pin))
        constructed[destination_pin] = tuple(values)

    return constructed