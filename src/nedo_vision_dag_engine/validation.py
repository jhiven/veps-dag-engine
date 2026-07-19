"""Structural and type validation.

This module implements the two validation phases that run after parsing
and normalization (``specification.py``) and before cycle detection and
execution-step construction (``compiler.py``):

- **Structural validation**: duplicate node identifiers, unknown
  processor types, malformed edge references, unknown pins, missing
  required inputs, unsupported multiple producers, and invalid processor
  configurations.
- **Type validation**: for every structurally valid edge, evaluate the
  payload type-assignability relation ``tau(p_s) <= tau(p_d)``, resolving
  generic pins through one ``GenericTypeResolver`` shared across the
  whole workflow.

Both phases collect every violation they find into one
``ValidationResult`` rather than raising on the first error, so that a
caller (the compiler, or a test) can observe the complete set of problems
in one candidate, and report a structured failure with reasons instead of
an opaque raised exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from nedo_vision_dag_engine.registry import RegistrySnapshot
from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    ProcessorConfiguration,
    WorkflowSpecification,
)
from nedo_vision_dag_engine.type_system import GenericResolutionError, GenericTypeResolver, assignable

__all__ = [
    "ConfigSchema",
    "ValidationErrorCode",
    "ValidationError",
    "ValidationResult",
    "validate_structure",
    "validate_types",
    "validate_workflow",
]


@runtime_checkable
class ConfigSchema(Protocol):
    """Optional contract a processor's ``config_schema`` type may
    implement so that structural validation can check a node's raw
    configuration against it.

    A ``config_schema`` that does not implement this protocol is treated
    as declaring no checkable constraints; pin- and edge-level structural
    validation still applies regardless.
    """

    @classmethod
    def validate_configuration(cls, configuration: ProcessorConfiguration) -> tuple[str, ...]:
        """Return a tuple of human-readable error messages; empty means valid."""
        ...


class ValidationErrorCode(Enum):
    DUPLICATE_NODE_ID = "duplicate_node_id"
    UNKNOWN_PROCESSOR_TYPE = "unknown_processor_type"
    UNKNOWN_SOURCE_NODE = "unknown_source_node"
    UNKNOWN_DESTINATION_NODE = "unknown_destination_node"
    UNKNOWN_SOURCE_PIN = "unknown_source_pin"
    UNKNOWN_DESTINATION_PIN = "unknown_destination_pin"
    MISSING_REQUIRED_INPUT = "missing_required_input"
    UNSUPPORTED_MULTIPLE_PRODUCERS = "unsupported_multiple_producers"
    INVALID_CONFIGURATION = "invalid_configuration"
    INCOMPATIBLE_PIN_TYPE = "incompatible_pin_type"
    GENERIC_TYPE_CONFLICT = "generic_type_conflict"


@dataclass(frozen=True, slots=True)
class ValidationError:
    code: ValidationErrorCode
    message: str
    node_id: str | None = None
    edge: Edge | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: tuple[ValidationError, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def _validate_configuration(config_schema: type[object], configuration: ProcessorConfiguration) -> tuple[str, ...]:
    if issubclass(config_schema, ConfigSchema):
        return config_schema.validate_configuration(configuration)
    return ()


def validate_structure(specification: WorkflowSpecification, registry: RegistrySnapshot) -> ValidationResult:
    """Structural validation: everything that can be checked about the
    graph's shape and references without looking at payload types."""
    errors: list[ValidationError] = []

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    nodes_by_id: dict[str, Node] = {}
    for node in specification.nodes:
        if node.node_id in seen_ids:
            duplicate_ids.add(node.node_id)
        else:
            nodes_by_id[node.node_id] = node
        seen_ids.add(node.node_id)

    for node_id in sorted(duplicate_ids):
        errors.append(
            ValidationError(
                ValidationErrorCode.DUPLICATE_NODE_ID,
                f"node id {node_id!r} is declared more than once.",
                node_id=node_id,
            )
        )

    def _is_canonical_occurrence(node: Node) -> bool:
        return node.node_id not in duplicate_ids or nodes_by_id.get(node.node_id) is node

    for node in specification.nodes:
        if not _is_canonical_occurrence(node):
            continue
        registration = registry.resolve(node.type_name)
        if registration is None:
            errors.append(
                ValidationError(
                    ValidationErrorCode.UNKNOWN_PROCESSOR_TYPE,
                    f"processor type {node.type_name!r} referenced by node {node.node_id!r} is not registered.",
                    node_id=node.node_id,
                )
            )
            continue
        for message in _validate_configuration(registration.descriptor.config_schema, node.configuration):
            errors.append(ValidationError(ValidationErrorCode.INVALID_CONFIGURATION, message, node_id=node.node_id))

    for edge in specification.edges:
        source_node = nodes_by_id.get(edge.source_node_id)
        destination_node = nodes_by_id.get(edge.destination_node_id)
        if source_node is None:
            errors.append(
                ValidationError(
                    ValidationErrorCode.UNKNOWN_SOURCE_NODE,
                    f"edge references unknown source node {edge.source_node_id!r}.",
                    edge=edge,
                )
            )
        elif source_node.output_pin(edge.source_pin) is None:
            errors.append(
                ValidationError(
                    ValidationErrorCode.UNKNOWN_SOURCE_PIN,
                    f"node {edge.source_node_id!r} has no output pin {edge.source_pin!r}.",
                    edge=edge,
                )
            )
        if destination_node is None:
            errors.append(
                ValidationError(
                    ValidationErrorCode.UNKNOWN_DESTINATION_NODE,
                    f"edge references unknown destination node {edge.destination_node_id!r}.",
                    edge=edge,
                )
            )
        elif destination_node.input_pin(edge.destination_pin) is None:
            errors.append(
                ValidationError(
                    ValidationErrorCode.UNKNOWN_DESTINATION_PIN,
                    f"node {edge.destination_node_id!r} has no input pin {edge.destination_pin!r}.",
                    edge=edge,
                )
            )

    producer_counts: dict[tuple[str, str], int] = {}
    for edge in specification.edges:
        source_node = nodes_by_id.get(edge.source_node_id)
        destination_node = nodes_by_id.get(edge.destination_node_id)
        if source_node is None or destination_node is None:
            continue
        if source_node.output_pin(edge.source_pin) is None:
            continue
        if destination_node.input_pin(edge.destination_pin) is None:
            continue
        key = (edge.destination_node_id, edge.destination_pin)
        producer_counts[key] = producer_counts.get(key, 0) + 1

    for node in specification.nodes:
        if not _is_canonical_occurrence(node):
            continue
        for pin in node.inputs:
            count = producer_counts.get((node.node_id, pin.name), 0)
            if pin.cardinality is PinCardinality.SINGLE and count > 1:
                errors.append(
                    ValidationError(
                        ValidationErrorCode.UNSUPPORTED_MULTIPLE_PRODUCERS,
                        f"input pin {pin.name!r} of node {node.node_id!r} has cardinality SINGLE "
                        f"but receives {count} producers.",
                        node_id=node.node_id,
                    )
                )
            if pin.requirement is PinRequirement.REQUIRED and count == 0:
                errors.append(
                    ValidationError(
                        ValidationErrorCode.MISSING_REQUIRED_INPUT,
                        f"required input pin {pin.name!r} of node {node.node_id!r} has no producer.",
                        node_id=node.node_id,
                    )
                )

    return ValidationResult(errors=tuple(errors))


def validate_types(
    specification: WorkflowSpecification,
    structural_result: ValidationResult,
) -> tuple[ValidationResult, GenericTypeResolver]:
    """Type validation: checks the payload type-assignability relation on
    every edge that structural validation did not already reject.

    Edges already reported as structurally invalid (unknown node or pin)
    are skipped here, since their payload types cannot be resolved; this
    avoids cascading, meaningless type errors on top of a structural one.
    """
    errors: list[ValidationError] = []
    resolver = GenericTypeResolver()
    structurally_invalid_edges: set[Edge] = {
        error.edge for error in structural_result.errors if error.edge is not None
    }
    nodes_by_id: dict[str, Node] = {}
    for node in specification.nodes:
        nodes_by_id.setdefault(node.node_id, node)

    for edge in specification.edges:
        if edge in structurally_invalid_edges:
            continue
        source_node = nodes_by_id.get(edge.source_node_id)
        destination_node = nodes_by_id.get(edge.destination_node_id)
        if source_node is None or destination_node is None:
            continue
        source_pin: Pin | None = source_node.output_pin(edge.source_pin)
        destination_pin: Pin | None = destination_node.input_pin(edge.destination_pin)
        if source_pin is None or destination_pin is None:
            continue
        try:
            is_assignable = assignable(source_pin.payload_type, destination_pin.payload_type, resolver)
        except GenericResolutionError as error:
            errors.append(ValidationError(ValidationErrorCode.GENERIC_TYPE_CONFLICT, str(error), edge=edge))
            continue
        if not is_assignable:
            errors.append(
                ValidationError(
                    ValidationErrorCode.INCOMPATIBLE_PIN_TYPE,
                    f"edge {edge.source_node_id}.{edge.source_pin} -> "
                    f"{edge.destination_node_id}.{edge.destination_pin}: "
                    f"{source_pin.payload_type!r} is not assignable to {destination_pin.payload_type!r}.",
                    edge=edge,
                )
            )

    return ValidationResult(errors=tuple(errors)), resolver


def validate_workflow(
    specification: WorkflowSpecification,
    registry: RegistrySnapshot,
) -> tuple[ValidationResult, GenericTypeResolver]:
    """Run structural validation followed by type validation and combine
    both result sets into one ``ValidationResult``.

    The returned ``GenericTypeResolver`` carries every generic-group
    binding discovered during type validation; the compiler reuses it
    when resolving generic pins prior to processor initialization.
    """
    structural_result = validate_structure(specification, registry)
    type_result, resolver = validate_types(specification, structural_result)
    combined = ValidationResult(errors=structural_result.errors + type_result.errors)
    return combined, resolver