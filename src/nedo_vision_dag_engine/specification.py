"""Immutable workflow-specification schema and canonical serialization.

This is the parsing and normalization layer: the workflow graph
G = (V, E) is represented as frozen dataclasses, and a specification can
be canonically serialized and hashed to produce a specification hash
``h_G = H(Canonicalize(G))`` that gets carried along as plan provenance.

This module performs no cross-referential validation. Whether node ids
are unique, whether referenced pins exist, whether the graph is acyclic,
and whether edges satisfy the type-assignability relation are structural
and type-validation concerns handled by ``validation.py`` and
``compiler.py``. A ``WorkflowSpecification`` here may therefore represent
a structurally invalid workflow; that is intentional. The validation
layer needs to be able to observe inputs like a duplicate node identifier
or an unknown pin and report them as a validation error with a reason,
rather than have them rejected silently at parse time before any
diagnostic can even be produced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from nedo_vision_dag_engine.type_system import ConcreteType, GenericParameter, PayloadType

__all__ = [
    "JsonValue",
    "ProcessorConfiguration",
    "freeze_json_value",
    "to_processor_configuration",
    "PinCardinality",
    "PinRequirement",
    "Pin",
    "Node",
    "Edge",
    "WorkflowSpecification",
    "canonicalize",
    "specification_hash",
]

type JsonValue = None | bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue]
"""A frozen JSON-shaped value: a parsed JSON array becomes a tuple and a
parsed JSON object becomes an immutable mapping. Processor configuration
payloads are represented with this type: this is the explicit
"node configuration may accept dict-shaped data" exception, kept
immutable rather than a plain mutable ``dict``."""

type ProcessorConfiguration = Mapping[str, JsonValue]


def freeze_json_value(value: object) -> JsonValue:
    """Recursively convert plain JSON-like Python data (e.g. the output of
    ``json.loads``) into the immutable ``JsonValue`` representation.

    Already-frozen values (``MappingProxyType``, ``tuple`` of frozen
    values) pass through without being copied again. Raises ``TypeError``
    for any value that is not JSON-shaped.
    """
    match value:
        case None | bool() | int() | float() | str():
            return value
        case MappingProxyType():
            return value # type: ignore
        case Mapping():
            return MappingProxyType({str(key): freeze_json_value(item) for key, item in value.items()}) # type: ignore
        case tuple() | list():
            return tuple(freeze_json_value(item) for item in value) # type: ignore
        case _:
            raise TypeError(f"value {value!r} of type {type(value)!r} is not JSON-shaped")


def to_processor_configuration(data: Mapping[str, object]) -> ProcessorConfiguration:
    """Freeze a plain top-level JSON-object mapping into a ``ProcessorConfiguration``."""
    frozen = freeze_json_value(dict(data))
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("processor configuration must be a JSON object at the top level")
    return frozen


class PinCardinality(Enum):
    """Whether a pin is bound to exactly one value or a sequence of values."""

    SINGLE = "single"
    MULTIPLE = "multiple"


class PinRequirement(Enum):
    """Whether a pin must be bound for its node to be executable."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class Pin:
    """One declared input or output pin on a node."""

    name: str
    payload_type: PayloadType
    cardinality: PinCardinality
    requirement: PinRequirement


@dataclass(frozen=True, slots=True)
class Node:
    """A processor node ``n = (id, type, configuration, I, O)``."""

    node_id: str
    type_name: str
    configuration: ProcessorConfiguration
    inputs: tuple[Pin, ...]
    outputs: tuple[Pin, ...]

    def input_pin(self, name: str) -> Pin | None:
        return next((pin for pin in self.inputs if pin.name == name), None)

    def output_pin(self, name: str) -> Pin | None:
        return next((pin for pin in self.outputs if pin.name == name), None)


@dataclass(frozen=True, slots=True)
class Edge:
    """A data dependency ``e = (n_s, p_s, n_d, p_d)``."""

    source_node_id: str
    source_pin: str
    destination_node_id: str
    destination_pin: str


@dataclass(frozen=True, slots=True)
class WorkflowSpecification:
    """A workflow graph ``G = (V, E)``.

    Nodes and edges are stored exactly as supplied, in whatever order the
    caller provided; this class performs no ordering or uniqueness
    enforcement. Use ``canonicalize`` to obtain the order-independent
    representation used for hashing and comparison.
    """

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def _serialize_payload_type(payload_type: PayloadType) -> str:
    match payload_type:
        case ConcreteType(python_type=python_type):
            return f"concrete:{python_type.__module__}.{python_type.__qualname__}"
        case GenericParameter(name=name, group=group):
            return f"generic:{group}:{name}"
        case _:
            raise TypeError(f"unrecognized payload type variant: {payload_type!r}")


def _serialize_pin(pin: Pin) -> dict[str, JsonValue]:
    return {
        "name": pin.name,
        "payload_type": _serialize_payload_type(pin.payload_type),
        "cardinality": pin.cardinality.value,
        "requirement": pin.requirement.value,
    }


def _serialize_node(node: Node) -> dict[str, JsonValue]:
    return {
        "node_id": node.node_id,
        "type_name": node.type_name,
        "configuration": node.configuration,
        "inputs": tuple(_serialize_pin(pin) for pin in sorted(node.inputs, key=lambda pin: pin.name)),
        "outputs": tuple(_serialize_pin(pin) for pin in sorted(node.outputs, key=lambda pin: pin.name)),
    }


def _serialize_edge(edge: Edge) -> dict[str, JsonValue]:
    return {
        "source_node_id": edge.source_node_id,
        "source_pin": edge.source_pin,
        "destination_node_id": edge.destination_node_id,
        "destination_pin": edge.destination_pin,
    }


def _edge_sort_key(edge: Edge) -> tuple[str, str, str, str]:
    return (edge.source_node_id, edge.source_pin, edge.destination_node_id, edge.destination_pin)


def _json_default(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return dict(value) # type: ignore
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def canonicalize(specification: WorkflowSpecification) -> str:
    """Produce the order-independent canonical serialization of a workflow.

    Node ordering in the source document must not determine runtime
    execution order, so canonicalization sorts nodes by ``node_id``,
    edges by their full tuple, and each node's pins by name, then
    serializes to JSON with sorted object keys and no incidental
    whitespace, so that equivalent syntactic representations normalize to
    one canonical string.
    """
    sorted_nodes = sorted(specification.nodes, key=lambda node: node.node_id)
    sorted_edges = sorted(specification.edges, key=_edge_sort_key)
    canonical_document: dict[str, JsonValue] = {
        "nodes": tuple(_serialize_node(node) for node in sorted_nodes),
        "edges": tuple(_serialize_edge(edge) for edge in sorted_edges),
    }
    return json.dumps(
        canonical_document,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def specification_hash(specification: WorkflowSpecification) -> str:
    """Compute ``h_G = H(Canonicalize(G))`` using SHA-256."""
    canonical = canonicalize(specification)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()