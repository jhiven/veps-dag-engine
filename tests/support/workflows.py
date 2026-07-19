"""Workflow specification builders for the test suite.

Provides typed factory functions to construct ``WorkflowSpecification``
objects without repeated boilerplate. Each function returns a concrete,
structurally valid specification that can be passed directly to the
compiler or reconfiguration controller.
"""

from __future__ import annotations

from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    WorkflowSpecification,
    to_processor_configuration,
)
from nedo_vision_dag_engine.type_system import ConcreteType
from tests.support.processors import (
    STATELESS_PASS_DESCRIPTOR,
    STATELESS_SOURCE_DESCRIPTOR,
    PassOutput,
)


def _source_node(
    node_id: str = "source",
    type_name: str = STATELESS_SOURCE_DESCRIPTOR.type_name,
) -> Node:
    return Node(
        node_id=node_id,
        type_name=type_name,
        configuration=to_processor_configuration({}),
        inputs=(),
        outputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(PassOutput),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )


def _pass_node(
    node_id: str = "pass",
    type_name: str = STATELESS_PASS_DESCRIPTOR.type_name,
) -> Node:
    return Node(
        node_id=node_id,
        type_name=type_name,
        configuration=to_processor_configuration({}),
        inputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(PassOutput),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
        outputs=(
            Pin(
                name="value",
                payload_type=ConcreteType(PassOutput),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )


def source_only(
    source_id: str = "source",
    source_type: str = STATELESS_SOURCE_DESCRIPTOR.type_name,
) -> WorkflowSpecification:
    """A single-node workflow with no incoming edges."""
    return WorkflowSpecification(
        nodes=(_source_node(source_id, source_type),),
        edges=(),
    )


def linear(
    source_id: str = "source",
    source_type: str = STATELESS_SOURCE_DESCRIPTOR.type_name,
    pass_id: str = "pass",
    pass_type: str = STATELESS_PASS_DESCRIPTOR.type_name,
) -> WorkflowSpecification:
    """source → pass: a two-node linear workflow."""
    return WorkflowSpecification(
        nodes=(_source_node(source_id, source_type), _pass_node(pass_id, pass_type)),
        edges=(
            Edge(
                source_node_id=source_id,
                source_pin="value",
                destination_node_id=pass_id,
                destination_pin="value",
            ),
        ),
    )


def tracker_only(
    tracker_type: str,
    node_id: str = "tracker",
) -> WorkflowSpecification:
    """A single-node workflow whose processor is a stateful tracker."""
    return WorkflowSpecification(
        nodes=(
            Node(
                node_id=node_id,
                type_name=tracker_type,
                configuration=to_processor_configuration({}),
                inputs=(),
                outputs=(
                    Pin(
                        name="tracks",
                        payload_type=ConcreteType(object),
                        cardinality=PinCardinality.SINGLE,
                        requirement=PinRequirement.REQUIRED,
                    ),
                ),
            ),
        ),
        edges=(),
    )


def tracker_then_pass(
    tracker_type: str,
    pass_type: str = STATELESS_PASS_DESCRIPTOR.type_name,
    tracker_id: str = "tracker",
    pass_id: str = "downstream",
) -> WorkflowSpecification:
    """tracker → downstream pass node."""
    tracker_node = Node(
        node_id=tracker_id,
        type_name=tracker_type,
        configuration=to_processor_configuration({}),
        inputs=(),
        outputs=(
            Pin(
                name="tracks",
                payload_type=ConcreteType(object),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )
    pass_node = Node(
        node_id=pass_id,
        type_name=pass_type,
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
    return WorkflowSpecification(
        nodes=(tracker_node, pass_node),
        edges=(
            Edge(
                source_node_id=tracker_id,
                source_pin="tracks",
                destination_node_id=pass_id,
                destination_pin="value",
            ),
        ),
    )
