"""The workflow compiler.

This module turns a parsed, structurally- and type-checked workflow into
a runnable execution plan. It has two halves:

- ordering: turning the node/edge graph into one deterministic execution
  order, and detecting whether that is even possible (i.e. whether the
  graph is acyclic);
- ``WorkflowCompiler``: the orchestration that ties validation, ordering,
  a diff against whatever plan is currently active, and execution-step
  construction into one call that produces either a ready candidate plan
  or a structured failure.

Ordering strategy
------------------
We use Kahn's algorithm: repeatedly pick a node whose dependencies have
all already been placed in the order, place it, and then re-check its
dependents. A node with no remaining unresolved dependency is a
candidate; once a node is placed, its outgoing edges are "spent" and its
dependents move one step closer to becoming candidates themselves.

If every node eventually gets placed, the order is valid and the graph
is acyclic. If the algorithm runs out of candidates while nodes remain
unplaced, those remaining nodes form (or are entangled in) a cycle: each
one is still waiting on a dependency that will never be resolved,
because that dependency is itself waiting on something downstream.

Two graphs that only differ in how their nodes or edges happen to be
listed should compile to the exact same order. Plain Kahn's algorithm
does not guarantee this on its own, because whenever more than one node
is simultaneously ready, the choice of which to place first is
arbitrary. We remove that arbitrariness by always placing the
lexicographically smallest ready node id, using a min-heap as the
"ready" set instead of an unordered queue or stack.

Diffing against a previous plan
--------------------------------
``compile`` only receives a previous *plan*, not the specification that
produced it -- a compiled plan intentionally does not carry the original
node configurations or type names, only the resolved steps. To classify
"what changed" per node, we still need that original specification. The
``WorkflowCompiler`` instance therefore keeps a small private cache
mapping plan version to the specification that produced it, populated
every time a compilation succeeds. ``forget_version`` lets a caller evict
an entry once a plan has been fully retired, so this cache does not grow
without bound over a long-running process.

Classifying each node
----------------------
For every node id present in the new specification, we compare it
against the same node id in the cached previous specification (if any)
and decide one of:

- ``ADDED`` -- the node id did not exist before; a new processor instance
  must be constructed, set up, and health-checked from scratch.
- ``REMOVED`` -- the node id existed before but is gone now; its old
  processor instance is scheduled for later retirement, but is not
  touched here (it may still be serving the currently active plan).
- ``UNCHANGED`` / ``RECONFIGURED`` -- the processor type is the same, and
  either the underlying processor has no state to worry about and its
  configuration did not change (``UNCHANGED``), or it does have state and
  it declares this specific transition compatible even though its
  configuration changed (``RECONFIGURED``). Either way the exact same
  live processor instance is carried over into the new plan, untouched:
  it is not reconstructed, re-configured, or health-checked again.
- ``REPLACED`` -- a new processor instance is required: either the
  processor type itself changed, or a stateless processor's configuration
  changed (a stateless processor has no "preserve" concept to fall back
  on -- new parameters require a fresh instance), or a stateful node's
  reconfiguration was given an explicit reset directive.

A stateful processor whose type changed, or whose declared compatibility
predicate rejects this specific transition, and for which no explicit
reset was requested, fails the whole candidate rather than silently
discarding accumulated state or silently constructing a replacement: the
caller asked for continuity and we cannot promise it, so we say so
instead of guessing.

Two structural signals ("configuration changed", "incoming edges
changed") are computed independently of the ADDED/REMOVED/UNCHANGED/
RECONFIGURED/REPLACED classification and stored alongside it, since a
node can be, say, unchanged in every other respect yet rewired to a
different upstream producer; that is orthogonal to whether its processor
instance itself can be reused.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from nedo_vision_dag_engine.lifecycle import ProcessorStagingArea
from nedo_vision_dag_engine.plan import (
    ActivationCondition,
    ExecutionPlan,
    ExecutionStep,
    InputBinding,
    ReadinessRule,
    WorkspaceView,
)
from nedo_vision_dag_engine.processor import SetupContext, StatefulProcessor, TransitionContext
from nedo_vision_dag_engine.registry import RegistrySnapshot
from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    PinCardinality,
    PinRequirement,
    WorkflowSpecification,
)
from nedo_vision_dag_engine.specification import specification_hash as _compute_specification_hash
from nedo_vision_dag_engine.type_system import MISSING, StatePolicy, StateTransitionPolicy
from nedo_vision_dag_engine.validation import ValidationError, validate_workflow

__all__ = [
    "CycleDetected",
    "topological_order",
    "NodeChangeKind",
    "NodeClassification",
    "StateDirective",
    "PendingStateTransition",
    "CompiledCandidate",
    "CompilationFailureKind",
    "CompilationFailure",
    "CandidatePlan",
    "ValidatedWorkflow",
    "WorkflowCompiler",
]


@dataclass(frozen=True, slots=True)
class ValidatedWorkflow:
    """A workflow specification that has passed structural and type validation,
    along with its pre-computed topological execution order.
    """

    specification: WorkflowSpecification
    topological_order: tuple[str, ...]
    registry: RegistrySnapshot


class CycleDetected(Exception):
    """Raised when the node/edge graph cannot be fully ordered because it
    contains a cycle.

    ``remaining_node_ids`` holds every node that Kahn's algorithm could
    not place; this is not necessarily the minimal cycle itself, but it
    is exactly the set of nodes whose dependencies never fully resolved,
    which is enough to point a caller at the offending part of the graph.
    """

    def __init__(self, remaining_node_ids: frozenset[str]) -> None:
        self.remaining_node_ids = remaining_node_ids
        super().__init__(
            f"workflow graph contains a cycle; the following node(s) could never be "
            f"scheduled because their dependencies never fully resolved: "
            f"{sorted(remaining_node_ids)!r}"
        )


def topological_order(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> tuple[str, ...]:
    """Compute a deterministic topological order of node ids.

    This function assumes the graph has already passed structural
    validation: node ids are unique, and every edge's source and
    destination node actually exists. An edge referencing a node id not
    present in ``nodes`` is silently ignored here (it will already have
    been reported elsewhere as a structural problem); this function's
    only job is ordering and cycle detection.
    """
    node_ids = tuple(node.node_id for node in nodes)
    known_node_ids = frozenset(node_ids)

    in_degree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        if edge.source_node_id not in known_node_ids or edge.destination_node_id not in known_node_ids:
            continue
        in_degree[edge.destination_node_id] += 1
        dependents[edge.source_node_id].append(edge.destination_node_id)

    ready: list[str] = [node_id for node_id, degree in in_degree.items() if degree == 0]
    heapq.heapify(ready)

    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for dependent in sorted(dependents[node_id]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(order) != len(node_ids):
        unresolved = frozenset(node_ids) - frozenset(order)
        raise CycleDetected(unresolved)

    return tuple(order)


class NodeChangeKind(Enum):
    """How one node id's processor instance is handled when compiling a
    new plan against a previous one."""

    UNCHANGED = "unchanged"
    RECONFIGURED = "reconfigured"
    ADDED = "added"
    REMOVED = "removed"
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class NodeClassification:
    """The full classification of one node id between two plan versions."""

    node_id: str
    kind: NodeChangeKind
    configuration_changed: bool
    edges_changed: bool


@dataclass(frozen=True, slots=True)
class StateDirective:
    """Per-request instructions about which stateful nodes should have
    their state explicitly reset rather than preserved, even if
    preservation would otherwise have been possible."""

    reset_node_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PendingStateTransition:
    """A state reset that happened during candidate preparation.

    This is deliberately lighter than a full audit event: at compile time
    we do not yet know whether this candidate will ever be committed, so
    we do not know the commit timestamp or whether "old plan version" and
    "new plan version" will end up meaning anything (a candidate that is
    later rejected as stale never becomes anyone's old or new version).
    Whoever actually commits the candidate is responsible for turning
    this into a full, timestamped record at that point.
    """

    node_id: str
    policy: StateTransitionPolicy
    reason: str


@dataclass(frozen=True, slots=True)
class CompiledCandidate:
    """A successfully compiled, fully staged candidate plan."""

    plan: ExecutionPlan
    classifications: Mapping[str, NodeClassification]
    pending_state_transitions: tuple[PendingStateTransition, ...]
    reused_node_ids: frozenset[str]
    staged_node_ids: frozenset[str]
    retired_node_ids: frozenset[str]


class CompilationFailureKind(Enum):
    """Distinguishes a failure caused by the submitted workflow itself
    (a bad graph, an incompatible reconfiguration request) from one
    caused by something going wrong in the environment while trying to
    honor an otherwise legitimate request (a processor that raised while
    being staged, an internal bookkeeping inconsistency). The
    reconfiguration controller maps these to different terminal request
    states.
    """

    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CompilationFailure:
    """Compilation could not produce a candidate.

    ``errors`` holds every structural/type validation problem found, if
    the failure came from that phase; it is empty for failures detected
    later (cycle detection, an unreconcilable stateful transition, or a
    processor that raised while being staged), which instead only carry a
    human-readable ``reason``.
    """

    errors: tuple[ValidationError, ...]
    reason: str
    kind: CompilationFailureKind = CompilationFailureKind.REJECTED


type CandidatePlan = CompiledCandidate | CompilationFailure


def _incoming_edge_signature(edges: tuple[Edge, ...], node_id: str) -> frozenset[tuple[str, str, str]]:
    return frozenset(
        (edge.source_node_id, edge.source_pin, edge.destination_pin)
        for edge in edges
        if edge.destination_node_id == node_id
    )


def _classify_nodes(
    specification: WorkflowSpecification,
    previous_specification: WorkflowSpecification | None,
    previous_plan: ExecutionPlan | None,
    registry: RegistrySnapshot,
    directive: StateDirective,
) -> dict[str, NodeClassification] | CompilationFailure:
    new_nodes_by_id = {node.node_id: node for node in specification.nodes}
    previous_nodes_by_id: dict[str, Node] = {}
    if previous_specification is not None:
        previous_nodes_by_id = {node.node_id: node for node in previous_specification.nodes}

    classifications: dict[str, NodeClassification] = {}

    for node_id, new_node in new_nodes_by_id.items():
        if previous_plan is None or node_id not in previous_nodes_by_id:
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=NodeChangeKind.ADDED,
                configuration_changed=True,
                edges_changed=True,
            )
            continue

        previous_node = previous_nodes_by_id[node_id]
        previous_step = previous_plan.step_by_node_id(node_id)
        if previous_step is None:
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=NodeChangeKind.ADDED,
                configuration_changed=True,
                edges_changed=True,
            )
            continue

        previous_descriptor = previous_step.processor_ref.descriptor

        new_registration = registry.resolve(new_node.type_name)
        if new_registration is None:
            return CompilationFailure(
                errors=(),
                reason=f"node {node_id!r} references processor type {new_node.type_name!r}, "
                "which is not present in the given registry snapshot.",
            )
        new_descriptor = new_registration.descriptor

        type_changed = previous_descriptor.type_name != new_descriptor.type_name
        configuration_changed = previous_node.configuration != new_node.configuration
        assert previous_specification is not None
        edges_changed = _incoming_edge_signature(
            previous_specification.edges, node_id
        ) != _incoming_edge_signature(specification.edges, node_id)
        explicit_reset_requested = node_id in directive.reset_node_ids

        is_previous_stateful = previous_descriptor.state_policy in (
            StatePolicy.PRESERVABLE,
            StatePolicy.RESETTABLE,
        )

        if type_changed:
            if is_previous_stateful and not explicit_reset_requested:
                return CompilationFailure(
                    errors=(),
                    reason=(
                        f"node {node_id!r} changes processor type from "
                        f"{previous_descriptor.type_name!r} to {new_descriptor.type_name!r} while its "
                        "current processor holds preservable/resettable state, and no explicit reset "
                        "was requested for this node; this transition cannot be reconciled implicitly."
                    ),
                )
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=NodeChangeKind.REPLACED,
                configuration_changed=configuration_changed,
                edges_changed=edges_changed,
            )
            continue

        if previous_descriptor.state_policy != new_descriptor.state_policy:
            return CompilationFailure(
                errors=(),
                reason=(
                    f"node {node_id!r} keeps processor type {new_descriptor.type_name!r} but that type "
                    f"now declares state policy {new_descriptor.state_policy!r} instead of "
                    f"{previous_descriptor.state_policy!r}; the registry is inconsistent across "
                    "compilations for a type name that is supposed to be stable."
                ),
                kind=CompilationFailureKind.FAILED,
            )

        is_new_stateful = new_descriptor.state_policy in (StatePolicy.PRESERVABLE, StatePolicy.RESETTABLE)

        if not is_new_stateful:
            kind = NodeChangeKind.REPLACED if configuration_changed else NodeChangeKind.UNCHANGED
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=kind,
                configuration_changed=configuration_changed,
                edges_changed=edges_changed,
            )
            continue

        stateful_descriptor = new_registration.stateful_descriptor
        assert stateful_descriptor is not None

        if explicit_reset_requested:
            if StateTransitionPolicy.RESET not in stateful_descriptor.supported_transition_policies:
                return CompilationFailure(
                    errors=(),
                    reason=f"node {node_id!r} was asked to reset, but its processor type does not "
                    "support the RESET transition policy.",
                )
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=NodeChangeKind.REPLACED,
                configuration_changed=configuration_changed,
                edges_changed=edges_changed,
            )
            continue

        if StateTransitionPolicy.PRESERVE not in stateful_descriptor.supported_transition_policies:
            return CompilationFailure(
                errors=(),
                reason=f"node {node_id!r} would need state preservation but its processor type does "
                "not support the PRESERVE transition policy, and no explicit reset was requested.",
            )

        assert previous_descriptor.state_schema_version is not None
        assert new_descriptor.state_schema_version is not None

        transition_context = TransitionContext(
            node_id=node_id,
            previous_processor_type=previous_descriptor.type_name,
            new_processor_type=new_descriptor.type_name,
            previous_state_schema_version=previous_descriptor.state_schema_version,
            new_state_schema_version=new_descriptor.state_schema_version,
            previous_input_payload_types={pin.name: pin.payload_type for pin in previous_node.inputs},
            new_input_payload_types={pin.name: pin.payload_type for pin in new_node.inputs},
            upstream_semantic_coordinate_system_unchanged=not edges_changed,
            label_taxonomy_compatible=not edges_changed,
            source_identity_unchanged=not edges_changed,
            explicit_reset_requested=False,
        )

        is_compatible = stateful_descriptor.preserves_state_for(
            previous_node.configuration, new_node.configuration, transition_context
        )
        if not is_compatible:
            return CompilationFailure(
                errors=(),
                reason=f"node {node_id!r} declined to preserve its state for this reconfiguration and "
                "no explicit reset was requested; the transition is rejected.",
            )

        kind = NodeChangeKind.RECONFIGURED if configuration_changed else NodeChangeKind.UNCHANGED
        classifications[node_id] = NodeClassification(
            node_id=node_id,
            kind=kind,
            configuration_changed=configuration_changed,
            edges_changed=edges_changed,
        )

    for node_id in previous_nodes_by_id:
        if node_id not in new_nodes_by_id:
            classifications[node_id] = NodeClassification(
                node_id=node_id,
                kind=NodeChangeKind.REMOVED,
                configuration_changed=True,
                edges_changed=True,
            )

    return classifications


def _at_least_one_present(indices: tuple[int, ...]) -> ActivationCondition:
    def check(workspace: WorkspaceView) -> bool:
        return any(workspace[index] is not MISSING for index in indices)

    return check


def _combine_conditions(conditions: tuple[ActivationCondition, ...]) -> ActivationCondition | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]

    def combined(workspace: WorkspaceView) -> bool:
        return all(condition(workspace) for condition in conditions)

    return combined


def _build_plan(
    specification: WorkflowSpecification,
    order: tuple[str, ...],
    registry: RegistrySnapshot,
    previous_plan: ExecutionPlan | None,
    classifications: Mapping[str, NodeClassification],
    compiler_version: str,
    new_version: int,
) -> (
    tuple[ExecutionPlan, tuple[PendingStateTransition, ...], frozenset[str], frozenset[str], frozenset[str]]
    | CompilationFailure
):
    nodes_by_id = {node.node_id: node for node in specification.nodes}
    position_by_node_id = {node_id: position for position, node_id in enumerate(order)}

    incoming_by_destination: dict[tuple[str, str], list[Edge]] = {}
    for edge in specification.edges:
        incoming_by_destination.setdefault((edge.destination_node_id, edge.destination_pin), []).append(edge)

    staging = ProcessorStagingArea()
    reused_node_ids: set[str] = set()
    pending_transitions: list[PendingStateTransition] = []
    steps: list[ExecutionStep] = []

    def fail(
        reason: str,
        kind: CompilationFailureKind = CompilationFailureKind.FAILED,
    ) -> CompilationFailure:
        cleanup_report = staging.rollback()
        if cleanup_report.failures:
            cleanup_details = "; ".join(
                f"{failure.node_id}: {failure.error}" for failure in cleanup_report.failures
            )
            reason = f"{reason} Staging rollback also reported cleanup failure(s): {cleanup_details}."
        return CompilationFailure(errors=(), reason=reason, kind=kind)

    try:
        for node_id in order:
            node = nodes_by_id[node_id]
            classification = classifications[node_id]
            registration = registry.resolve(node.type_name)
            if registration is None:
                return fail(
                    f"node {node_id!r} references unregistered processor type {node.type_name!r}.",
                    CompilationFailureKind.REJECTED,
                )

            if classification.kind in (NodeChangeKind.UNCHANGED, NodeChangeKind.RECONFIGURED):
                if previous_plan is None:
                    return fail(
                        f"node {node_id!r} was classified as reusable without a previous execution plan."
                    )
                previous_step = previous_plan.step_by_node_id(node_id)
                if previous_step is None:
                    return fail(
                        f"node {node_id!r} was classified as reusable but is absent from previous "
                        f"plan version {previous_plan.version}."
                    )
                processor_ref = previous_step.processor_ref
                reused_node_ids.add(node_id)
            else:
                try:
                    processor_ref = registration.factory()
                except Exception as error:
                    return fail(
                        f"constructing processor for node {node_id!r} "
                        f"(type {node.type_name!r}) failed: {error!r}"
                    )

                try:
                    staging.register(node_id, processor_ref)
                except Exception as error:
                    return fail(
                        f"registering staged processor for node {node_id!r} "
                        f"(type {node.type_name!r}) failed: {error!r}"
                    )

                if processor_ref.descriptor != registration.descriptor:
                    return fail(
                        f"processor factory for type {node.type_name!r} returned an instance whose "
                        "descriptor does not match the immutable registry snapshot."
                    )

                registered_stateful_descriptor = registration.stateful_descriptor
                if registered_stateful_descriptor is not None:
                    if not isinstance(processor_ref, StatefulProcessor):
                        return fail(
                            f"processor factory for stateful type {node.type_name!r} returned an "
                            "instance that does not implement StatefulProcessor."
                        )
                    if processor_ref.stateful_descriptor != registered_stateful_descriptor:
                        return fail(
                            f"processor factory for stateful type {node.type_name!r} returned an "
                            "instance whose stateful descriptor does not match the registry snapshot."
                        )

                try:
                    processor_ref.setup(
                        SetupContext(
                            node_id=node_id,
                            configuration=node.configuration,
                            registry_snapshot_id=registry.snapshot_id,
                            candidate_plan_version=new_version,
                            warm_up_requested=True,
                        )
                    )
                    processor_ref.healthcheck()
                except Exception as error:
                    return fail(
                        f"preparing processor for node {node_id!r} "
                        f"(type {node.type_name!r}) failed: {error!r}"
                    )

                if classification.kind is NodeChangeKind.REPLACED and previous_plan is not None:
                    previous_step_for_node = previous_plan.step_by_node_id(node_id)
                    if previous_step_for_node is not None:
                        previous_state_policy = previous_step_for_node.processor_ref.descriptor.state_policy
                        if previous_state_policy in (StatePolicy.PRESERVABLE, StatePolicy.RESETTABLE):
                            pending_transitions.append(
                                PendingStateTransition(
                                    node_id=node_id,
                                    policy=StateTransitionPolicy.RESET,
                                    reason=(
                                        "processor type changed, configuration required a fresh "
                                        "instance, or an explicit reset was requested"
                                    ),
                                )
                            )

            input_bindings: list[InputBinding] = []
            required_single_indices: list[int] = []
            multiple_required_groups: list[tuple[int, ...]] = []

            for pin in node.inputs:
                edges_for_pin = incoming_by_destination.get((node_id, pin.name), [])
                pin_source_indices: list[int] = []
                for edge in edges_for_pin:
                    source_index = position_by_node_id[edge.source_node_id]
                    input_bindings.append(
                        InputBinding(
                            destination_pin=pin.name,
                            source_index=source_index,
                            source_pin=edge.source_pin,
                            required=pin.requirement is PinRequirement.REQUIRED,
                        )
                    )
                    pin_source_indices.append(source_index)

                if pin.requirement is PinRequirement.REQUIRED:
                    if pin.cardinality is PinCardinality.SINGLE:
                        required_single_indices.extend(pin_source_indices)
                    else:
                        multiple_required_groups.append(tuple(pin_source_indices))

            activation_condition = _combine_conditions(
                tuple(_at_least_one_present(group) for group in multiple_required_groups)
            )
            readiness_rule = ReadinessRule(
                required_source_indices=tuple(required_single_indices),
                activation_condition=activation_condition,
            )

            steps.append(
                ExecutionStep(
                    node_id=node_id,
                    output_index=position_by_node_id[node_id],
                    processor_ref=processor_ref,
                    input_bindings=tuple(input_bindings),
                    readiness_rule=readiness_rule,
                    output_schema=registration.descriptor.output_schema,
                )
            )

        retired_node_ids = frozenset(
            node_id
            for node_id, classification in classifications.items()
            if classification.kind is NodeChangeKind.REMOVED
        )

        plan = ExecutionPlan(
            version=new_version,
            specification_hash=_compute_specification_hash(specification),
            registry_snapshot_id=registry.snapshot_id,
            compiler_version=compiler_version,
            steps=tuple(steps),
            output_slot_count=len(steps),
            processor_ids=frozenset(order),
        )
    except Exception as error:
        return fail(f"building execution plan version {new_version} failed: {error!r}")

    staged_node_ids = staging.release()
    return (
        plan,
        tuple(pending_transitions),
        frozenset(reused_node_ids),
        staged_node_ids,
        retired_node_ids,
    )


class WorkflowCompiler:
    """Turns a workflow specification into a candidate plan.

    An instance keeps a small private cache of every specification that
    produced a plan version currently in use somewhere (the active plan,
    or a candidate still being evaluated), so that a later call can diff
    a new specification against whichever plan it says it is replacing.
    Call ``forget_version`` once a plan version is fully retired so the
    cache does not grow without bound.
    """

    def __init__(self, compiler_version: str) -> None:
        self._compiler_version = compiler_version
        self._specifications_by_version: dict[int, WorkflowSpecification] = {}

    def forget_version(self, version: int) -> None:
        self._specifications_by_version.pop(version, None)

    def validate(
        self,
        specification: WorkflowSpecification,
        registry: RegistrySnapshot,
    ) -> ValidatedWorkflow | CompilationFailure:
        combined_validation, _resolver = validate_workflow(specification, registry)
        if not combined_validation.is_valid:
            return CompilationFailure(
                errors=combined_validation.errors, reason="structural or type validation failed"
            )

        try:
            order = topological_order(specification.nodes, specification.edges)
        except CycleDetected as error:
            return CompilationFailure(errors=(), reason=str(error))

        return ValidatedWorkflow(
            specification=specification,
            topological_order=order,
            registry=registry,
        )

    def compile_validated(
        self,
        validated: ValidatedWorkflow,
        previous_plan: ExecutionPlan | None = None,
        state_directive: StateDirective | None = None,
    ) -> CandidatePlan:
        directive = state_directive if state_directive is not None else StateDirective()
        specification = validated.specification
        registry = validated.registry
        order = validated.topological_order

        previous_specification: WorkflowSpecification | None = None
        if previous_plan is not None:
            previous_specification = self._specifications_by_version.get(previous_plan.version)
            if previous_specification is None:
                return CompilationFailure(
                    errors=(),
                    reason=(
                        f"no cached specification is available for previous plan version "
                        f"{previous_plan.version}; cannot compute a differential candidate against it."
                    ),
                    kind=CompilationFailureKind.FAILED,
                )

        classification_result = _classify_nodes(specification, previous_specification, previous_plan, registry, directive)
        if isinstance(classification_result, CompilationFailure):
            return classification_result
        classifications = classification_result

        new_version = previous_plan.version + 1 if previous_plan is not None else 1
        build_result = _build_plan(
            specification=specification,
            order=order,
            registry=registry,
            previous_plan=previous_plan,
            classifications=classifications,
            compiler_version=self._compiler_version,
            new_version=new_version,
        )
        if isinstance(build_result, CompilationFailure):
            return build_result
        plan, pending_transitions, reused_node_ids, staged_node_ids, retired_node_ids = build_result

        self._specifications_by_version[plan.version] = specification

        return CompiledCandidate(
            plan=plan,
            classifications=classifications,
            pending_state_transitions=pending_transitions,
            reused_node_ids=reused_node_ids,
            staged_node_ids=staged_node_ids,
            retired_node_ids=retired_node_ids,
        )

    def compile(
        self,
        specification: WorkflowSpecification,
        registry: RegistrySnapshot,
        previous_plan: ExecutionPlan | None = None,
        state_directive: StateDirective | None = None,
    ) -> CandidatePlan:
        validated_result = self.validate(specification, registry)
        if isinstance(validated_result, CompilationFailure):
            return validated_result

        return self.compile_validated(
            validated=validated_result,
            previous_plan=previous_plan,
            state_directive=state_directive,
        )