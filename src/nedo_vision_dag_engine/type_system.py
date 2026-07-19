"""Core type-system primitives shared across the typed DAG runtime.

This module defines the vocabulary every other module in this package
depends on:

- ``MISSING``, the sentinel used instead of ``None`` to represent an
  unavailable workspace slot value. ``None`` is a valid application-level
  payload and therefore cannot double as "no value".
- the payload type-assignability relation ``tau(p_s) <= tau(p_d)``,
  including resolution of generic pin types.
- the ``StatePolicy`` and ``StateTransitionPolicy`` enumerations shared by
  processor descriptors and the reconfiguration controller.

These primitives must not be redefined elsewhere; every other module
imports them from here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final, final

__all__ = [
    "MISSING",
    "MissingType",
    "PayloadType",
    "ConcreteType",
    "GenericParameter",
    "GenericResolutionError",
    "GenericTypeResolver",
    "assignable",
    "StatePolicy",
    "StateTransitionPolicy",
]


@final
class MissingType:
    """Singleton sentinel type. Use the module-level ``MISSING`` instance.

    Direct instantiation elsewhere is prevented in practice by always
    importing the shared instance; equality and identity both hold since
    only one instance is ever constructed.
    """

    _instance: "MissingType | None" = None

    def __new__(cls) -> "MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self) -> tuple[type["MissingType"], tuple[()]]:
        return (MissingType, ())


MISSING: Final = MissingType()


class PayloadType:
    """Base of the sealed payload-type hierarchy used to declare pins.

    The only variants are :class:`ConcreteType` and :class:`GenericParameter`.
    :func:`assignable` pattern-matches on exactly these two variants, so
    this hierarchy is closed by convention rather than by a runtime guard.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ConcreteType(PayloadType):
    """A payload type bound to one concrete Python type."""

    python_type: type[object]


@dataclass(frozen=True, slots=True)
class GenericParameter(PayloadType):
    """An unresolved generic pin type, scoped to one resolution group.

    ``group`` identifies the set of pins whose generic parameter must
    resolve to the same concrete type — for example, all pins sharing one
    type variable within a single node's declared schema. ``name`` is a
    human-readable label only; ``group`` is what the resolver keys on.
    """

    name: str
    group: str


class GenericResolutionError(Exception):
    """Raised when a generic parameter group resolves to inconsistent types.

    A generic parameter group that gets bound to two different concrete
    types across the edges that use it is a deterministic compilation
    error, not something to silently paper over by picking one of them.
    """

    def __init__(self, group: str, first: type[object], conflicting: type[object]) -> None:
        self.group = group
        self.first = first
        self.conflicting = conflicting
        super().__init__(
            f"generic parameter group {group!r} resolved to inconsistent "
            f"concrete types: {first!r} and {conflicting!r}"
        )


@dataclass(slots=True)
class GenericTypeResolver:
    """Accumulates concrete-type bindings for generic parameter groups.

    One resolver instance is scoped to a single compilation attempt. Each
    edge that connects a concrete source pin to a generic destination pin
    contributes one binding via :func:`assignable`. Bindings within a
    group must be exactly equal; the resolver never silently treats an
    unresolved group as accepting an arbitrary type.
    """

    _bindings: dict[str, type[object]] = field(default_factory=lambda: {})

    def bind(self, group: str, concrete: type[object]) -> None:
        existing = self._bindings.get(group)
        if existing is None:
            self._bindings[group] = concrete
            return
        if existing is not concrete:
            raise GenericResolutionError(group, existing, concrete)

    def resolve(self, group: str) -> type[object] | None:
        return self._bindings.get(group)

    def resolved_groups(self) -> dict[str, type[object]]:
        return dict(self._bindings)


def assignable(
    source: PayloadType,
    destination: PayloadType,
    resolver: GenericTypeResolver | None = None,
) -> bool:
    """Evaluate ``tau(p_s) <= tau(p_d)``, the pin type-assignability relation.

    Rules, by (source, destination) variant:

    - concrete -> concrete: accepted only if the source type is the
      destination type or a subtype of it. A consumer declaring a base
      class may accept a more specific producer type.
    - concrete -> generic: requires a resolver. The concrete type is
      recorded as a binding for the destination's generic group; a
      :class:`GenericResolutionError` is raised immediately if that group
      is already bound to a different concrete type. Without a resolver
      the edge is rejected outright, since an unresolved generic
      destination must never be silently treated as accepting any value.
    - generic -> concrete: always rejected. A generic source pin cannot be
      statically known to satisfy a concrete consumer.
    - generic -> generic: accepted only when both sides name the same
      resolution group.
    """
    match source, destination:
        case ConcreteType(python_type=source_type), ConcreteType(python_type=destination_type):
            return issubclass(source_type, destination_type)
        case ConcreteType(python_type=source_type), GenericParameter(group=group):
            if resolver is None:
                return False
            resolver.bind(group, source_type)
            return True
        case GenericParameter(group=source_group), GenericParameter(group=destination_group):
            return source_group == destination_group
        case GenericParameter(), ConcreteType():
            return False
        case _:
            return False


class StatePolicy(enum.Enum):
    """Declares how a processor's cross-frame state may be treated across
    plan versions.

    ``MIGRATABLE`` is reserved for future general state-transformation
    support and is not implemented by the initial runtime; any component
    that encounters it on a live processor must reject the transition
    rather than attempt a transformation.
    """

    STATELESS = "stateless"
    PRESERVABLE = "preservable"
    RESETTABLE = "resettable"
    MIGRATABLE = "migratable"


class StateTransitionPolicy(enum.Enum):
    """The transition policy requested or applied for one stateful node
    across a single reconfiguration.

    ``MIGRATE`` is reserved for a future implementation that would support
    general state transformation across incompatible schema versions; it
    is out of scope for now and must be treated as unsupported by the
    reconfiguration controller.
    """

    PRESERVE = "preserve"
    RESET = "reset"
    REJECT = "reject"
    MIGRATE = "migrate"