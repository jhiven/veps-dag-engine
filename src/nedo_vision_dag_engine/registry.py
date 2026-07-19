"""Processor registry and immutable registry snapshot.

Processor resolution is performed through a registry
``R: type_name -> ProcessorDescriptor``. Compilation must use an
*immutable* snapshot of that registry so that a concurrent processor
registration cannot change the meaning of a candidate plan while it is
being compiled.

This module therefore separates two concerns:

- ``RegistryBuilder``, a plain mutable class used at process start-up (or
  in tests) to accumulate processor-type registrations;
- ``RegistrySnapshot``, produced by freezing a builder, which is the only
  form the compiler is ever given. Its ``snapshot_id`` is a deterministic
  content hash of every registered descriptor, so that the same
  normalized workflow specification, compiler version, and registry
  snapshot always produce the same structural plan representation.

``snapshot_id`` is computed over each ``ProcessorDescriptor`` and
``StatefulProcessorDescriptor``'s declarative fields only. The
``factory`` callable and the ``preserves_state_for`` predicate are
ordinary Python callables with no stable, serializable identity, so they
are deliberately excluded from the hash; they affect processor
*behavior*, not the *structural* shape of a compiled plan, which is what
this hash is meant to capture reproducibly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from nedo_vision_dag_engine.processor import Processor, ProcessorDescriptor, StatefulProcessorDescriptor
from nedo_vision_dag_engine.type_system import StatePolicy

__all__ = [
    "ProcessorFactory",
    "RegisteredProcessorType",
    "RegistrySnapshot",
    "RegistryBuilder",
]

type ProcessorFactory = Callable[[], Processor]


@dataclass(frozen=True, slots=True)
class RegisteredProcessorType:
    """One registry entry: a processor type's descriptor together with the
    factory used to construct new instances of it (construction,
    configuration validation, setup, and so on happen using this factory
    when a new instance is actually needed).
    """

    descriptor: ProcessorDescriptor
    factory: ProcessorFactory
    stateful_descriptor: StatefulProcessorDescriptor | None = None

    def __post_init__(self) -> None:
        is_stateful_policy = self.descriptor.state_policy in (
            StatePolicy.PRESERVABLE,
            StatePolicy.RESETTABLE,
        )
        if is_stateful_policy and self.stateful_descriptor is None:
            raise ValueError(
                f"processor {self.descriptor.type_name!r} declares state policy "
                f"{self.descriptor.state_policy!r} and must supply a stateful_descriptor."
            )
        if not is_stateful_policy and self.stateful_descriptor is not None:
            raise ValueError(
                f"processor {self.descriptor.type_name!r} declares state policy "
                f"{self.descriptor.state_policy!r} and must not supply a stateful_descriptor."
            )
        if (
            self.stateful_descriptor is not None
            and self.stateful_descriptor.state_schema_version != self.descriptor.state_schema_version
        ):
            raise ValueError(
                f"processor {self.descriptor.type_name!r} has mismatched state_schema_version "
                f"between ProcessorDescriptor ({self.descriptor.state_schema_version!r}) and "
                f"StatefulProcessorDescriptor ({self.stateful_descriptor.state_schema_version!r})."
            )


def _type_reference(python_type: type[object]) -> str:
    return f"{python_type.__module__}.{python_type.__qualname__}"


def _serialize_registration(registration: RegisteredProcessorType) -> dict[str, object]:
    descriptor = registration.descriptor
    record: dict[str, object] = {
        "type_name": descriptor.type_name,
        "input_schema": _type_reference(descriptor.input_schema),
        "output_schema": _type_reference(descriptor.output_schema),
        "config_schema": _type_reference(descriptor.config_schema),
        "state_policy": descriptor.state_policy.value,
        "state_schema_version": descriptor.state_schema_version,
        "stateful": None,
    }
    stateful = registration.stateful_descriptor
    if stateful is not None:
        record["stateful"] = {
            "state_schema_version": stateful.state_schema_version,
            "supported_transition_policies": sorted(
                policy.value for policy in stateful.supported_transition_policies
            ),
        }
    return record


def _compute_snapshot_id(entries: Mapping[str, RegisteredProcessorType]) -> str:
    canonical_document = {
        type_name: _serialize_registration(registration)
        for type_name, registration in sorted(entries.items())
    }
    canonical = json.dumps(canonical_document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """An immutable, content-hashed view of the processor registry.
    The compiler receives only this type, never a mutable
    ``RegistryBuilder``.
    """

    snapshot_id: str
    entries: Mapping[str, RegisteredProcessorType]

    def resolve(self, type_name: str) -> RegisteredProcessorType | None:
        return self.entries.get(type_name)

    def __contains__(self, type_name: str) -> bool:
        return type_name in self.entries

    def __getitem__(self, type_name: str) -> RegisteredProcessorType:
        try:
            return self.entries[type_name]
        except KeyError:
            raise KeyError(
                f"processor type {type_name!r} is not present in registry snapshot {self.snapshot_id!r}"
            ) from None

    def __len__(self) -> int:
        return len(self.entries)

    def type_names(self) -> frozenset[str]:
        return frozenset(self.entries.keys())


class RegistryBuilder:
    """Mutable accumulator of processor-type registrations.

    Not used by the compiler directly; call :meth:`snapshot` to obtain the
    immutable :class:`RegistrySnapshot` that compilation actually consumes.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredProcessorType] = {}

    def register(self, registration: RegisteredProcessorType) -> "RegistryBuilder":
        type_name = registration.descriptor.type_name
        if type_name in self._entries:
            raise ValueError(f"processor type {type_name!r} is already registered.")
        self._entries[type_name] = registration
        return self

    def snapshot(self) -> RegistrySnapshot:
        frozen_entries = MappingProxyType(dict(self._entries))
        snapshot_id = _compute_snapshot_id(frozen_entries)
        return RegistrySnapshot(snapshot_id=snapshot_id, entries=frozen_entries)