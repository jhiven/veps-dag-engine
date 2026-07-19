"""Deterministic synthetic tracker for stateful conformance testing.

This module implements a fully synthetic object tracker that maintains
clearly observable cross-frame state without depending on any external
libraries, neural-network models, or image data.

``SyntheticTracker`` satisfies both ``Processor`` and ``StatefulProcessor``
protocols and declares a preservation predicate that makes an explicit,
non-trivial distinction between compatible and incompatible configuration
transitions.

State model
-----------
The tracker receives a list of ``SyntheticDetection`` values each frame
and produces a list of ``SyntheticTrack`` values. Detection identity is
based on ``object_key`` (a string label). The tracker:

- assigns a new, monotonically increasing ``track_id`` to each object key
  the first time it appears;
- reuses the same ``track_id`` for the same key on subsequent frames;
- appends each detection's ``(x, y)`` to the key's coordinate history.

Because identity is purely key-based and the algorithm has no temporal
prediction step, the tracker is completely deterministic: the same
sequence of inputs always produces the same output and the same internal
state.

Preservation compatibility
--------------------------
Two tracker configurations are state-compatible when:

- the tracker algorithm identifier is unchanged;
- the source identifier is unchanged;
- image width and height are unchanged;
- the coordinate convention is unchanged;
- the label taxonomy version is unchanged;
- the matching threshold is unchanged (or the predicate is explicitly
  asked to ignore it by the caller).

Changing any of the above invalidates the accumulated track state and
must be rejected unless an explicit RESET is requested.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
    StatefulProcessorDescriptor,
    TransitionContext,
)
from nedo_vision_dag_engine.specification import ProcessorConfiguration
from nedo_vision_dag_engine.type_system import StatePolicy, StateTransitionPolicy

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticDetection:
    """One detected object in an input frame."""

    object_key: str
    class_id: int
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class SyntheticTrack:
    """One maintained track with its full coordinate history."""

    track_id: int
    object_key: str
    history: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TrackerInput:
    """Input schema for ``SyntheticTracker``."""

    detections: tuple[SyntheticDetection, ...]


@dataclass(frozen=True, slots=True)
class TrackerOutput:
    """Output schema for ``SyntheticTracker``."""

    tracks: tuple[SyntheticTrack, ...]
    frame_count: int


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Typed configuration for ``SyntheticTracker``.

    Every field is state-compatibility-relevant. Changing any field while
    a tracker holds live state must either be declared compatible by the
    predicate (none are, in the default implementation) or the caller must
    provide an explicit RESET directive.
    """

    algorithm_id: str = "synthetic-v1"
    source_id: str = "default"
    image_width: int = 1920
    image_height: int = 1080
    coordinate_convention: str = "top_left_origin"
    label_taxonomy_version: str = "coco-v1"
    matching_threshold: float = 0.5


def _config_to_tracker_config(raw: ProcessorConfiguration) -> TrackerConfig:
    """Extract a ``TrackerConfig`` from a ``ProcessorConfiguration`` mapping."""
    return TrackerConfig(
        algorithm_id=str(raw.get("algorithm_id", "synthetic-v1")),
        source_id=str(raw.get("source_id", "default")),
        image_width=int(raw.get("image_width", 1920)),  # type: ignore[arg-type]
        image_height=int(raw.get("image_height", 1080)),  # type: ignore[arg-type]
        coordinate_convention=str(raw.get("coordinate_convention", "top_left_origin")),
        label_taxonomy_version=str(raw.get("label_taxonomy_version", "coco-v1")),
        matching_threshold=float(raw.get("matching_threshold", 0.5)),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Preservation predicate
# ---------------------------------------------------------------------------


def _preserves_tracker_state(
    previous_configuration: ProcessorConfiguration,
    new_configuration: ProcessorConfiguration,
    context: TransitionContext,
) -> bool:
    """Return True only if neither the configuration nor structural context changed.

    Every field of ``TrackerConfig`` is state-relevant. Any field change
    invalidates accumulated track state, so this predicate rejects all
    configuration changes while still allowing a no-op reconfiguration
    (same spec, different downstream graph) to preserve state when the
    structural conditions also hold.
    """
    if not context.structurally_preservable:
        return False
    if previous_configuration == new_configuration:
        return True
    prev = _config_to_tracker_config(previous_configuration)
    next_ = _config_to_tracker_config(new_configuration)
    return prev == next_


# ---------------------------------------------------------------------------
# Descriptor constants
# ---------------------------------------------------------------------------

TRACKER_STATE_SCHEMA_VERSION: Final = "synthetic-tracker-v1"

SYNTHETIC_TRACKER_DESCRIPTOR: Final = ProcessorDescriptor(
    type_name="synthetic_tracker",
    input_schema=object,
    output_schema=TrackerOutput,
    config_schema=object,
    state_policy=StatePolicy.PRESERVABLE,
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
)

SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR: Final = StatefulProcessorDescriptor(
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
    supported_transition_policies=frozenset(
        {StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}
    ),
    preserves_state_for=_preserves_tracker_state,
)

# A second tracker type used to test type-change rejection.
SYNTHETIC_TRACKER_V2_DESCRIPTOR: Final = ProcessorDescriptor(
    type_name="synthetic_tracker_v2",
    input_schema=object,
    output_schema=TrackerOutput,
    config_schema=object,
    state_policy=StatePolicy.PRESERVABLE,
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
)

SYNTHETIC_TRACKER_V2_STATEFUL_DESCRIPTOR: Final = StatefulProcessorDescriptor(
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
    supported_transition_policies=frozenset(
        {StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}
    ),
    preserves_state_for=_preserves_tracker_state,
)

# ---------------------------------------------------------------------------
# Instance tracking for leak detection
# ---------------------------------------------------------------------------


class _InstanceRegistry:
    """Tracks how many tracker instances are currently alive.

    Tests inspect ``live_count()`` and ``ever_created()`` to detect leaks
    or premature cleanup without depending on process memory measurements.
    """

    __slots__ = ("_lock", "_all_ids", "_live_ids")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live_ids: set[int] = set()
        self._all_ids: set[int] = set()

    def register(self, instance_id: int) -> None:
        with self._lock:
            self._live_ids.add(instance_id)
            self._all_ids.add(instance_id)

    def unregister(self, instance_id: int) -> None:
        with self._lock:
            self._live_ids.discard(instance_id)

    def live_count(self) -> int:
        with self._lock:
            return len(self._live_ids)

    def ever_created(self) -> int:
        with self._lock:
            return len(self._all_ids)

    def live_ids(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._live_ids)


GLOBAL_TRACKER_REGISTRY: Final = _InstanceRegistry()


# ---------------------------------------------------------------------------
# Synthetic tracker processor
# ---------------------------------------------------------------------------


class SyntheticTracker:
    """A deterministic, self-contained object tracker.

    Mutable state is kept private and is never exposed directly. Tests can
    observe state by examining the ``TrackerOutput`` produced each frame,
    or by calling ``snapshot_state()`` which returns an immutable view of
    the internal state at that moment.

    Thread safety: mutable state is protected by ``_state_lock``. The lock
    is held only during state reads and writes, not during process() as a
    whole, so the concurrency test can detect whether two threads enter
    process() simultaneously (via ``max_concurrent_invocations``).
    """

    __slots__ = (
        "_cleanup_count",
        "_config",
        "_entry_count",
        "_entry_lock",
        "_frame_count",
        "_key_to_history",
        "_key_to_track_id",
        "_last_frame_id",
        "_next_track_id",
        "_setup_count",
        "_state_lock",
        "descriptor",
        "stateful_descriptor",
    )

    def __init__(
        self,
        descriptor: ProcessorDescriptor = SYNTHETIC_TRACKER_DESCRIPTOR,
        stateful_descriptor: StatefulProcessorDescriptor = SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    ) -> None:
        self.descriptor = descriptor
        self.stateful_descriptor = stateful_descriptor
        self._state_lock = threading.Lock()
        self._entry_lock = threading.Lock()
        self._next_track_id: int = 1
        self._key_to_track_id: dict[str, int] = {}
        self._key_to_history: dict[str, list[tuple[int, int]]] = {}
        self._frame_count: int = 0
        self._last_frame_id: int | None = None
        self._setup_count: int = 0
        self._cleanup_count: int = 0
        self._entry_count: int = 0
        self._config: TrackerConfig = TrackerConfig()
        GLOBAL_TRACKER_REGISTRY.register(id(self))

    def setup(self, context: SetupContext) -> None:
        with self._state_lock:
            self._setup_count += 1
            self._config = _config_to_tracker_config(context.configuration)

    def process(self, inputs: object, context: FrameContext) -> object:
        with self._entry_lock:
            self._entry_count += 1

        try:
            detections: Sequence[SyntheticDetection] = ()
            if isinstance(inputs, TrackerInput):
                detections = inputs.detections
            elif isinstance(inputs, dict):
                raw = inputs.get("detections", ()) # type: ignore
                if isinstance(raw, (list, tuple)):
                    detections = tuple(
                        d for d in raw if isinstance(d, SyntheticDetection) # type: ignore
                    )

            with self._state_lock:
                self._frame_count += 1
                self._last_frame_id = context.frame_id

                tracks: list[SyntheticTrack] = []
                for detection in detections:
                    key = detection.object_key
                    if key not in self._key_to_track_id:
                        self._key_to_track_id[key] = self._next_track_id
                        self._next_track_id += 1
                        self._key_to_history[key] = []
                    self._key_to_history[key].append((detection.x, detection.y))
                    tracks.append(
                        SyntheticTrack(
                            track_id=self._key_to_track_id[key],
                            object_key=key,
                            history=tuple(self._key_to_history[key]),
                        )
                    )
                frame_count = self._frame_count
            return TrackerOutput(tracks=tuple(tracks), frame_count=frame_count)
        finally:
            with self._entry_lock:
                self._entry_count -= 1

    def healthcheck(self) -> None:
        pass

    def cleanup(self) -> None:
        with self._state_lock:
            self._cleanup_count += 1
        GLOBAL_TRACKER_REGISTRY.unregister(id(self))

    def snapshot_state(self) -> "TrackerStateSnapshot":
        """Return an immutable view of the current tracker state for test assertions."""
        with self._state_lock:
            return TrackerStateSnapshot(
                next_track_id=self._next_track_id,
                key_to_track_id=dict(self._key_to_track_id),
                frame_count=self._frame_count,
                last_frame_id=self._last_frame_id,
                setup_count=self._setup_count,
                cleanup_count=self._cleanup_count,
            )

    def max_concurrent_entries(self) -> int:
        """Return the maximum observed concurrent entry count inside process()."""
        return self._entry_count


@dataclass(frozen=True, slots=True)
class TrackerStateSnapshot:
    """An immutable view of a tracker's internal state at a point in time."""

    next_track_id: int
    key_to_track_id: dict[str, int]
    frame_count: int
    last_frame_id: int | None
    setup_count: int
    cleanup_count: int


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_synthetic_tracker() -> SyntheticTracker:
    """Factory compatible with ``ProcessorFactory`` for the default tracker type."""
    return SyntheticTracker(
        descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
        stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    )


def make_synthetic_tracker_v2() -> SyntheticTracker:
    """Factory for the alternative (v2) tracker type, used to test type-change rejection."""
    return SyntheticTracker(
        descriptor=SYNTHETIC_TRACKER_V2_DESCRIPTOR,
        stateful_descriptor=SYNTHETIC_TRACKER_V2_STATEFUL_DESCRIPTOR,
    )


# ---------------------------------------------------------------------------
# Predicate that raises unexpectedly (for predicate-failure injection)
# ---------------------------------------------------------------------------


def _raising_predicate(
    previous_configuration: ProcessorConfiguration,
    new_configuration: ProcessorConfiguration,
    context: TransitionContext,
) -> bool:
    raise RuntimeError("injected predicate failure")


TRACKER_RAISING_PREDICATE_STATEFUL_DESCRIPTOR: Final = StatefulProcessorDescriptor(
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
    supported_transition_policies=frozenset(
        {StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}
    ),
    preserves_state_for=_raising_predicate,
)

TRACKER_RAISING_DESCRIPTOR: Final = ProcessorDescriptor(
    type_name="synthetic_tracker_raising",
    input_schema=object,
    output_schema=TrackerOutput,
    config_schema=object,
    state_policy=StatePolicy.PRESERVABLE,
    state_schema_version=TRACKER_STATE_SCHEMA_VERSION,
)


def make_raising_predicate_tracker() -> SyntheticTracker:
    """Factory for a tracker whose preservation predicate always raises."""
    return SyntheticTracker(
        descriptor=TRACKER_RAISING_DESCRIPTOR,
        stateful_descriptor=TRACKER_RAISING_PREDICATE_STATEFUL_DESCRIPTOR,
    )
