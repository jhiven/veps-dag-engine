"""Domain contracts, protocols, and immutable data structures for video analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from enum import Enum

__all__ = [
    "ExecutionMode",
    "BackendKind",
    "VideoMetadata",
    "FramePacket",
    "BoundingBox",
    "Detection",
    "DetectionBatch",
    "Track",
    "TrackBatch",
    "ModelIdentity",
    "TrackerStateSnapshot",
    "FrameSource",
    "ResultSink",
    "DetectorBackend",
    "TrackerBackend",
]


class ExecutionMode(str, Enum):
    """Execution profile mode for video analytics benchmark."""

    SMOKE = "smoke"
    PUBLICATION = "publication"


class BackendKind(str, Enum):
    """Backend implementation kind indicating fake or production status."""

    FAKE = "fake"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata describing a video source."""

    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_ns: int | None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Video dimensions must be positive, got {self.width}x{self.height}")
        if self.fps <= 0.0:
            raise ValueError(f"Video FPS must be positive, got {self.fps}")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A single decoded video frame payload."""

    frame_id: int
    source_timestamp_ns: int
    image_bgr: npt.NDArray[np.uint8]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.frame_id <= 0:
            raise ValueError(f"frame_id must be positive, got {self.frame_id}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got {self.width}x{self.height}")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box coordinates in frame space (pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result."""

    box: BoundingBox
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Batch of detections for a single frame."""

    frame_id: int
    source_timestamp_ns: int
    detector_id: str
    plan_version: int | None
    detections: tuple[Detection, ...]
    admission_timestamp_ns: int = 0

    @property
    def output(self) -> DetectionBatch:
        """Alias property matching output pin name for DAG step extraction."""
        return self


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object instance."""

    track_id: int
    class_id: int
    class_name: str
    box: BoundingBox
    score: float


@dataclass(frozen=True, slots=True)
class TrackBatch:
    """Batch of active tracks for a single frame."""

    frame_id: int
    source_timestamp_ns: int
    detector_id: str
    tracker_instance_id: str
    tracks: tuple[Track, ...]
    admission_timestamp_ns: int = 0
    plan_version: int | None = None

    @property
    def output(self) -> TrackBatch:
        """Alias property matching output pin name for DAG step extraction."""
        return self


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Identity and configuration descriptor for a detection model."""

    model_id: str
    architecture: str
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class TrackerStateSnapshot:
    """Snapshot of stateful tracker observables for audit and verification."""

    instance_id: str
    reset_count: int
    processed_frame_count: int
    live_track_ids: tuple[int, ...]


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for frame sources."""

    @property
    def backend_kind(self) -> BackendKind: ...
    def open(self) -> VideoMetadata: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


@runtime_checkable
class ResultSink(Protocol):
    """Protocol for result sinks."""

    @property
    def backend_kind(self) -> BackendKind: ...
    def open(self, metadata: VideoMetadata) -> None: ...
    def consume(self, result: TrackBatch) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class DetectorBackend(Protocol):
    """Protocol for detection backends."""

    @property
    def backend_kind(self) -> BackendKind: ...
    @property
    def model_id(self) -> str: ...
    def prepare(self, sample_frame: FramePacket) -> None: ...
    def infer(self, frame: FramePacket) -> tuple[Detection, ...]: ...
    def close(self) -> None: ...


@runtime_checkable
class TrackerBackend(Protocol):
    """Protocol for tracking backends."""

    @property
    def backend_kind(self) -> BackendKind: ...
    @property
    def instance_id(self) -> str: ...
    @property
    def reset_count(self) -> int: ...
    @property
    def processed_frame_count(self) -> int: ...
    def update(
        self,
        frame_id: int,
        source_timestamp_ns: int,
        detections: tuple[Detection, ...],
    ) -> tuple[Track, ...]: ...
    def snapshot(self) -> TrackerStateSnapshot: ...
    def close(self) -> None: ...

