"""Configuration dataclasses and validation for the video analytics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "RTDETRConfig",
    "ByteTrackConfig",
    "NullSinkConfig",
    "FileVideoSourceConfig",
    "VideoAnalyticsConfig",
    "validate_config",
]


@dataclass(frozen=True, slots=True)
class RTDETRConfig:
    """Configuration for RT-DETR object detector."""

    model_id: str = "PekingU/rtdetr_r18vd"
    device: str = "cpu"
    dtype: str = "float32"
    confidence_threshold: float = 0.5
    person_class_name: str = "person"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        if not (self.device.startswith("cpu") or self.device.startswith("cuda")):
            raise ValueError(f"Unsupported device {self.device!r}. Must start with 'cpu' or 'cuda'.")


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    """Configuration for stateful ByteTrack tracker."""

    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30

    def __post_init__(self) -> None:
        if not (0.0 <= self.track_thresh <= 1.0):
            raise ValueError(f"track_thresh must be between 0.0 and 1.0, got {self.track_thresh}")
        if self.track_buffer <= 0:
            raise ValueError(f"track_buffer must be positive, got {self.track_buffer}")


@dataclass(frozen=True, slots=True)
class NullSinkConfig:
    """Configuration for NullSink."""

    validate_monotonic_frame_ids: bool = True


@dataclass(frozen=True, slots=True)
class FileVideoSourceConfig:
    """Configuration for FileVideoSource."""

    video_path: str
    fallback_fps: float | None = None
    enable_pacing: bool = True

    def __post_init__(self) -> None:
        if not self.video_path:
            raise ValueError("video_path must not be empty")
        if self.fallback_fps is not None and self.fallback_fps <= 0.0:
            raise ValueError(f"fallback_fps must be positive, got {self.fallback_fps}")


@dataclass(frozen=True, slots=True)
class VideoAnalyticsConfig:
    """Top-level configuration for video analytics pipeline execution."""

    source: FileVideoSourceConfig
    initial_detector: RTDETRConfig = field(default_factory=RTDETRConfig)
    candidate_detector: RTDETRConfig = field(
        default_factory=lambda: RTDETRConfig(model_id="PekingU/rtdetr_r50vd")
    )
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    sink: NullSinkConfig = field(default_factory=NullSinkConfig)
    update_frame_id: int = 100
    total_frames_to_process: int | None = None


def validate_config(config: VideoAnalyticsConfig) -> None:
    """Validate video analytics configuration parameters."""
    if config.update_frame_id <= 0:
        raise ValueError(f"update_frame_id must be positive, got {config.update_frame_id}")
    if config.total_frames_to_process is not None and config.total_frames_to_process <= 0:
        raise ValueError(f"total_frames_to_process must be positive, got {config.total_frames_to_process}")
    if config.total_frames_to_process is not None and config.total_frames_to_process <= config.update_frame_id:
        raise ValueError(
            f"total_frames_to_process ({config.total_frames_to_process}) must be greater than update_frame_id ({config.update_frame_id})"
        )
