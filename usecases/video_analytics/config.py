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
    "resolve_default_device",
]


def resolve_default_device(requested_device: str | None = None) -> str:
    """Resolve compute device, defaulting to 'cuda:0' if CUDA is available, else 'cpu'."""
    if requested_device and requested_device != "auto":
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True, slots=True)
class RTDETRConfig:
    """Configuration for RT-DETR object detector."""

    model_id: str = "PekingU/rtdetr_r18vd"
    device: str = "auto"
    dtype: str = "float32"
    confidence_threshold: float = 0.5
    person_class_name: str = "person"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.device == "auto":
            object.__setattr__(self, "device", resolve_default_device("auto"))
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        if not (self.device.startswith("cpu") or self.device.startswith("cuda")):
            raise ValueError(f"Unsupported device {self.device!r}. Must start with 'cpu', 'cuda', or 'auto'.")


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

    video_path: str = "fake_video.mp4"
    enable_pacing: bool = True
    fallback_fps: float = 30.0

    def __post_init__(self) -> None:
        if not self.video_path.strip():
            raise ValueError("video_path cannot be empty")
        if self.fallback_fps <= 0.0:
            raise ValueError(f"fallback_fps must be positive, got {self.fallback_fps}")


@dataclass(frozen=True, slots=True)
class VideoAnalyticsConfig:
    """Top-level pipeline configuration."""

    source: FileVideoSourceConfig = field(default_factory=FileVideoSourceConfig)
    initial_detector: RTDETRConfig = field(default_factory=RTDETRConfig)
    candidate_detector: RTDETRConfig = field(default_factory=lambda: RTDETRConfig(model_id="PekingU/rtdetr_r50vd"))
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    sink: NullSinkConfig = field(default_factory=NullSinkConfig)
    update_frame_id: int = 50
    total_frames_to_process: int = 150


def validate_config(config: VideoAnalyticsConfig) -> None:
    """Validate top-level configuration consistency."""
    if config.update_frame_id <= 0:
        raise ValueError(f"update_frame_id must be positive, got {config.update_frame_id}")
    if config.total_frames_to_process <= config.update_frame_id:
        raise ValueError(
            f"total_frames_to_process ({config.total_frames_to_process}) must be greater than update_frame_id ({config.update_frame_id})"
        )
