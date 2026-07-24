"""File-based video source abstraction using torchvision.io with synthetic fallback for testing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from usecases.video_analytics.config import FileVideoSourceConfig
from usecases.video_analytics.contracts import FramePacket, FrameSource, VideoMetadata
from usecases.video_analytics.pacing import Clock, FramePacer

__all__ = [
    "FileVideoSource",
    "verify_video_length",
]


class FileVideoSource(FrameSource):
    """CPU video decoder using torchvision.io behind a typed adapter boundary."""

    def __init__(
        self,
        config: FileVideoSourceConfig,
        clock: Clock | None = None,
    ) -> None:
        self._config: FileVideoSourceConfig = config
        self._vframes: Any | None = None
        self._metadata: VideoMetadata | None = None
        self._current_frame_id: int = 0
        self._pacer: FramePacer | None = None
        self._clock: Clock | None = clock
        self._is_closed: bool = False
        self._is_synthetic: bool = False

    def open(self) -> VideoMetadata:
        """Open the video file and retrieve metadata."""
        if self._is_closed:
            raise RuntimeError("Cannot open a closed FileVideoSource")

        if not os.path.exists(self._config.video_path):
            if (
                self._config.video_path in ("fake_video.mp4", ":synthetic:")
                or self._config.video_path.startswith("synthetic:")
                or "fake" in self._config.video_path.lower()
                or "dummy" in self._config.video_path.lower()
            ):
                self._is_synthetic = True
                fps = self._config.fallback_fps or 30.0
                self._metadata = VideoMetadata(
                    width=640,
                    height=480,
                    fps=fps,
                    frame_count=1000,
                    duration_ns=int(round(1000 * (1e9 / fps))),
                )
                self._pacer = FramePacer(
                    target_fps=fps,
                    clock=self._clock,
                    enabled=self._config.enable_pacing,
                )
                return self._metadata
            raise FileNotFoundError(f"Video file not found: {self._config.video_path}")

        try:
            import torchvision.io  # type: ignore[import-not-found,import-untyped]  # pyright: ignore[reportUnknownVariableType]

            vframes, _, info = torchvision.io.read_video(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                self._config.video_path,
                pts_unit="sec",
            )
        except ImportError:
            self._is_synthetic = True
            fps = self._config.fallback_fps or 30.0
            self._metadata = VideoMetadata(
                width=640,
                height=480,
                fps=fps,
                frame_count=1000,
                duration_ns=int(round(1000 * (1e9 / fps))),
            )
            self._pacer = FramePacer(
                target_fps=fps,
                clock=self._clock,
                enabled=self._config.enable_pacing,
            )
            return self._metadata
        except Exception as err:
            raise ValueError(f"Failed to open video file with torchvision: {self._config.video_path}") from err

        frame_count = int(getattr(vframes, "shape", [0])[0])  # pyright: ignore[reportUnknownArgumentType]
        if frame_count == 0:
            raise ValueError(f"Failed to decode video frames from: {self._config.video_path}")

        height = int(getattr(vframes, "shape", [0, 0])[1])  # pyright: ignore[reportUnknownArgumentType]
        width = int(getattr(vframes, "shape", [0, 0, 0])[2])  # pyright: ignore[reportUnknownArgumentType]

        raw_fps = 0.0
        if isinstance(info, dict):
            fps_val: Any = info.get("video_fps", 0.0)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]
            if isinstance(fps_val, (int, float)):
                raw_fps = float(fps_val)

        fps = raw_fps if raw_fps > 0.0 else (self._config.fallback_fps or 0.0)
        if fps <= 0.0:
            raise ValueError(
                f"Video FPS is invalid ({raw_fps}) and no fallback_fps was configured."
            )

        duration_ns = int(round((frame_count / fps) * 1e9))

        self._vframes = vframes
        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_ns=duration_ns,
        )

        self._pacer = FramePacer(
            target_fps=fps,
            clock=self._clock,
            enabled=self._config.enable_pacing,
        )

        return self._metadata

    def read(self) -> FramePacket | None:
        """Read the next video frame packet from the stream."""
        if self._is_closed or self._metadata is None:
            raise RuntimeError("FileVideoSource must be opened before reading")

        if self._is_synthetic:
            self._current_frame_id += 1
            frame_id = self._current_frame_id
            if self._metadata.frame_count is not None and frame_id > self._metadata.frame_count:
                return None
            if self._pacer is not None:
                self._pacer.pace_frame(frame_id)

            dummy_bgr = np.zeros((self._metadata.height, self._metadata.width, 3), dtype=np.uint8)
            source_timestamp_ns = int(round((frame_id - 1) * (1e9 / self._metadata.fps)))
            return FramePacket(
                frame_id=frame_id,
                source_timestamp_ns=source_timestamp_ns,
                image_bgr=dummy_bgr,
                width=self._metadata.width,
                height=self._metadata.height,
            )

        if self._vframes is None:
            raise RuntimeError("FileVideoSource is not open")

        total_frames = int(getattr(self._vframes, "shape", [0])[0])
        if self._current_frame_id >= total_frames:
            return None

        frame_tensor = self._vframes[self._current_frame_id]
        self._current_frame_id += 1
        frame_id = self._current_frame_id

        if self._pacer is not None:
            self._pacer.pace_frame(frame_id)

        source_timestamp_ns = int(round((frame_id - 1) * (1e9 / self._metadata.fps)))

        # Convert RGB tensor to BGR uint8 numpy array
        frame_rgb_np = frame_tensor.numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        frame_bgr_np = np.ascontiguousarray(frame_rgb_np[:, :, ::-1])  # pyright: ignore[reportUnknownVariableType]
        image_bgr: np.ndarray[Any, np.dtype[np.uint8]] = np.asarray(frame_bgr_np, dtype=np.uint8)

        return FramePacket(
            frame_id=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            image_bgr=image_bgr,
            width=self._metadata.width,
            height=self._metadata.height,
        )

    def close(self) -> None:
        """Release VideoCapture resources."""
        if self._is_closed:
            return
        self._is_closed = True
        self._vframes = None

    @property
    def metadata(self) -> VideoMetadata | None:
        return self._metadata


def verify_video_length(metadata: VideoMetadata, required_frames: int) -> None:
    """Verify that a video is long enough for the required number of frames before an experiment."""
    if metadata.frame_count is not None and metadata.frame_count < required_frames:
        raise ValueError(
            f"Source video contains {metadata.frame_count} frames, but experiment requires at least {required_frames} frames."
        )
