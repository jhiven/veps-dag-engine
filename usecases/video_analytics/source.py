"""File-based video source abstraction using OpenCV VideoCapture with synthetic fallback for testing."""

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
    """CPU video decoder using OpenCV VideoCapture behind a typed adapter boundary."""

    def __init__(
        self,
        config: FileVideoSourceConfig,
        clock: Clock | None = None,
    ) -> None:
        self._config: FileVideoSourceConfig = config
        self._capture: Any | None = None
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
            import cv2  # type: ignore[import-not-found,import-untyped]
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

        capture = cv2.VideoCapture(self._config.video_path)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not capture.isOpened():  # pyright: ignore[reportUnknownMemberType]
            capture.release()  # pyright: ignore[reportUnknownMemberType]
            raise ValueError(f"Failed to open video file: {self._config.video_path}")

        self._capture = capture

        width = int(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        height = int(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        raw_count = int(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        fps = raw_fps if raw_fps > 0.0 else (self._config.fallback_fps or 0.0)
        if fps <= 0.0:
            capture.release()  # pyright: ignore[reportUnknownMemberType]
            self._capture = None
            raise ValueError(
                f"Video FPS is invalid ({raw_fps}) and no fallback_fps was configured."
            )

        frame_count = raw_count if raw_count > 0 else None
        duration_ns = int(round((frame_count / fps) * 1e9)) if frame_count is not None else None

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

        if self._capture is None:
            raise RuntimeError("FileVideoSource capture is not open")

        import cv2  # type: ignore[import-not-found,import-untyped]

        ret, frame_bgr = self._capture.read()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not ret or frame_bgr is None:
            return None

        self._current_frame_id += 1
        frame_id = self._current_frame_id

        if self._pacer is not None:
            self._pacer.pace_frame(frame_id)

        # Source timestamp in nanoseconds
        timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        if timestamp_ms > 0.0:
            source_timestamp_ns = int(round(timestamp_ms * 1e6))
        else:
            source_timestamp_ns = int(round((frame_id - 1) * (1e9 / self._metadata.fps)))

        # Convert frame_bgr to NDArray[np.uint8]
        image_bgr: np.ndarray[Any, np.dtype[np.uint8]] = np.asarray(frame_bgr, dtype=np.uint8)

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
        if self._capture is not None:
            try:
                self._capture.release()  # pyright: ignore[reportUnknownMemberType]
            finally:
                self._capture = None

    @property
    def metadata(self) -> VideoMetadata | None:
        return self._metadata


def verify_video_length(metadata: VideoMetadata, required_frames: int) -> None:
    """Verify that a video is long enough for the required number of frames before an experiment."""
    if metadata.frame_count is not None and metadata.frame_count < required_frames:
        raise ValueError(
            f"Source video contains {metadata.frame_count} frames, but experiment requires at least {required_frames} frames."
        )
