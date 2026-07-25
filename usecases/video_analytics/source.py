"""File-based video source abstraction using torchvision.io with synthetic fallback for testing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from usecases.video_analytics.config import FileVideoSourceConfig
from usecases.video_analytics.contracts import BackendKind, FramePacket, FrameSource, VideoMetadata
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
        require_production: bool = False,
    ) -> None:
        self._config: FileVideoSourceConfig = config
        self._vframes: Any | None = None
        self._metadata: VideoMetadata | None = None
        self._current_frame_id: int = 0
        self._pacer: FramePacer | None = None
        self._clock: Clock | None = clock
        self._is_closed: bool = False
        self._is_synthetic: bool = False
        self._require_production: bool = require_production

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.FAKE if self._is_synthetic else BackendKind.PRODUCTION

    def open(self) -> VideoMetadata:
        """Open the video file and retrieve metadata."""
        if self._is_closed:
            raise RuntimeError("Cannot open a closed FileVideoSource")

        if not os.path.exists(self._config.video_path):
            if not self._require_production and (
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

        vframes_bgr: np.ndarray[Any, Any] | None = None
        fps: float = 0.0
        err_messages: list[str] = []

        # 1. Try torchvision.io.read_video
        try:
            import torchvision.io  # type: ignore[import-not-found,import-untyped]  # pyright: ignore[reportUnknownVariableType]

            read_video_fn: Any = getattr(torchvision.io, "read_video", None)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            if read_video_fn is not None:
                vf, _, info = read_video_fn(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                    self._config.video_path,
                    pts_unit="sec",
                )
                raw_fps = 0.0
                if isinstance(info, dict):
                    fps_val: Any = info.get("video_fps", 0.0)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]
                    if isinstance(fps_val, (int, float)):
                        raw_fps = float(fps_val)
                fps = raw_fps if raw_fps > 0.0 else (self._config.fallback_fps or 30.0)
                vf_np = vf.numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                vframes_bgr = np.ascontiguousarray(vf_np[:, :, :, ::-1])
        except Exception as err:
            err_messages.append(f"torchvision: {err}")

        # 2. Try ffmpeg fallback
        if vframes_bgr is None:
            try:
                vframes_bgr, fps = self._decode_ffmpeg(self._config.video_path)
            except Exception as err:
                err_messages.append(f"ffmpeg: {err}")

        if vframes_bgr is None:
            combined_err = "; ".join(err_messages)
            if self._require_production:
                raise RuntimeError(
                    f"Production video decoding failed for {self._config.video_path}: {combined_err}"
                )
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

        frame_count = int(vframes_bgr.shape[0])
        height = int(vframes_bgr.shape[1])
        width = int(vframes_bgr.shape[2])

        if fps <= 0.0:
            fps = self._config.fallback_fps or 30.0

        duration_ns = int(round((frame_count / fps) * 1e9))

        self._vframes = vframes_bgr
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

    def _decode_ffmpeg(self, video_path: str) -> tuple[np.ndarray[Any, Any], float]:
        """Decode video file into BGR uint8 numpy array using ffmpeg/ffprobe CLI."""
        import json
        import subprocess

        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            video_path,
        ]
        probe_output = subprocess.check_output(probe_cmd).decode("utf-8")
        info: dict[str, Any] = json.loads(probe_output)
        streams: list[dict[str, Any]] = info.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        w = int(video_stream["width"])
        h = int(video_stream["height"])
        fps_str = str(video_stream.get("r_frame_rate", "30/1"))
        if "/" in fps_str:
            num_s, denom_s = fps_str.split("/")
            fps = float(num_s) / float(denom_s) if float(denom_s) != 0 else 30.0
        else:
            fps = float(fps_str)

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = h * w * 3
        frames: list[np.ndarray[Any, Any]] = []
        assert proc.stdout is not None
        while True:
            raw_frame = proc.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
            frames.append(frame)
        proc.wait()

        if not frames:
            raise ValueError(f"No frames decoded from {video_path} via ffmpeg")

        vframes_bgr = np.stack(frames, axis=0)
        return vframes_bgr, fps

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

        frame_raw: Any = self._vframes[self._current_frame_id]
        self._current_frame_id += 1
        frame_id = self._current_frame_id

        if self._pacer is not None:
            self._pacer.pace_frame(frame_id)

        source_timestamp_ns = int(round((frame_id - 1) * (1e9 / self._metadata.fps)))

        if isinstance(frame_raw, np.ndarray):
            image_bgr: np.ndarray[Any, np.dtype[np.uint8]] = np.asarray(frame_raw, dtype=np.uint8)
        else:
            # Convert RGB tensor to BGR uint8 numpy array
            frame_rgb_np = frame_raw.numpy()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            frame_bgr_np = np.ascontiguousarray(frame_rgb_np[:, :, ::-1])  # pyright: ignore[reportUnknownVariableType]
            image_bgr = np.asarray(frame_bgr_np, dtype=np.uint8)

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
