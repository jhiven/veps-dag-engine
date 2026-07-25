"""File-based and RTSP-based live video source abstractions using ffmpeg / torchvision with background capture ingress."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from usecases.video_analytics.config import FileVideoSourceConfig
from usecases.video_analytics.contracts import BackendKind, DropReason, FramePacket, FrameSource, VideoMetadata
from usecases.video_analytics.ingress import RTSPCaptureIngress
from usecases.video_analytics.pacing import Clock, FramePacer

__all__ = [
    "FileVideoSource",
    "RTSPVideoSource",
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
        if (
            self._is_synthetic
            or self._config.video_path in ("fake_video.mp4", ":synthetic:")
            or self._config.video_path.startswith("synthetic:")
            or "fake" in self._config.video_path.lower()
            or "dummy" in self._config.video_path.lower()
        ):
            return BackendKind.FAKE
        return BackendKind.PRODUCTION

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
        frame_bytes = w * h * 3
        frames_list: list[np.ndarray[Any, Any]] = []

        while True:
            assert proc.stdout is not None
            raw_frame = proc.stdout.read(frame_bytes)
            if len(raw_frame) < frame_bytes:
                break
            frame_arr = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
            frames_list.append(frame_arr)

        proc.wait()
        if not frames_list:
            raise RuntimeError(f"FFmpeg failed to decode any frames from {video_path}")

        vframes = np.stack(frames_list, axis=0)
        return vframes, fps

    def read(self) -> FramePacket | None:
        """Read the next video frame synchronously."""
        if self._is_closed or self._metadata is None:
            raise RuntimeError("FileVideoSource must be opened before reading")

        if self._is_synthetic or self._vframes is None:
            if self._current_frame_id >= 1000:
                return None
            self._current_frame_id += 1
            if self._pacer is not None:
                self._pacer.pace_frame(self._current_frame_id)

            h, w = self._metadata.height, self._metadata.width
            img = np.zeros((h, w, 3), dtype=np.uint8)

            now_ns = time.monotonic_ns()
            return FramePacket(
                frame_id=self._current_frame_id,
                source_timestamp_ns=now_ns,
                image_bgr=img,
                width=w,
                height=h,
            )

        if self._current_frame_id >= self._metadata.frame_count:  # type: ignore[operator]
            return None

        self._current_frame_id += 1
        if self._pacer is not None:
            self._pacer.pace_frame(self._current_frame_id)

        idx = self._current_frame_id - 1
        img = self._vframes[idx]  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue,reportIndexIssue]

        now_ns = time.monotonic_ns()
        return FramePacket(
            frame_id=self._current_frame_id,
            source_timestamp_ns=now_ns,
            image_bgr=img,  # pyright: ignore[reportArgumentType]
            width=self._metadata.width,
            height=self._metadata.height,
        )

    def close(self) -> None:
        """Close the video source."""
        self._is_closed = True
        self._vframes = None


class RTSPVideoSource(FrameSource):
    """Production RTSP live-stream frame source backed by independent RTSPCaptureIngress background thread."""

    def __init__(
        self,
        rtsp_url: str,
        video_path: str = "sample_video.mp4",
        queue_capacity: int = 4,
        on_drop_callback: Callable[[FramePacket, DropReason], None] | None = None,
        fallback_fps: float = 30.0,
    ) -> None:
        self._rtsp_url: str = rtsp_url
        self._video_path: str = video_path
        self._queue_capacity: int = queue_capacity
        self._on_drop_callback: Callable[[FramePacket, DropReason], None] | None = on_drop_callback
        self._fallback_fps: float = fallback_fps

        self._metadata: VideoMetadata | None = None
        self._ingress: RTSPCaptureIngress | None = None
        self._decoder_proc: subprocess.Popen[bytes] | None = None
        self._current_frame_id: int = 0
        self._is_closed: bool = False
        self._lock: threading.Lock = threading.Lock()

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.PRODUCTION

    @property
    def rtsp_url(self) -> str:
        return self._rtsp_url

    @property
    def ingress(self) -> RTSPCaptureIngress | None:
        return self._ingress

    def open(self) -> VideoMetadata:
        """Probe RTSP metadata and launch independent background RTSPCaptureIngress thread."""
        if self._is_closed:
            raise RuntimeError("Cannot open a closed RTSPVideoSource")

        width, height, fps = self._probe_metadata()
        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=None,
            duration_ns=None,
        )

        frame_bytes = width * height * 3
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self._rtsp_url,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ]

        try:
            self._decoder_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10 * frame_bytes,
            )
        except Exception as err:
            raise RuntimeError(f"Failed to launch FFmpeg RTSP decoder process for {self._rtsp_url}: {err}") from err

        proc = self._decoder_proc

        def frame_decoder() -> FramePacket | None:
            if proc.poll() is not None or proc.stdout is None:
                return None
            try:
                raw_data = proc.stdout.read(frame_bytes)
                if len(raw_data) < frame_bytes:
                    return None

                with self._lock:
                    self._current_frame_id += 1
                    fid = self._current_frame_id

                img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                now_ns = time.monotonic_ns()

                return FramePacket(
                    frame_id=fid,
                    source_timestamp_ns=now_ns,
                    image_bgr=img_np,
                    width=width,
                    height=height,
                )
            except Exception:
                return None

        self._ingress = RTSPCaptureIngress(
            frame_decoder=frame_decoder,
            queue_capacity=self._queue_capacity,
            on_drop_callback=self._on_drop_callback,
        )
        self._ingress.start()

        return self._metadata

    def read(self) -> FramePacket | None:
        """Pop the latest frame non-blockingly from the bounded drop-oldest ingress queue."""
        if self._is_closed or self._ingress is None:
            raise RuntimeError("RTSPVideoSource must be opened before reading")
        return self._ingress.read()

    def close(self) -> None:
        """Stop capture thread and terminate decoder process."""
        if self._is_closed:
            return
        self._is_closed = True

        if self._ingress is not None:
            self._ingress.stop()
            self._ingress = None

        if self._decoder_proc is not None:
            if self._decoder_proc.poll() is None:
                self._decoder_proc.terminate()
                try:
                    self._decoder_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._decoder_proc.kill()
            self._decoder_proc = None

    def _probe_metadata(self) -> tuple[int, int, float]:
        target_path = self._video_path if os.path.exists(self._video_path) else self._rtsp_url
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                target_path,
            ]
            output = subprocess.check_output(probe_cmd, timeout=5.0).decode("utf-8")
            info: dict[str, Any] = json.loads(output)
            streams: list[dict[str, Any]] = info.get("streams", [])
            v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            if v_stream is not None:
                w = int(v_stream.get("width", 640))
                h = int(v_stream.get("height", 480))
                fps_str = str(v_stream.get("r_frame_rate", "30/1"))
                if "/" in fps_str:
                    n_str, d_str = fps_str.split("/")
                    fps = float(n_str) / float(d_str) if float(d_str) != 0 else 30.0
                else:
                    fps = float(fps_str)
                return w, h, fps
        except Exception:
            pass
        return 640, 480, self._fallback_fps


def verify_video_length(metadata: VideoMetadata, required_frames: int) -> None:
    """Verify that a video is long enough for the required number of frames before an experiment."""
    if metadata.frame_count is not None and metadata.frame_count < required_frames:
        raise ValueError(
            f"Video file contains {metadata.frame_count} frames, but experiment requires at least {required_frames}"
        )
