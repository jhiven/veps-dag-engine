"""FFmpeg RTSP publisher abstraction for live-stream video ingress."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from urllib.parse import urlparse

from usecases.video_analytics.contracts import RTSPPublisherProtocol

__all__ = [
    "FFmpegRTSPPublisher",
    "FakeRTSPPublisher",
    "build_ffmpeg_publisher_command",
    "verify_mediamtx_reachable",
]


def build_ffmpeg_publisher_command(
    video_path: str,
    rtsp_url: str,
) -> list[str]:
    """Construct FFmpeg CLI command for real-time RTSP stream publishing over TCP."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        rtsp_url,
    ]


def verify_mediamtx_reachable(rtsp_base_url: str, timeout_seconds: float = 2.0) -> None:
    """Verify that the MediaMTX RTSP server port is accepting TCP connections."""
    parsed = urlparse(rtsp_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8554

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_seconds)
    try:
        s.connect((host, port))
    except Exception as err:
        raise RuntimeError(
            f"MediaMTX RTSP server is not reachable at {rtsp_base_url!r} ({host}:{port}): {err}. "
            "Please ensure MediaMTX server is running before executing the publication benchmark."
        ) from err
    finally:
        s.close()


class FFmpegRTSPPublisher(RTSPPublisherProtocol):
    """Subprocess manager for publishing video over RTSP using FFmpeg."""

    def __init__(
        self,
        video_path: str,
        rtsp_base_url: str = "rtsp://127.0.0.1:8554",
        sub_run_id: str | None = None,
    ) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video file not found for RTSP publisher: {video_path}")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg executable not found in PATH. Please install FFmpeg to run RTSP benchmarks.")

        self._video_path: str = video_path
        self._base_url: str = rtsp_base_url.rstrip("/")
        path_suffix = sub_run_id or uuid.uuid4().hex[:8]
        self._rtsp_url: str = f"{self._base_url}/live_{path_suffix}"
        self._process: subprocess.Popen[bytes] | None = None
        self._captured_stderr: str = ""

    @property
    def rtsp_url(self) -> str:
        return self._rtsp_url

    @property
    def captured_stderr(self) -> str:
        return self._captured_stderr

    def start(self) -> None:
        """Verify MediaMTX availability and launch FFmpeg RTSP publisher process."""
        if self._process is not None and self._process.poll() is None:
            return

        verify_mediamtx_reachable(self._base_url)

        cmd = build_ffmpeg_publisher_command(self._video_path, self._rtsp_url)
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as err:
            raise RuntimeError(f"Failed to launch FFmpeg RTSP publisher subprocess: {err}") from err

    def wait_until_ready(self, timeout_seconds: float = 5.0) -> None:
        """Confirm FFmpeg subprocess is running and RTSP stream endpoint is accepting connections."""
        if self._process is None:
            raise RuntimeError("Cannot wait for an unstarted FFmpeg publisher.")

        start_t = time.monotonic()
        parsed = urlparse(self._rtsp_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8554

        ready = False
        while time.monotonic() - start_t < timeout_seconds:
            if self._process.poll() is not None:
                # Process exited prematurely
                stderr_bytes = b""
                if self._process.stderr is not None:
                    stderr_bytes = self._process.stderr.read()
                self._captured_stderr = stderr_bytes.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"FFmpeg publisher exited prematurely with code {self._process.returncode}. Stderr:\n{self._captured_stderr}"
                )

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                ready = True
                s.close()
                break
            except Exception:
                s.close()
                time.sleep(0.1)

        if not ready:
            self.stop()
            raise RuntimeError(
                f"Timed out waiting {timeout_seconds}s for RTSP stream at {self._rtsp_url!r} to become readable."
            )

    def stop(self) -> None:
        """Terminate the FFmpeg publisher process cleanly with timeout and kill fallback."""
        if self._process is None:
            return

        proc = self._process
        self._process = None

        if proc.poll() is not None:
            if proc.stderr is not None:
                self._captured_stderr = proc.stderr.read().decode("utf-8", errors="replace")
            return

        proc.terminate()
        try:
            _, stderr_bytes = proc.communicate(timeout=3.0)
            if stderr_bytes:
                self._captured_stderr = stderr_bytes.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_bytes = proc.communicate()
            if stderr_bytes:
                self._captured_stderr = stderr_bytes.decode("utf-8", errors="replace")


class FakeRTSPPublisher(RTSPPublisherProtocol):
    """Synthetic fake RTSP publisher for deterministic unit testing."""

    def __init__(
        self,
        rtsp_url: str = "rtsp://127.0.0.1:8554/fake_live",
        should_fail_startup: bool = False,
        fake_stderr: str = "Simulated FFmpeg startup error",
    ) -> None:
        self._rtsp_url: str = rtsp_url
        self._should_fail_startup: bool = should_fail_startup
        self._fake_stderr: str = fake_stderr
        self.is_running: bool = False
        self.captured_stderr: str = ""

    @property
    def rtsp_url(self) -> str:
        return self._rtsp_url

    def start(self) -> None:
        if self._should_fail_startup:
            self.captured_stderr = self._fake_stderr
            raise RuntimeError(f"FFmpeg publisher startup failed. Stderr:\n{self._fake_stderr}")
        self.is_running = True

    def wait_until_ready(self, timeout_seconds: float = 5.0) -> None:
        if not self.is_running:
            raise RuntimeError("Publisher is not running")

    def stop(self) -> None:
        self.is_running = False
