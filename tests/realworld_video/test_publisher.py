"""Unit tests for FFmpeg RTSP publisher abstraction."""

from __future__ import annotations

import pytest

from usecases.video_analytics.publisher import (
    FakeRTSPPublisher,
    build_ffmpeg_publisher_command,
)


def test_ffmpeg_command_construction() -> None:
    """Test 1: FFmpeg command construction matches canonical RTSP over TCP specification."""
    cmd = build_ffmpeg_publisher_command("sample.mp4", "rtsp://127.0.0.1:8554/live_test")
    assert cmd[0] == "ffmpeg"
    assert "-re" in cmd
    assert "-rtsp_transport" in cmd
    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert cmd[-1] == "rtsp://127.0.0.1:8554/live_test"


def test_publisher_startup_failure_includes_stderr() -> None:
    """Test 2: Publisher startup failure captures and includes FFmpeg stderr."""
    pub = FakeRTSPPublisher(
        should_fail_startup=True,
        fake_stderr="Error: Invalid codec or connection refused",
    )
    with pytest.raises(RuntimeError) as exc_info:
        pub.start()

    assert "FFmpeg publisher startup failed" in str(exc_info.value)
    assert "Invalid codec" in pub.captured_stderr


def test_publisher_termination_and_kill_fallback() -> None:
    """Test 3: Publisher termination handles stop cleanly."""
    pub = FakeRTSPPublisher()
    pub.start()
    assert pub.is_running
    pub.stop()
    assert not pub.is_running
