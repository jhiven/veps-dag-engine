"""Unit tests for realworld-video CLI flags and preflight validations."""

from __future__ import annotations

import pytest

from benchmarks.cli import run_benchmarks


def test_realworld_video_cli_invalid_rtsp_url() -> None:
    """Test CLI rejects invalid RTSP URLs."""
    with pytest.raises(ValueError) as exc_info:
        run_benchmarks(
            suite="realworld-video",
            profile="smoke",
            rtsp_base_url="http://127.0.0.1:8554",
        )
    assert "Invalid RTSP URL" in str(exc_info.value)


def test_realworld_video_cli_invalid_queue_capacity() -> None:
    """Test CLI rejects non-positive queue capacity."""
    with pytest.raises(ValueError) as exc_info:
        run_benchmarks(
            suite="realworld-video",
            profile="smoke",
            queue_capacity=0,
        )
    assert "Queue capacity must be positive" in str(exc_info.value)
