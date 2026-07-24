"""Unit tests for FileVideoSource and VideoMetadata validation."""

from __future__ import annotations

import pytest

from usecases.video_analytics.config import FileVideoSourceConfig
from usecases.video_analytics.contracts import VideoMetadata
from usecases.video_analytics.source import FileVideoSource, verify_video_length


def test_missing_video_file_raises_error() -> None:
    """Verify that opening a non-existent video file raises FileNotFoundError."""
    source = FileVideoSource(FileVideoSourceConfig(video_path="non_existent_video_123.mp4"))
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        source.open()


def test_verify_video_length_check() -> None:
    """Verify video length verification helper."""
    meta = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=100,
        duration_ns=3_333_333_333,
    )
    # Valid
    verify_video_length(meta, 50)

    # Insufficient length raises ValueError
    with pytest.raises(ValueError, match="contains 100 frames, but experiment requires at least 200"):
        verify_video_length(meta, 200)


def test_closed_source_cannot_be_opened_or_read() -> None:
    """Verify closed FileVideoSource rejects open and read operations."""
    source = FileVideoSource(FileVideoSourceConfig(video_path="test.mp4"))
    source.close()

    with pytest.raises(RuntimeError, match="closed"):
        source.open()

    with pytest.raises(RuntimeError, match="opened before reading"):
        source.read()
