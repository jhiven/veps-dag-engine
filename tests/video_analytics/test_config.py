"""Unit tests for video analytics configuration validation."""

from __future__ import annotations

import pytest

from usecases.video_analytics.config import (
    ByteTrackConfig,
    FileVideoSourceConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
    validate_config,
)


def test_rtdetr_config_validation() -> None:
    """Verify RTDETRConfig validation for confidence threshold and device."""
    # Valid
    cfg = RTDETRConfig(confidence_threshold=0.7, device="cuda:0")
    assert cfg.confidence_threshold == 0.7
    assert cfg.device == "cuda:0"

    # Invalid confidence threshold
    with pytest.raises(ValueError, match="confidence_threshold"):
        RTDETRConfig(confidence_threshold=1.5)

    # Unsupported device
    with pytest.raises(ValueError, match="Unsupported device"):
        RTDETRConfig(device="tpu:0")


def test_bytetrack_config_validation() -> None:
    """Verify ByteTrackConfig validation."""
    cfg = ByteTrackConfig(track_thresh=0.6, track_buffer=20)
    assert cfg.track_thresh == 0.6

    with pytest.raises(ValueError, match="track_thresh"):
        ByteTrackConfig(track_thresh=-0.1)

    with pytest.raises(ValueError, match="track_buffer"):
        ByteTrackConfig(track_buffer=0)


def test_source_config_validation() -> None:
    """Verify FileVideoSourceConfig validation."""
    with pytest.raises(ValueError, match="video_path"):
        FileVideoSourceConfig(video_path="")

    with pytest.raises(ValueError, match="fallback_fps"):
        FileVideoSourceConfig(video_path="test.mp4", fallback_fps=-5.0)


def test_top_level_config_validation() -> None:
    """Verify VideoAnalyticsConfig validation logic."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="test.mp4"),
        update_frame_id=100,
        total_frames_to_process=200,
    )
    validate_config(cfg)

    # Invalid update frame
    invalid_update = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="test.mp4"),
        update_frame_id=0,
    )
    with pytest.raises(ValueError, match="update_frame_id"):
        validate_config(invalid_update)

    # Total frames less than update frame
    invalid_frames = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="test.mp4"),
        update_frame_id=100,
        total_frames_to_process=50,
    )
    with pytest.raises(ValueError, match="total_frames_to_process"):
        validate_config(invalid_frames)
