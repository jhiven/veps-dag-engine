"""Unit tests for realworld_video benchmark runner."""

from __future__ import annotations

import os
import tempfile

import pytest

from benchmarks.runners.realworld_video import (
    _probe_video_frame_count,  # pyright: ignore[reportPrivateUsage]
    run_realworld_video_suite,
)
from usecases.video_analytics.source import RTSPVideoSource


def test_realworld_video_runner_smoke() -> None:
    """Verify run_realworld_video_suite produces output CSVs using fake backends across Stop, Pause, VEPS."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, _ = run_realworld_video_suite(
            run_id="run_rw_test",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=2,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
        )

        assert len(samples) == 6  # 2 reps * 3 mechanisms (Stop, Pause, VEPS)
        assert os.path.exists(csv_path)
        assert os.path.exists(os.path.join(tmp_dir, "realworld-video-frame-samples.csv"))

        mechs = {s.mechanism for s in samples}
        assert mechs == {"Stop", "Pause", "VEPS"}
        assert all(s.tracker_reset_count_before == 0 for s in samples)
        assert all(s.tracker_reset_count_after == 0 for s in samples)
        assert all(s.request_trigger_receiver_position == 5 for s in samples)
        assert all(
            s.request_timestamp_ns is not None
            and s.candidate_prep_start_ns is not None
            and s.request_timestamp_ns <= s.candidate_prep_start_ns
            for s in samples
        )
        assert all(
            s.measurement_end_timestamp_ns is not None
            and s.gpu_cutoff_sample_timestamp_ns is not None
            and s.gpu_post_window_sync_timestamp_ns is not None
            and s.measurement_end_timestamp_ns <= s.gpu_cutoff_sample_timestamp_ns
            < s.gpu_post_window_sync_timestamp_ns
            for s in samples
        )

        for repetition in (1, 2):
            matched = [s for s in samples if s.repetition == repetition]
            assert len({s.fixed_window_start_media_frame_index for s in matched}) == 1
            assert len({s.fixed_window_end_media_frame_index for s in matched}) == 1


def test_replacement_video_has_expected_deterministic_loop_length() -> None:
    assert _probe_video_frame_count("sample_video.mp4") == 300


def test_rtsp_measurement_can_be_armed_at_future_media_boundary() -> None:
    source = RTSPVideoSource("rtsp://127.0.0.1:8554/test")
    source._current_frame_id = 30  # pyright: ignore[reportPrivateUsage]

    source.start_measurement_window(180, start_media_frame_index=301)

    assert source.measurement_start_timestamp_ns is None
    assert source._measurement_start_media_frame_index == 301  # pyright: ignore[reportPrivateUsage]

    source._current_frame_id = 301  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="already passed"):
        source.start_measurement_window(180, start_media_frame_index=301)
