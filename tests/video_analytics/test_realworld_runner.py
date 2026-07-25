"""Unit tests for realworld_video benchmark runner."""

from __future__ import annotations

import os
import tempfile

from benchmarks.runners.realworld_video import run_realworld_video_suite


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
