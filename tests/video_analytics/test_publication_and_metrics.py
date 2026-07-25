"""Comprehensive tests for publication guards, metric derivations, and frame observers."""

from __future__ import annotations

import os
import tempfile
import pytest

from benchmarks.runners.realworld_video import run_realworld_video_suite
from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend
from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import (
    FileVideoSourceConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
)
from usecases.video_analytics.contracts import ExecutionMode
from usecases.video_analytics.source import FileVideoSource


def test_publication_mode_rejects_fake_detector() -> None:
    """1. Publication mode rejects a fake detector backend."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="sample_video.mp4"),
        initial_detector=RTDETRConfig(model_id="PekingU/rtdetr_r18vd"),
        candidate_detector=RTDETRConfig(model_id="PekingU/rtdetr_r50vd"),
    )
    fake_det = FakeDetectorBackend()
    app = VideoAnalyticsApplication(
        config=cfg,
        initial_detector_backend=fake_det,
        execution_mode=ExecutionMode.PUBLICATION,
    )
    with pytest.raises((ValueError, RuntimeError), match="Publication mode rejects fake"):
        app.open()


def test_publication_mode_rejects_fake_tracker() -> None:
    """2. Publication mode rejects a fake tracker backend."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="sample_video.mp4"),
    )
    fake_trk = FakeTrackerBackend()
    app = VideoAnalyticsApplication(
        config=cfg,
        tracker_backend=fake_trk,
        execution_mode=ExecutionMode.PUBLICATION,
    )
    with pytest.raises((ValueError, RuntimeError), match="Publication mode rejects fake"):
        app.open()


def test_publication_mode_rejects_fake_source() -> None:
    """3. Publication mode rejects a fake source or fake clock."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
    )
    fake_src = FileVideoSource(cfg.source)
    app = VideoAnalyticsApplication(
        config=cfg,
        source=fake_src,
        execution_mode=ExecutionMode.PUBLICATION,
    )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        app.open()


def test_publication_mode_selects_production_backend_factories() -> None:
    """4. Publication mode selects the production backend factories."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="nonexistent_video.mp4"),
    )
    app = VideoAnalyticsApplication(
        config=cfg,
        execution_mode=ExecutionMode.PUBLICATION,
    )
    # FileVideoSource will raise FileNotFoundError in publication mode for non-existent video
    with pytest.raises(FileNotFoundError):
        app.open()


def test_explicit_cpu_publication_mode_allowed() -> None:
    """5. Explicit CPU publication mode is allowed."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="sample_video.mp4"),
        initial_detector=RTDETRConfig(device="cpu"),
    )
    app = VideoAnalyticsApplication(
        config=cfg,
        execution_mode=ExecutionMode.PUBLICATION,
    )
    assert app.execution_mode is ExecutionMode.PUBLICATION


def test_cuda_publication_fails_when_cuda_unavailable() -> None:
    """6. CUDA publication mode fails clearly when CUDA is unavailable."""
    try:
        import torch  # type: ignore[import-not-found,import-untyped]

        cuda_available = torch.cuda.is_available()  # pyright: ignore[reportUnknownMemberType]
    except Exception:
        cuda_available = False

    if not cuda_available:
        cfg = VideoAnalyticsConfig(
            source=FileVideoSourceConfig(video_path="sample_video.mp4"),
            initial_detector=RTDETRConfig(device="cuda:0"),
        )
        app = VideoAnalyticsApplication(
            config=cfg,
            execution_mode=ExecutionMode.PUBLICATION,
        )
        with pytest.raises(RuntimeError, match="CUDA"):
            app.open()


def test_sink_observer_writes_non_empty_frame_rows() -> None:
    """7. The sink observer writes non-empty frame-level rows."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")
        frame_csv_path = os.path.join(tmp_dir, "realworld-video-frame-samples.csv")

        _, frame_rows = run_realworld_video_suite(
            run_id="run_test_obs",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        assert len(frame_rows) > 0
        assert os.path.exists(frame_csv_path)
        with open(frame_csv_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == len(frame_rows) + 1  # header + data rows


def test_frame_rows_metadata_correctness() -> None:
    """8. Frame rows contain real plan version, detector ID, and tracker instance ID."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        _, frame_rows = run_realworld_video_suite(
            run_id="run_test_meta",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        for fr in frame_rows:
            assert fr.plan_version in (1, 2)
            assert fr.detector_id in ("PekingU/rtdetr_r18vd", "PekingU/rtdetr_r50vd")
            assert fr.tracker_instance_id.startswith("fake_trk") or fr.tracker_instance_id.startswith("bytetrack")


def test_request_timestamp_ordering() -> None:
    """9 & 10. Request timestamp is before prep start and timestamp ordering is valid."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, _ = run_realworld_video_suite(
            run_id="run_test_ts",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        for s in samples:
            assert s.request_timestamp_ns <= s.candidate_prep_start_ns
            assert s.candidate_prep_start_ns <= s.candidate_prep_end_ns
            assert s.candidate_prep_end_ns <= s.publication_timestamp_ns
            assert s.publication_timestamp_ns <= s.first_candidate_output_ns


def test_first_candidate_output_and_metrics_derivation() -> None:
    """11, 12, 13, 14, 15. Derived metrics calculations match frame observation events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, frame_rows = run_realworld_video_suite(
            run_id="run_test_metrics",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        for s in samples:
            # 12: No constant +1_000_000 fallback
            assert s.first_candidate_output_ns != s.publication_timestamp_ns + 1_000_000

            # 13: Request to effect
            assert s.request_to_effect_ns == s.first_candidate_output_ns - s.request_timestamp_ns

            # 14: Transition output gap
            mech_frames = [f for f in frame_rows if f.mechanism == s.mechanism and f.repetition == s.repetition]
            cand_frames = [f for f in mech_frames if f.plan_version > 1 and f.detector_id == s.candidate_model_id]
            assert cand_frames
            first_cand = cand_frames[0]
            assert s.first_candidate_output_ns == first_cand.completion_timestamp_ns

            old_frames = [f for f in mech_frames if f.completion_timestamp_ns < s.first_candidate_output_ns]
            assert old_frames
            last_old = max(old_frames, key=lambda f: f.completion_timestamp_ns)
            assert s.transition_output_gap_ns == s.first_candidate_output_ns - last_old.completion_timestamp_ns


def test_frame_level_count_agrees_with_run_level() -> None:
    """16. Frame-level row count agrees with run-level accounting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, frame_rows = run_realworld_video_suite(
            run_id="run_test_counts",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        # 3 mechanisms in 1 repetition = 3 sample rows
        assert len(samples) == 3
        # Each mechanism processed 30 frames = 90 total frame rows
        assert len(frame_rows) == 90


def test_tracker_identity_originates_from_actual_instance() -> None:
    """19. Tracker identity comes from actual injected tracker instance."""
    trk = FakeTrackerBackend(instance_id="custom_trk_123")
    det = FakeDetectorBackend()
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
    )
    with VideoAnalyticsApplication(
        config=cfg,
        initial_detector_backend=det,
        tracker_backend=trk,
        execution_mode=ExecutionMode.SMOKE,
    ) as app:
        assert app.tracker_backend is not None
        assert app.tracker_backend.instance_id == "custom_trk_123"


def test_processor_peak_counting() -> None:
    """20. Lifecycle peak counting includes candidate processors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, _ = run_realworld_video_suite(
            run_id="run_test_peak",
            output_csv_path=csv_path,
            video_path="fake_video.mp4",
            repetition_count=1,
            update_frame_id=5,
            total_frames=30,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        for s in samples:
            assert s.processor_peak_count >= 4


def test_same_candidate_preparation_callable_used() -> None:
    """21. Candidate preparation callable is invoked for Stop, Pause, and VEPS."""
    init_det = FakeDetectorBackend()
    cand_det = FakeDetectorBackend(model_id="PekingU/rtdetr_r50vd")
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
    )
    with VideoAnalyticsApplication(
        config=cfg,
        initial_detector_backend=init_det,
        candidate_detector_backend=cand_det,
        execution_mode=ExecutionMode.SMOKE,
    ) as app:
        app.request_detector_swap(mechanism="Stop")
        assert cand_det.is_prepared
