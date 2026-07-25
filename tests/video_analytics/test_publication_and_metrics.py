"""Integration tests for video analytics publication mode, metrics, and contracts."""

from __future__ import annotations

import os
import tempfile

import pytest

from benchmarks.runners.realworld_video import run_realworld_video_suite
from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend
from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import (
    ByteTrackConfig,
    FileVideoSourceConfig,
    NullSinkConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
)
from usecases.video_analytics.contracts import BackendKind, ExecutionMode
from usecases.video_analytics.sink import NullSink
from usecases.video_analytics.source import FileVideoSource


class DummyProdSource:
    backend_kind = BackendKind.PRODUCTION


def test_publication_mode_rejects_fake_detector() -> None:
    """1. Publication mode rejects FakeDetectorBackend."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
        initial_detector=RTDETRConfig(model_id="PekingU/rtdetr_r18vd"),
        candidate_detector=RTDETRConfig(model_id="PekingU/rtdetr_r50vd"),
        tracker=ByteTrackConfig(),
        sink=NullSinkConfig(),
    )
    sink = NullSink(cfg.sink)
    fake_det = FakeDetectorBackend()
    fake_trk = FakeTrackerBackend()

    app = VideoAnalyticsApplication(
        config=cfg,
        source=DummyProdSource(),  # type: ignore
        sink=sink,
        initial_detector_backend=fake_det,
        candidate_detector_backend=fake_det,
        tracker_backend=fake_trk,
        execution_mode=ExecutionMode.PUBLICATION,
    )

    with pytest.raises(ValueError) as exc_info:
        app.open()
    assert "Publication mode rejects fake initial DetectorBackend" in str(exc_info.value)


def test_publication_mode_rejects_fake_tracker() -> None:
    """2. Publication mode rejects FakeTrackerBackend."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
        initial_detector=RTDETRConfig(model_id="PekingU/rtdetr_r18vd"),
        candidate_detector=RTDETRConfig(model_id="PekingU/rtdetr_r50vd"),
        tracker=ByteTrackConfig(),
        sink=NullSinkConfig(),
    )
    sink = NullSink(cfg.sink)
    fake_trk = FakeTrackerBackend()

    app = VideoAnalyticsApplication(
        config=cfg,
        source=DummyProdSource(),  # type: ignore
        sink=sink,
        tracker_backend=fake_trk,
        execution_mode=ExecutionMode.PUBLICATION,
    )

    with pytest.raises(ValueError) as exc_info:
        app.open()
    assert "Publication mode rejects fake TrackerBackend" in str(exc_info.value)


def test_publication_mode_rejects_fake_source() -> None:
    """3. Publication mode rejects synthetic/fake VideoSource."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
        initial_detector=RTDETRConfig(model_id="PekingU/rtdetr_r18vd"),
        candidate_detector=RTDETRConfig(model_id="PekingU/rtdetr_r50vd"),
        tracker=ByteTrackConfig(),
        sink=NullSinkConfig(),
    )
    fake_source = FileVideoSource(cfg.source, require_production=False)

    app = VideoAnalyticsApplication(
        config=cfg,
        source=fake_source,
        execution_mode=ExecutionMode.PUBLICATION,
    )

    with pytest.raises(ValueError) as exc_info:
        app.open()
    assert "Publication mode rejects fake" in str(exc_info.value)


def test_explicit_cpu_publication_mode_allowed() -> None:
    """5. Explicit CPU device is permitted in publication mode."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="fake_video.mp4"),
        initial_detector=RTDETRConfig(model_id="PekingU/rtdetr_r18vd", device="cpu"),
        candidate_detector=RTDETRConfig(model_id="PekingU/rtdetr_r50vd", device="cpu"),
        tracker=ByteTrackConfig(),
        sink=NullSinkConfig(),
    )
    app = VideoAnalyticsApplication(
        config=cfg,
        source=DummyProdSource(),  # type: ignore
        execution_mode=ExecutionMode.PUBLICATION,
    )

    app.validate_publication_mode()


def test_sink_observer_writes_non_empty_frame_rows() -> None:
    """7. Sink observer captures non-empty frame sample rows."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, frame_rows = run_realworld_video_suite(
            run_id="run_test_observer",
            output_csv_path=csv_path,
            repetition_count=1,
            update_frame_id=5,
            total_frames=10,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        assert len(samples) == 3  # VEPS, Pause, Stop
        assert len(frame_rows) > 0


def test_first_candidate_output_and_metrics_derivation() -> None:
    """12-15. Candidate output metrics and transition gaps come from real frame rows."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, frame_rows = run_realworld_video_suite(
            run_id="run_test_metrics",
            output_csv_path=csv_path,
            repetition_count=1,
            update_frame_id=5,
            total_frames=10,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        for s in samples:
            assert s.first_candidate_output_ns is not None
            assert s.request_to_effect_ns is not None
            assert s.transition_output_gap_ns is not None
            assert s.request_timestamp_ns is not None

            assert s.first_candidate_output_ns > 0
            assert s.request_to_effect_ns > 0
            assert s.transition_output_gap_ns > 0

            # 13: Request to effect
            assert s.request_to_effect_ns == s.first_candidate_output_ns - s.request_timestamp_ns

            # 14: Transition output gap
            mech_frames = [f for f in frame_rows if f.mechanism == s.mechanism and f.repetition == s.repetition]
            cand_frames = [
                f for f in mech_frames
                if f.plan_version is not None and f.plan_version > 1 and f.detector_id == s.candidate_model_id
            ]
            assert cand_frames
            first_cand = cand_frames[0]
            assert s.first_candidate_output_ns == first_cand.completion_timestamp_ns

            old_frames = [
                f for f in mech_frames
                if f.completion_timestamp_ns is not None and f.completion_timestamp_ns < s.first_candidate_output_ns
            ]
            assert old_frames
            last_old = max(old_frames, key=lambda f: f.completion_timestamp_ns or 0)
            assert last_old.completion_timestamp_ns is not None
            assert s.transition_output_gap_ns == s.first_candidate_output_ns - last_old.completion_timestamp_ns


def test_frame_level_count_agrees_with_run_level() -> None:
    """16. Frame-level row count agrees with run-level accounting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "realworld-video-samples.csv")

        samples, frame_rows = run_realworld_video_suite(
            run_id="run_test_counts",
            output_csv_path=csv_path,
            repetition_count=1,
            update_frame_id=5,
            total_frames=10,
            use_fake_backends=True,
            execution_mode=ExecutionMode.SMOKE,
        )

        assert len(samples) == 3
        for s in samples:
            m_frames = [f for f in frame_rows if f.mechanism == s.mechanism and f.repetition == s.repetition and f.inside_measurement_window]
            # Phase-normalized window: 5 pre-request + transition + 5 post-effect.
            # Total measured frames may exceed the old fixed 10-frame window.
            assert len(m_frames) >= 10
            # Verify phase invariants on the summary row.
            assert s.pre_request_source_frames_received == s.pre_request_source_frame_target
            assert s.post_effect_source_frames_received == s.post_effect_source_frame_target
            assert s.total_measurement_source_frames_received == len(m_frames)
