"""Unit tests for VideoAnalyticsApplication API."""

from __future__ import annotations

from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend
from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import FileVideoSourceConfig, RTDETRConfig, VideoAnalyticsConfig
from usecases.video_analytics.sink import NullSink


class FakeSource:
    def __init__(self, frame_count: int = 100) -> None:
        self.frame_count: int = frame_count
        self.read_count: int = 0
        self.is_opened: bool = False
        self.is_closed: bool = False

    def open(self):
        from usecases.video_analytics.contracts import VideoMetadata

        self.is_opened = True
        return VideoMetadata(width=640, height=480, fps=30.0, frame_count=self.frame_count, duration_ns=3333333333)

    def read(self):
        if not self.is_opened or self.is_closed or self.read_count >= self.frame_count:
            return None
        self.read_count += 1
        import numpy as np
        from usecases.video_analytics.contracts import FramePacket

        return FramePacket(
            frame_id=self.read_count,
            source_timestamp_ns=self.read_count * 33333333,
            image_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
            width=640,
            height=480,
        )

    def close(self) -> None:
        self.is_closed = True


def test_application_lifecycle_and_swap_with_fakes() -> None:
    """Verify application startup, frame processing, detector swap, and shutdown using fakes."""
    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path="dummy.mp4", enable_pacing=False),
        initial_detector=RTDETRConfig(model_id="r18"),
        candidate_detector=RTDETRConfig(model_id="r50"),
        update_frame_id=10,
    )

    source = FakeSource(frame_count=30)
    sink = NullSink()
    init_backend = FakeDetectorBackend(model_id="r18")
    cand_backend = FakeDetectorBackend(model_id="r50")
    trk_backend = FakeTrackerBackend(instance_id="app_trk_1")

    with VideoAnalyticsApplication(
        config=cfg,
        source=source,  # type: ignore
        sink=sink,
        initial_detector_backend=init_backend,
        candidate_detector_backend=cand_backend,
        tracker_backend=trk_backend,
    ) as app:
        p1 = app.process_until_frame(10)
        assert p1 == 10
        assert app.processed_frames == 10

        app.request_detector_swap(cfg.candidate_detector, mechanism="VEPS", candidate_backend=cand_backend)

        p2 = app.process_remaining()
        assert p2 > 0
        assert app.processed_frames == 30

    assert trk_backend.instance_id == "app_trk_1"
    assert trk_backend.reset_count == 0
    assert sink.consumed_count == 30
