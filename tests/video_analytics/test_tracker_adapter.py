"""Unit tests for ByteTrack tracker backend adapter."""

from __future__ import annotations

from usecases.video_analytics.backends.bytetrack import ByteTrackerBackend, FakeTrackerBackend, calculate_box_iou
from usecases.video_analytics.config import ByteTrackConfig
from usecases.video_analytics.contracts import BoundingBox, Detection


def test_calculate_box_iou() -> None:
    """Verify IoU calculation between bounding boxes."""
    boxA = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    boxB = BoundingBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0)

    # Intersection: 5x10 = 50, Union: 100 + 100 - 50 = 150 -> IoU = 1/3
    iou = calculate_box_iou(boxA, boxB)
    assert abs(iou - (1.0 / 3.0)) < 1e-5

    # Non-overlapping
    boxC = BoundingBox(x1=20.0, y1=20.0, x2=30.0, y2=30.0)
    assert calculate_box_iou(boxA, boxC) == 0.0


def test_bytetracker_backend_lifecycle_and_observables() -> None:
    """Verify ByteTrackerBackend updates processed frame count, preserves instance ID, and snapshots state."""
    cfg = ByteTrackConfig(track_thresh=0.5)
    tracker = ByteTrackerBackend(cfg)

    inst_id = tracker.instance_id
    assert inst_id.startswith("bytetrack_")
    assert tracker.reset_count == 0
    assert tracker.processed_frame_count == 0

    det1 = Detection(
        box=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=100.0),
        score=0.9,
        class_id=0,
        class_name="person",
    )

    tracks1 = tracker.update(frame_id=1, source_timestamp_ns=1000, detections=(det1,))
    assert len(tracks1) == 1
    assert tracker.processed_frame_count == 1
    assert tracker.instance_id == inst_id

    # Frame 2 with slight movement
    det2 = Detection(
        box=BoundingBox(x1=12.0, y1=10.0, x2=52.0, y2=100.0),
        score=0.92,
        class_id=0,
        class_name="person",
    )
    tracks2 = tracker.update(frame_id=2, source_timestamp_ns=2000, detections=(det2,))
    assert len(tracks2) == 1
    assert tracks2[0].track_id == tracks1[0].track_id
    assert tracker.processed_frame_count == 2

    # Snapshot check
    snap = tracker.snapshot()
    assert snap.instance_id == inst_id
    assert snap.processed_frame_count == 2
    assert snap.reset_count == 0
    assert snap.live_track_ids == (tracks1[0].track_id,)

    tracker.close()


def test_fake_tracker_backend() -> None:
    """Verify FakeTrackerBackend functionality."""
    tracker = FakeTrackerBackend(instance_id="fake_123")
    assert tracker.instance_id == "fake_123"

    det = Detection(
        box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        score=0.8,
        class_id=0,
        class_name="person",
    )
    tracks = tracker.update(1, 100, (det,))
    assert len(tracks) == 1
    assert tracker.processed_frame_count == 1

    snap = tracker.snapshot()
    assert snap.instance_id == "fake_123"
    tracker.close()
    assert tracker.is_closed
