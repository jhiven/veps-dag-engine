"""Unit tests for bounded drop-oldest ingress queue and capture thread independence."""

from __future__ import annotations

import numpy as np

from usecases.video_analytics.contracts import DropReason, FramePacket
from usecases.video_analytics.ingress import BoundedIngressQueue, FakeRTSPCaptureIngress


def _make_dummy_frame(frame_id: int) -> FramePacket:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    return FramePacket(
        frame_id=frame_id,
        source_timestamp_ns=frame_id * 1_000_000,
        image_bgr=img,
        width=10,
        height=10,
    )


def test_drop_oldest_queue_behavior() -> None:
    """Test 4: Bounded ingress queue drops oldest frame when capacity is exceeded."""
    dropped_frames: list[tuple[FramePacket, DropReason]] = []

    def on_drop(frame: FramePacket, reason: DropReason) -> None:
        dropped_frames.append((frame, reason))

    queue: BoundedIngressQueue[FramePacket] = BoundedIngressQueue(
        capacity=3,
        on_drop_callback=on_drop,
    )

    f1, f2, f3, f4 = [_make_dummy_frame(i) for i in range(1, 5)]

    queue.put(f1)
    queue.put(f2)
    queue.put(f3)
    assert queue.size() == 3
    assert len(dropped_frames) == 0

    # Put 4th frame -> f1 should be dropped
    queue.put(f4)
    assert queue.size() == 3
    assert len(dropped_frames) == 1
    assert dropped_frames[0][0].frame_id == 1
    assert dropped_frames[0][1] is DropReason.INGRESS_OVERFLOW

    popped = queue.get()
    assert popped is not None
    assert popped.frame_id == 2


def test_capture_thread_remains_active_during_pause() -> None:
    """Test 5: Ingress capture thread remains active while DAG admission is paused."""
    frames = [_make_dummy_frame(i) for i in range(1, 6)]
    ingress = FakeRTSPCaptureIngress(frames=frames, queue_capacity=4)

    ingress.start()
    assert ingress.is_active
    # DAG is paused, but ingress source counter and queue stay populated
    assert ingress.source_frames_received == 5
    assert ingress.queue.size() == 4
    ingress.stop()
    assert not ingress.is_active


def test_capture_thread_remains_active_during_stop_reconstruction() -> None:
    """Test 6: Ingress capture thread remains active while Stop reconstruction occurs."""
    frames = [_make_dummy_frame(i) for i in range(1, 10)]
    ingress = FakeRTSPCaptureIngress(frames=frames, queue_capacity=10)

    ingress.start()
    assert ingress.is_active
    # Simulate Stop reconstruction clearing queued DAG admissions
    cancelled = ingress.queue.clear_and_cancel_all()
    assert len(cancelled) == 9
    assert ingress.is_active
    ingress.stop()
