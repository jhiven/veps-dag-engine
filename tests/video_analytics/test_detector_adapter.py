"""Unit tests for detector backend adapter using FakeDetectorBackend."""

from __future__ import annotations

import numpy as np

from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import RTDETRConfig
from usecases.video_analytics.contracts import BoundingBox, Detection, FramePacket
from usecases.video_analytics.processors.detector import DetectorProcessor


def test_fake_detector_backend_prepare_and_infer() -> None:
    """Verify FakeDetectorBackend prepare, infer, and close lifecycle."""
    custom_det = Detection(
        box=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=250.0),
        score=0.95,
        class_id=0,
        class_name="person",
    )
    backend = FakeDetectorBackend(model_id="test_r18", detections=(custom_det,))

    assert backend.model_id == "test_r18"
    assert not backend.is_prepared

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = FramePacket(
        frame_id=1,
        source_timestamp_ns=1000,
        image_bgr=dummy_bgr,
        width=640,
        height=480,
    )

    backend.prepare(frame)
    assert backend.is_prepared

    res = backend.infer(frame)
    assert len(res) == 1
    assert res[0].score == 0.95
    assert res[0].box.x1 == 50.0

    backend.close()
    assert backend.is_closed


def test_detector_processor_integration() -> None:
    """Verify DetectorProcessor setup, process, healthcheck, and cleanup."""
    backend = FakeDetectorBackend()
    cfg = RTDETRConfig()
    proc = DetectorProcessor(backend, cfg)

    assert proc.descriptor.type_name == "rt_detr_detector"

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = FramePacket(
        frame_id=5,
        source_timestamp_ns=5000,
        image_bgr=dummy_bgr,
        width=640,
        height=480,
    )

    from nedo_vision_dag_engine.processor import FrameContext, SetupContext

    setup_ctx = SetupContext(
        node_id="detector",
        configuration={},
        registry_snapshot_id="reg1",
        candidate_plan_version=1,
        warm_up_requested=True,
    )
    proc.setup(setup_ctx)

    frame_ctx = FrameContext(frame_id=5, plan_version=1, admitted_at_ns=5000)
    output = proc.process(frame, frame_ctx)

    from usecases.video_analytics.contracts import DetectionBatch

    assert isinstance(output, DetectionBatch)
    assert output.frame_id == 5
    assert len(output.detections) == 1

    proc.healthcheck()
    proc.cleanup()
