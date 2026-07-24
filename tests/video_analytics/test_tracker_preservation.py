"""Unit tests verifying ByteTrack state preservation across compatible detector replacement."""

from __future__ import annotations

import time

from nedo_vision_dag_engine.compiler import CompilationFailure, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)

from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend
from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import NullSinkConfig, RTDETRConfig
from usecases.video_analytics.contracts import BoundingBox, Detection, VideoMetadata
from usecases.video_analytics.graph import (
    build_video_analytics_registry,
    build_video_analytics_specification,
)
from usecases.video_analytics.sink import NullSink


def test_tracker_preservation_across_detector_swap() -> None:
    """Verify that ByteTrack tracker processor instance and state are preserved across compatible detector swap."""
    init_det_config = RTDETRConfig(model_id="PekingU/rtdetr_r18vd")
    cand_det_config = RTDETRConfig(model_id="PekingU/rtdetr_r50vd")
    snk_config = NullSinkConfig()

    init_det_backend = FakeDetectorBackend(
        model_id="PekingU/rtdetr_r18vd",
        detections=(
            Detection(box=BoundingBox(10, 10, 50, 100), score=0.9, class_id=0, class_name="person"),
        ),
    )
    cand_det_backend = FakeDetectorBackend(
        model_id="PekingU/rtdetr_r50vd",
        detections=(
            Detection(box=BoundingBox(12, 10, 52, 100), score=0.95, class_id=0, class_name="person"),
        ),
    )
    trk_backend = FakeTrackerBackend(instance_id="stable_tracker_inst_1")
    sink = NullSink(snk_config)
    sink.open(VideoMetadata(width=640, height=480, fps=30.0, frame_count=100, duration_ns=3333333333))

    # Initial graph
    spec_v1 = build_video_analytics_specification(detector_config=init_det_config)
    reg_v1 = build_video_analytics_registry(
        detector_backend=init_det_backend,
        tracker_backend=trk_backend,
        sink_instance=sink,
        detector_config=init_det_config,
    )

    compiler = WorkflowCompiler("0.1.0")
    cand_v1 = compiler.compile(spec_v1, registry=reg_v1)
    assert not isinstance(cand_v1, CompilationFailure)
    plan_v1 = cand_v1.plan

    executor = PipelineExecutor(initial_plan=plan_v1)
    controller = ReconfigurationController(executor=executor, compiler=compiler, registry=reg_v1)

    try:
        # Process frames on plan v1
        for frame_id in range(1, 11):
            now_ns = time.monotonic_ns()
            controller.admit_frame(admitted_at_ns=now_ns, frame_id=frame_id)

        assert trk_backend.processed_frame_count == 10
        assert trk_backend.reset_count == 0
        assert trk_backend.instance_id == "stable_tracker_inst_1"

        # Candidate graph with detector r50
        spec_v2 = build_video_analytics_specification(detector_config=cand_det_config)
        reg_v2 = build_video_analytics_registry(
            detector_backend=cand_det_backend,
            tracker_backend=trk_backend,
            sink_instance=sink,
            detector_config=cand_det_config,
        )

        controller.update_registry(reg_v2)

        reconfig_req = ReconfigurationRequest(
            request_id="swap_det_1",
            base_version=1,
            target_specification=spec_v2,
            state_directive=StateDirective(),
            submitted_at_ns=time.monotonic_ns(),
        )
        controller.submit(reconfig_req)
        rec_ready = controller.wait_for_status(
            "swap_det_1",
            statuses=frozenset({ReconfigurationStatus.READY}),
            timeout_seconds=5.0,
        )

        assert rec_ready is not None
        assert rec_ready.status is ReconfigurationStatus.READY
        controller.commit_ready()

        # Process frames on plan v2
        for frame_id in range(11, 21):
            now_ns = time.monotonic_ns()
            controller.admit_frame(admitted_at_ns=now_ns, frame_id=frame_id)

        # Verification of state preservation invariants
        assert trk_backend.instance_id == "stable_tracker_inst_1"
        assert trk_backend.reset_count == 0
        assert trk_backend.processed_frame_count == 20
        assert executor.active_plan.version == 2

    finally:
        controller.close()
        sink.close()
