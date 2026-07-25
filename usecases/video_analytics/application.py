"""Video analytics application layer coordinating pipeline execution and reconfiguration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

import numpy as np

from nedo_vision_dag_engine.compiler import CompilationFailure, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import FrameStatus
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)

from usecases.video_analytics.config import RTDETRConfig, VideoAnalyticsConfig
from usecases.video_analytics.contracts import (
    BackendKind,
    DetectorBackend,
    ExecutionMode,
    FramePacket,
    FrameSource,
    ResultSink,
    TrackerBackend,
    VideoMetadata,
)
from usecases.video_analytics.graph import (
    build_video_analytics_registry,
    build_video_analytics_specification,
)
from usecases.video_analytics.lifecycle import cleanup_repetition_resources
from usecases.video_analytics.sink import NullSink
from usecases.video_analytics.source import FileVideoSource, verify_video_length

logger = logging.getLogger(__name__)

__all__ = [
    "VideoAnalyticsApplication",
    "SwapTiming",
]


@dataclass(frozen=True, slots=True)
class SwapTiming:
    request_timestamp_ns: int
    candidate_prep_start_ns: int
    candidate_prep_end_ns: int
    publication_timestamp_ns: int


class VideoAnalyticsApplication:
    """Application API coordinating video analytics execution and live detector reconfigurations."""

    def __init__(
        self,
        config: VideoAnalyticsConfig,
        source: FrameSource | None = None,
        sink: ResultSink | None = None,
        initial_detector_backend: DetectorBackend | None = None,
        candidate_detector_backend: DetectorBackend | None = None,
        tracker_backend: TrackerBackend | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SMOKE,
    ) -> None:
        self.config: VideoAnalyticsConfig = config
        self.execution_mode: ExecutionMode = execution_mode
        self._source: FrameSource | None = source
        self._sink: ResultSink | None = sink
        self._initial_detector_backend: DetectorBackend | None = initial_detector_backend
        self._candidate_detector_backend: DetectorBackend | None = candidate_detector_backend
        self._tracker_backend: TrackerBackend | None = tracker_backend

        self._metadata: VideoMetadata | None = None
        self._compiler: WorkflowCompiler | None = None
        self._executor: PipelineExecutor | None = None
        self._controller: ReconfigurationController | None = None

        self._current_plan_version: int = 1
        self._is_opened: bool = False
        self._is_closed: bool = False
        self._processed_frames: int = 0
        self._last_swap_timing: SwapTiming | None = None

    def validate_publication_mode(self) -> None:
        """Validate that all injected/constructed components satisfy publication contracts."""
        if self.execution_mode is not ExecutionMode.PUBLICATION:
            return

        if self.config.initial_detector.device.startswith("cuda") or self.config.candidate_detector.device.startswith("cuda"):
            import torch 

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self.config.initial_detector.device!r} was requested in publication mode, but CUDA is not available."
                )

        if self._source is not None and getattr(self._source, "backend_kind", None) is not BackendKind.PRODUCTION:
            raise ValueError("Publication mode rejects fake or synthetic FrameSource.")

        if self._initial_detector_backend is not None and getattr(self._initial_detector_backend, "backend_kind", None) is not BackendKind.PRODUCTION:
            raise ValueError("Publication mode rejects fake initial DetectorBackend.")

        if self._candidate_detector_backend is not None and getattr(self._candidate_detector_backend, "backend_kind", None) is not BackendKind.PRODUCTION:
            raise ValueError("Publication mode rejects fake candidate DetectorBackend.")

        if self._tracker_backend is not None and getattr(self._tracker_backend, "backend_kind", None) is not BackendKind.PRODUCTION:
            raise ValueError("Publication mode rejects fake TrackerBackend.")

        if self._sink is not None and getattr(self._sink, "backend_kind", None) is not BackendKind.PRODUCTION:
            raise ValueError("Publication mode rejects fake ResultSink.")

    def open(self) -> VideoMetadata:
        """Open video source, construct initial DAG, and initialize execution pipeline."""
        if self._is_opened or self._is_closed:
            raise RuntimeError("Application is already opened or closed")

        if self._source is None:
            self._source = FileVideoSource(
                self.config.source,
                require_production=(self.execution_mode is ExecutionMode.PUBLICATION),
            )

        # Initialize detector backend if not injected
        if self._initial_detector_backend is None:
            from usecases.video_analytics.backends.rtdetr import RTDETRDetectorBackend

            self._initial_detector_backend = RTDETRDetectorBackend(self.config.initial_detector)

        # Initialize tracker backend if not injected
        if self._tracker_backend is None:
            from usecases.video_analytics.backends.bytetrack import ByteTrackerBackend

            self._tracker_backend = ByteTrackerBackend(self.config.tracker)

        if self._sink is None:
            self._sink = NullSink(self.config.sink)

        self.validate_publication_mode()

        self._metadata = self._source.open()

        verify_video_length(self._metadata, self.config.total_frames_to_process)

        self._sink.open(self._metadata)

        # Build initial specification & registry
        spec = build_video_analytics_specification(
            detector_config=self.config.initial_detector,
            tracker_config=self.config.tracker,
            sink_config=self.config.sink,
        )

        registry = build_video_analytics_registry(
            detector_backend=self._initial_detector_backend,
            tracker_backend=self._tracker_backend,
            sink_instance=self._sink,
            source_instance=self._source,
            detector_config=self.config.initial_detector,
            tracker_config=self.config.tracker,
        )

        self._compiler = WorkflowCompiler("0.1.0")
        cand_result = self._compiler.compile(spec, registry=registry)
        if isinstance(cand_result, CompilationFailure):
            raise RuntimeError(f"Initial video analytics DAG compilation failed: {cand_result.reason}")

        initial_plan = cand_result.plan
        self._executor = PipelineExecutor(initial_plan=initial_plan)
        self._controller = ReconfigurationController(
            executor=self._executor,
            compiler=self._compiler,
            registry=registry,
        )

        self._is_opened = True
        return self._metadata

    def process_until_frame(self, target_frame_id: int) -> int:
        """Admit and process video frames until frame_id reaches target_frame_id."""
        if not self._is_opened or self._is_closed:
            raise RuntimeError("Application must be opened before processing frames")

        assert self._controller is not None

        processed = 0
        while self._processed_frames < target_frame_id:
            now_ns = time.monotonic_ns()
            res = self._controller.admit_frame(
                admitted_at_ns=now_ns,
                frame_id=self._processed_frames + 1,
            )
            if res.status is not FrameStatus.COMPLETED:
                retried = False
                for _ in range(5):
                    time.sleep(0.01)
                    retry_res = self._controller.admit_frame(
                        admitted_at_ns=time.monotonic_ns(),
                        frame_id=self._processed_frames + 1,
                    )
                    if retry_res.status is FrameStatus.COMPLETED:
                        retried = True
                        break
                if not retried:
                    break
            self._processed_frames += 1
            processed += 1

        return processed

    def request_detector_swap(
        self,
        new_detector_config: RTDETRConfig | None = None,
        mechanism: Literal["VEPS", "Pause", "Stop"] = "VEPS",
        candidate_backend: DetectorBackend | None = None,
    ) -> SwapTiming:
        """Perform live or offline detector replacement using configured mechanism."""
        if not self._is_opened or self._is_closed:
            raise RuntimeError("Application must be opened before requesting detector swap")

        det_config = new_detector_config or self.config.candidate_detector

        if candidate_backend is not None:
            cand_backend = candidate_backend
        elif self._candidate_detector_backend is not None:
            cand_backend = self._candidate_detector_backend
        else:
            from usecases.video_analytics.backends.rtdetr import RTDETRDetectorBackend

            cand_backend = RTDETRDetectorBackend(det_config)
            self._candidate_detector_backend = cand_backend

        self.validate_publication_mode()

        assert self._source is not None
        assert self._sink is not None
        assert self._tracker_backend is not None
        assert self._controller is not None
        assert self._compiler is not None
        assert self._executor is not None

        # Build candidate specification
        candidate_spec = build_video_analytics_specification(
            detector_config=det_config,
            tracker_config=self.config.tracker,
            sink_config=self.config.sink,
        )

        # Build candidate registry with new candidate detector backend
        candidate_registry = build_video_analytics_registry(
            detector_backend=cand_backend,
            tracker_backend=self._tracker_backend,
            sink_instance=self._sink,
            source_instance=self._source,
            detector_config=det_config,
            tracker_config=self.config.tracker,
        )

        t_req = time.monotonic_ns()

        if mechanism == "VEPS":
            req_id = f"swap_to_{det_config.model_id}_{time.monotonic_ns()}"
            reconfig_req = ReconfigurationRequest(
                request_id=req_id,
                base_version=self._executor.active_plan.version,
                target_specification=candidate_spec,
                state_directive=StateDirective(),
                submitted_at_ns=t_req,
            )
            self._controller.update_registry(candidate_registry)
            self._controller.submit(reconfig_req)

            # Continue processing frames while candidate is preparing off-path
            while True:
                rec = self._controller.record(req_id)
                if rec.status in (
                    ReconfigurationStatus.READY,
                    ReconfigurationStatus.COMMITTED,
                    ReconfigurationStatus.REJECTED,
                    ReconfigurationStatus.FAILED,
                    ReconfigurationStatus.ABORTED,
                ):
                    break
                now_ns = time.monotonic_ns()
                res = self._controller.admit_frame(
                    admitted_at_ns=now_ns,
                    frame_id=self._processed_frames + 1,
                )
                if res.status is not FrameStatus.COMPLETED:
                    break
                self._processed_frames += 1
                time.sleep(0.001)

            rec_ready = self._controller.wait_for_status(
                req_id,
                statuses=frozenset({ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED}),
                timeout_seconds=60.0,
            )
            if rec_ready is None or rec_ready.status not in (ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED):
                fail_reason = rec_ready.failure_reason if rec_ready else "Timeout waiting for READY/COMMITTED"
                raise RuntimeError(f"VEPS candidate preparation failed: {fail_reason}")

            if rec_ready.status is ReconfigurationStatus.READY:
                self._controller.commit_ready()
            rec_committed = self._controller.record(req_id)
            self._current_plan_version += 1

            t_prep_start = rec_committed.preparation_started_ns or t_req
            t_prep_end = rec_committed.preparation_completed_ns or rec_committed.ready_ns or t_prep_start
            t_pub = rec_committed.commit_ns or rec_committed.ready_ns or t_prep_end

        elif mechanism == "Pause":
            req_id = f"pause_swap_{det_config.model_id}_{time.monotonic_ns()}"
            reconfig_req = ReconfigurationRequest(
                request_id=req_id,
                base_version=self._executor.active_plan.version,
                target_specification=candidate_spec,
                state_directive=StateDirective(),
                submitted_at_ns=t_req,
            )
            self._controller.update_registry(candidate_registry)
            self._controller.submit(reconfig_req)
            rec_ready = self._controller.wait_for_status(
                req_id,
                statuses=frozenset({ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED}),
                timeout_seconds=60.0,
            )
            if rec_ready is None or rec_ready.status not in (ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED):
                fail_reason = rec_ready.failure_reason if rec_ready else "Timeout waiting for READY/COMMITTED"
                raise RuntimeError(f"Pause candidate preparation failed: {fail_reason}")

            if rec_ready.status is ReconfigurationStatus.READY:
                self._controller.commit_ready()
            rec_committed = self._controller.record(req_id)
            self._current_plan_version += 1

            t_prep_start = rec_committed.preparation_started_ns or t_req
            t_prep_end = rec_committed.preparation_completed_ns or rec_committed.ready_ns or t_prep_start
            t_pub = rec_committed.commit_ns or rec_committed.ready_ns or t_prep_end

        elif mechanism == "Stop":
            t_prep_start = time.monotonic_ns()
            dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            sample_packet = FramePacket(
                frame_id=1,
                source_timestamp_ns=0,
                image_bgr=dummy_bgr,
                width=640,
                height=480,
            )
            cand_backend.prepare(sample_packet)
            t_prep_end = time.monotonic_ns()

            old_plan = self._executor.active_plan
            self._controller.close()
            cand_res = self._compiler.compile(
                candidate_spec,
                registry=candidate_registry,
                previous_plan=old_plan,
            )
            if isinstance(cand_res, CompilationFailure):
                raise RuntimeError(f"Stop-rebuild compilation failed: {cand_res.reason}")

            compiled_plan = cand_res.plan
            self._current_plan_version = compiled_plan.version
            self._executor = PipelineExecutor(initial_plan=compiled_plan)
            self._controller = ReconfigurationController(
                executor=self._executor,
                compiler=self._compiler,
                registry=candidate_registry,
            )
            t_pub = time.monotonic_ns()

        timing = SwapTiming(
            request_timestamp_ns=t_req,
            candidate_prep_start_ns=t_prep_start,
            candidate_prep_end_ns=t_prep_end,
            publication_timestamp_ns=t_pub,
        )
        self._last_swap_timing = timing
        return timing

    def process_remaining(self) -> int:
        """Process all remaining frames in the video source."""
        if not self._is_opened or self._is_closed:
            raise RuntimeError("Application must be opened before processing frames")

        assert self._controller is not None

        processed = 0
        while True:
            if self._processed_frames >= self.config.total_frames_to_process:
                break
            now_ns = time.monotonic_ns()
            res = self._controller.admit_frame(
                admitted_at_ns=now_ns,
                frame_id=self._processed_frames + 1,
            )
            if res.status is not FrameStatus.COMPLETED:
                retried = False
                for _ in range(5):
                    time.sleep(0.01)
                    retry_res = self._controller.admit_frame(
                        admitted_at_ns=time.monotonic_ns(),
                        frame_id=self._processed_frames + 1,
                    )
                    if retry_res.status is FrameStatus.COMPLETED:
                        retried = True
                        break
                if not retried:
                    break
            self._processed_frames += 1
            processed += 1

        return processed

    def close(self) -> None:
        """Gracefully close controller, executor, backends, source, and sink."""
        if self._is_closed:
            return
        self._is_closed = True

        if self._controller is not None:
            try:
                self._controller.close()
            except Exception as err:
                logger.warning("Error closing ReconfigurationController: %s", err)

        cleanup_repetition_resources(
            source=self._source,
            sink=self._sink,
            initial_detector=self._initial_detector_backend,
            candidate_detector=self._candidate_detector_backend,
            tracker=self._tracker_backend,
        )

    def __enter__(self) -> VideoAnalyticsApplication:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def processed_frames(self) -> int:
        return self._processed_frames

    @property
    def tracker_backend(self) -> TrackerBackend | None:
        return self._tracker_backend

    @property
    def executor(self) -> PipelineExecutor | None:
        return self._executor
