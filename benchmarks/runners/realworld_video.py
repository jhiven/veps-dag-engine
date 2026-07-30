"""Benchmark runner for real-world video analytics live reconfiguration (Section 9)."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Literal, cast, Any

import numpy as np

from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.backends.bytetrack import FakeTrackerBackend
from usecases.video_analytics.backends.rtdetr import FakeDetectorBackend
from usecases.video_analytics.config import (
    ByteTrackConfig,
    FileVideoSourceConfig,
    NullSinkConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
    resolve_default_device,
)
from usecases.video_analytics.contracts import (
    CUDAMemorySamplerProtocol,
    CUDAMemorySnapshot,
    DetectorBackend,
    DropReason,
    ExecutionMode,
    FramePacket,
    FrameSource,
    RTSPPublisherProtocol,
    TerminalStatus,
    TrackBatch,
    TrackerBackend,
)
from usecases.video_analytics.gpu_memory import FakeCUDAMemorySampler, PyTorchCUDAMemorySampler
from usecases.video_analytics.lifecycle import cleanup_repetition_resources
from usecases.video_analytics.metrics import calculate_flow_metrics
from usecases.video_analytics.publisher import FFmpegRTSPPublisher, FakeRTSPPublisher
from usecases.video_analytics.sink import NullSink
from usecases.video_analytics.source import FileVideoSource, RTSPVideoSource

__all__ = [
    "RealworldVideoSampleRow",
    "RealworldVideoFrameSampleRow",
    "REALWORLD_VIDEO_HEADERS",
    "REALWORLD_VIDEO_FRAME_HEADERS",
    "run_realworld_video_suite",
]

REALWORLD_VIDEO_HEADERS = (
    "run_id",
    "repetition",
    "execution_order_position",
    "mechanism",
    "source_path",
    "source_file_hash",
    "video_fps",
    "resolution",
    "initial_model_id",
    "candidate_model_id",
    "device",
    "dtype",
    "confidence_threshold",
    "tracker_config",
    "request_timestamp_ns",
    "candidate_prep_start_ns",
    "candidate_prep_end_ns",
    "publication_timestamp_ns",
    "first_candidate_output_ns",
    "request_to_effect_ns",
    "transition_output_gap_ns",
    "old_plan_frames_completed_during_prep",
    "source_frames_received",
    "frames_admitted",
    "frames_completed",
    "ingress_overflow_drop_count",
    "admission_rejection_count",
    "execution_cancelled_count",
    "frames_in_flight_at_window_end",
    "total_dropped_frame_count",
    "drop_rate",
    "dropped_frame_count",
    "duplicated_frame_count",
    "tracker_instance_id_before",
    "tracker_instance_id_after",
    "tracker_reset_count_before",
    "tracker_reset_count_after",
    "processor_peak_count",
    "gpu_memory_before_prep_allocated_bytes",
    "gpu_memory_before_prep_reserved_bytes",
    "gpu_memory_before_prep_peak_allocated_bytes",
    "gpu_memory_before_prep_peak_reserved_bytes",
    "gpu_memory_during_coexistence_allocated_bytes",
    "gpu_memory_during_coexistence_reserved_bytes",
    "gpu_memory_during_coexistence_peak_allocated_bytes",
    "gpu_memory_during_coexistence_peak_reserved_bytes",
    "gpu_memory_after_pub_allocated_bytes",
    "gpu_memory_after_pub_reserved_bytes",
    "gpu_memory_after_pub_peak_allocated_bytes",
    "gpu_memory_after_pub_peak_reserved_bytes",
    "gpu_memory_after_ret_allocated_bytes",
    "gpu_memory_after_ret_reserved_bytes",
    "gpu_memory_after_ret_peak_allocated_bytes",
    "gpu_memory_after_ret_peak_reserved_bytes",
    "gpu_memory_before_prep_bytes",
    "gpu_memory_during_coexistence_bytes",
    "gpu_memory_after_pub_bytes",
    "gpu_memory_after_ret_bytes",
    "warmup_completed_frames",
    "measurement_source_frame_target",
    "reconfiguration_trigger_frame_offset",
    "measurement_source_frames_received",
    "measurement_start_timestamp_ns",
    "measurement_end_timestamp_ns",
    "drain_start_timestamp_ns",
    "drain_end_timestamp_ns",
    "drops_before_request",
    "drops_during_candidate_preparation",
    "drops_between_preparation_and_publication",
    "drops_between_publication_and_first_candidate_output",
    "drops_after_first_candidate_output",
    "active_detector_id_after_retirement",
    "active_plan_version_after_retirement",
    "active_detector_instance_count_after_retirement",
    "peak_live_plan_count",
    "peak_live_detector_instance_count",
    "live_plan_count_after_publication",
    "live_plan_count_after_retirement",
    "live_detector_count_after_publication",
    "live_detector_count_after_retirement",
    "measurement_start_media_frame_index",
    "measurement_start_media_pts_ns",
    "measurement_end_media_frame_index",
    "measurement_end_media_pts_ns",
    "request_trigger_media_frame_index",
    "request_trigger_media_pts_ns",
    "measurement_start_source_sequence",
    "measurement_end_source_sequence",
    "pre_request_source_frame_target",
    "pre_request_source_frames_received",
    "transition_source_frames_received",
    "post_effect_source_frame_target",
    "post_effect_source_frames_received",
    "total_measurement_source_frames_received",
    "source_frames_before_request",
    "source_frames_during_candidate_preparation",
    "source_frames_between_preparation_and_publication",
    "source_frames_between_publication_and_first_candidate_output",
    "source_frames_after_first_candidate_output",
    "drop_rate_before_request",
    "drop_rate_during_candidate_preparation",
    "drop_rate_between_preparation_and_publication",
    "drop_rate_between_publication_and_first_candidate_output",
    "drop_rate_after_first_candidate_output",
    "gpu_memory_transition_max_allocated_bytes",
    "gpu_memory_transition_max_reserved_bytes",
    "fixed_window_start_media_frame_index",
    "fixed_window_end_media_frame_index",
    "fixed_window_expected_source_frames",
    "fixed_window_observed_source_frames",
    "fixed_window_admitted",
    "fixed_window_completed",
    "fixed_window_dropped",
    "fixed_window_duplicated",
    "fixed_window_old_plan_completions",
    "fixed_window_new_plan_completions",
    "fixed_window_accounting_residual",
    "fixed_window_accounting_valid",
)

REALWORLD_VIDEO_FRAME_HEADERS = (
    "run_id",
    "repetition",
    "mechanism",
    "frame_id",
    "source_timestamp_ns",
    "admission_timestamp_ns",
    "completion_timestamp_ns",
    "plan_version",
    "detector_id",
    "tracker_instance_id",
    "queue_occupancy_before_enqueue",
    "queue_occupancy_after_enqueue",
    "queue_capacity",
    "terminal_status",
    "drop_reason",
    "dropped",
    "duplicated",
    "inside_measurement_window",
    "media_pts_ns",
    "receiver_ingress_timestamp_ns",
    "enqueue_decision_timestamp_ns",
    "drop_decision_timestamp_ns",
    "media_frame_index",
)


@dataclass(frozen=True, slots=True)
class RealworldVideoSampleRow:
    run_id: str
    repetition: int
    execution_order_position: int
    mechanism: str  # Stop, Pause, VEPS
    source_path: str
    source_file_hash: str
    video_fps: float
    resolution: str
    initial_model_id: str
    candidate_model_id: str
    device: str
    dtype: str
    confidence_threshold: float
    tracker_config: str
    request_timestamp_ns: int | None
    candidate_prep_start_ns: int | None
    candidate_prep_end_ns: int | None
    publication_timestamp_ns: int | None
    first_candidate_output_ns: int | None
    request_to_effect_ns: int | None
    transition_output_gap_ns: int | None
    old_plan_frames_completed_during_prep: int
    source_frames_received: int
    frames_admitted: int
    frames_completed: int
    ingress_overflow_drop_count: int
    admission_rejection_count: int
    execution_cancelled_count: int
    frames_in_flight_at_window_end: int
    total_dropped_frame_count: int
    drop_rate: float | None
    dropped_frame_count: int
    duplicated_frame_count: int
    tracker_instance_id_before: str
    tracker_instance_id_after: str
    tracker_reset_count_before: int
    tracker_reset_count_after: int
    processor_peak_count: int
    gpu_before_snap: CUDAMemorySnapshot
    gpu_coexist_snap: CUDAMemorySnapshot
    gpu_pub_snap: CUDAMemorySnapshot
    gpu_ret_snap: CUDAMemorySnapshot
    warmup_completed_frames: int
    measurement_source_frame_target: int
    reconfiguration_trigger_frame_offset: int
    measurement_source_frames_received: int
    measurement_start_timestamp_ns: int | None
    measurement_end_timestamp_ns: int | None
    drain_start_timestamp_ns: int | None
    drain_end_timestamp_ns: int | None
    drops_before_request: int
    drops_during_candidate_preparation: int
    drops_between_preparation_and_publication: int
    drops_between_publication_and_first_candidate_output: int
    drops_after_first_candidate_output: int
    active_detector_id_after_retirement: str
    active_plan_version_after_retirement: int | None
    active_detector_instance_count_after_retirement: int
    peak_live_plan_count: int
    peak_live_detector_instance_count: int
    live_plan_count_after_publication: int
    live_plan_count_after_retirement: int
    live_detector_count_after_publication: int
    live_detector_count_after_retirement: int
    measurement_start_media_frame_index: int | None
    measurement_start_media_pts_ns: int | None
    measurement_end_media_frame_index: int | None
    measurement_end_media_pts_ns: int | None
    request_trigger_media_frame_index: int | None
    request_trigger_media_pts_ns: int | None
    measurement_start_source_sequence: int | None
    measurement_end_source_sequence: int | None
    pre_request_source_frame_target: int
    pre_request_source_frames_received: int
    transition_source_frames_received: int
    post_effect_source_frame_target: int
    post_effect_source_frames_received: int
    total_measurement_source_frames_received: int
    source_frames_before_request: int
    source_frames_during_candidate_preparation: int
    source_frames_between_preparation_and_publication: int
    source_frames_between_publication_and_first_candidate_output: int
    source_frames_after_first_candidate_output: int
    drop_rate_before_request: float | None
    drop_rate_during_candidate_preparation: float | None
    drop_rate_between_preparation_and_publication: float | None
    drop_rate_between_publication_and_first_candidate_output: float | None
    drop_rate_after_first_candidate_output: float | None
    gpu_memory_transition_max_allocated_bytes: int | None
    gpu_memory_transition_max_reserved_bytes: int | None
    # Fixed source-frame-index window (reviewer P0.6):
    # 60 frames before request trigger + 150 after = 210 expected total
    fixed_window_start_media_frame_index: int = 0
    fixed_window_end_media_frame_index: int = 0
    fixed_window_expected_source_frames: int = 0
    fixed_window_observed_source_frames: int = 0
    fixed_window_admitted: int = 0
    fixed_window_completed: int = 0
    fixed_window_dropped: int = 0
    fixed_window_duplicated: int = 0
    fixed_window_old_plan_completions: int = 0
    fixed_window_new_plan_completions: int = 0
    fixed_window_accounting_residual: int = 0
    fixed_window_accounting_valid: bool = False


@dataclass(frozen=True, slots=True)
class RealworldVideoFrameSampleRow:
    run_id: str
    repetition: int
    mechanism: str
    frame_id: int
    source_timestamp_ns: int | None
    admission_timestamp_ns: int | None
    completion_timestamp_ns: int | None
    plan_version: int | None
    detector_id: str
    tracker_instance_id: str
    queue_occupancy_before_enqueue: int
    queue_occupancy_after_enqueue: int
    queue_capacity: int
    terminal_status: str
    drop_reason: str
    dropped: bool
    duplicated: bool
    inside_measurement_window: bool
    media_pts_ns: int | None
    receiver_ingress_timestamp_ns: int | None
    enqueue_decision_timestamp_ns: int | None
    drop_decision_timestamp_ns: int | None
    media_frame_index: int | None


def _calculate_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return "fake_hash"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_fixed_window_metrics(
    frame_rows: list[RealworldVideoFrameSampleRow],
    request_trigger_media_frame_index: int,
    baseline_frame_count: int = 60,
    transition_frame_count: int = 150,
) -> dict[str, int | bool]:
    """Compute source-frame-index-based fixed-window metrics.

    Uses identical source-frame window for all mechanisms:
    - Baseline: indices [trigger - 60, trigger)
    - Transition: indices [trigger, trigger + 150)
    - Total expected: 210 source frames

    This replaces mechanism-dependent wall-clock windows with a fair,
    mechanism-independent source-frame-index comparison (reviewer P0.6).
    """
    window_start = max(0, request_trigger_media_frame_index - baseline_frame_count)
    window_end = request_trigger_media_frame_index + transition_frame_count
    expected_frames = baseline_frame_count + transition_frame_count  # 210

    observed = 0
    admitted = 0
    completed = 0
    dropped_count = 0
    duplicated = 0
    old_plan_completions = 0
    new_plan_completions = 0

    # Determine old vs new plan version
    old_plan_version: int | None = None
    new_plan_version: int | None = None
    for row in frame_rows:
        if row.plan_version is not None:
            if old_plan_version is None or row.plan_version < old_plan_version:
                old_plan_version = row.plan_version
            if new_plan_version is None or row.plan_version > new_plan_version:
                new_plan_version = row.plan_version

    for row in frame_rows:
        mfi = row.media_frame_index
        if mfi is None:
            continue
        if mfi < window_start or mfi >= window_end:
            continue

        observed += 1
        if row.duplicated:
            duplicated += 1
        if row.dropped:
            dropped_count += 1
        else:
            admitted += 1
            if row.completion_timestamp_ns is not None:
                completed += 1
                if row.plan_version is not None:
                    if old_plan_version is not None and row.plan_version == old_plan_version:
                        old_plan_completions += 1
                    elif new_plan_version is not None and row.plan_version == new_plan_version:
                        new_plan_completions += 1

    # Accounting: observed = completed + dropped + residual (in-flight)
    accounted = completed + dropped_count
    residual = observed - accounted
    in_flight = admitted - completed
    accounting_valid = (residual >= 0 and duplicated == 0
                        and residual == in_flight)

    return {
        "fixed_window_start_media_frame_index": window_start,
        "fixed_window_end_media_frame_index": window_end,
        "fixed_window_expected_source_frames": expected_frames,
        "fixed_window_observed_source_frames": observed,
        "fixed_window_admitted": admitted,
        "fixed_window_completed": completed,
        "fixed_window_dropped": dropped_count,
        "fixed_window_duplicated": duplicated,
        "fixed_window_old_plan_completions": old_plan_completions,
        "fixed_window_new_plan_completions": new_plan_completions,
        "fixed_window_accounting_residual": residual,
        "fixed_window_accounting_valid": accounting_valid,
    }


def run_realworld_video_suite(
    run_id: str,
    output_csv_path: str,
    video_path: str = "sample_video.mp4",
    rtsp_base_url: str = "rtsp://127.0.0.1:8554",
    initial_model: str = "PekingU/rtdetr_r18vd",
    candidate_model: str = "PekingU/rtdetr_r50vd",
    device: str = "auto",
    dtype: str = "float32",
    repetition_count: int = 5,
    warmup_completed_frames: int = 30,
    measurement_source_frames: int = 180,
    reconfiguration_trigger_frame_offset: int = 60,
    drain_timeout_seconds: float = 2.0,
    queue_capacity: int = 4,
    use_fake_backends: bool = False,
    random_seed: int = 42,
    execution_mode: ExecutionMode | None = None,
    # Compatibility arguments:
    update_frame_id: int | None = None,
    total_frames: int | None = None,
) -> tuple[list[RealworldVideoSampleRow], list[RealworldVideoFrameSampleRow]]:
    """Orchestrate the real-world video analytics benchmark suite across Stop, Pause, and VEPS."""
    if update_frame_id is not None:
        reconfiguration_trigger_frame_offset = update_frame_id
    if total_frames is not None:
        measurement_source_frames = total_frames

    device = resolve_default_device(device)
    mode = execution_mode or (ExecutionMode.SMOKE if use_fake_backends else ExecutionMode.PUBLICATION)

    from nedo_vision_dag_engine.compiler import StateDirective, CompilationFailure
    from nedo_vision_dag_engine.reconfiguration import ReconfigurationRequest, ReconfigurationStatus, ReconfigurationController
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from usecases.video_analytics.graph import build_video_analytics_specification, build_video_analytics_registry
    from usecases.video_analytics.lifecycle import CoexistenceTracker
    import threading

    if mode is ExecutionMode.PUBLICATION:
        if use_fake_backends:
            raise ValueError("Publication mode rejects use_fake_backends=True")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Publication mode video path not found: {video_path}")
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {device!r} was requested in publication mode, but CUDA is not available."
                )

        from usecases.video_analytics.backends.rtdetr import preload_rtdetr_models_to_ram
        from usecases.video_analytics.cli import prepare_assets_cmd

        print(
            f"Pre-downloading/verifying model checkpoints: initial={initial_model!r}, candidate={candidate_model!r}..."
        )
        prepare_assets_cmd(initial_model=initial_model, candidate_model=candidate_model)

        print("Caching model weight state_dicts into CPU RAM memory cache...")
        preload_rtdetr_models_to_ram([initial_model, candidate_model])
        print(
            "RAM model cache initialized. Each sub-run will construct a fresh candidate backend from RAM cache."
        )

    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    os.makedirs(output_dir, exist_ok=True)
    frame_csv_path = os.path.join(output_dir, "realworld-video-frame-samples.csv")

    source_hash = _calculate_file_hash(video_path)
    mechanisms = ["VEPS", "Pause", "Stop"]
    rng = random.Random(random_seed)

    sample_rows: list[RealworldVideoSampleRow] = []
    frame_rows: list[RealworldVideoFrameSampleRow] = []

    def classify_drops(
        frame_records: list[RealworldVideoFrameSampleRow],
        t_req: int | None,
        t_prep_end: int | None,
        t_pub: int | None,
        t_first_cand_out: int | None,
    ) -> tuple[int, int, int, int, int]:
        """Classify each dropped frame into a phase using drop_decision_timestamp_ns (local monotonic)."""
        d_before = 0
        d_prep = 0
        d_between_prep_pub = 0
        d_between_pub_first = 0
        d_after = 0

        for r in frame_records:
            if not r.inside_measurement_window or not r.dropped:
                continue
            ts = r.drop_decision_timestamp_ns
            if ts is None:
                continue

            if t_req is None or ts < t_req:
                d_before += 1
            elif t_prep_end is None or ts < t_prep_end:
                d_prep += 1
            elif t_pub is None or ts < t_pub:
                d_between_prep_pub += 1
            elif t_first_cand_out is None or ts < t_first_cand_out:
                d_between_pub_first += 1
            else:
                d_after += 1

        return d_before, d_prep, d_between_prep_pub, d_between_pub_first, d_after

    def classify_source_frames(
        frame_records: list[RealworldVideoFrameSampleRow],
        t_req: int | None,
        t_prep_end: int | None,
        t_pub: int | None,
        t_first_cand_out: int | None,
    ) -> tuple[int, int, int, int, int]:
        """Classify each source frame into a phase using receiver_ingress_timestamp_ns (local monotonic)."""
        s_before = 0
        s_prep = 0
        s_between_prep_pub = 0
        s_between_pub_first = 0
        s_after = 0

        for r in frame_records:
            if not r.inside_measurement_window:
                continue
            ts = r.receiver_ingress_timestamp_ns
            if ts is None:
                continue

            if t_req is None or ts < t_req:
                s_before += 1
            elif t_prep_end is None or ts < t_prep_end:
                s_prep += 1
            elif t_pub is None or ts < t_pub:
                s_between_prep_pub += 1
            elif t_first_cand_out is None or ts < t_first_cand_out:
                s_between_pub_first += 1
            else:
                s_after += 1

        return s_before, s_prep, s_between_prep_pub, s_between_pub_first, s_after

    try:
        for rep in range(1, repetition_count + 1):
            mech_order = list(mechanisms)
            rng.shuffle(mech_order)

            for pos, mech in enumerate(mech_order, start=1):
                local_files = mode is ExecutionMode.PUBLICATION

                publisher: RTSPPublisherProtocol
                if mode is ExecutionMode.PUBLICATION:
                    sub_id = f"r{rep}_p{pos}_{mech.lower()}"
                    publisher = FFmpegRTSPPublisher(video_path, rtsp_base_url, sub_run_id=sub_id)
                else:
                    publisher = FakeRTSPPublisher()

                publisher.start()
                if mode is ExecutionMode.PUBLICATION:
                    publisher.wait_until_ready(timeout_seconds=5.0)

                cfg = VideoAnalyticsConfig(
                    source=FileVideoSourceConfig(video_path=video_path, enable_pacing=not use_fake_backends),
                    initial_detector=RTDETRConfig(
                        model_id=initial_model,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files,
                    ),
                    candidate_detector=RTDETRConfig(
                        model_id=candidate_model,
                        device=device,
                        dtype=dtype,
                        local_files_only=local_files,
                    ),
                    tracker=ByteTrackConfig(),
                    sink=NullSinkConfig(),
                    update_frame_id=reconfiguration_trigger_frame_offset,
                    total_frames_to_process=measurement_source_frames + warmup_completed_frames + 50,
                )

                init_backend: DetectorBackend | None = None
                cand_backend: DetectorBackend | None = None
                trk_backend: TrackerBackend | None = None

                if use_fake_backends or mode is ExecutionMode.SMOKE:
                    init_backend = FakeDetectorBackend(model_id=initial_model)
                    cand_backend = FakeDetectorBackend(model_id=candidate_model)
                    trk_backend = FakeTrackerBackend(instance_id=f"fake_trk_rep_{rep}")

                memory_sampler: CUDAMemorySamplerProtocol
                if device.startswith("cuda"):
                    memory_sampler = PyTorchCUDAMemorySampler(device=device)
                    memory_sampler.initialize()
                else:
                    memory_sampler = FakeCUDAMemorySampler(is_cuda=False)

                # Reset peaks and synchronize at the start of each sub-run to get repetition-local peak stats
                memory_sampler.synchronize()
                memory_sampler.reset_peak_stats()

                CoexistenceTracker.reset()
                CoexistenceTracker.plan_created(1) # initial version 1

                rep_frame_rows: list[RealworldVideoFrameSampleRow] = []
                seen_frame_ids: set[int] = set()
                dup_count: int = 0
                frame_tuples: list[tuple[int, TerminalStatus, DropReason, bool, bool]] = []
                telemetry_lock = threading.Lock()

                def on_drop_callback(packet: FramePacket, reason: DropReason) -> None:
                    status = (
                        TerminalStatus.DROPPED_INGRESS_OVERFLOW
                        if reason is DropReason.INGRESS_OVERFLOW
                        else TerminalStatus.CANCELLED_ON_STOP
                    )
                    drop_ts = packet.drop_decision_timestamp_ns or time.monotonic_ns()
                    drop_ts = max(drop_ts, packet.enqueue_decision_timestamp_ns or 0)
                    fr_row = RealworldVideoFrameSampleRow(
                        run_id=run_id,
                        repetition=rep,
                        mechanism=mech,
                        frame_id=packet.frame_id,
                        source_timestamp_ns=packet.source_timestamp_ns,
                        admission_timestamp_ns=None,
                        completion_timestamp_ns=None,
                        plan_version=None,
                        detector_id="",
                        tracker_instance_id="",
                        queue_occupancy_before_enqueue=packet.queue_occupancy_before_enqueue or 0,
                        queue_occupancy_after_enqueue=packet.queue_occupancy_after_enqueue or 0,
                        queue_capacity=packet.queue_capacity or queue_capacity,
                        terminal_status=status.value,
                        drop_reason=reason.value,
                        dropped=True,
                        duplicated=False,
                        inside_measurement_window=packet.inside_measurement_window,
                        media_pts_ns=packet.media_pts_ns,
                        receiver_ingress_timestamp_ns=packet.receiver_ingress_timestamp_ns,
                        enqueue_decision_timestamp_ns=packet.enqueue_decision_timestamp_ns,
                        drop_decision_timestamp_ns=drop_ts,
                        media_frame_index=packet.media_frame_index,
                    )
                    with telemetry_lock:
                        rep_frame_rows.append(fr_row)
                        if packet.inside_measurement_window:
                            frame_tuples.append(
                                (packet.frame_id, status, reason, False, False)
                            )

                source_obj: FrameSource
                source_path_val: str
                if mode is ExecutionMode.PUBLICATION:
                    source_obj = RTSPVideoSource(
                        rtsp_url=publisher.rtsp_url,
                        video_path=video_path,
                        queue_capacity=queue_capacity,
                        on_drop_callback=on_drop_callback,
                    )
                    source_path_val = publisher.rtsp_url
                else:
                    source_obj = FileVideoSource(cfg.source)
                    source_path_val = video_path

                def sink_observer(batch: TrackBatch, completion_ns: int) -> None:
                    nonlocal dup_count
                    is_dup = batch.frame_id in seen_frame_ids
                    if is_dup:
                        with telemetry_lock:
                            if batch.inside_measurement_window:
                                dup_count += 1
                    seen_frame_ids.add(batch.frame_id)

                    plan_ver = batch.plan_version if batch.plan_version is not None else 1
                    admission_ns = batch.admission_timestamp_ns if batch.admission_timestamp_ns is not None and batch.admission_timestamp_ns > 0 else None
                    comp_ns = max(completion_ns, admission_ns or 0) if admission_ns is not None else completion_ns

                    if (
                        batch.plan_version is not None
                        and batch.plan_version > 1
                        and batch.detector_id == candidate_model
                        and batch.inside_measurement_window
                    ):
                        if hasattr(source_obj, "mark_first_candidate_output_completed"):
                            getattr(source_obj, "mark_first_candidate_output_completed")()

                    fr_row = RealworldVideoFrameSampleRow(
                        run_id=run_id,
                        repetition=rep,
                        mechanism=mech,
                        frame_id=batch.frame_id,
                        source_timestamp_ns=batch.source_timestamp_ns,
                        admission_timestamp_ns=admission_ns,
                        completion_timestamp_ns=comp_ns,
                        plan_version=plan_ver,
                        detector_id=batch.detector_id,
                        tracker_instance_id=batch.tracker_instance_id,
                        queue_occupancy_before_enqueue=batch.queue_occupancy_before_enqueue or 0,
                        queue_occupancy_after_enqueue=batch.queue_occupancy_after_enqueue or 0,
                        queue_capacity=batch.queue_capacity or queue_capacity,
                        terminal_status=TerminalStatus.COMPLETED.value,
                        drop_reason=DropReason.NONE.value,
                        dropped=False,
                        duplicated=is_dup,
                        inside_measurement_window=batch.inside_measurement_window,
                        media_pts_ns=batch.media_pts_ns,
                        receiver_ingress_timestamp_ns=batch.receiver_ingress_timestamp_ns,
                        enqueue_decision_timestamp_ns=batch.enqueue_decision_timestamp_ns,
                        drop_decision_timestamp_ns=batch.drop_decision_timestamp_ns,
                        media_frame_index=batch.media_frame_index,
                    )
                    with telemetry_lock:
                        rep_frame_rows.append(fr_row)
                        if batch.inside_measurement_window:
                            frame_tuples.append(
                                (batch.frame_id, TerminalStatus.COMPLETED, DropReason.NONE, True, is_dup)
                            )

                sink_obj = NullSink(cfg.sink, observer=sink_observer)

                app = VideoAnalyticsApplication(
                    config=cfg,
                    source=source_obj,
                    sink=sink_obj,
                    initial_detector_backend=init_backend,
                    candidate_detector_backend=cand_backend,
                    tracker_backend=trk_backend,
                    execution_mode=mode,
                )

                try:
                    meta = app.open()
                    if app.tracker_backend is not None:
                        trk_inst_before = app.tracker_backend.instance_id
                        reset_before = app.tracker_backend.reset_count
                    else:
                        trk_inst_before = "unknown"
                        reset_before = 0

                    # 3. Warm up the initial detector outside the measured window.
                    #    Feed non-measured frames through the pipeline until the
                    #    initial detector has processed warmup_completed_frames
                    #    outputs.  Candidate detector must NOT be loaded yet.
                    warmup_completed = 0
                    warmup_admitted = 0
                    from nedo_vision_dag_engine.instrumentation import FrameStatus
                    while warmup_completed < warmup_completed_frames:
                        now_ns = time.monotonic_ns()
                        app_controller_warmup: Any = getattr(app, "_controller")
                        res = app_controller_warmup.admit_frame(
                            admitted_at_ns=now_ns,
                            frame_id=warmup_admitted + 1,
                        )
                        warmup_admitted += 1
                        if res.status is FrameStatus.COMPLETED:
                            warmup_completed += 1

                    # Capture the media frame index of the next frame that will
                    # be read — this is our deterministic measurement origin.
                    _measurement_start_media_frame_index: int | None = None
                    _measurement_start_media_pts_ns: int | None = None
                    _measurement_end_media_frame_index: int | None = None
                    _measurement_end_media_pts_ns: int | None = None
                    _request_trigger_media_frame_index: int | None = None
                    _request_trigger_media_pts_ns: int | None = None
                    _measurement_start_source_sequence: int | None = None

                    # 4. Start phase-normalized measurement window.
                    getattr(source_obj, "start_measurement_window")(measurement_source_frames)
                    # Configure phase targets so the source uses the correct
                    # pre-request / post-effect frame counts for this run.
                    if hasattr(source_obj, "_pre_request_target"):
                        source_obj._pre_request_target = reconfiguration_trigger_frame_offset  # type: ignore[attr-defined]
                    if hasattr(source_obj, "_post_effect_target"):
                        source_obj._post_effect_target = reconfiguration_trigger_frame_offset  # type: ignore[attr-defined]
                    measurement_start_timestamp_ns = getattr(source_obj, "measurement_start_timestamp_ns", None)

                    # Telemetry snapshots & states
                    gpu_before_snap = CUDAMemorySnapshot(None, None, None, None)
                    gpu_coexist_snap = CUDAMemorySnapshot(None, None, None, None)
                    gpu_pub_snap = CUDAMemorySnapshot(None, None, None, None)
                    gpu_ret_snap = CUDAMemorySnapshot(None, None, None, None)

                    peak_live_plan_count = 1
                    peak_live_detector_instance_count = 1
                    live_plan_count_after_publication = 1
                    live_plan_count_after_retirement = 1
                    live_detector_count_after_publication = 1
                    live_detector_count_after_retirement = 1

                    active_detector_id_after_retirement = ""
                    active_plan_version_after_retirement = None
                    active_detector_instance_count_after_retirement = 1

                    measured_admitted = 0
                    swap_requested = False
                    t_req_swap = None
                    t_prep_start = None
                    t_prep_end = None
                    t_pub = None

                    typed_mech = cast(Literal["VEPS", "Pause", "Stop"], mech)

                    # Measured window loop
                    while True:
                        if getattr(source_obj, "measurement_stopped", False):
                            break

                        if not swap_requested and getattr(source_obj, "measurement_source_frames_received") >= reconfiguration_trigger_frame_offset:
                            swap_requested = True

                            # Before-prep snapshot
                            memory_sampler.synchronize()
                            gpu_before_snap = memory_sampler.sample()

                            # Initialize candidate if not already
                            if cand_backend is None:
                                from usecases.video_analytics.backends.rtdetr import RTDETRDetectorBackend
                                cand_backend = RTDETRDetectorBackend(cfg.candidate_detector)
                                setattr(app, "_candidate_detector_backend", cand_backend)

                            if typed_mech == "Stop":
                                t_req_swap = time.monotonic_ns()
                                t_prep_start = time.monotonic_ns()
                                
                                # Warm up the candidate detector instance
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

                                # Coexistence snapshot
                                memory_sampler.synchronize()
                                gpu_coexist_snap = memory_sampler.sample()
                                peak_live_plan_count = max(peak_live_plan_count, CoexistenceTracker.get_live_plan_count())
                                peak_live_detector_instance_count = max(peak_live_detector_instance_count, CoexistenceTracker.get_live_detector_count())

                                app_executor: Any = getattr(app, "_executor")
                                app_controller: Any = getattr(app, "_controller")
                                old_plan = app_executor.active_plan
                                app_controller.close()
                                CoexistenceTracker.plan_retired(old_plan.version)

                                candidate_spec = build_video_analytics_specification(
                                    detector_config=cfg.candidate_detector,
                                    tracker_config=cfg.tracker,
                                    sink_config=cfg.sink,
                                )
                                app_tracker_backend = cast(TrackerBackend, getattr(app, "_tracker_backend"))
                                app_sink = getattr(app, "_sink")
                                candidate_registry = build_video_analytics_registry(
                                    detector_backend=cand_backend,
                                    tracker_backend=app_tracker_backend,
                                    sink_instance=app_sink,
                                    source_instance=source_obj,
                                    detector_config=cfg.candidate_detector,
                                    tracker_config=cfg.tracker,
                                )
                                app_compiler: Any = getattr(app, "_compiler")
                                cand_res = app_compiler.compile(
                                    candidate_spec,
                                    registry=candidate_registry,
                                    previous_plan=old_plan,
                                )
                                if isinstance(cand_res, CompilationFailure):
                                    raise RuntimeError(f"Stop-rebuild compilation failed: {cand_res.reason}")

                                compiled_plan = cand_res.plan
                                CoexistenceTracker.plan_created(compiled_plan.version)

                                setattr(app, "_current_plan_version", compiled_plan.version)
                                app_executor = PipelineExecutor(initial_plan=compiled_plan)
                                setattr(app, "_executor", app_executor)
                                app_controller = ReconfigurationController(
                                    executor=app_executor,
                                    compiler=app_compiler,
                                    registry=candidate_registry,
                                )
                                setattr(app, "_controller", app_controller)
                                t_pub = time.monotonic_ns()

                                # After-publication snapshot
                                memory_sampler.synchronize()
                                gpu_pub_snap = memory_sampler.sample()
                                live_plan_count_after_publication = CoexistenceTracker.get_live_plan_count()
                                live_detector_count_after_publication = CoexistenceTracker.get_live_detector_count()

                                # Now retire superseded processors!
                                from nedo_vision_dag_engine.lifecycle import retire_superseded_processors
                                retire_superseded_processors(old_plan, compiled_plan)

                                # Stop retires old plan immediately
                                memory_sampler.synchronize()
                                gpu_ret_snap = memory_sampler.sample()
                                live_plan_count_after_retirement = CoexistenceTracker.get_live_plan_count()
                                live_detector_count_after_retirement = CoexistenceTracker.get_live_detector_count()

                                for step in app_executor.active_plan.steps:
                                    if hasattr(step.processor_ref, "backend"):
                                        active_detector_id_after_retirement = step.processor_ref.backend.model_id
                                        break
                                active_plan_version_after_retirement = app_executor.active_plan.version
                                active_detector_instance_count_after_retirement = CoexistenceTracker.get_live_detector_count()

                                if getattr(cand_backend, "is_closed", False):
                                    raise RuntimeError("Candidate detector is closed after retirement snapshot")

                            else: # Pause or VEPS
                                candidate_spec = build_video_analytics_specification(
                                    detector_config=cfg.candidate_detector,
                                    tracker_config=cfg.tracker,
                                    sink_config=cfg.sink,
                                )
                                app_tracker_backend = cast(TrackerBackend, getattr(app, "_tracker_backend"))
                                app_sink = getattr(app, "_sink")
                                candidate_registry = build_video_analytics_registry(
                                    detector_backend=cand_backend,
                                    tracker_backend=app_tracker_backend,
                                    sink_instance=app_sink,
                                    source_instance=source_obj,
                                    detector_config=cfg.candidate_detector,
                                    tracker_config=cfg.tracker,
                                )
                                app_controller: Any = getattr(app, "_controller")
                                app_executor: Any = getattr(app, "_executor")
                                app_controller.update_registry(candidate_registry)

                                t_req = time.monotonic_ns()
                                t_req_swap = t_req

                                req_id = f"swap_{typed_mech.lower()}_{t_req}"
                                reconfig_req = ReconfigurationRequest(
                                    request_id=req_id,
                                    base_version=app_executor.active_plan.version,
                                    target_specification=candidate_spec,
                                    state_directive=StateDirective(),
                                    submitted_at_ns=t_req,
                                )
                                app_controller.submit(reconfig_req)

                                if typed_mech == "VEPS":
                                    while True:
                                        rec: Any = app_controller.record(req_id)
                                        if rec.status in (
                                            ReconfigurationStatus.READY,
                                            ReconfigurationStatus.COMMITTED,
                                            ReconfigurationStatus.REJECTED,
                                            ReconfigurationStatus.FAILED,
                                            ReconfigurationStatus.ABORTED,
                                        ):
                                            break

                                        if getattr(source_obj, "measurement_stopped", False):
                                            break

                                        if getattr(source_obj, "measurement_source_frames_received") >= measurement_source_frames:
                                            break

                                        now_ns = time.monotonic_ns()
                                        res = app_controller.admit_frame(
                                            admitted_at_ns=now_ns,
                                            frame_id=warmup_admitted + measured_admitted + 1,
                                        )
                                        if res.status is not FrameStatus.COMPLETED:
                                            print(f"DEBUG VEPS: frame_id={warmup_admitted + measured_admitted + 1} failed: status={res.status}, error={res.error}")
                                            break
                                        measured_admitted += 1
                                        time.sleep(0.001)

                                rec_ready: Any = app_controller.wait_for_status(
                                    req_id,
                                    statuses=frozenset({ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED}),
                                    timeout_seconds=60.0,
                                )
                                if rec_ready is None or rec_ready.status not in (ReconfigurationStatus.READY, ReconfigurationStatus.COMMITTED):
                                    raise RuntimeError(f"{typed_mech} candidate preparation failed")

                                t_prep_start = rec_ready.preparation_started_ns or t_req
                                t_prep_end = rec_ready.preparation_completed_ns or rec_ready.ready_ns or t_prep_start

                                CoexistenceTracker.plan_created(rec_ready.candidate_version or 2)

                                # Coexistence snapshot
                                memory_sampler.synchronize()
                                gpu_coexist_snap = memory_sampler.sample()
                                peak_live_plan_count = max(peak_live_plan_count, CoexistenceTracker.get_live_plan_count())
                                peak_live_detector_instance_count = max(peak_live_detector_instance_count, CoexistenceTracker.get_live_detector_count())

                                if rec_ready.status is ReconfigurationStatus.READY:
                                    app_controller.commit_ready()

                                rec_committed: Any = app_controller.record(req_id)
                                t_pub = rec_committed.commit_ns or rec_committed.ready_ns or t_prep_end
                                setattr(app, "_current_plan_version", rec_committed.candidate_version or 2)

                                # After publication snapshot
                                memory_sampler.synchronize()
                                gpu_pub_snap = memory_sampler.sample()
                                live_plan_count_after_publication = CoexistenceTracker.get_live_plan_count()
                                live_detector_count_after_publication = CoexistenceTracker.get_live_detector_count()

                                ret_rec: Any = app_controller.wait_for_retirement(req_id, timeout_seconds=drain_timeout_seconds)
                                if ret_rec is not None:
                                    CoexistenceTracker.plan_retired(ret_rec.base_version)

                                # After retirement snapshot
                                memory_sampler.synchronize()
                                gpu_ret_snap = memory_sampler.sample()
                                live_plan_count_after_retirement = CoexistenceTracker.get_live_plan_count()
                                live_detector_count_after_retirement = CoexistenceTracker.get_live_detector_count()

                                for step in app_executor.active_plan.steps:
                                    if hasattr(step.processor_ref, "backend"):
                                        active_detector_id_after_retirement = step.processor_ref.backend.model_id
                                        break
                                active_plan_version_after_retirement = app_executor.active_plan.version
                                active_detector_instance_count_after_retirement = CoexistenceTracker.get_live_detector_count()

                                if getattr(cand_backend, "is_closed", False):
                                    raise RuntimeError("Candidate detector is closed after retirement snapshot")

                            peak_live_plan_count = max(peak_live_plan_count, CoexistenceTracker.get_peak_plan_count())
                            peak_live_detector_instance_count = max(peak_live_detector_instance_count, CoexistenceTracker.get_peak_detector_count())
                            continue

                        now_ns = time.monotonic_ns()
                        app_controller: Any = getattr(app, "_controller")
                        res = app_controller.admit_frame(
                            admitted_at_ns=now_ns,
                            frame_id=warmup_admitted + measured_admitted + 1,
                        )
                        measured_admitted += 1

                    # 8. Perform a bounded drain
                    drain_start_timestamp_ns = time.monotonic_ns()
                    if isinstance(source_obj, RTSPVideoSource):
                        while True:
                            if source_obj.ingress is not None and source_obj.ingress.queue.size() == 0:
                                break
                            if time.monotonic_ns() - drain_start_timestamp_ns >= drain_timeout_seconds * 1e9:
                                break

                            now_ns = time.monotonic_ns()
                            app_controller: Any = getattr(app, "_controller")
                            res = app_controller.admit_frame(
                                admitted_at_ns=now_ns,
                                frame_id=warmup_admitted + measured_admitted + 1,
                            )
                            measured_admitted += 1

                    drain_end_timestamp_ns = time.monotonic_ns()
                    measurement_end_timestamp_ns = getattr(source_obj, "measurement_end_timestamp_ns", None) or drain_start_timestamp_ns

                    # Clear remaining frames in the queue
                    if isinstance(source_obj, RTSPVideoSource) and source_obj.ingress is not None:
                        remaining_frames = source_obj.ingress.queue.clear_and_cancel_all()
                        for pkt in remaining_frames:
                            fr_row = RealworldVideoFrameSampleRow(
                                run_id=run_id,
                                repetition=rep,
                                mechanism=mech,
                                frame_id=pkt.frame_id,
                                source_timestamp_ns=pkt.source_timestamp_ns,
                                admission_timestamp_ns=None,
                                completion_timestamp_ns=None,
                                plan_version=None,
                                detector_id="",
                                tracker_instance_id="",
                                queue_occupancy_before_enqueue=pkt.queue_occupancy_before_enqueue or 0,
                                queue_occupancy_after_enqueue=pkt.queue_occupancy_after_enqueue or 0,
                                queue_capacity=pkt.queue_capacity or queue_capacity,
                                terminal_status=TerminalStatus.IN_FLIGHT_AT_WINDOW_END.value,
                                drop_reason=DropReason.NONE.value,
                                dropped=False,
                                duplicated=False,
                                inside_measurement_window=True,
                                media_pts_ns=pkt.media_pts_ns,
                                receiver_ingress_timestamp_ns=pkt.receiver_ingress_timestamp_ns,
                                enqueue_decision_timestamp_ns=pkt.enqueue_decision_timestamp_ns,
                                drop_decision_timestamp_ns=pkt.drop_decision_timestamp_ns,
                                media_frame_index=pkt.media_frame_index,
                            )
                            with telemetry_lock:
                                rep_frame_rows.append(fr_row)
                                frame_tuples.append(
                                    (pkt.frame_id, TerminalStatus.IN_FLIGHT_AT_WINDOW_END, DropReason.NONE, False, False)
                                )

                    if app.tracker_backend is not None:
                        trk_inst_after = app.tracker_backend.instance_id
                        reset_after = app.tracker_backend.reset_count
                    else:
                        trk_inst_after = "unknown"
                        reset_after = 0

                    app.close()

                    # Find first candidate output frame
                    first_cand_row = next(
                        (r for r in rep_frame_rows if r.plan_version is not None and r.plan_version > 1 and r.detector_id == candidate_model and r.inside_measurement_window),
                        None,
                    )
                    first_candidate_output_ns = None
                    if first_cand_row is not None and first_cand_row.completion_timestamp_ns is not None:
                        first_candidate_output_ns = first_cand_row.completion_timestamp_ns

                    if first_candidate_output_ns is None:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No candidate output reached the sink."
                        )

                    # ----- Measurement media alignment ---------------------------------
                    # Derive alignment anchors from measured frame records (media time).
                    measured_frames_sorted = sorted(
                        [r for r in rep_frame_rows if r.inside_measurement_window],
                        key=lambda r: r.frame_id,
                    )
                    if measured_frames_sorted:
                        _measurement_start_media_frame_index = measured_frames_sorted[0].media_frame_index
                        _measurement_start_media_pts_ns = measured_frames_sorted[0].media_pts_ns
                        _measurement_end_media_frame_index = measured_frames_sorted[-1].media_frame_index
                        _measurement_end_media_pts_ns = measured_frames_sorted[-1].media_pts_ns
                        _measurement_start_source_sequence = 1
                        # Request trigger aligns to the 60th measured source frame.
                        if len(measured_frames_sorted) >= 60:
                            _request_trigger_media_frame_index = measured_frames_sorted[59].media_frame_index
                            _request_trigger_media_pts_ns = measured_frames_sorted[59].media_pts_ns
                    # ------------------------------------------------------------------

                    # Find last old-plan output occurring before first candidate output
                    old_rows_before_cand = [
                        r for r in rep_frame_rows if r.completion_timestamp_ns is not None and r.completion_timestamp_ns < first_candidate_output_ns and r.inside_measurement_window
                    ]
                    if not old_rows_before_cand:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No old-plan output reached the sink before first candidate output."
                        )
                    last_old_row = max(old_rows_before_cand, key=lambda r: r.completion_timestamp_ns or 0)
                    last_old_output_ns = last_old_row.completion_timestamp_ns

                    # Derive transition metrics
                    request_to_effect_ns = None
                    if t_req_swap is not None:
                        request_to_effect_ns = first_candidate_output_ns - t_req_swap

                    transition_output_gap_ns = None
                    if last_old_output_ns is not None:
                        transition_output_gap_ns = first_candidate_output_ns - last_old_output_ns

                    # Count old plan frames completed during prep
                    old_prep_frames = 0
                    if t_prep_start is not None and t_prep_end is not None:
                        old_prep_frames = sum(
                            1
                            for r in rep_frame_rows
                            if r.plan_version == 1
                            and r.completion_timestamp_ns is not None
                            and t_prep_start <= r.completion_timestamp_ns <= t_prep_end
                            and r.inside_measurement_window
                        )

                    # Validate timestamp ordering
                    if t_req_swap is not None and t_prep_start is not None and t_prep_end is not None and t_pub is not None:
                        if not (t_req_swap <= t_prep_start <= t_prep_end <= t_pub <= first_candidate_output_ns):
                            raise RuntimeError(
                                f"Repetition {rep} mechanism {mech}: Timestamp ordering violation: "
                                f"req={t_req_swap}, prep_start={t_prep_start}, prep_end={t_prep_end}, "
                                f"pub={t_pub}, first_cand_output={first_candidate_output_ns}"
                            )

                    # Calculate Flow Accounting Metrics
                    flow_summary = calculate_flow_metrics(frame_tuples)

                    # Compute phase drop counts
                    d_before, d_prep, d_between_prep_pub, d_between_pub_first, d_after = classify_drops(
                        rep_frame_rows,
                        t_req_swap,
                        t_prep_end,
                        t_pub,
                        first_candidate_output_ns,
                    )
                    sum_phase_drops = d_before + d_prep + d_between_prep_pub + d_between_pub_first + d_after
                    if sum_phase_drops != flow_summary.total_dropped_frame_count:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: Sum of phase-specific drops ({sum_phase_drops}) "
                            f"does not equal total dropped count ({flow_summary.total_dropped_frame_count})"
                        )

                    # Compute phase source-frame counts
                    s_before, s_prep, s_prep_pub, s_pub_first, s_after = classify_source_frames(
                        rep_frame_rows,
                        t_req_swap,
                        t_prep_end,
                        t_pub,
                        first_candidate_output_ns,
                    )
                    total_measured = getattr(source_obj, "measurement_source_frames_received")
                    pre_req_rec = getattr(source_obj, "pre_request_source_frames_received")
                    trans_rec = getattr(source_obj, "transition_source_frames_received")
                    post_eff_rec = getattr(source_obj, "post_effect_source_frames_received")

                    # Phase-normalized drop rates (None when denominator is zero).
                    def _rate(num: int, den: int) -> float | None:
                        if den == 0:
                            return None
                        return num / den

                    drop_rate_before = _rate(d_before, pre_req_rec)
                    drop_rate_prep = _rate(d_prep, s_prep)
                    drop_rate_prep_pub = _rate(d_between_prep_pub, s_prep_pub)
                    drop_rate_pub_first = _rate(d_between_pub_first, s_pub_first)
                    drop_rate_after = _rate(d_after, s_after)

                    # GPU transition maximum (peak across all sampled checkpoints).
                    _gpu_alloc_vals = [
                        gpu_before_snap.allocated_bytes,
                        gpu_coexist_snap.allocated_bytes,
                        gpu_pub_snap.allocated_bytes,
                        gpu_ret_snap.allocated_bytes,
                    ]
                    _gpu_resv_vals = [
                        gpu_before_snap.reserved_bytes,
                        gpu_coexist_snap.reserved_bytes,
                        gpu_pub_snap.reserved_bytes,
                        gpu_ret_snap.reserved_bytes,
                    ]
                    gpu_trans_max_alloc = max((v for v in _gpu_alloc_vals if v is not None), default=None)
                    gpu_trans_max_resv = max((v for v in _gpu_resv_vals if v is not None), default=None)

                    sample_row = RealworldVideoSampleRow(
                        run_id=run_id,
                        repetition=rep,
                        execution_order_position=pos,
                        mechanism=mech,
                        source_path=source_path_val,
                        source_file_hash=source_hash,
                        video_fps=meta.fps,
                        resolution=f"{meta.width}x{meta.height}",
                        initial_model_id=initial_model,
                        candidate_model_id=candidate_model,
                        device=device,
                        dtype=dtype,
                        confidence_threshold=cfg.initial_detector.confidence_threshold,
                        tracker_config="bytetrack_default",
                        request_timestamp_ns=t_req_swap,
                        candidate_prep_start_ns=t_prep_start,
                        candidate_prep_end_ns=t_prep_end,
                        publication_timestamp_ns=t_pub,
                        first_candidate_output_ns=first_candidate_output_ns,
                        request_to_effect_ns=request_to_effect_ns,
                        transition_output_gap_ns=transition_output_gap_ns,
                        old_plan_frames_completed_during_prep=old_prep_frames,
                        source_frames_received=flow_summary.source_frames_received,
                        frames_admitted=flow_summary.frames_admitted,
                        frames_completed=flow_summary.frames_completed,
                        ingress_overflow_drop_count=flow_summary.ingress_overflow_drop_count,
                        admission_rejection_count=flow_summary.admission_rejection_count,
                        execution_cancelled_count=flow_summary.execution_cancelled_count,
                        frames_in_flight_at_window_end=flow_summary.frames_in_flight_at_window_end,
                        total_dropped_frame_count=flow_summary.total_dropped_frame_count,
                        drop_rate=flow_summary.drop_rate,
                        dropped_frame_count=flow_summary.total_dropped_frame_count,
                        duplicated_frame_count=dup_count,
                        tracker_instance_id_before=trk_inst_before,
                        tracker_instance_id_after=trk_inst_after,
                        tracker_reset_count_before=reset_before,
                        tracker_reset_count_after=reset_after,
                        processor_peak_count=5,
                        gpu_before_snap=gpu_before_snap,
                        gpu_coexist_snap=gpu_coexist_snap,
                        gpu_pub_snap=gpu_pub_snap,
                        gpu_ret_snap=gpu_ret_snap,
                        warmup_completed_frames=warmup_completed_frames,
                        measurement_source_frame_target=measurement_source_frames,
                        reconfiguration_trigger_frame_offset=reconfiguration_trigger_frame_offset,
                        measurement_source_frames_received=getattr(source_obj, "measurement_source_frames_received"),
                        measurement_start_timestamp_ns=measurement_start_timestamp_ns,
                        measurement_end_timestamp_ns=measurement_end_timestamp_ns,
                        drain_start_timestamp_ns=drain_start_timestamp_ns,
                        drain_end_timestamp_ns=drain_end_timestamp_ns,
                        drops_before_request=d_before,
                        drops_during_candidate_preparation=d_prep,
                        drops_between_preparation_and_publication=d_between_prep_pub,
                        drops_between_publication_and_first_candidate_output=d_between_pub_first,
                        drops_after_first_candidate_output=d_after,
                        active_detector_id_after_retirement=active_detector_id_after_retirement,
                        active_plan_version_after_retirement=active_plan_version_after_retirement,
                        active_detector_instance_count_after_retirement=active_detector_instance_count_after_retirement,
                        peak_live_plan_count=peak_live_plan_count,
                        peak_live_detector_instance_count=peak_live_detector_instance_count,
                        live_plan_count_after_publication=live_plan_count_after_publication,
                        live_plan_count_after_retirement=live_plan_count_after_retirement,
                        live_detector_count_after_publication=live_detector_count_after_publication,
                        live_detector_count_after_retirement=live_detector_count_after_retirement,
                        measurement_start_media_frame_index=_measurement_start_media_frame_index,
                        measurement_start_media_pts_ns=_measurement_start_media_pts_ns,
                        measurement_end_media_frame_index=_measurement_end_media_frame_index,
                        measurement_end_media_pts_ns=_measurement_end_media_pts_ns,
                        request_trigger_media_frame_index=_request_trigger_media_frame_index,
                        request_trigger_media_pts_ns=_request_trigger_media_pts_ns,
                        measurement_start_source_sequence=_measurement_start_source_sequence,
                        measurement_end_source_sequence=getattr(source_obj, "measurement_source_frames_received"),
                        pre_request_source_frame_target=reconfiguration_trigger_frame_offset,
                        pre_request_source_frames_received=pre_req_rec,
                        transition_source_frames_received=trans_rec,
                        post_effect_source_frame_target=reconfiguration_trigger_frame_offset,
                        post_effect_source_frames_received=post_eff_rec,
                        total_measurement_source_frames_received=total_measured,
                        source_frames_before_request=s_before,
                        source_frames_during_candidate_preparation=s_prep,
                        source_frames_between_preparation_and_publication=s_prep_pub,
                        source_frames_between_publication_and_first_candidate_output=s_pub_first,
                        source_frames_after_first_candidate_output=s_after,
                        drop_rate_before_request=drop_rate_before,
                        drop_rate_during_candidate_preparation=drop_rate_prep,
                        drop_rate_between_preparation_and_publication=drop_rate_prep_pub,
                        drop_rate_between_publication_and_first_candidate_output=drop_rate_pub_first,
                        drop_rate_after_first_candidate_output=drop_rate_after,
                        gpu_memory_transition_max_allocated_bytes=gpu_trans_max_alloc,
                        gpu_memory_transition_max_reserved_bytes=gpu_trans_max_resv,
                    )
                    if sample_row.request_trigger_media_frame_index is not None:
                        fw = compute_fixed_window_metrics(
                            frame_rows=rep_frame_rows,
                            request_trigger_media_frame_index=sample_row.request_trigger_media_frame_index,
                        )
                        sample_row = replace(
                            sample_row,
                            fixed_window_start_media_frame_index=int(fw["fixed_window_start_media_frame_index"]),  # type: ignore[arg-type]
                            fixed_window_end_media_frame_index=int(fw["fixed_window_end_media_frame_index"]),  # type: ignore[arg-type]
                            fixed_window_expected_source_frames=int(fw["fixed_window_expected_source_frames"]),  # type: ignore[arg-type]
                            fixed_window_observed_source_frames=int(fw["fixed_window_observed_source_frames"]),  # type: ignore[arg-type]
                            fixed_window_admitted=int(fw["fixed_window_admitted"]),  # type: ignore[arg-type]
                            fixed_window_completed=int(fw["fixed_window_completed"]),  # type: ignore[arg-type]
                            fixed_window_dropped=int(fw["fixed_window_dropped"]),  # type: ignore[arg-type]
                            fixed_window_duplicated=int(fw["fixed_window_duplicated"]),  # type: ignore[arg-type]
                            fixed_window_old_plan_completions=int(fw["fixed_window_old_plan_completions"]),  # type: ignore[arg-type]
                            fixed_window_new_plan_completions=int(fw["fixed_window_new_plan_completions"]),  # type: ignore[arg-type]
                            fixed_window_accounting_residual=int(fw["fixed_window_accounting_residual"]),  # type: ignore[arg-type]
                            fixed_window_accounting_valid=bool(fw["fixed_window_accounting_valid"]),  # type: ignore[arg-type]
                        )
                    sample_rows.append(sample_row)
                    frame_rows.extend(rep_frame_rows)

                finally:
                    publisher.stop()
                    cleanup_repetition_resources()

    finally:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REALWORLD_VIDEO_HEADERS)
            for r in sample_rows:
                gb = r.gpu_before_snap
                gc = r.gpu_coexist_snap
                gp = r.gpu_pub_snap
                gr = r.gpu_ret_snap
                writer.writerow([
                    r.run_id,
                    r.repetition,
                    r.execution_order_position,
                    r.mechanism,
                    r.source_path,
                    r.source_file_hash,
                    f"{r.video_fps:.2f}",
                    r.resolution,
                    r.initial_model_id,
                    r.candidate_model_id,
                    r.device,
                    r.dtype,
                    f"{r.confidence_threshold:.2f}",
                    r.tracker_config,
                    r.request_timestamp_ns if r.request_timestamp_ns is not None else "",
                    r.candidate_prep_start_ns if r.candidate_prep_start_ns is not None else "",
                    r.candidate_prep_end_ns if r.candidate_prep_end_ns is not None else "",
                    r.publication_timestamp_ns if r.publication_timestamp_ns is not None else "",
                    r.first_candidate_output_ns if r.first_candidate_output_ns is not None else "",
                    r.request_to_effect_ns if r.request_to_effect_ns is not None else "",
                    r.transition_output_gap_ns if r.transition_output_gap_ns is not None else "",
                    r.old_plan_frames_completed_during_prep,
                    r.source_frames_received,
                    r.frames_admitted,
                    r.frames_completed,
                    r.ingress_overflow_drop_count,
                    r.admission_rejection_count,
                    r.execution_cancelled_count,
                    r.frames_in_flight_at_window_end,
                    r.total_dropped_frame_count,
                    f"{r.drop_rate:.6f}" if r.drop_rate is not None else "",
                    r.dropped_frame_count,
                    r.duplicated_frame_count,
                    r.tracker_instance_id_before,
                    r.tracker_instance_id_after,
                    r.tracker_reset_count_before,
                    r.tracker_reset_count_after,
                    r.processor_peak_count,
                    gb.allocated_bytes if gb.allocated_bytes is not None else "",
                    gb.reserved_bytes if gb.reserved_bytes is not None else "",
                    gb.peak_allocated_bytes if gb.peak_allocated_bytes is not None else "",
                    gb.peak_reserved_bytes if gb.peak_reserved_bytes is not None else "",
                    gc.allocated_bytes if gc.allocated_bytes is not None else "",
                    gc.reserved_bytes if gc.reserved_bytes is not None else "",
                    gc.peak_allocated_bytes if gc.peak_allocated_bytes is not None else "",
                    gc.peak_reserved_bytes if gc.peak_reserved_bytes is not None else "",
                    gp.allocated_bytes if gp.allocated_bytes is not None else "",
                    gp.reserved_bytes if gp.reserved_bytes is not None else "",
                    gp.peak_allocated_bytes if gp.peak_allocated_bytes is not None else "",
                    gp.peak_reserved_bytes if gp.peak_reserved_bytes is not None else "",
                    gr.allocated_bytes if gr.allocated_bytes is not None else "",
                    gr.reserved_bytes if gr.reserved_bytes is not None else "",
                    gr.peak_allocated_bytes if gr.peak_allocated_bytes is not None else "",
                    gr.peak_reserved_bytes if gr.peak_reserved_bytes is not None else "",
                    gb.allocated_bytes if gb.allocated_bytes is not None else "",
                    gc.allocated_bytes if gc.allocated_bytes is not None else "",
                    gp.allocated_bytes if gp.allocated_bytes is not None else "",
                    gr.allocated_bytes if gr.allocated_bytes is not None else "",
                    r.warmup_completed_frames,
                    r.measurement_source_frame_target,
                    r.reconfiguration_trigger_frame_offset,
                    r.measurement_source_frames_received,
                    r.measurement_start_timestamp_ns if r.measurement_start_timestamp_ns is not None else "",
                    r.measurement_end_timestamp_ns if r.measurement_end_timestamp_ns is not None else "",
                    r.drain_start_timestamp_ns if r.drain_start_timestamp_ns is not None else "",
                    r.drain_end_timestamp_ns if r.drain_end_timestamp_ns is not None else "",
                    r.drops_before_request,
                    r.drops_during_candidate_preparation,
                    r.drops_between_preparation_and_publication,
                    r.drops_between_publication_and_first_candidate_output,
                    r.drops_after_first_candidate_output,
                    r.active_detector_id_after_retirement,
                    r.active_plan_version_after_retirement if r.active_plan_version_after_retirement is not None else "",
                    r.active_detector_instance_count_after_retirement,
                    r.peak_live_plan_count,
                    r.peak_live_detector_instance_count,
                    r.live_plan_count_after_publication,
                    r.live_plan_count_after_retirement,
                    r.live_detector_count_after_publication,
                    r.live_detector_count_after_retirement,
                    r.measurement_start_media_frame_index if r.measurement_start_media_frame_index is not None else "",
                    r.measurement_start_media_pts_ns if r.measurement_start_media_pts_ns is not None else "",
                    r.measurement_end_media_frame_index if r.measurement_end_media_frame_index is not None else "",
                    r.measurement_end_media_pts_ns if r.measurement_end_media_pts_ns is not None else "",
                    r.request_trigger_media_frame_index if r.request_trigger_media_frame_index is not None else "",
                    r.request_trigger_media_pts_ns if r.request_trigger_media_pts_ns is not None else "",
                    r.measurement_start_source_sequence if r.measurement_start_source_sequence is not None else "",
                    r.measurement_end_source_sequence if r.measurement_end_source_sequence is not None else "",
                    r.pre_request_source_frame_target,
                    r.pre_request_source_frames_received,
                    r.transition_source_frames_received,
                    r.post_effect_source_frame_target,
                    r.post_effect_source_frames_received,
                    r.total_measurement_source_frames_received,
                    r.source_frames_before_request,
                    r.source_frames_during_candidate_preparation,
                    r.source_frames_between_preparation_and_publication,
                    r.source_frames_between_publication_and_first_candidate_output,
                    r.source_frames_after_first_candidate_output,
                    f"{r.drop_rate_before_request:.6f}" if r.drop_rate_before_request is not None else "",
                    f"{r.drop_rate_during_candidate_preparation:.6f}" if r.drop_rate_during_candidate_preparation is not None else "",
                    f"{r.drop_rate_between_preparation_and_publication:.6f}" if r.drop_rate_between_preparation_and_publication is not None else "",
                    f"{r.drop_rate_between_publication_and_first_candidate_output:.6f}" if r.drop_rate_between_publication_and_first_candidate_output is not None else "",
                    f"{r.drop_rate_after_first_candidate_output:.6f}" if r.drop_rate_after_first_candidate_output is not None else "",
                    r.gpu_memory_transition_max_allocated_bytes if r.gpu_memory_transition_max_allocated_bytes is not None else "",
                    r.gpu_memory_transition_max_reserved_bytes if r.gpu_memory_transition_max_reserved_bytes is not None else "",
                    r.fixed_window_start_media_frame_index,
                    r.fixed_window_end_media_frame_index,
                    r.fixed_window_expected_source_frames,
                    r.fixed_window_observed_source_frames,
                    r.fixed_window_admitted,
                    r.fixed_window_completed,
                    r.fixed_window_dropped,
                    r.fixed_window_duplicated,
                    r.fixed_window_old_plan_completions,
                    r.fixed_window_new_plan_completions,
                    r.fixed_window_accounting_residual,
                    r.fixed_window_accounting_valid,
                ])

        with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REALWORLD_VIDEO_FRAME_HEADERS)
            for fr in frame_rows:
                writer.writerow([
                    fr.run_id,
                    fr.repetition,
                    fr.mechanism,
                    fr.frame_id,
                    fr.source_timestamp_ns if fr.source_timestamp_ns is not None else "",
                    fr.admission_timestamp_ns if fr.admission_timestamp_ns is not None else "",
                    fr.completion_timestamp_ns if fr.completion_timestamp_ns is not None else "",
                    fr.plan_version if fr.plan_version is not None else "",
                    fr.detector_id,
                    fr.tracker_instance_id,
                    fr.queue_occupancy_before_enqueue,
                    fr.queue_occupancy_after_enqueue,
                    fr.queue_capacity,
                    fr.terminal_status,
                    fr.drop_reason,
                    1 if fr.dropped else 0,
                    1 if fr.duplicated else 0,
                    1 if fr.inside_measurement_window else 0,
                    fr.media_pts_ns if fr.media_pts_ns is not None else "",
                    fr.receiver_ingress_timestamp_ns if fr.receiver_ingress_timestamp_ns is not None else "",
                    fr.enqueue_decision_timestamp_ns if fr.enqueue_decision_timestamp_ns is not None else "",
                    fr.drop_decision_timestamp_ns if fr.drop_decision_timestamp_ns is not None else "",
                    fr.media_frame_index if fr.media_frame_index is not None else "",
                ])

    return sample_rows, frame_rows
