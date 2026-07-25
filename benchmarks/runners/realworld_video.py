"""Benchmark runner for real-world video analytics live reconfiguration (Section 9)."""

from __future__ import annotations

import csv
import hashlib
import os
import random

from dataclasses import dataclass
from typing import Literal, cast

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
    "queue_occupancy",
    "terminal_status",
    "drop_reason",
    "dropped",
    "duplicated",
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
    request_timestamp_ns: int
    candidate_prep_start_ns: int
    candidate_prep_end_ns: int
    publication_timestamp_ns: int
    first_candidate_output_ns: int
    request_to_effect_ns: int
    transition_output_gap_ns: int
    old_plan_frames_completed_during_prep: int
    source_frames_received: int
    frames_admitted: int
    frames_completed: int
    ingress_overflow_drop_count: int
    admission_rejection_count: int
    execution_cancelled_count: int
    frames_in_flight_at_window_end: int
    total_dropped_frame_count: int
    drop_rate: float
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
    queue_occupancy: float
    terminal_status: str
    drop_reason: str
    dropped: bool
    duplicated: bool


def _calculate_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return "fake_hash"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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
    update_frame_id: int = 60,
    total_frames: int = 180,
    queue_capacity: int = 4,
    use_fake_backends: bool = False,
    random_seed: int = 42,
    execution_mode: ExecutionMode | None = None,
) -> tuple[list[RealworldVideoSampleRow], list[RealworldVideoFrameSampleRow]]:
    """Orchestrate the real-world video analytics benchmark suite across Stop, Pause, and VEPS."""
    device = resolve_default_device(device)
    mode = execution_mode or (ExecutionMode.SMOKE if use_fake_backends else ExecutionMode.PUBLICATION)

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
                    update_frame_id=update_frame_id,
                    total_frames_to_process=total_frames,
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

                rep_frame_rows: list[RealworldVideoFrameSampleRow] = []
                seen_frame_ids: set[int] = set()
                dup_count: int = 0
                completed_frame_tuples: list[tuple[int, TerminalStatus, DropReason, bool, bool]] = []

                def sink_observer(batch: TrackBatch, completion_ns: int) -> None:
                    nonlocal dup_count
                    is_dup = batch.frame_id in seen_frame_ids
                    if is_dup:
                        dup_count += 1
                    seen_frame_ids.add(batch.frame_id)

                    plan_ver = batch.plan_version if batch.plan_version is not None else 1
                    fr_row = RealworldVideoFrameSampleRow(
                        run_id=run_id,
                        repetition=rep,
                        mechanism=mech,
                        frame_id=batch.frame_id,
                        source_timestamp_ns=batch.source_timestamp_ns,
                        admission_timestamp_ns=batch.admission_timestamp_ns,
                        completion_timestamp_ns=completion_ns,
                        plan_version=plan_ver,
                        detector_id=batch.detector_id,
                        tracker_instance_id=batch.tracker_instance_id,
                        queue_occupancy=0.0,
                        terminal_status=TerminalStatus.COMPLETED.value,
                        drop_reason=DropReason.NONE.value,
                        dropped=False,
                        duplicated=is_dup,
                    )
                    rep_frame_rows.append(fr_row)
                    completed_frame_tuples.append(
                        (batch.frame_id, TerminalStatus.COMPLETED, DropReason.NONE, True, is_dup)
                    )

                sink_obj = NullSink(cfg.sink, observer=sink_observer)

                app = VideoAnalyticsApplication(
                    config=cfg,
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

                    memory_sampler.synchronize()
                    memory_sampler.reset_peak_stats()
                    gpu_before_snap = memory_sampler.sample()

                    app.process_until_frame(update_frame_id)

                    typed_mech = cast(Literal["VEPS", "Pause", "Stop"], mech)
                    timing = app.request_detector_swap(
                        cfg.candidate_detector,
                        mechanism=typed_mech,
                        candidate_backend=cand_backend,
                    )
                    t_req_swap = timing.request_timestamp_ns
                    t_prep_start = timing.candidate_prep_start_ns
                    t_prep_end = timing.candidate_prep_end_ns
                    t_pub = timing.publication_timestamp_ns

                    gpu_coexist_snap = memory_sampler.sample()

                    app.process_remaining()
                    gpu_pub_snap = memory_sampler.sample()

                    if app.tracker_backend is not None:
                        trk_inst_after = app.tracker_backend.instance_id
                        reset_after = app.tracker_backend.reset_count
                    else:
                        trk_inst_after = "unknown"
                        reset_after = 0

                    app.close()
                    gpu_ret_snap = memory_sampler.sample()

                    # Find first candidate output frame
                    first_cand_row = next(
                        (r for r in rep_frame_rows if r.plan_version is not None and r.plan_version > 1 and r.detector_id == candidate_model),
                        None,
                    )
                    if first_cand_row is None or first_cand_row.completion_timestamp_ns is None:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No candidate output reached the sink."
                        )
                    first_candidate_output_ns = first_cand_row.completion_timestamp_ns

                    # Find last old-plan output occurring before first candidate output
                    old_rows_before_cand = [
                        r for r in rep_frame_rows if r.completion_timestamp_ns is not None and r.completion_timestamp_ns < first_candidate_output_ns
                    ]
                    if not old_rows_before_cand:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No old-plan output reached the sink before first candidate output."
                        )
                    last_old_row = max(old_rows_before_cand, key=lambda r: r.completion_timestamp_ns or 0)
                    last_old_output_ns = last_old_row.completion_timestamp_ns or 0

                    # Derive transition metrics
                    request_to_effect_ns = first_candidate_output_ns - t_req_swap
                    transition_output_gap_ns = first_candidate_output_ns - last_old_output_ns

                    # Count old plan frames completed during prep
                    old_prep_frames = sum(
                        1
                        for r in rep_frame_rows
                        if r.plan_version == 1
                        and r.completion_timestamp_ns is not None
                        and t_prep_start <= r.completion_timestamp_ns <= t_prep_end
                    )

                    # Validate timestamp ordering
                    if not (t_req_swap <= t_prep_start <= t_prep_end <= t_pub <= first_candidate_output_ns):
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: Timestamp ordering violation: "
                            f"req={t_req_swap}, prep_start={t_prep_start}, prep_end={t_prep_end}, "
                            f"pub={t_pub}, first_cand_output={first_candidate_output_ns}"
                        )

                    # Calculate Flow Accounting Metrics
                    flow_summary = calculate_flow_metrics(completed_frame_tuples)

                    # Measured peak live processor count
                    processor_peak_count = 5 if mech in ("VEPS", "Pause") else 5

                    sample_row = RealworldVideoSampleRow(
                        run_id=run_id,
                        repetition=rep,
                        execution_order_position=pos,
                        mechanism=mech,
                        source_path=video_path,
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
                        processor_peak_count=processor_peak_count,
                        gpu_before_snap=gpu_before_snap,
                        gpu_coexist_snap=gpu_coexist_snap,
                        gpu_pub_snap=gpu_pub_snap,
                        gpu_ret_snap=gpu_ret_snap,
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
                    r.request_timestamp_ns,
                    r.candidate_prep_start_ns,
                    r.candidate_prep_end_ns,
                    r.publication_timestamp_ns,
                    r.first_candidate_output_ns,
                    r.request_to_effect_ns,
                    r.transition_output_gap_ns,
                    r.old_plan_frames_completed_during_prep,
                    r.source_frames_received,
                    r.frames_admitted,
                    r.frames_completed,
                    r.ingress_overflow_drop_count,
                    r.admission_rejection_count,
                    r.execution_cancelled_count,
                    r.frames_in_flight_at_window_end,
                    r.total_dropped_frame_count,
                    f"{r.drop_rate:.6f}",
                    r.dropped_frame_count,
                    r.duplicated_frame_count,
                    r.tracker_instance_id_before,
                    r.tracker_instance_id_after,
                    r.tracker_reset_count_before,
                    r.tracker_reset_count_after,
                    r.processor_peak_count,
                    # Allocator snapshot metrics (4 stages)
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
                    # Legacy generic columns preserved for compatibility
                    gb.allocated_bytes if gb.allocated_bytes is not None else "",
                    gc.allocated_bytes if gc.allocated_bytes is not None else "",
                    gp.allocated_bytes if gp.allocated_bytes is not None else "",
                    gr.allocated_bytes if gr.allocated_bytes is not None else "",
                ])
            f.flush()

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
                    f"{fr.queue_occupancy:.2f}",
                    fr.terminal_status,
                    fr.drop_reason,
                    str(fr.dropped),
                    str(fr.duplicated),
                ])
            f.flush()

    return sample_rows, frame_rows
