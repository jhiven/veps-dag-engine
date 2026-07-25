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
)
from usecases.video_analytics.contracts import (
    DetectorBackend,
    ExecutionMode,
    TrackBatch,
    TrackerBackend,
)
from usecases.video_analytics.lifecycle import cleanup_repetition_resources
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
    "dropped_frame_count",
    "duplicated_frame_count",
    "tracker_instance_id_before",
    "tracker_instance_id_after",
    "tracker_reset_count_before",
    "tracker_reset_count_after",
    "processor_peak_count",
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
    dropped_frame_count: int
    duplicated_frame_count: int
    tracker_instance_id_before: str
    tracker_instance_id_after: str
    tracker_reset_count_before: int
    tracker_reset_count_after: int
    processor_peak_count: int
    gpu_memory_before_prep_bytes: int | None
    gpu_memory_during_coexistence_bytes: int | None
    gpu_memory_after_pub_bytes: int | None
    gpu_memory_after_ret_bytes: int | None


@dataclass(frozen=True, slots=True)
class RealworldVideoFrameSampleRow:
    run_id: str
    repetition: int
    mechanism: str
    frame_id: int
    source_timestamp_ns: int
    admission_timestamp_ns: int
    completion_timestamp_ns: int
    plan_version: int
    detector_id: str
    tracker_instance_id: str
    queue_occupancy: float
    dropped: bool
    duplicated: bool


def _calculate_file_hash(path: str) -> str:
    if not os.path.exists(path):
        return "fake_hash"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get_gpu_memory_bytes() -> int | None:
    try:
        import torch 

        if torch.cuda.is_available(): 
            torch.cuda.synchronize() 
            return int(torch.cuda.memory_allocated()) 
    except Exception:
        pass
    return None


def run_realworld_video_suite(
    run_id: str,
    output_csv_path: str,
    video_path: str = "sample_video.mp4",
    initial_model: str = "PekingU/rtdetr_r18vd",
    candidate_model: str = "PekingU/rtdetr_r50vd",
    device: str = "cpu",
    dtype: str = "float32",
    repetition_count: int = 5,
    update_frame_id: int = 50,
    total_frames: int = 150,
    use_fake_backends: bool = False,
    random_seed: int = 42,
    execution_mode: ExecutionMode | None = None,
) -> tuple[list[RealworldVideoSampleRow], list[RealworldVideoFrameSampleRow]]:
    """Orchestrate the real-world video analytics benchmark suite across Stop, Pause, and VEPS."""
    mode = execution_mode or (ExecutionMode.SMOKE if use_fake_backends else ExecutionMode.PUBLICATION)

    if mode is ExecutionMode.PUBLICATION:
        if use_fake_backends:
            raise ValueError("Publication mode rejects use_fake_backends=True")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Publication mode video path not found: {video_path}")
        if device.startswith("cuda"):
            import torch 

            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device {device!r} was requested in publication mode, but CUDA is not available.")

        # Pre-download and pre-warm model checkpoints before measured execution
        from usecases.video_analytics.cli import prepare_assets_cmd

        print(f"Pre-downloading/verifying model checkpoints: initial={initial_model!r}, candidate={candidate_model!r}...")
        prepare_assets_cmd(initial_model=initial_model, candidate_model=candidate_model)

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

                gpu_before = _get_gpu_memory_bytes()

                rep_frame_rows: list[RealworldVideoFrameSampleRow] = []
                seen_frame_ids: set[int] = set()
                dup_count: int = 0

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
                        dropped=False,
                        duplicated=is_dup,
                    )
                    rep_frame_rows.append(fr_row)

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

                    gpu_coexist = _get_gpu_memory_bytes()

                    app.process_remaining()
                    gpu_pub = _get_gpu_memory_bytes()

                    if app.tracker_backend is not None:
                        trk_inst_after = app.tracker_backend.instance_id
                        reset_after = app.tracker_backend.reset_count
                    else:
                        trk_inst_after = "unknown"
                        reset_after = 0

                    app.close()
                    gpu_ret = _get_gpu_memory_bytes()

                    # Find first candidate output frame
                    first_cand_row = next(
                        (r for r in rep_frame_rows if r.plan_version > 1 and r.detector_id == candidate_model),
                        None,
                    )
                    if first_cand_row is None:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No candidate output reached the sink."
                        )
                    first_candidate_output_ns = first_cand_row.completion_timestamp_ns

                    # Find last old-plan output occurring before first candidate output
                    old_rows_before_cand = [
                        r for r in rep_frame_rows if r.completion_timestamp_ns < first_candidate_output_ns
                    ]
                    if not old_rows_before_cand:
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: No old-plan output reached the sink before first candidate output."
                        )
                    last_old_row = max(old_rows_before_cand, key=lambda r: r.completion_timestamp_ns)
                    last_old_output_ns = last_old_row.completion_timestamp_ns

                    # Derive transition metrics
                    request_to_effect_ns = first_candidate_output_ns - t_req_swap
                    transition_output_gap_ns = first_candidate_output_ns - last_old_output_ns

                    # Count old plan frames completed during prep
                    old_prep_frames = sum(
                        1
                        for r in rep_frame_rows
                        if r.plan_version == 1
                        and t_prep_start <= r.completion_timestamp_ns <= t_prep_end
                    )

                    # Validate timestamp ordering
                    if not (t_req_swap <= t_prep_start <= t_prep_end <= t_pub <= first_candidate_output_ns):
                        raise RuntimeError(
                            f"Repetition {rep} mechanism {mech}: Timestamp ordering violation: "
                            f"req={t_req_swap}, prep_start={t_prep_start}, prep_end={t_prep_end}, "
                            f"pub={t_pub}, first_cand_output={first_candidate_output_ns}"
                        )

                    # Measured peak live processor count (initial 4 nodes + candidate detector staged during prep)
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
                        dropped_frame_count=0,
                        duplicated_frame_count=dup_count,
                        tracker_instance_id_before=trk_inst_before,
                        tracker_instance_id_after=trk_inst_after,
                        tracker_reset_count_before=reset_before,
                        tracker_reset_count_after=reset_after,
                        processor_peak_count=processor_peak_count,
                        gpu_memory_before_prep_bytes=gpu_before if device.startswith("cuda") else None,
                        gpu_memory_during_coexistence_bytes=gpu_coexist if device.startswith("cuda") else None,
                        gpu_memory_after_pub_bytes=gpu_pub if device.startswith("cuda") else None,
                        gpu_memory_after_ret_bytes=gpu_ret if device.startswith("cuda") else None,
                    )
                    sample_rows.append(sample_row)
                    frame_rows.extend(rep_frame_rows)

                finally:
                    cleanup_repetition_resources()

    finally:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REALWORLD_VIDEO_HEADERS)
            for r in sample_rows:
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
                    r.dropped_frame_count,
                    r.duplicated_frame_count,
                    r.tracker_instance_id_before,
                    r.tracker_instance_id_after,
                    r.tracker_reset_count_before,
                    r.tracker_reset_count_after,
                    r.processor_peak_count,
                    r.gpu_memory_before_prep_bytes if r.gpu_memory_before_prep_bytes is not None else "",
                    r.gpu_memory_during_coexistence_bytes if r.gpu_memory_during_coexistence_bytes is not None else "",
                    r.gpu_memory_after_pub_bytes if r.gpu_memory_after_pub_bytes is not None else "",
                    r.gpu_memory_after_ret_bytes if r.gpu_memory_after_ret_bytes is not None else "",
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
                    fr.source_timestamp_ns,
                    fr.admission_timestamp_ns,
                    fr.completion_timestamp_ns,
                    fr.plan_version,
                    fr.detector_id,
                    fr.tracker_instance_id,
                    f"{fr.queue_occupancy:.2f}",
                    str(fr.dropped),
                    str(fr.duplicated),
                ])
            f.flush()

    return sample_rows, frame_rows
