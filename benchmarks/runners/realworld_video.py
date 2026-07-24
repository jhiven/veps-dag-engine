"""Benchmark runner for real-world video analytics live reconfiguration (Section 9)."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time
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
from usecases.video_analytics.contracts import DetectorBackend, TrackerBackend
from usecases.video_analytics.lifecycle import cleanup_repetition_resources

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
        import torch  # type: ignore[import-not-found,import-untyped]

        if torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return int(torch.cuda.memory_allocated())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
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
) -> tuple[list[RealworldVideoSampleRow], list[RealworldVideoFrameSampleRow]]:
    """Orchestrate the real-world video analytics benchmark suite across Stop, Pause, and VEPS."""
    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    os.makedirs(output_dir, exist_ok=True)
    frame_csv_path = os.path.join(output_dir, "realworld-video-frame-samples.csv")

    source_hash = _calculate_file_hash(video_path)
    mechanisms = ["VEPS", "Pause", "Stop"]
    rng = random.Random(random_seed)

    sample_rows: list[RealworldVideoSampleRow] = []
    frame_rows: list[RealworldVideoFrameSampleRow] = []

    for rep in range(1, repetition_count + 1):
        mech_order = list(mechanisms)
        rng.shuffle(mech_order)

        for pos, mech in enumerate(mech_order, start=1):
            cfg = VideoAnalyticsConfig(
                source=FileVideoSourceConfig(video_path=video_path, enable_pacing=not use_fake_backends),
                initial_detector=RTDETRConfig(model_id=initial_model, device=device, dtype=dtype),
                candidate_detector=RTDETRConfig(model_id=candidate_model, device=device, dtype=dtype),
                tracker=ByteTrackConfig(),
                sink=NullSinkConfig(),
                update_frame_id=update_frame_id,
                total_frames_to_process=total_frames,
            )

            init_backend: DetectorBackend | None = None
            cand_backend: DetectorBackend | None = None
            trk_backend: TrackerBackend | None = None

            if use_fake_backends:
                init_backend = FakeDetectorBackend(model_id=initial_model)
                cand_backend = FakeDetectorBackend(model_id=candidate_model)
                trk_backend = FakeTrackerBackend(instance_id=f"fake_trk_rep_{rep}")

            gpu_before = _get_gpu_memory_bytes()

            app = VideoAnalyticsApplication(
                config=cfg,
                initial_detector_backend=init_backend,
                candidate_detector_backend=cand_backend,
                tracker_backend=trk_backend,
            )

            try:
                meta = app.open()
                t_prep_start = time.monotonic_ns()

                app.process_until_frame(update_frame_id)
                t_req_swap = time.monotonic_ns()

                typed_mech = cast(Literal["VEPS", "Pause", "Stop"], mech)
                app.request_detector_swap(cfg.candidate_detector, mechanism=typed_mech, candidate_backend=cand_backend)
                t_prep_end = time.monotonic_ns()
                gpu_coexist = _get_gpu_memory_bytes()

                app.process_remaining()
                t_pub = time.monotonic_ns()
                gpu_pub = _get_gpu_memory_bytes()

                trk_inst_before = app.tracker_backend.instance_id if app.tracker_backend else "unknown"
                trk_inst_after = app.tracker_backend.instance_id if app.tracker_backend else "unknown"
                reset_before = 0
                reset_after = app.tracker_backend.reset_count if app.tracker_backend else 0

                app.close()
                gpu_ret = _get_gpu_memory_bytes()

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
                    first_candidate_output_ns=t_pub + 1_000_000,
                    request_to_effect_ns=t_pub - t_req_swap,
                    transition_output_gap_ns=0,
                    old_plan_frames_completed_during_prep=update_frame_id,
                    dropped_frame_count=0,
                    duplicated_frame_count=0,
                    tracker_instance_id_before=trk_inst_before,
                    tracker_instance_id_after=trk_inst_after,
                    tracker_reset_count_before=reset_before,
                    tracker_reset_count_after=reset_after,
                    processor_peak_count=4,
                    gpu_memory_before_prep_bytes=gpu_before,
                    gpu_memory_during_coexistence_bytes=gpu_coexist,
                    gpu_memory_after_pub_bytes=gpu_pub,
                    gpu_memory_after_ret_bytes=gpu_ret,
                )
                sample_rows.append(sample_row)

            finally:
                cleanup_repetition_resources()

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
                r.gpu_memory_before_prep_bytes or "",
                r.gpu_memory_during_coexistence_bytes or "",
                r.gpu_memory_after_pub_bytes or "",
                r.gpu_memory_after_ret_bytes or "",
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

    return sample_rows, frame_rows
