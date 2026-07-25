"""Standalone CLI for video analytics asset preflight and standalone pipeline execution."""

from __future__ import annotations

import argparse
import sys
import time

from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.config import (
    FileVideoSourceConfig,
    NullSinkConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
    validate_config,
)

__all__ = [
    "prepare_assets_cmd",
    "run_cmd",
    "main",
]


def prepare_assets_cmd(
    initial_model: str = "PekingU/rtdetr_r18vd",
    candidate_model: str = "PekingU/rtdetr_r50vd",
) -> None:
    """Preflight asset command to verify or download model checkpoints."""
    print(f"Preparing assets for initial model {initial_model!r} and candidate model {candidate_model!r}...")

    try:
        from transformers import (
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        )
    except ImportError:
        print("Error: PyTorch and Transformers are required for asset preparation.", file=sys.stderr)
        print("Install dependencies with: uv sync --group realworld-video", file=sys.stderr)
        sys.exit(1)

    for mid in (initial_model, candidate_model):
        print(f"  Downloading/verifying checkpoint: {mid}")
        RTDetrImageProcessor.from_pretrained(mid) # pyright: ignore[reportUnknownMemberType]
        RTDetrForObjectDetection.from_pretrained(mid)  # pyright: ignore[reportUnknownMemberType]

    print("All model assets successfully prepared and verified.")


def run_cmd(
    video_path: str,
    initial_model: str = "PekingU/rtdetr_r18vd",
    candidate_model: str = "PekingU/rtdetr_r50vd",
    device: str = "cpu",
    confidence_threshold: float = 0.5,
    update_frame_id: int = 100,
    total_frames: int | None = None,
) -> None:
    """Run standalone video analytics execution with a detector swap."""
    # Ensure model checkpoints are pre-downloaded and verified before starting pipeline
    prepare_assets_cmd(initial_model=initial_model, candidate_model=candidate_model)

    cfg = VideoAnalyticsConfig(
        source=FileVideoSourceConfig(video_path=video_path),
        initial_detector=RTDETRConfig(
            model_id=initial_model,
            device=device,
            confidence_threshold=confidence_threshold,
            local_files_only=True,
        ),
        candidate_detector=RTDETRConfig(
            model_id=candidate_model,
            device=device,
            confidence_threshold=confidence_threshold,
            local_files_only=True,
        ),
        sink=NullSinkConfig(),
        update_frame_id=update_frame_id,
        total_frames_to_process=total_frames if total_frames is not None else 150,
    )
    validate_config(cfg)

    t0 = time.monotonic_ns()
    print(f"Opening video analytics pipeline on video: {video_path}")
    with VideoAnalyticsApplication(cfg) as app:
        print(f"  Processing pre-update frames up to frame {update_frame_id}...")
        app.process_until_frame(update_frame_id)

        print(f"  Requesting detector swap: {initial_model} -> {candidate_model} (VEPS)...")
        app.request_detector_swap(cfg.candidate_detector, mechanism="VEPS")

        print("  Processing remaining frames...")
        _ = app.process_remaining()

    dur_sec = (time.monotonic_ns() - t0) / 1e9
    print(f"Execution complete in {dur_sec:.2f} seconds.")
    print(f"  Total frames processed: {app.processed_frames}")
    if app.tracker_backend is not None:
        print(f"  Tracker instance ID: {app.tracker_backend.instance_id}")
        print(f"  Tracker reset count: {app.tracker_backend.reset_count}")
        print(f"  Tracker processed frames: {app.tracker_backend.processed_frame_count}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Video Analytics Pipeline CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # prepare-assets subcommand
    prep_parser = subparsers.add_parser("prepare-assets", help="Download and verify model checkpoints")
    prep_parser.add_argument("--initial-model", type=str, default="PekingU/rtdetr_r18vd")
    prep_parser.add_argument("--candidate-model", type=str, default="PekingU/rtdetr_r50vd")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run standalone video analytics use case")
    run_parser.add_argument("--video", type=str, required=True, help="Path to input H.264 MP4 video")
    run_parser.add_argument("--initial-model", type=str, default="PekingU/rtdetr_r18vd")
    run_parser.add_argument("--candidate-model", type=str, default="PekingU/rtdetr_r50vd")
    run_parser.add_argument("--device", type=str, default="cpu")
    run_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    run_parser.add_argument("--update-frame", type=int, default=100)
    run_parser.add_argument("--total-frames", type=int, default=None)

    args = parser.parse_args()

    if args.subcommand == "prepare-assets":
        prepare_assets_cmd(
            initial_model=args.initial_model,
            candidate_model=args.candidate_model,
        )
    elif args.subcommand == "run":
        run_cmd(
            video_path=args.video,
            initial_model=args.initial_model,
            candidate_model=args.candidate_model,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            update_frame_id=args.update_frame,
            total_frames=args.total_frames,
        )


if __name__ == "__main__":
    main()
