"""Tests for execution provenance collection and its publication-mode gate."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from benchmarks.provenance import UNKNOWN, collect_provenance, missing_required_provenance


def _provenance_for(video_path: str, suites: tuple[str, ...] = ("realworld-video",)) -> dict[str, Any]:
    return collect_provenance(
        selected_suites=suites,
        device="cpu",
        source_video=video_path,
        rtsp_base_url="rtsp://127.0.0.1:8554",
        model_ids=("PekingU/rtdetr_r18vd",),
    )


def test_source_video_hash_is_the_file_digest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "clip.mp4")
        payload = b"not really a video, but it hashes the same way"
        with open(path, "wb") as file:
            file.write(payload)

        provenance = _provenance_for(path)
        assert provenance["source_video_sha256"] == hashlib.sha256(payload).hexdigest()
        assert provenance["source_video_path"] == path


def test_absent_video_is_recorded_as_unknown_not_as_a_default() -> None:
    provenance = _provenance_for(os.path.join(tempfile.gettempdir(), "no-such-clip.mp4"))
    assert provenance["source_video_sha256"] == UNKNOWN


def test_publication_requires_video_and_toolchain_provenance() -> None:
    provenance = _provenance_for(os.path.join(tempfile.gettempdir(), "no-such-clip.mp4"))
    missing = missing_required_provenance(provenance, ("realworld-video",), "cpu")
    assert "source_video_sha256" in missing


def test_suites_without_video_do_not_require_media_provenance() -> None:
    provenance = _provenance_for(
        os.path.join(tempfile.gettempdir(), "no-such-clip.mp4"),
        suites=("steady-state", "reconfiguration"),
    )
    assert missing_required_provenance(provenance, ("steady-state", "reconfiguration"), "cpu") == ()


def test_cuda_runs_additionally_require_driver_and_device_identity() -> None:
    """A CUDA campaign may not proceed while the driver or GPU is unrecorded."""
    provenance: dict[str, Any] = {
        "ffmpeg_version": "ffmpeg version n9.0.1",
        "ffprobe_version": "ffprobe version n9.0.1",
        "source_video_sha256": "0" * 64,
        "torch_version": "2.13.0",
        "cuda_runtime_version": UNKNOWN,
        "nvidia_driver_version": UNKNOWN,
        "gpu_name": UNKNOWN,
        "models": {"m": {"revision": "abc", "config_sha256": "def"}},
    }

    assert missing_required_provenance(provenance, ("realworld-video",), "cpu") == ()

    cuda_missing = missing_required_provenance(provenance, ("realworld-video",), "cuda:0")
    assert set(cuda_missing) == {"cuda_runtime_version", "nvidia_driver_version", "gpu_name"}


def test_unresolved_model_checkpoints_block_publication() -> None:
    provenance: dict[str, Any] = {
        "ffmpeg_version": "ffmpeg version n9.0.1",
        "ffprobe_version": "ffprobe version n9.0.1",
        "source_video_sha256": "0" * 64,
        "torch_version": "2.13.0",
        "models": {"PekingU/rtdetr_r18vd": {"revision": UNKNOWN, "config_sha256": UNKNOWN}},
    }

    missing = missing_required_provenance(provenance, ("realworld-video",), "cpu")
    assert "models.PekingU/rtdetr_r18vd.revision" in missing
    assert "models.PekingU/rtdetr_r18vd.config_sha256" in missing
