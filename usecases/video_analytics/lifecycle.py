"""Lifecycle management, resource cleanup, and failure recovery for video analytics runs."""

from __future__ import annotations

import gc
import logging
from typing import Any

from usecases.video_analytics.contracts import DetectorBackend, FrameSource, ResultSink, TrackerBackend

logger = logging.getLogger(__name__)

__all__ = [
    "cleanup_repetition_resources",
    "cleanup_gpu_memory",
]


def cleanup_gpu_memory() -> None:
    """Perform GPU garbage collection, empty cache, and synchronize CUDA streams.

    Must be invoked outside the measured transition path.
    """
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found,import-untyped]

        if torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            torch.cuda.empty_cache()  # pyright: ignore[reportUnknownMemberType]
            torch.cuda.synchronize()  # pyright: ignore[reportUnknownMemberType]
    except (ImportError, Exception) as err:
        logger.debug("GPU cleanup skipped: %s", err)


def cleanup_repetition_resources(
    source: FrameSource | None = None,
    sink: ResultSink | None = None,
    initial_detector: DetectorBackend | None = None,
    candidate_detector: DetectorBackend | None = None,
    tracker: TrackerBackend | None = None,
    extra_objects: tuple[Any, ...] = (),
) -> None:
    """Safely release resources after a benchmark repetition or on failure."""
    if source is not None:
        try:
            source.close()
        except Exception as err:
            logger.warning("Error closing source: %s", err)

    if sink is not None:
        try:
            sink.close()
        except Exception as err:
            logger.warning("Error closing sink: %s", err)

    if initial_detector is not None:
        try:
            initial_detector.close()
        except Exception as err:
            logger.warning("Error closing initial_detector: %s", err)

    if candidate_detector is not None:
        try:
            candidate_detector.close()
        except Exception as err:
            logger.warning("Error closing candidate_detector: %s", err)

    if tracker is not None:
        try:
            tracker.close()
        except Exception as err:
            logger.warning("Error closing tracker: %s", err)

    for obj in extra_objects:
        if hasattr(obj, "close") and callable(getattr(obj, "close")):
            try:
                getattr(obj, "close")()
            except Exception as err:
                logger.warning("Error closing extra object %r: %s", obj, err)

    cleanup_gpu_memory()
