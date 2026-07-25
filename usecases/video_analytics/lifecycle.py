"""Lifecycle management, resource cleanup, and failure recovery for video analytics runs."""

from __future__ import annotations

import gc
import logging
from typing import Any

from usecases.video_analytics.contracts import DetectorBackend, FrameSource, ResultSink, TrackerBackend

import threading

logger = logging.getLogger(__name__)

__all__ = [
    "cleanup_repetition_resources",
    "cleanup_gpu_memory",
    "CoexistenceTracker",
]


class CoexistenceTracker:
    """Thread-safe global tracker for live execution plans and detector instances."""

    _lock = threading.Lock()
    _live_plans: set[int] = set()
    _live_detectors: set[int] = set()
    _peak_plan_count: int = 0
    _peak_detector_count: int = 0

    @classmethod
    def reset(cls) -> None:
        """Reset all tracking metrics to baseline state."""
        with cls._lock:
            cls._live_plans.clear()
            cls._live_detectors.clear()
            cls._peak_plan_count = 0
            cls._peak_detector_count = 0

    @classmethod
    def plan_created(cls, version: int) -> None:
        """Record plan creation/activation event."""
        with cls._lock:
            cls._live_plans.add(version)
            cls._peak_plan_count = max(cls._peak_plan_count, len(cls._live_plans))

    @classmethod
    def plan_retired(cls, version: int) -> None:
        """Record plan retirement event."""
        with cls._lock:
            cls._live_plans.discard(version)

    @classmethod
    def detector_created(cls, detector_id: int) -> None:
        """Record detector instance instantiation event."""
        with cls._lock:
            cls._live_detectors.add(detector_id)
            cls._peak_detector_count = max(cls._peak_detector_count, len(cls._live_detectors))

    @classmethod
    def detector_closed(cls, detector_id: int) -> None:
        """Record detector instance close/destruction event."""
        with cls._lock:
            cls._live_detectors.discard(detector_id)

    @classmethod
    def get_live_plan_count(cls) -> int:
        """Retrieve count of currently active execution plans."""
        with cls._lock:
            return len(cls._live_plans)

    @classmethod
    def get_live_detector_count(cls) -> int:
        """Retrieve count of currently active detector backend instances."""
        with cls._lock:
            return len(cls._live_detectors)

    @classmethod
    def get_peak_plan_count(cls) -> int:
        """Retrieve peak count of concurrent live plans reached."""
        with cls._lock:
            return cls._peak_plan_count

    @classmethod
    def get_peak_detector_count(cls) -> int:
        """Retrieve peak count of concurrent live detector instances reached."""
        with cls._lock:
            return cls._peak_detector_count



def cleanup_gpu_memory() -> None:
    """Perform GPU garbage collection, empty cache, and synchronize CUDA streams.

    Must be invoked outside the measured transition path.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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
