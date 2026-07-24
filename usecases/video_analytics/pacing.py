"""Pacing clock abstractions and frame rate scheduling."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

__all__ = [
    "Clock",
    "SystemClock",
    "FakeClock",
    "FramePacer",
]


class Clock(Protocol):
    """Protocol for nanosecond clock and sleep operations."""

    def now_ns(self) -> int: ...
    def sleep_ns(self, duration_ns: int) -> None: ...


class SystemClock:
    """Real monotonic system clock using time.monotonic_ns."""

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns > 0:
            time.sleep(duration_ns / 1e9)


class FakeClock:
    """Deterministic fake clock for testing without real-time sleeps."""

    __slots__ = ("_current_ns", "sleep_calls")

    def __init__(self, initial_ns: int = 0) -> None:
        self._current_ns: int = initial_ns
        self.sleep_calls: list[int] = []

    def now_ns(self) -> int:
        return self._current_ns

    def advance_ns(self, duration_ns: int) -> None:
        if duration_ns < 0:
            raise ValueError(f"duration_ns must be non-negative, got {duration_ns}")
        self._current_ns += duration_ns

    def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns > 0:
            self.sleep_calls.append(duration_ns)
            self._current_ns += duration_ns


class FramePacer:
    """Frame rate pacer pacing frame submission according to source video FPS."""

    __slots__ = (
        "target_fps",
        "clock",
        "enabled",
        "frame_period_ns",
        "start_ns",
        "lateness_history_ns",
        "total_frames_paced",
    )

    def __init__(
        self,
        target_fps: float,
        clock: Clock | None = None,
        enabled: bool = True,
    ) -> None:
        if target_fps <= 0.0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        self.target_fps: float = target_fps
        self.clock: Clock = clock if clock is not None else SystemClock()
        self.enabled: bool = enabled
        self.frame_period_ns: int = int(round(1e9 / target_fps))
        self.start_ns: int | None = None
        self.lateness_history_ns: list[float] = []
        self.total_frames_paced: int = 0

    def pace_frame(self, frame_id: int) -> None:
        """Pace the submission of frame_id (1-indexed) according to target FPS."""
        if not self.enabled:
            return

        now = self.clock.now_ns()
        if self.start_ns is None:
            self.start_ns = now

        target_deadline_ns = self.start_ns + (frame_id - 1) * self.frame_period_ns
        lateness_ns = now - target_deadline_ns
        self.lateness_history_ns.append(float(lateness_ns))
        self.total_frames_paced += 1

        if now < target_deadline_ns:
            self.clock.sleep_ns(target_deadline_ns - now)

    @property
    def median_lateness_ns(self) -> float:
        if not self.lateness_history_ns:
            return 0.0
        return float(np.median(self.lateness_history_ns))

    @property
    def p95_lateness_ns(self) -> float:
        if not self.lateness_history_ns:
            return 0.0
        return float(np.percentile(self.lateness_history_ns, 95.0))
