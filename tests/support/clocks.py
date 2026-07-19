"""Deterministic monotonic clock for tests.

Replaces ``time.monotonic_ns`` with a simple counter so tests can assert
on exact timestamps without relying on wall-clock duration.
"""

from __future__ import annotations

from threading import Lock


class IncrementingClock:
    """Thread-safe monotonically incrementing clock.

    Each call returns the current value and then advances by one, so
    consecutive calls always produce distinct, increasing nanosecond
    timestamps regardless of how quickly they run.
    """

    __slots__ = ("_lock", "_value")

    def __init__(self, initial: int = 1_000) -> None:
        self._value = initial
        self._lock = Lock()

    def __call__(self) -> int:
        with self._lock:
            value = self._value
            self._value += 1
            return value

    def peek(self) -> int:
        """Return the next value without advancing the counter."""
        with self._lock:
            return self._value
