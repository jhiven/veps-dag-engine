"""Fail-closed admission and worker coordination for benchmark harnesses."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Condition, Lock

__all__ = [
    "AdmissionGate",
    "FrameAccountingSnapshot",
    "OfferedFrameAccounting",
    "WorkerErrors",
    "drain_queue",
    "wait_until",
]


_TERMINAL_OUTCOMES = frozenset(
    {
        "completed",
        "failed_execution",
        "ingress_overflow",
        "intentional_queue_cancellation",
        "admission_rejection",
    }
)


@dataclass(frozen=True, slots=True)
class FrameAccountingSnapshot:
    offered: int
    completed: int
    failed_execution: int
    ingress_overflow: int
    intentional_queue_cancellation: int
    admission_rejection: int
    still_queued_or_in_flight: int
    residual: int


class OfferedFrameAccounting:
    """Classify each offered frame once at a defined cutoff."""

    __slots__ = ("_lock", "_offered", "_terminal")

    def __init__(self) -> None:
        self._lock = Lock()
        self._offered: set[int] = set()
        self._terminal: dict[int, str] = {}

    def offer(self, frame_id: int) -> None:
        with self._lock:
            if frame_id in self._offered:
                raise RuntimeError(f"frame {frame_id} was offered more than once")
            self._offered.add(frame_id)

    def classify(self, frame_id: int, outcome: str) -> None:
        if outcome not in _TERMINAL_OUTCOMES:
            raise ValueError(f"unknown offered-frame outcome: {outcome!r}")
        with self._lock:
            if frame_id not in self._offered:
                raise RuntimeError(f"frame {frame_id} was classified before it was offered")
            previous = self._terminal.get(frame_id)
            if previous is not None:
                raise RuntimeError(
                    f"frame {frame_id} already has terminal outcome {previous!r}"
                )
            self._terminal[frame_id] = outcome

    def snapshot(self, still_queued_or_in_flight: set[int] | frozenset[int]) -> FrameAccountingSnapshot:
        with self._lock:
            unresolved = self._offered.difference(self._terminal)
            if unresolved != set(still_queued_or_in_flight):
                missing = unresolved.difference(still_queued_or_in_flight)
                extra = set(still_queued_or_in_flight).difference(unresolved)
                raise RuntimeError(
                    "offered-frame cutoff classification mismatch: "
                    f"unclassified={sorted(missing)!r}, unknown_pending={sorted(extra)!r}"
                )
            counts = {outcome: 0 for outcome in _TERMINAL_OUTCOMES}
            for outcome in self._terminal.values():
                counts[outcome] += 1
            pending = len(still_queued_or_in_flight)
            accounted = sum(counts.values()) + pending
            offered = len(self._offered)
            return FrameAccountingSnapshot(
                offered=offered,
                completed=counts["completed"],
                failed_execution=counts["failed_execution"],
                ingress_overflow=counts["ingress_overflow"],
                intentional_queue_cancellation=counts["intentional_queue_cancellation"],
                admission_rejection=counts["admission_rejection"],
                still_queued_or_in_flight=pending,
                residual=offered - accounted,
            )


class AdmissionGate:
    """Linearize baseline admission with Stop/Pause boundaries.

    ``try_enter`` and ``close_and_drain`` use one condition lock.  After
    closure, a dequeued frame cannot cross the gate.  Closure waits for all
    executions that crossed earlier to leave before destructive lifecycle
    work begins.
    """

    __slots__ = ("_condition", "_in_flight", "_open")

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._in_flight = 0
        self._open = True

    def try_enter(self) -> bool:
        with self._condition:
            if not self._open:
                return False
            self._in_flight += 1
            return True

    def leave(self) -> None:
        with self._condition:
            if self._in_flight <= 0:
                raise RuntimeError("unbalanced benchmark admission-gate leave")
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    def close_and_drain(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            self._open = False
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(timeout=remaining):
                    raise TimeoutError(
                        f"timed out draining {self._in_flight} admitted benchmark frame(s)"
                    )

    def open(self) -> None:
        with self._condition:
            if self._open:
                raise RuntimeError("benchmark admission gate is already open")
            self._open = True
            self._condition.notify_all()

    @property
    def in_flight(self) -> int:
        with self._condition:
            return self._in_flight


class WorkerErrors:
    """Thread-safe first-error channel for benchmark workers."""

    __slots__ = ("_errors",)

    def __init__(self) -> None:
        self._errors: Queue[BaseException] = Queue()

    def capture(self, error: BaseException) -> None:
        self._errors.put(error)

    def raise_if_any(self) -> None:
        try:
            error = self._errors.get_nowait()
        except Empty:
            return
        raise RuntimeError("benchmark worker failed") from error


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float,
    description: str,
    errors: WorkerErrors | None = None,
) -> None:
    """Wait for a benchmark condition with a mandatory deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if errors is not None:
            errors.raise_if_any()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.0001)
    if errors is not None:
        errors.raise_if_any()


def drain_queue(queue: Queue[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Remove and return every queued frame for explicit cancellation."""
    drained: list[tuple[int, int]] = []
    while True:
        try:
            drained.append(queue.get_nowait())
        except Empty:
            return tuple(drained)
