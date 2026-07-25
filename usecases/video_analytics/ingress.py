"""Bounded drop-oldest ingress queue and independent RTSP capture thread."""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable, Generic, TypeVar

from usecases.video_analytics.contracts import DropReason, FramePacket 

__all__ = [
    "BoundedIngressQueue",
    "RTSPCaptureIngress",
    "FakeRTSPCaptureIngress",
]

T = TypeVar("T")


class BoundedIngressQueue(Generic[T]):
    """Thread-safe bounded queue with drop-oldest eviction policy."""

    def __init__(
        self,
        capacity: int = 4,
        on_drop_callback: Callable[[T, DropReason], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Queue capacity must be positive, got {capacity}")

        self._capacity: int = capacity
        self._on_drop_callback: Callable[[T, DropReason], None] | None = on_drop_callback
        self._queue: collections.deque[T] = collections.deque()
        self._lock: threading.Lock = threading.Lock()
        self._overflow_drop_count: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overflow_drop_count(self) -> int:
        with self._lock:
            return self._overflow_drop_count

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def occupancy_ratio(self) -> float:
        with self._lock:
            return len(self._queue) / self._capacity

    def put(self, item: T) -> None:
        """Push an item into the queue. If full, evict the oldest item first (drop-oldest)."""
        import dataclasses
        from typing import Any, cast
        dropped_item: T | None = None
        with self._lock:
            occ_before = len(self._queue)
            if occ_before >= self._capacity:
                raw_dropped = self._queue.popleft()
                self._overflow_drop_count += 1
                occ_after = self._capacity
                
                if raw_dropped is not None and hasattr(raw_dropped, "queue_occupancy_before_enqueue"):
                    dropped_item = cast(T, dataclasses.replace(
                        cast(Any, raw_dropped),
                        queue_occupancy_before_enqueue=self._capacity,
                        queue_occupancy_after_enqueue=self._capacity,
                        queue_capacity=self._capacity
                    ))
                else:
                    dropped_item = raw_dropped
            else:
                occ_after = occ_before + 1
            
            if hasattr(item, "queue_occupancy_before_enqueue"):
                updated_item = cast(T, dataclasses.replace(
                    cast(Any, item),
                    queue_occupancy_before_enqueue=occ_before,
                    queue_occupancy_after_enqueue=occ_after,
                    queue_capacity=self._capacity
                ))
            else:
                updated_item = item
            self._queue.append(updated_item)

        if dropped_item is not None and self._on_drop_callback is not None:
            self._on_drop_callback(dropped_item, DropReason.INGRESS_OVERFLOW)

    def get(self) -> T | None:
        """Non-blocking pop from the front of the queue."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def clear_and_cancel_all(self, cancel_callback: Callable[[T, DropReason], None] | None = None) -> list[T]:
        """Clear all queued items and invoke cancellation callback."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()

        if cancel_callback is not None:
            for item in items:
                cancel_callback(item, DropReason.CANCELLED_ON_STOP)
        return items


class RTSPCaptureIngress:
    """Independent background capture thread feeding a bounded drop-oldest ingress queue."""

    def __init__(
        self,
        frame_decoder: Callable[[], FramePacket | None],
        queue_capacity: int = 4,
        on_drop_callback: Callable[[FramePacket, DropReason], None] | None = None,
    ) -> None:
        self._frame_decoder: Callable[[], FramePacket | None] = frame_decoder
        self._queue: BoundedIngressQueue[FramePacket] = BoundedIngressQueue(
            capacity=queue_capacity,
            on_drop_callback=on_drop_callback,
        )
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_frames_received: int = 0
        self._lock: threading.Lock = threading.Lock()

    @property
    def queue(self) -> BoundedIngressQueue[FramePacket]:
        return self._queue

    @property
    def source_frames_received(self) -> int:
        with self._lock:
            return self._source_frames_received

    def start(self) -> None:
        """Launch background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="RTSPCaptureThread")
        self._thread.start()

    def _capture_loop(self) -> None:
        """Continuous background capture loop independent of DAG execution plan."""
        while not self._stop_event.is_set():
            frame = self._frame_decoder()
            if frame is None:
                # End of stream or temporary delay
                time.sleep(0.005)
                continue

            with self._lock:
                self._source_frames_received += 1

            self._queue.put(frame)

    def read(self) -> FramePacket | None:
        """Consumer pop from bounded queue."""
        return self._queue.get()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        """Stop background capture thread cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None


class FakeRTSPCaptureIngress:
    """Synthetic fake capture ingress for deterministic unit testing."""

    def __init__(
        self,
        frames: list[FramePacket],
        queue_capacity: int = 4,
        on_drop_callback: Callable[[FramePacket, DropReason], None] | None = None,
    ) -> None:
        self._queue: BoundedIngressQueue[FramePacket] = BoundedIngressQueue(
            capacity=queue_capacity,
            on_drop_callback=on_drop_callback,
        )
        self._frames: list[FramePacket] = list(frames)
        self.source_frames_received: int = len(frames)
        self.is_active: bool = False

    @property
    def queue(self) -> BoundedIngressQueue[FramePacket]:
        return self._queue

    def start(self) -> None:
        self.is_active = True
        for frame in self._frames:
            self._queue.put(frame)

    def read(self) -> FramePacket | None:
        return self._queue.get()

    def stop(self) -> None:
        self.is_active = False
