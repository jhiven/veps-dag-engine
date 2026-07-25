import time
from typing import Callable

from usecases.video_analytics.config import NullSinkConfig
from usecases.video_analytics.contracts import BackendKind, ResultSink, TrackBatch, VideoMetadata

__all__ = [
    "NullSink",
]


class NullSink(ResultSink):
    """A zero-overhead NullSink for measured benchmark execution."""

    def __init__(
        self,
        config: NullSinkConfig | None = None,
        observer: Callable[[TrackBatch, int], None] | None = None,
    ) -> None:
        self._config: NullSinkConfig = config or NullSinkConfig()
        self._observer: Callable[[TrackBatch, int], None] | None = observer
        self._metadata: VideoMetadata | None = None
        self._last_frame_id: int = 0
        self._is_closed: bool = False
        self._consumed_count: int = 0

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.PRODUCTION

    def set_observer(self, observer: Callable[[TrackBatch, int], None] | None) -> None:
        """Register a frame completion observer callback."""
        self._observer = observer

    def open(self, metadata: VideoMetadata) -> None:
        """Open the sink with video metadata."""
        if self._is_closed:
            raise RuntimeError("Cannot open a closed NullSink")
        self._metadata = metadata
        self._last_frame_id = 0
        self._consumed_count = 0

    def consume(self, result: TrackBatch) -> None:
        """Consume a frame's tracking result."""
        if self._is_closed:
            raise RuntimeError("Cannot consume into a closed NullSink")

        completion_ns = time.monotonic_ns()

        if self._config.validate_monotonic_frame_ids:
            if result.frame_id <= self._last_frame_id:
                raise ValueError(
                    f"Non-monotonic frame_id received in NullSink: got {result.frame_id}, last was {self._last_frame_id}"
                )

        self._last_frame_id = result.frame_id
        self._consumed_count += 1

        if self._observer is not None:
            self._observer(result, completion_ns)

    def close(self) -> None:
        """Close the sink idempotently."""
        if self._is_closed:
            return
        self._is_closed = True

    @property
    def consumed_count(self) -> int:
        return self._consumed_count

    @property
    def last_frame_id(self) -> int:
        return self._last_frame_id

