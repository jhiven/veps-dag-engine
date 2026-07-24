"""Unit tests for NullSink implementation."""

from __future__ import annotations

import pytest

from usecases.video_analytics.config import NullSinkConfig
from usecases.video_analytics.contracts import TrackBatch, VideoMetadata
from usecases.video_analytics.sink import NullSink


def test_null_sink_no_io_and_monotonicity() -> None:
    """Verify NullSink consumes results without I/O and validates monotonic frame IDs."""
    sink = NullSink(NullSinkConfig(validate_monotonic_frame_ids=True))
    meta = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=100, duration_ns=3333333333)

    sink.open(meta)
    assert sink.consumed_count == 0
    assert sink.last_frame_id == 0

    tb1 = TrackBatch(
        frame_id=1,
        source_timestamp_ns=1000,
        detector_id="det_1",
        tracker_instance_id="trk_1",
        tracks=(),
    )
    sink.consume(tb1)
    assert sink.consumed_count == 1
    assert sink.last_frame_id == 1

    tb2 = TrackBatch(
        frame_id=2,
        source_timestamp_ns=2000,
        detector_id="det_1",
        tracker_instance_id="trk_1",
        tracks=(),
    )
    sink.consume(tb2)
    assert sink.consumed_count == 2
    assert sink.last_frame_id == 2

    # Duplicate or out-of-order frame_id raises ValueError when validation is enabled
    tb_dup = TrackBatch(
        frame_id=2,
        source_timestamp_ns=2000,
        detector_id="det_1",
        tracker_instance_id="trk_1",
        tracks=(),
    )
    with pytest.raises(ValueError, match="Non-monotonic frame_id"):
        sink.consume(tb_dup)

    # Close idempotency
    sink.close()
    sink.close()
