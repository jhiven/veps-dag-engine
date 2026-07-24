"""Unit tests for FramePacer and FakeClock scheduling."""

from __future__ import annotations

import pytest

from usecases.video_analytics.pacing import FakeClock, FramePacer, SystemClock


def test_fake_clock_advance_and_sleep() -> None:
    """Verify FakeClock advances deterministically and records sleeps."""
    clock = FakeClock(initial_ns=1_000_000)
    assert clock.now_ns() == 1_000_000

    clock.sleep_ns(500_000)
    assert clock.now_ns() == 1_500_000
    assert clock.sleep_calls == [500_000]

    with pytest.raises(ValueError, match="non-negative"):
        clock.advance_ns(-100)


def test_frame_pacer_deadline_calculation_with_fake_clock() -> None:
    """Verify FramePacer computes deadlines from target FPS without negative sleeps."""
    fake_clock = FakeClock(initial_ns=0)
    pacer = FramePacer(target_fps=30.0, clock=fake_clock, enabled=True)

    # Frame 1: sets start_ns = 0, deadline = 0 -> no sleep needed
    pacer.pace_frame(1)
    assert len(fake_clock.sleep_calls) == 0

    # Frame 2: deadline = 33,333,333 ns. Clock is at 0 -> sleeps 33,333,333 ns
    pacer.pace_frame(2)
    assert fake_clock.sleep_calls == [33333333]
    assert fake_clock.now_ns() == 33333333

    # Frame 3: deadline = 66,666,666 ns
    pacer.pace_frame(3)
    assert fake_clock.sleep_calls == [33333333, 33333333]
    assert pacer.total_frames_paced == 3
    assert isinstance(pacer.median_lateness_ns, float)
    assert isinstance(pacer.p95_lateness_ns, float)


def test_disabled_pacer_does_not_sleep() -> None:
    """Verify disabled FramePacer bypasses pacing."""
    fake_clock = FakeClock(initial_ns=0)
    pacer = FramePacer(target_fps=30.0, clock=fake_clock, enabled=False)

    pacer.pace_frame(1)
    pacer.pace_frame(2)
    assert len(fake_clock.sleep_calls) == 0
    assert pacer.total_frames_paced == 0


def test_system_clock_basic() -> None:
    """Verify SystemClock returns positive nanosecond timestamp."""
    clock = SystemClock()
    t1 = clock.now_ns()
    t2 = clock.now_ns()
    assert t1 > 0
    assert t2 >= t1
