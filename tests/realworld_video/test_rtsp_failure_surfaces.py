"""A dead RTSP decoder must fail the run, not stall it.

A decoder that attaches before the publisher registered its path exits at once.
When that exit was indistinguishable from an empty queue, the benchmark's
warm-up loop retried forever: one publication run sat on sub-run 1 of 90 for
eight hours with no error and no artifact. These tests pin the two properties
that make such a stall impossible: the decoder's own diagnosis reaches the
caller, and no frame-waiting loop is unbounded.
"""

from __future__ import annotations

import tempfile
import time

import pytest

from usecases.video_analytics.source import RTSPVideoSource


class _ExitedProcess:
    """Stands in for an FFmpeg decoder that has already exited."""

    def __init__(self, returncode: int = 8) -> None:
        self.returncode = returncode
        self.stdout = None

    def poll(self) -> int:
        return self.returncode


class _EmptyIngress:
    """An ingress whose queue never yields a frame."""

    class _Queue:
        def size(self) -> int:
            return 0

        def clear_and_cancel_all(self) -> tuple[()]:
            return ()

    def __init__(self) -> None:
        self.queue = _EmptyIngress._Queue()
        self.source_frames_received = 0

    def read(self) -> None:
        return None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _source_with_dead_decoder() -> RTSPVideoSource:
    source = RTSPVideoSource(rtsp_url="rtsp://127.0.0.1:8554/gone", video_path="sample_video.mp4")
    setattr(source, "_ingress", _EmptyIngress())
    setattr(source, "_decoder_proc", _ExitedProcess())
    return source


def test_read_raises_once_the_decoder_has_exited() -> None:
    """Returning None here is what let the caller retry forever."""
    source = _source_with_dead_decoder()

    with pytest.raises(RuntimeError, match="no longer running"):
        source.read()


def test_read_reports_the_decoder_exit_code() -> None:
    source = _source_with_dead_decoder()

    with pytest.raises(RuntimeError, match="exited with code 8"):
        source.read()


def test_read_is_bounded_even_when_it_raises() -> None:
    """The dead-decoder check must not wait out a long retry window."""
    source = _source_with_dead_decoder()

    started = time.monotonic()
    with pytest.raises(RuntimeError):
        source.read()
    assert time.monotonic() - started < 1.0


def test_read_still_returns_none_while_the_decoder_lives() -> None:
    """A momentarily empty queue is not a failure."""

    class _RunningProcess:
        returncode = None
        stdout = None

        def poll(self) -> None:
            return None

    source = RTSPVideoSource(rtsp_url="rtsp://127.0.0.1:8554/live", video_path="sample_video.mp4")
    setattr(source, "_ingress", _EmptyIngress())
    setattr(source, "_decoder_proc", _RunningProcess())

    assert source.read() is None


def test_decoder_stderr_is_captured_not_discarded() -> None:
    """The reason FFmpeg gave has to survive as far as the error message."""
    source = _source_with_dead_decoder()
    handle = tempfile.TemporaryFile()
    handle.write(b"method DESCRIBE failed: 404 (Not Found)\n")
    setattr(source, "_decoder_stderr", handle)

    with pytest.raises(RuntimeError, match="404"):
        source.read()
    handle.close()


def test_open_refuses_a_path_that_is_not_published() -> None:
    """End to end against a real FFmpeg: a 404 must surface, and quickly.

    This is the exact condition that hung the publication run.
    """
    source = RTSPVideoSource(
        rtsp_url="rtsp://127.0.0.1:8554/path_that_does_not_exist_for_tests",
        video_path="sample_video.mp4",
    )
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError) as failure:
            source.open()
    except FileNotFoundError:  # pragma: no cover - environment without ffmpeg
        pytest.skip("ffmpeg is not installed")
    finally:
        try:
            source.close()
        except Exception:
            pass

    elapsed = time.monotonic() - started
    assert elapsed < 30.0, f"a missing RTSP path took {elapsed:.1f}s to fail"
    assert "No frame arrived" in str(failure.value)
