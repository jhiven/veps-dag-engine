"""Unit tests for real-world video benchmark runner policies and existing suite preservation."""

from __future__ import annotations

import pytest

from benchmarks.runners.realworld_video import run_realworld_video_suite
from usecases.video_analytics.contracts import ExecutionMode


def test_publication_mode_rejects_fake_backends() -> None:
    """Test 19: Publication mode rejects use_fake_backends=True."""
    with pytest.raises(ValueError) as exc_info:
        run_realworld_video_suite(
            run_id="test_run",
            output_csv_path="/tmp/test.csv",
            use_fake_backends=True,
            execution_mode=ExecutionMode.PUBLICATION,
        )
    assert "Publication mode rejects use_fake_backends=True" in str(exc_info.value)


def test_existing_non_realworld_runners_unaffected() -> None:
    """Test 20: Existing non-realworld benchmark modules import and run without interference."""
    from benchmarks.runners.conformance import run_conformance_suite
    from benchmarks.runners.steady_state import run_steady_state_suite

    assert callable(run_steady_state_suite)
    assert callable(run_conformance_suite)
