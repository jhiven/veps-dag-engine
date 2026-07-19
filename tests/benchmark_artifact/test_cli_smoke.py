"""CLI smoke integration test."""

from __future__ import annotations

import os
import tempfile

from benchmarks.cli import report_cmd, run_benchmarks, verify_cmd


def test_cli_smoke_run_and_verify() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = run_benchmarks(suite="all", profile="smoke", base_output_dir=tmp_dir, seed=42)

        assert os.path.exists(run_dir)
        assert os.path.exists(os.path.join(run_dir, "run.json"))
        assert os.path.exists(os.path.join(run_dir, "completion.json"))

        # Verify artifact
        verify_cmd(run_dir)

        # Re-run report generation
        report_cmd(run_dir)
