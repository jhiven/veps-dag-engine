"""CLI smoke integration test."""

from __future__ import annotations

import os
import tempfile

from benchmarks.cli import report_cmd, run_benchmarks, verify_cmd
from benchmarks.storage import read_reconfiguration_rows


def test_cli_smoke_run_and_verify() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = run_benchmarks(suite="all", profile="smoke", base_output_dir=tmp_dir, seed=42)

        assert os.path.exists(run_dir)
        assert os.path.exists(os.path.join(run_dir, "run.json"))
        assert os.path.exists(os.path.join(run_dir, "completion.json"))

        reconfig_csv = os.path.join(run_dir, "reconfiguration-samples.csv")
        stress_csv = os.path.join(run_dir, "reconfiguration-stress-samples.csv")

        assert os.path.exists(reconfig_csv)
        assert os.path.exists(stress_csv)

        reconfig_rows = read_reconfiguration_rows(reconfig_csv)
        stress_rows = read_reconfiguration_rows(stress_csv)

        assert len(reconfig_rows) > 0
        assert len(stress_rows) > 0

        for r in reconfig_rows + stress_rows:
            assert r.admission_stop_ns is not None
            assert r.executor_teardown_ns is not None
            assert r.executor_reconstruction_ns is not None
            assert r.publication_ns is not None
            assert r.executor_restart_ns is not None
            assert r.first_admission_wait_ns is not None
            assert r.first_completion_wait_ns is not None
            assert r.retirement_ns is not None
            assert r.total_synchronous_ns is not None
            assert r.instrumented_phase_sum_ns is not None
            assert r.unattributed_request_time_ns is not None

        # Verify artifact
        verify_cmd(run_dir)

        # Re-run report generation
        report_cmd(run_dir)
