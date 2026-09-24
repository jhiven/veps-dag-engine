"""Tests for the artifact validation gate and its CLI subcommand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from benchmarks.conformance_trace import SCHEMA_VERSION
from benchmarks.runners.conformance import run_conformance_suite
from benchmarks.validation_gate import validate_benchmark_directory

REQUIRED_CAMPAIGNS = (
    "failure_atomicity",
    "state_policy",
    "stateless_publication",
    "mutable_handoff",
    "candidate_failure",
    "cleanup_failure",
    "frame_exception",
    "successive_generation_ownership",
)
RANDOMIZED_SEEDS = (42, 314159, 271828, 161803, 20260922)


def _write_metadata(run_dir: str, selected_suites: list[str]) -> None:
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as file:
        json.dump({"run_id": "run_test", "selected_suites": selected_suites}, file)
    with open(os.path.join(run_dir, "completion.json"), "w", encoding="utf-8") as file:
        json.dump({"run_id": "run_test", "status": "COMPLETED"}, file)


def _write_full_conformance(run_dir: str) -> str:
    directory = os.path.join(run_dir, "conformance")
    run_conformance_suite("run_test", directory, "smoke", RANDOMIZED_SEEDS)
    return directory


def test_gate_reports_a_missing_directory() -> None:
    report = validate_benchmark_directory(os.path.join(tempfile.gettempdir(), "no-such-run-dir"))
    assert not report.all_passed
    assert report.errors


def test_gate_rejects_a_run_that_recorded_a_failure() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, [])
        with open(os.path.join(run_dir, "failure.json"), "w", encoding="utf-8") as file:
            json.dump({"status": "FAILED"}, file)

        report = validate_benchmark_directory(run_dir)
        assert not report.all_passed
        assert any(not result.passed for result in report.results)


def test_gate_accepts_a_complete_conformance_directory() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        _write_full_conformance(run_dir)

        report = validate_benchmark_directory(run_dir)
        assert report.all_passed, report.to_dict()


def test_gate_rejects_a_missing_required_campaign() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        directory = _write_full_conformance(run_dir)
        os.remove(os.path.join(directory, "cleanup_failure-seed-42.jsonl"))

        report = validate_benchmark_directory(run_dir)
        assert not report.all_passed
        assert any("cleanup_failure" in result.detail for result in report.results)


def test_gate_rejects_a_missing_randomized_seed() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        directory = _write_full_conformance(run_dir)
        os.remove(os.path.join(directory, "randomized_consistency-seed-271828.jsonl"))

        report = validate_benchmark_directory(run_dir)
        assert not report.all_passed
        assert any("seeds mismatch" in result.detail for result in report.results)


def test_gate_rejects_a_tampered_trace() -> None:
    """A trace edited after the fact must not replay cleanly."""
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        directory = _write_full_conformance(run_dir)

        tampered = os.path.join(directory, "frame_exception-seed-42.jsonl")
        with open(tampered, "r", encoding="utf-8") as file:
            events = [json.loads(line) for line in file if line.strip()]
        # Drop the lease release so the frame keeps an unreleased lease.
        events = [event for event in events if event["event_kind"] != "lease_released"]
        for sequence, event in enumerate(events, 1):
            event["event_sequence"] = sequence
        with open(tampered, "w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

        report = validate_benchmark_directory(run_dir)
        assert not report.all_passed
        assert any("unreleased leases" in result.detail for result in report.results)


def test_gate_rejects_an_unsupported_schema_version() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        directory = _write_full_conformance(run_dir)

        path = os.path.join(directory, "state_policy-seed-42.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        assert SCHEMA_VERSION in content
        with open(path, "w", encoding="utf-8") as file:
            file.write(content.replace(SCHEMA_VERSION, "1.0.0"))

        report = validate_benchmark_directory(run_dir)
        assert not report.all_passed
        assert any("schema_version" in result.detail for result in report.results)


@pytest.mark.parametrize("passing", [True, False])
def test_validate_gate_subcommand_is_reachable(passing: bool) -> None:
    """The subcommand must dispatch and exit non-zero on a failing directory."""
    with tempfile.TemporaryDirectory() as run_dir:
        _write_metadata(run_dir, ["conformance"])
        directory = _write_full_conformance(run_dir)
        if not passing:
            os.remove(os.path.join(directory, "mutable_handoff-seed-42.jsonl"))

        # Match the parent's explicit free-threaded mode. A child Python
        # process does not inherit the parent's -Xgil=0 interpreter option.
        gil_probe = getattr(sys, "_is_gil_enabled", None)
        interpreter_options = ["-Xgil=0"] if callable(gil_probe) and not gil_probe() else []
        completed = subprocess.run(
            [sys.executable, *interpreter_options, "-m", "benchmarks.cli", "validate-gate", run_dir],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

        assert completed.returncode == (0 if passing else 1), completed.stdout + completed.stderr
