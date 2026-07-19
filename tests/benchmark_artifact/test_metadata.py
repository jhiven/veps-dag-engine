"""Tests for run and completion metadata serialization and verification."""

from __future__ import annotations

import tempfile
from benchmarks.metadata import (
    CompletionMetadata,
    RunMetadata,
    load_completion_json,
    load_run_json,
    write_completion_json,
    write_run_json,
)
from benchmarks.system import collect_system_environment


def test_metadata_serialization_roundtrip() -> None:
    env = collect_system_environment()
    run_meta = RunMetadata.create(
        run_id="test_run_1",
        utc_start_datetime="2026-07-19T12:00:00Z",
        benchmark_command="uv run python -m benchmarks.cli run all --smoke",
        selected_suites=("steady-state", "reconfiguration", "conformance"),
        selected_scenarios=("linear_5", "branch_merge_9"),
        env=env,
        benchmark_seed=42,
        warmup_count=5,
        profile="smoke",
        repetition_count=5,
        operations_per_repetition=50,
        workflow_specification_hashes={"spec1": "hash1"},
        registry_snapshot_identifiers={"reg1": "id1"},
        compiler_version="0.1.0",
        workload_calibration={"minimal": 0, "approximately_1_ms": 1000, "approximately_5_ms": 5000},
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        path = f.name

    write_run_json(path, run_meta)
    loaded_meta = load_run_json(path)
    assert loaded_meta == run_meta


def test_completion_metadata_serialization_roundtrip() -> None:
    comp_meta = CompletionMetadata(
        run_id="test_run_1",
        utc_completion_datetime="2026-07-19T12:01:00Z",
        status="SUCCESS",
        total_duration_seconds=60.0,
        raw_file_row_counts={"steady-state-samples.csv": 10},
        raw_file_sha256={"steady-state-samples.csv": "abc"},
        generated_table_paths=("tables/table1.csv",),
        generated_figure_paths=("figures/fig1.pdf",),
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        path = f.name

    write_completion_json(path, comp_meta)
    loaded_meta = load_completion_json(path)
    assert loaded_meta == comp_meta
