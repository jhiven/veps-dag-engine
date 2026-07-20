"""Run, completion, and failure metadata models and JSON persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.scenarios import WorkloadCalibrationDetail
from benchmarks.system import SystemEnvironment

__all__ = [
    "RunMetadata",
    "CompletionMetadata",
    "FailureMetadata",
    "write_run_json",
    "load_run_json",
    "write_completion_json",
    "load_completion_json",
    "write_failure_json",
]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    artifact_schema_version: str
    benchmark_suite_version: str
    run_id: str
    utc_start_datetime: str
    benchmark_command: str
    selected_suites: tuple[str, ...]
    selected_scenarios: tuple[str, ...]
    git_commit: str
    dirty_working_tree: bool
    python_implementation: str
    python_version: str
    uv_lock_sha256: str
    operating_system: str
    kernel_version: str
    hostname: str
    cpu_model: str
    logical_cpu_count: int
    available_cpu_affinity: tuple[int, ...]
    applied_cpu_affinity: tuple[int, ...]
    timer_implementation: str
    timer_resolution_ns: float
    benchmark_seed: int
    warmup_count: int
    publication_or_smoke_profile: str
    repetition_count: int
    operations_per_repetition: int
    garbage_collector_state: bool
    workflow_specification_hashes: dict[str, str]
    registry_snapshot_identifiers: dict[str, str]
    compiler_version: str
    equivalence_margins: dict[str, float]
    scenario_ordering_policy: str
    workload_calibration: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_id: str,
        utc_start_datetime: str,
        benchmark_command: str,
        selected_suites: tuple[str, ...],
        selected_scenarios: tuple[str, ...],
        env: SystemEnvironment,
        benchmark_seed: int,
        warmup_count: int,
        profile: str,
        repetition_count: int,
        operations_per_repetition: int,
        workflow_specification_hashes: dict[str, str],
        registry_snapshot_identifiers: dict[str, str],
        compiler_version: str,
        workload_calibration: dict[str, Any],
    ) -> RunMetadata:
        return cls(
            artifact_schema_version="1.4.0",
            benchmark_suite_version="0.1.0",
            run_id=run_id,
            utc_start_datetime=utc_start_datetime,
            benchmark_command=benchmark_command,
            selected_suites=selected_suites,
            selected_scenarios=selected_scenarios,
            git_commit=env.git_commit,
            dirty_working_tree=env.dirty_working_tree,
            python_implementation=env.python_implementation,
            python_version=env.python_version,
            uv_lock_sha256=env.uv_lock_sha256,
            operating_system=env.operating_system,
            kernel_version=env.kernel_version,
            hostname=env.hostname,
            cpu_model=env.cpu_model,
            logical_cpu_count=env.logical_cpu_count,
            available_cpu_affinity=env.available_cpu_affinity,
            applied_cpu_affinity=env.applied_cpu_affinity,
            timer_implementation=env.timer_implementation,
            timer_resolution_ns=env.timer_resolution_ns,
            benchmark_seed=benchmark_seed,
            warmup_count=warmup_count,
            publication_or_smoke_profile=profile,
            repetition_count=repetition_count,
            operations_per_repetition=operations_per_repetition,
            garbage_collector_state=True,
            workflow_specification_hashes=workflow_specification_hashes,
            registry_snapshot_identifiers=registry_snapshot_identifiers,
            compiler_version=compiler_version,
            equivalence_margins={"relative_margin": 0.01},
            scenario_ordering_policy="counterbalanced_seeded_random",
            workload_calibration=workload_calibration,
        )


@dataclass(frozen=True, slots=True)
class CompletionMetadata:
    run_id: str
    utc_completion_datetime: str
    status: str
    total_duration_seconds: float
    raw_file_row_counts: dict[str, int]
    raw_file_sha256: dict[str, str]
    generated_table_paths: tuple[str, ...]
    generated_figure_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FailureMetadata:
    run_id: str
    utc_failure_datetime: str
    status: str
    error_message: str


def write_run_json(path: str, metadata: RunMetadata) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def _parse_workload_calibration(val: Any) -> WorkloadCalibrationDetail | Any:
    if hasattr(val, "get") and callable(getattr(val, "get", None)):
        return WorkloadCalibrationDetail(
            workload_id=str(val.get("workload_id")),
            calibrated_iterations=int(str(val.get("calibrated_iterations"))),
            median_ns=float(str(val.get("median_ns"))),
            p95_ns=float(str(val.get("p95_ns"))),
            calibration_repetitions=int(str(val.get("calibration_repetitions"))),
            target_description=str(val.get("target_description")),
        )
    return val


def load_run_json(path: str) -> RunMetadata:
    with open(path, "r", encoding="utf-8") as file:
        d: dict[str, Any] = json.load(file)
    return RunMetadata(
        artifact_schema_version=str(d["artifact_schema_version"]),
        benchmark_suite_version=str(d["benchmark_suite_version"]),
        run_id=str(d["run_id"]),
        utc_start_datetime=str(d["utc_start_datetime"]),
        benchmark_command=str(d["benchmark_command"]),
        selected_suites=tuple(str(x) for x in d["selected_suites"]),
        selected_scenarios=tuple(str(x) for x in d["selected_scenarios"]),
        git_commit=str(d["git_commit"]),
        dirty_working_tree=bool(d["dirty_working_tree"]),
        python_implementation=str(d["python_implementation"]),
        python_version=str(d["python_version"]),
        uv_lock_sha256=str(d["uv_lock_sha256"]),
        operating_system=str(d["operating_system"]),
        kernel_version=str(d["kernel_version"]),
        hostname=str(d["hostname"]),
        cpu_model=str(d["cpu_model"]),
        logical_cpu_count=int(d["logical_cpu_count"]),
        available_cpu_affinity=tuple(int(x) for x in d["available_cpu_affinity"]),
        applied_cpu_affinity=tuple(int(x) for x in d["applied_cpu_affinity"]),
        timer_implementation=str(d["timer_implementation"]),
        timer_resolution_ns=float(d["timer_resolution_ns"]),
        benchmark_seed=int(d["benchmark_seed"]),
        warmup_count=int(d["warmup_count"]),
        publication_or_smoke_profile=str(d["publication_or_smoke_profile"]),
        repetition_count=int(d["repetition_count"]),
        operations_per_repetition=int(d["operations_per_repetition"]),
        garbage_collector_state=bool(d["garbage_collector_state"]),
        workflow_specification_hashes={
            str(k): str(v) for k, v in d["workflow_specification_hashes"].items()
        },
        registry_snapshot_identifiers={
            str(k): str(v) for k, v in d["registry_snapshot_identifiers"].items()
        },
        compiler_version=str(d["compiler_version"]),
        equivalence_margins={
            str(k): float(v) for k, v in d["equivalence_margins"].items()
        },
        scenario_ordering_policy=str(d["scenario_ordering_policy"]),
        workload_calibration={
            str(k): _parse_workload_calibration(v)
            for k, v in d["workload_calibration"].items()
        },
    )


def write_completion_json(path: str, metadata: CompletionMetadata) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def load_completion_json(path: str) -> CompletionMetadata:
    with open(path, "r", encoding="utf-8") as file:
        d: dict[str, Any] = json.load(file)
    return CompletionMetadata(
        run_id=str(d["run_id"]),
        utc_completion_datetime=str(d["utc_completion_datetime"]),
        status=str(d["status"]),
        total_duration_seconds=float(d["total_duration_seconds"]),
        raw_file_row_counts={
            str(k): int(v) for k, v in d["raw_file_row_counts"].items()
        },
        raw_file_sha256={str(k): str(v) for k, v in d["raw_file_sha256"].items()},
        generated_table_paths=tuple(str(x) for x in d["generated_table_paths"]),
        generated_figure_paths=tuple(str(x) for x in d["generated_figure_paths"]),
    )


def write_failure_json(path: str, metadata: FailureMetadata) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)
