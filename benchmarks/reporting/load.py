"""Data loading helper for benchmark raw CSV files."""

from __future__ import annotations

import os
from dataclasses import dataclass

from benchmarks.model import (
    ConformanceResultRow,
    ReconfigurationSampleRow,
    SteadyStateSampleRow,
)
from benchmarks.storage import (
    read_conformance_rows,
    read_reconfiguration_rows,
    read_steady_state_rows,
)

__all__ = ["BenchmarkArtifactData", "load_benchmark_artifact"]


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactData:
    run_directory: str
    steady_state_samples: tuple[SteadyStateSampleRow, ...]
    reconfiguration_samples: tuple[ReconfigurationSampleRow, ...]
    conformance_results: tuple[ConformanceResultRow, ...]


def load_benchmark_artifact(run_directory: str) -> BenchmarkArtifactData:
    steady_path = os.path.join(run_directory, "steady-state-samples.csv")
    reconfig_path = os.path.join(run_directory, "reconfiguration-samples.csv")
    conformance_path = os.path.join(run_directory, "conformance-results.csv")

    steady_samples = read_steady_state_rows(steady_path) if os.path.exists(steady_path) else ()
    reconfig_samples = read_reconfiguration_rows(reconfig_path) if os.path.exists(reconfig_path) else ()
    conformance_results = read_conformance_rows(conformance_path) if os.path.exists(conformance_path) else ()

    return BenchmarkArtifactData(
        run_directory=run_directory,
        steady_state_samples=steady_samples,
        reconfiguration_samples=reconfig_samples,
        conformance_results=conformance_results,
    )
