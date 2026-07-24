"""Command line interface for benchmark suite execution, report generation, and artifact verification."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys
import time
import uuid

from benchmarks.metadata import (
    CompletionMetadata,
    FailureMetadata,
    RunMetadata,
    load_completion_json,
    load_run_json,
    write_completion_json,
    write_failure_json,
    write_run_json,
)
from benchmarks.reporting.ablation import generate_ablation_summary_files
from benchmarks.reporting.figures import generate_all_figures
from benchmarks.reporting.interference import generate_interference_summary_file
from benchmarks.reporting.load import load_benchmark_artifact
from benchmarks.reporting.summarize import generate_summary_csv
from benchmarks.reporting.tables import generate_all_tables
from benchmarks.runners.ablation import run_ablation_suite
from benchmarks.runners.conformance import run_conformance_suite
from benchmarks.runners.interference import run_interference_suite
from benchmarks.runners.reconfiguration import run_reconfiguration_suite
from benchmarks.runners.reconfiguration_stress import run_reconfiguration_stress_suite
from benchmarks.runners.steady_state import run_steady_state_suite
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    build_branch_merge_9_spec,
    build_linear_5_spec,
    calibrate_workload_iterations,
    create_reconfiguration_registry,
    create_workload_registry,
    generate_layered_dag,
    make_reconfiguration_base_spec,
)
from benchmarks.reporting.validation import (
    validate_experiment_artifacts,
    write_experiment_validation_json,
)
from benchmarks.storage import (
    read_ablation_rows,
    read_interference_rows,
    read_reconfiguration_rows,
)
from benchmarks.system import collect_system_environment
from nedo_vision_dag_engine.specification import specification_hash
from tests.support.workflows import linear, tracker_only

__all__ = [
    "run_benchmarks",
    "report_cmd",
    "verify_cmd",
    "main",
]


def _calculate_file_sha256(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def _count_csv_data_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    return max(0, len(lines) - 1)


def run_benchmarks(
    suite: str,
    profile: str,
    base_output_dir: str = "benchmark-results",
    seed: int = 42,
) -> str:
    run_id = f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    utc_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start_time_ns = time.monotonic_ns()

    valid_suite_names = {"steady-state", "reconfiguration", "reconfiguration-stress", "conformance", "ablation", "interference"}
    if suite == "all":
        selected_suites = ("steady-state", "reconfiguration", "conformance", "reconfiguration-stress", "ablation", "interference")
    else:
        parts = [s.strip() for s in suite.split(",") if s.strip()]
        for p in parts:
            if p not in valid_suite_names:
                raise ValueError(
                    f"Unknown benchmark suite {p!r}. Valid options: 'all', 'steady-state', 'reconfiguration', 'conformance', 'ablation', 'interference' (or comma-separated combination)."
                )
        selected_suites = tuple(parts)

    pin_cpus = (0,) if selected_suites == ("steady-state",) else (0, 2)
    env = collect_system_environment(pin_cpus=pin_cpus)

    selected_scenarios: list[str] = []
    if "steady-state" in selected_suites:
        selected_scenarios.extend(["linear_5", "branch_merge_9"])
    if "reconfiguration" in selected_suites:
        selected_scenarios.extend([
            "insert_stateless_node",
            "remove_stateless_node",
            "rewire_stateless_edge",
            "compatible_edit_preserving_tracker",
        ])
    if "conformance" in selected_suites:
        selected_scenarios.extend(["frame_consistency", "failure_atomicity", "stateful"])
    if "reconfiguration-stress" in selected_suites:
        selected_scenarios.extend([
            "stress_delay0ms",
            "stress_delay20ms",
            "stress_delay50ms",
        ])
    if "ablation" in selected_suites:
        selected_scenarios.extend(["ablation_remove_stateless_node", "ablation_compatible_edit_preserving_tracker"])
    if "interference" in selected_suites:
        selected_scenarios.extend([
            "interference_no_candidate_preparation",
            "interference_sleep_preparation_20ms",
            "interference_sleep_preparation_50ms",
            "interference_cpu_bound_preparation_20ms",
            "interference_cpu_bound_preparation_50ms",
        ])

    repetition_count = 5 if profile == "smoke" else 30
    warmup_count = 5 if profile == "smoke" else 50
    ops_per_rep = 50 if profile == "smoke" else 1000

    # Calibrate workloads
    calibrated_details = calibrate_workload_iterations(
        calibration_repetitions=5 if profile == "smoke" else 30
    )
    calibrated_iters = {k: v.calibrated_iterations for k, v in calibrated_details.items()}

    # Compute canonical specification hashes
    base_spec = make_reconfiguration_base_spec()
    wf_hashes = {
        "linear_5": specification_hash(build_linear_5_spec()),
        "branch_merge_9": specification_hash(build_branch_merge_9_spec()),
        "reconfiguration_base": specification_hash(base_spec),
        "reconfig_insert_stateless_node": specification_hash(apply_reconfiguration_edit(base_spec, "insert_stateless_node")),
        "reconfig_remove_stateless_node": specification_hash(apply_reconfiguration_edit(base_spec, "remove_stateless_node")),
        "reconfig_rewire_stateless_edge": specification_hash(apply_reconfiguration_edit(base_spec, "rewire_stateless_edge")),
        "reconfig_compatible_edit_preserving_tracker": specification_hash(apply_reconfiguration_edit(base_spec, "compatible_edit_preserving_tracker")),
        "graph_size_5": specification_hash(generate_layered_dag(5)),
        "graph_size_25": specification_hash(generate_layered_dag(25)),
        "graph_size_100": specification_hash(generate_layered_dag(100)),
        "conformance_stress": specification_hash(base_spec),
        "conformance_failure_atomicity": specification_hash(linear()),
        "conformance_stateful": specification_hash(tracker_only("synthetic_tracker")),
    }

    # Compute registry snapshot identifiers
    reg_snapshot_ids = {
        "steady_state": f"registry:sha256:{create_workload_registry(calibrated_iters['approximately_1_ms']).snapshot_id}",
        "reconfiguration": f"registry:sha256:{create_reconfiguration_registry(extra_tracker=True).snapshot_id}",
        "conformance": f"registry:sha256:{create_reconfiguration_registry().snapshot_id}",
    }

    run_meta = RunMetadata.create(
        run_id=run_id,
        utc_start_datetime=utc_start,
        benchmark_command=f"uv run python -m benchmarks.cli run {suite} --{profile}",
        selected_suites=selected_suites,
        selected_scenarios=tuple(selected_scenarios),
        env=env,
        benchmark_seed=seed,
        warmup_count=warmup_count,
        profile=profile,
        repetition_count=repetition_count,
        operations_per_repetition=ops_per_rep,
        workflow_specification_hashes=wf_hashes,
        registry_snapshot_identifiers=reg_snapshot_ids,
        compiler_version="0.1.0",
        workload_calibration=calibrated_details,
    )

    run_json_path = os.path.join(run_dir, "run.json")
    write_run_json(run_json_path, run_meta)

    row_counts: dict[str, int] = {}
    sha256_dict: dict[str, str] = {}

    try:
        if "steady-state" in selected_suites:
            steady_csv = os.path.join(run_dir, "steady-state-samples.csv")
            run_steady_state_suite(
                run_id=run_id,
                output_csv_path=steady_csv,
                profile=profile,
                repetition_count=repetition_count,
                calibrated_iterations=calibrated_iters,
                seed=seed,
            )
            row_counts["steady-state-samples.csv"] = _count_csv_data_rows(steady_csv)
            sha256_dict["steady-state-samples.csv"] = _calculate_file_sha256(steady_csv)

        if "reconfiguration" in selected_suites:
            reconfig_csv = os.path.join(run_dir, "reconfiguration-samples.csv")
            run_reconfiguration_suite(
                run_id=run_id,
                output_csv_path=reconfig_csv,
                profile=profile,
                repetition_count=repetition_count,
                seed=seed,
            )
            row_counts["reconfiguration-samples.csv"] = _count_csv_data_rows(reconfig_csv)
            sha256_dict["reconfiguration-samples.csv"] = _calculate_file_sha256(reconfig_csv)

        if "reconfiguration-stress" in selected_suites:
            stress_csv = os.path.join(run_dir, "reconfiguration-stress-samples.csv")
            run_reconfiguration_stress_suite(
                run_id=run_id,
                output_csv_path=stress_csv,
                profile=profile,
                repetition_count=repetition_count,
                seed=seed,
            )
            row_counts["reconfiguration-stress-samples.csv"] = _count_csv_data_rows(stress_csv)
            sha256_dict["reconfiguration-stress-samples.csv"] = _calculate_file_sha256(stress_csv)

        if "conformance" in selected_suites:
            conformance_csv = os.path.join(run_dir, "conformance-results.csv")
            run_conformance_suite(
                run_id=run_id,
                output_csv_path=conformance_csv,
                profile=profile,
                seed=seed,
            )
            row_counts["conformance-results.csv"] = _count_csv_data_rows(conformance_csv)
            sha256_dict["conformance-results.csv"] = _calculate_file_sha256(conformance_csv)

        if "ablation" in selected_suites:
            ablation_csv = os.path.join(run_dir, "ablation-samples.csv")
            run_ablation_suite(
                output_dir=run_dir,
                run_id=run_id,
                repetitions=repetition_count,
            )
            row_counts["ablation-samples.csv"] = _count_csv_data_rows(ablation_csv)
            sha256_dict["ablation-samples.csv"] = _calculate_file_sha256(ablation_csv)

        if "interference" in selected_suites:
            interference_csv = os.path.join(run_dir, "interference-samples.csv")
            run_interference_suite(
                run_id=run_id,
                output_csv_path=interference_csv,
                repetition_count=repetition_count,
            )
            row_counts["interference-samples.csv"] = _count_csv_data_rows(interference_csv)
            sha256_dict["interference-samples.csv"] = _calculate_file_sha256(interference_csv)

        # Generate summary, tables, and figures for selected suite data
        summary_csv = os.path.join(run_dir, "summary.csv")
        tables_dir = os.path.join(run_dir, "tables")
        figures_dir = os.path.join(run_dir, "figures")

        artifact_data = load_benchmark_artifact(run_dir)
        generate_summary_csv(artifact_data, summary_csv)
        table_paths = generate_all_tables(artifact_data, tables_dir)
        figure_paths = generate_all_figures(artifact_data, figures_dir)

        duration_sec = (time.monotonic_ns() - start_time_ns) / 1e9
        utc_comp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        completion_meta = CompletionMetadata(
            run_id=run_id,
            utc_completion_datetime=utc_comp,
            status="SUCCESS",
            total_duration_seconds=duration_sec,
            raw_file_row_counts=row_counts,
            raw_file_sha256=sha256_dict,
            generated_table_paths=tuple(table_paths),
            generated_figure_paths=tuple(figure_paths),
        )

        completion_json_path = os.path.join(run_dir, "completion.json")
        write_completion_json(completion_json_path, completion_meta)

        print(f"Benchmark run completed successfully: {run_dir}")
        return run_dir

    except Exception as err:
        fail_meta = FailureMetadata(
            run_id=run_id,
            utc_failure_datetime=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status="FAILED",
            error_message=repr(err),
        )
        write_failure_json(os.path.join(run_dir, "failure.json"), fail_meta)
        print(f"Benchmark run failed: {err}", file=sys.stderr)
        raise


def report_cmd(run_directory: str) -> None:
    artifact_data = load_benchmark_artifact(run_directory)
    summary_csv = os.path.join(run_directory, "summary.csv")
    tables_dir = os.path.join(run_directory, "tables")
    figures_dir = os.path.join(run_directory, "figures")

    generate_summary_csv(artifact_data, summary_csv)
    t_paths = generate_all_tables(artifact_data, tables_dir)
    f_paths = generate_all_figures(artifact_data, figures_dir)

    ablation_csv = os.path.join(run_directory, "ablation-samples.csv")
    if os.path.exists(ablation_csv):
        generate_ablation_summary_files(ablation_csv, run_directory)

    interference_csv = os.path.join(run_directory, "interference-samples.csv")
    if os.path.exists(interference_csv):
        interference_summary_csv = os.path.join(run_directory, "interference-summary.csv")
        generate_interference_summary_file(interference_csv, interference_summary_csv)

    ablation_rows = read_ablation_rows(ablation_csv) if os.path.exists(ablation_csv) else ()
    interference_rows = read_interference_rows(interference_csv) if os.path.exists(interference_csv) else ()

    if ablation_rows or interference_rows:
        val_report = validate_experiment_artifacts(
            ablation_rows=ablation_rows,
            interference_rows=interference_rows,
        )
        write_experiment_validation_json(val_report, os.path.join(run_directory, "experiment-validation.json"))

    print(f"Report generated for {run_directory}:")
    print(f"  Tables: {len(t_paths)} files")
    print(f"  Figures: {len(f_paths)} files")


def verify_cmd(run_directory: str) -> None:
    run_json = os.path.join(run_directory, "run.json")
    comp_json = os.path.join(run_directory, "completion.json")
    fail_json = os.path.join(run_directory, "failure.json")

    if os.path.exists(fail_json):
        raise ValueError(f"Run directory {run_directory} contains failure.json; run failed.")

    if not os.path.exists(run_json):
        raise ValueError(f"Missing run.json in {run_directory}")
    if not os.path.exists(comp_json):
        raise ValueError(f"Missing completion.json in {run_directory}")

    run_meta = load_run_json(run_json)
    comp_meta = load_completion_json(comp_json)

    if run_meta.run_id != comp_meta.run_id:
        raise ValueError(f"Run ID mismatch: run.json has {run_meta.run_id}, completion.json has {comp_meta.run_id}")

    for fname in comp_meta.raw_file_row_counts.keys():
        fpath = os.path.join(run_directory, fname)
        if not os.path.exists(fpath):
            raise ValueError(f"Missing expected raw CSV file: {fname}")

        expected_count = comp_meta.raw_file_row_counts.get(fname)
        actual_count = _count_csv_data_rows(fpath)
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(f"Row count mismatch for {fname}: expected {expected_count}, got {actual_count}")

        expected_sha = comp_meta.raw_file_sha256.get(fname)
        actual_sha = _calculate_file_sha256(fpath)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {fname}: expected {expected_sha}, got {actual_sha}")

    # Check non-negative intervals and interval relationships in reconfiguration samples
    reconfig_csv = os.path.join(run_directory, "reconfiguration-samples.csv")
    if os.path.exists(reconfig_csv):
        reconfig_samples = read_reconfiguration_rows(reconfig_csv)
        for s in reconfig_samples:
            for val in (s.validation_ns, s.preparation_ns, s.request_to_ready_ns, s.boundary_wait_ns, s.commit_ns, s.request_to_effect_ns, s.retirement_queue_delay_ns, s.retirement_duration_ns, s.commit_to_retirement_complete_ns, s.maximum_output_gap_ns):
                if val is not None and val < 0:
                    raise ValueError(f"Negative timing interval found in reconfiguration record: {s}")

            if s.request_to_ready_ns is not None and s.validation_ns is not None:
                if s.request_to_ready_ns < s.validation_ns:
                    raise ValueError(f"Invalid interval relationship: request_to_ready_ns ({s.request_to_ready_ns}) < validation_ns ({s.validation_ns})")

            if s.request_to_effect_ns is not None and s.request_to_ready_ns is not None:
                if s.request_to_effect_ns < s.request_to_ready_ns:
                    raise ValueError(f"Invalid interval relationship: request_to_effect_ns ({s.request_to_effect_ns}) < request_to_ready_ns ({s.request_to_ready_ns})")

    # Check generated tables and figures exist
    for t_path in comp_meta.generated_table_paths:
        if not os.path.exists(t_path):
            raise ValueError(f"Missing generated table file: {t_path}")
    for f_path in comp_meta.generated_figure_paths:
        if not os.path.exists(f_path):
            raise ValueError(f"Missing generated figure file: {f_path}")

    print(f"Verification PASSED for run artifact: {run_directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible DAG Engine Benchmark CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "suite",
        type=str,
        help="Benchmark suite(s) to run ('all', 'steady-state', 'reconfiguration', 'conformance', or comma-separated list e.g. 'reconfiguration,conformance')",
    )
    profile_group = run_parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument("--smoke", action="store_true", help="Run smoke profile for CI/validation")
    profile_group.add_argument("--publication", action="store_true", help="Run publication profile")

    # report subcommand
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run_directory", type=str)

    # verify subcommand
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("run_directory", type=str)

    args = parser.parse_args()

    if args.subcommand == "run":
        profile = "smoke" if args.smoke else "publication"
        run_benchmarks(suite=args.suite, profile=profile)
    elif args.subcommand == "report":
        report_cmd(args.run_directory)
    elif args.subcommand == "verify":
        verify_cmd(args.run_directory)


if __name__ == "__main__":
    main()
