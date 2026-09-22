# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Command line interface for running DAG engine benchmark suites."""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import time
from typing import Any

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
from benchmarks.provenance import collect_provenance, missing_required_provenance
from benchmarks.reporting.ablation import generate_ablation_summary_files
from benchmarks.reporting.figures import generate_all_figures
from benchmarks.reporting.interference import generate_interference_summary_file
from benchmarks.reporting.load import load_benchmark_artifact
from benchmarks.reporting.summarize import generate_summary_csv
from benchmarks.reporting.tables import generate_all_tables
from benchmarks.reporting.validation import validate_experiment_artifacts, write_experiment_validation_json
from benchmarks.runners.ablation import run_ablation_suite
from benchmarks.runners.conformance import run_conformance_suite
from benchmarks.runners.interference import run_interference_suite
from benchmarks.runners.realworld_video import run_realworld_video_suite
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
from benchmarks.storage import (
    ReconfigurationSampleRow,
    read_ablation_rows,
    read_interference_rows,
    read_reconfiguration_rows,
)
from benchmarks.system import collect_system_environment
from nedo_vision_dag_engine.specification import specification_hash
from tests.support.workflows import linear, tracker_only

__all__ = ["run_benchmarks", "main"]


def _count_csv_data_rows(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        return max(0, len(lines) - 1)


def _count_raw_artifact_rows(filepath: str) -> int:
    """Count data records in a raw artifact.

    JSONL traces carry one record per line; CSV files carry a header that is
    not a record. Both promotion and verification must count the same way or a
    run cannot verify itself.
    """
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as raw_file:
            return sum(1 for line in raw_file if line.strip())
    return _count_csv_data_rows(filepath)


def _calculate_file_sha256(filepath: str) -> str:
    import hashlib

    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_benchmarks(
    suite: str,
    profile: str = "smoke",
    output_dir: str = "benchmark-results",
    base_output_dir: str | None = None,
    seed: int = 42,
    source_video: str = "sample_video.mp4",
    rtsp_base_url: str = "rtsp://127.0.0.1:8554",
    queue_capacity: int = 4,
    warmup_completed_frames: int = 30,
    realworld_initial_model: str = "PekingU/rtdetr_r18vd",
    realworld_candidate_model: str = "PekingU/rtdetr_r50vd",
    device: str = "auto",
    conformance_seeds: tuple[int, ...] = (42, 314159, 271828, 161803, 20260922),
    baseline_receiver_frames: int = 60,
    transition_receiver_frames: int = 150,
    drain_timeout_seconds: float = 30.0,
    operation_timeout_seconds: float = 120.0,
    exact_argument_vector: tuple[str, ...] = (),
) -> str:
    """Execute selected benchmark suite(s) and save structured artifact run directory."""
    if base_output_dir is not None:
        output_dir = base_output_dir

    start_time_ns = time.monotonic_ns()
    now = datetime.datetime.now(datetime.timezone.utc)
    utc_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"run_{now.strftime('%Y-%m-%dT%H-%M-%S')}_{seed}"

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    suite_list = [s.strip() for s in suite.split(",") if s.strip()]
    if "all" in suite_list:
        selected_suites = (
            "steady-state",
            "reconfiguration",
            "conformance",
            "reconfiguration-stress",
            "ablation",
            "interference",
            "realworld-video",
        )
    else:
        parts: list[str] = []
        for s in suite_list:
            if s in (
                "steady-state",
                "reconfiguration",
                "conformance",
                "reconfiguration-stress",
                "ablation",
                "interference",
                "realworld-video",
            ):
                parts.append(s)
            else:
                raise ValueError(f"Unknown benchmark suite: {s}")
        selected_suites = tuple(parts)

    # Validations for realworld-video suite
    if "realworld-video" in selected_suites:
        if queue_capacity <= 0:
            raise ValueError(f"Queue capacity must be positive, got {queue_capacity}")
        if not rtsp_base_url.startswith("rtsp://"):
            raise ValueError(f"Invalid RTSP URL: {rtsp_base_url!r}. Must start with 'rtsp://'")
        if profile == "publication":
            if not os.path.exists(source_video):
                raise FileNotFoundError(f"Source video file not found: {source_video}")
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg executable not found in PATH for RTSP publication benchmark.")

    available_cpus = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else tuple(range(os.cpu_count() or 1))
    requested_cpu_count = 1 if selected_suites == ("steady-state",) else min(2, len(available_cpus))
    pin_cpus = available_cpus[:requested_cpu_count]
    env = collect_system_environment(pin_cpus=pin_cpus)

    # Model checkpoints must be on disk before provenance can resolve their
    # revisions, otherwise a first run on a clean host fails its own preflight
    # for assets it was about to download anyway.
    if profile == "publication" and "realworld-video" in selected_suites:
        from usecases.video_analytics.cli import prepare_assets_cmd

        prepare_assets_cmd(
            initial_model=realworld_initial_model,
            candidate_model=realworld_candidate_model,
        )

    # Collected after every benchmark import so the recorded toolchain is the
    # one the run actually loaded, not the one that was merely installed.
    provenance = collect_provenance(
        selected_suites=selected_suites,
        device=device,
        source_video=source_video,
        rtsp_base_url=rtsp_base_url,
        model_ids=(realworld_initial_model, realworld_candidate_model),
    )
    if profile == "publication":
        failures: list[str] = []
        failures.extend(
            f"provenance field {field!r} is unavailable"
            for field in missing_required_provenance(provenance, selected_suites, device)
        )
        if env.dirty_working_tree:
            failures.append("Git working tree is dirty")
        if env.git_commit in {"", "unknown"}:
            failures.append("source commit is unknown")
        if env.uv_lock_sha256 in {"", "none", "unknown"}:
            failures.append("dependency lock hash is unavailable")
        if env.applied_cpu_affinity != pin_cpus:
            failures.append(
                f"requested CPU affinity {pin_cpus!r} was not applied; got {env.applied_cpu_affinity!r}"
            )
        if env.gil_enabled:
            failures.append("the GIL is enabled after benchmark imports")
        if failures:
            error = RuntimeError("publication preflight failed: " + "; ".join(failures))
            write_failure_json(
                os.path.join(run_dir, "failure.json"),
                FailureMetadata(
                    run_id=run_id,
                    utc_failure_datetime=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    status="FAILED",
                    error_message=repr(error),
                ),
            )
            raise error

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
    if "realworld-video" in selected_suites:
        selected_scenarios.extend(["realworld_video_rtsp_reconfiguration"])

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
        exact_argument_vector=exact_argument_vector,
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
        provenance=provenance,
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
            conformance_dir = os.path.join(run_dir, "conformance")
            run_conformance_suite(
                run_id=run_id,
                output_directory=conformance_dir,
                profile=profile,
                seeds=conformance_seeds,
            )
            for name in sorted(os.listdir(conformance_dir)):
                if not name.endswith(".jsonl"):
                    continue
                relative = os.path.join("conformance", name)
                trace_path = os.path.join(run_dir, relative)
                row_counts[relative] = _count_raw_artifact_rows(trace_path)
                sha256_dict[relative] = _calculate_file_sha256(trace_path)

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
            interference_frame_csv = os.path.join(run_dir, "interference-frame-samples.csv")
            row_counts["interference-frame-samples.csv"] = _count_csv_data_rows(interference_frame_csv)
            sha256_dict["interference-frame-samples.csv"] = _calculate_file_sha256(interference_frame_csv)

        if "realworld-video" in selected_suites:
            from usecases.video_analytics.contracts import ExecutionMode

            rw_video_csv = os.path.join(run_dir, "realworld-video-samples.csv")
            run_realworld_video_suite(
                run_id=run_id,
                output_csv_path=rw_video_csv,
                video_path=source_video,
                rtsp_base_url=rtsp_base_url,
                # Passed explicitly so the checkpoints recorded in provenance
                # cannot drift from the ones the suite loads.
                initial_model=realworld_initial_model,
                candidate_model=realworld_candidate_model,
                device=device,
                repetition_count=repetition_count,
                warmup_completed_frames=warmup_completed_frames,
                measurement_source_frames=(
                    baseline_receiver_frames + transition_receiver_frames
                ),
                reconfiguration_trigger_frame_offset=baseline_receiver_frames,
                drain_timeout_seconds=drain_timeout_seconds,
                queue_capacity=queue_capacity,
                use_fake_backends=(profile == "smoke"),
                random_seed=seed,
                execution_mode=ExecutionMode.SMOKE if profile == "smoke" else ExecutionMode.PUBLICATION,
            )
            row_counts["realworld-video-samples.csv"] = _count_csv_data_rows(rw_video_csv)
            sha256_dict["realworld-video-samples.csv"] = _calculate_file_sha256(rw_video_csv)
            rw_video_frame_csv = os.path.join(run_dir, "realworld-video-frame-samples.csv")
            row_counts["realworld-video-frame-samples.csv"] = _count_csv_data_rows(rw_video_frame_csv)
            sha256_dict["realworld-video-frame-samples.csv"] = _calculate_file_sha256(rw_video_frame_csv)

        if "conformance" in selected_suites:
            from benchmarks.conformance_trace import validate_conformance_directory

            validate_conformance_directory(
                os.path.join(run_dir, "conformance"),
                required_campaigns=frozenset(
                    {
                        "randomized_consistency",
                        "failure_atomicity",
                        "state_policy",
                        "stateless_publication",
                        "mutable_handoff",
                        "candidate_failure",
                        "cleanup_failure",
                        "frame_exception",
                        "successive_generation_ownership",
                    }
                ),
                randomized_seeds=conformance_seeds,
            )

        # Fail closed before promotion: every declared raw file must still
        # match its row count and digest, and suite-specific accounting must
        # validate from the persisted evidence.
        for relative_path, expected_count in row_counts.items():
            raw_path = os.path.join(run_dir, relative_path)
            if not os.path.isfile(raw_path):
                raise RuntimeError(f"raw artifact is missing: {relative_path}")
            actual_count = _count_raw_artifact_rows(raw_path)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"raw artifact row count changed for {relative_path}: "
                    f"expected {expected_count}, got {actual_count}"
                )
            if _calculate_file_sha256(raw_path) != sha256_dict[relative_path]:
                raise RuntimeError(f"raw artifact hash changed for {relative_path}")

        for filename in ("reconfiguration-samples.csv", "reconfiguration-stress-samples.csv"):
            path = os.path.join(run_dir, filename)
            if os.path.exists(path):
                for row in read_reconfiguration_rows(path):
                    from benchmarks.model import validate_reconfiguration_sample

                    validate_reconfiguration_sample(row)

        if "realworld-video" in selected_suites:
            from usecases.video_analytics.validator import validate_benchmark_csvs

            valid, validation_errors = validate_benchmark_csvs(
                os.path.join(run_dir, "realworld-video-samples.csv"),
                os.path.join(run_dir, "realworld-video-frame-samples.csv"),
            )
            if not valid:
                raise RuntimeError(
                    "real-world raw artifact validation failed: "
                    + "; ".join(validation_errors)
                )

        duration_sec = (time.monotonic_ns() - start_time_ns) / 1e9
        utc_comp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        completion_meta = CompletionMetadata(
            run_id=run_id,
            utc_completion_datetime=utc_comp,
            status="SUCCESS",
            total_duration_seconds=duration_sec,
            raw_file_row_counts=row_counts,
            raw_file_sha256=sha256_dict,
            generated_table_paths=(),
            generated_figure_paths=(),
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


def report_cmd(run_directory: str, analysis_output_directory: str) -> None:
    artifact_data: Any = load_benchmark_artifact(run_directory)
    os.makedirs(analysis_output_directory, exist_ok=True)
    summary_csv = os.path.join(analysis_output_directory, "summary.csv")
    tables_dir = os.path.join(analysis_output_directory, "tables")
    figures_dir = os.path.join(analysis_output_directory, "figures")

    generate_summary_csv(artifact_data, summary_csv)
    t_paths_raw: Any = generate_all_tables(artifact_data, tables_dir)
    f_paths_raw: Any = generate_all_figures(artifact_data, figures_dir)
    t_paths: list[str] = [str(p) for p in t_paths_raw]
    f_paths: list[str] = [str(p) for p in f_paths_raw]

    ablation_csv = os.path.join(run_directory, "ablation-samples.csv")
    if os.path.exists(ablation_csv):
        generate_ablation_summary_files(ablation_csv, analysis_output_directory)

    interference_csv = os.path.join(run_directory, "interference-samples.csv")
    if os.path.exists(interference_csv):
        interference_summary_csv = os.path.join(analysis_output_directory, "interference-summary.csv")
        generate_interference_summary_file(interference_csv, interference_summary_csv)

    ablation_rows: Any = read_ablation_rows(ablation_csv) if os.path.exists(ablation_csv) else ()
    interference_rows: Any = read_interference_rows(interference_csv) if os.path.exists(interference_csv) else ()

    if ablation_rows or interference_rows:
        val_report: Any = validate_experiment_artifacts(
            ablation_rows=ablation_rows,
            interference_rows=interference_rows,
        )
        write_experiment_validation_json(val_report, os.path.join(analysis_output_directory, "experiment-validation.json"))

    print(f"Report generated from {run_directory} into {analysis_output_directory}:")
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

    run_meta: RunMetadata = load_run_json(run_json)
    comp_meta: CompletionMetadata = load_completion_json(comp_json)

    if run_meta.run_id != comp_meta.run_id:
        raise ValueError(f"Run ID mismatch: run.json has {run_meta.run_id}, completion.json has {comp_meta.run_id}")

    raw_counts: dict[str, int] = comp_meta.raw_file_row_counts
    raw_shas: dict[str, str] = comp_meta.raw_file_sha256

    for fname in list(raw_counts.keys()):
        fpath = os.path.join(run_directory, fname)
        if not os.path.exists(fpath):
            raise ValueError(f"Missing expected raw CSV file: {fname}")

        expected_count = raw_counts.get(fname)
        actual_count = _count_raw_artifact_rows(fpath)
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(f"Row count mismatch for {fname}: expected {expected_count}, got {actual_count}")

        expected_sha = raw_shas.get(fname)
        actual_sha = _calculate_file_sha256(fpath)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {fname}: expected {expected_sha}, got {actual_sha}")

    reconfig_csv = os.path.join(run_directory, "reconfiguration-samples.csv")
    if os.path.exists(reconfig_csv):
        reconfig_samples_raw: Any = read_reconfiguration_rows(reconfig_csv)
        reconfig_samples: list[ReconfigurationSampleRow] = list(reconfig_samples_raw)
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

    table_paths_raw: Any = comp_meta.generated_table_paths
    figure_paths_raw: Any = comp_meta.generated_figure_paths
    table_paths: list[str] = [str(p) for p in table_paths_raw]
    figure_paths: list[str] = [str(p) for p in figure_paths_raw]

    for t_path in table_paths:
        if not os.path.exists(t_path):
            raise ValueError(f"Missing generated table file: {t_path}")
    for f_path in figure_paths:
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
        help="Benchmark suite(s) to run ('all', 'steady-state', 'reconfiguration', 'conformance', 'realworld-video', or comma-separated list)",
    )
    profile_group = run_parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument("--smoke", action="store_true", help="Run smoke profile for CI/validation")
    profile_group.add_argument("--publication", action="store_true", help="Run publication profile")

    # Real-world video specific optional arguments
    run_parser.add_argument("--source", type=str, default="sample_video.mp4", help="Source video file path")
    run_parser.add_argument("--rtsp-base-url", type=str, default="rtsp://127.0.0.1:8554", help="MediaMTX RTSP base URL")
    run_parser.add_argument("--queue-capacity", type=int, default=4, help="Bounded ingress queue capacity")
    run_parser.add_argument("--device", type=str, default="auto", help="Device target ('auto', 'cuda:0', 'cpu')")
    run_parser.add_argument("--output-dir", type=str, default="benchmark-results")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--conformance-seeds",
        type=str,
        default="42,314159,271828,161803,20260922",
    )
    run_parser.add_argument("--warmup-completed-frames", type=int, default=30)
    run_parser.add_argument("--baseline-receiver-frames", type=int, default=60)
    run_parser.add_argument("--transition-receiver-frames", type=int, default=150)
    run_parser.add_argument("--drain-timeout-seconds", type=float, default=30.0)
    run_parser.add_argument("--operation-timeout-seconds", type=float, default=120.0)

    # report subcommand
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run_directory", type=str)
    report_parser.add_argument("--output-dir", required=True, type=str)

    # verify subcommand
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("run_directory", type=str)

    gate_parser = subparsers.add_parser("validate-gate")
    gate_parser.add_argument("run_directory", type=str)

    # validate-realworld subcommand
    validate_parser = subparsers.add_parser("validate-realworld")
    validate_parser.add_argument("run_directory", type=str)

    args = parser.parse_args()

    if args.subcommand == "run":
        profile = "smoke" if args.smoke else "publication"
        parsed_conformance_seeds = tuple(
            int(value.strip())
            for value in args.conformance_seeds.split(",")
            if value.strip()
        )
        if not parsed_conformance_seeds:
            raise ValueError("--conformance-seeds must contain at least one integer")
        run_benchmarks(
            suite=args.suite,
            profile=profile,
            source_video=args.source,
            rtsp_base_url=args.rtsp_base_url,
            queue_capacity=args.queue_capacity,
            warmup_completed_frames=args.warmup_completed_frames,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            conformance_seeds=parsed_conformance_seeds,
            baseline_receiver_frames=args.baseline_receiver_frames,
            transition_receiver_frames=args.transition_receiver_frames,
            drain_timeout_seconds=args.drain_timeout_seconds,
            operation_timeout_seconds=args.operation_timeout_seconds,
            exact_argument_vector=tuple(sys.argv),
        )
    elif args.subcommand == "report":
        report_cmd(args.run_directory, args.output_dir)
    elif args.subcommand == "verify":
        verify_cmd(args.run_directory)
    elif args.subcommand == "validate-gate":
        from benchmarks.validation_gate import validate_benchmark_directory
        report = validate_benchmark_directory(args.run_directory)
        if not report.all_passed:
            sys.exit(1)
    elif args.subcommand == "validate-realworld":
        from usecases.video_analytics.validator import validate_benchmark_csvs
        summary_csv = os.path.join(args.run_directory, "realworld-video-samples.csv")
        frame_csv = os.path.join(args.run_directory, "realworld-video-frame-samples.csv")
        success, _ = validate_benchmark_csvs(summary_csv, frame_csv)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
