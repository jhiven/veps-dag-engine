"""RQ2 structural reconfiguration performance and graph-size sensitivity runner."""

from __future__ import annotations

import gc
import time
from threading import Event, Thread

from benchmarks.model import ReconfigurationSampleRow
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    generate_layered_dag,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_reconfiguration_rows, write_reconfiguration_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.specification import WorkflowSpecification

__all__ = ["run_reconfiguration_suite"]


def run_reconfiguration_suite(
    run_id: str,
    output_csv_path: str,
    profile: str,
    repetition_count: int,
    seed: int = 42,
) -> tuple[ReconfigurationSampleRow, ...]:
    write_reconfiguration_header(output_csv_path)

    baselines: tuple[str, ...] = (
        "stop_rebuild_restart",
        "pause_compile_resume",
        "prepare_and_commit",
    )
    edit_types: tuple[str, ...] = (
        "insert_stateless_node",
        "remove_stateless_node",
        "rewire_stateless_edge",
        "compatible_edit_preserving_tracker",
    )

    all_rows: list[ReconfigurationSampleRow] = []

    # 1. Main 10-node edit scenarios across 3 baselines
    base_spec = make_reconfiguration_base_spec()
    graph_size = len(base_spec.nodes)

    for edit_type in edit_types:
        target_spec = apply_reconfiguration_edit(base_spec, edit_type)
        scenario_id = f"reconfig_{edit_type}_size{graph_size}"

        for rep in range(1, repetition_count + 1):
            for baseline in baselines:
                gc.collect()
                row = _run_reconfig_repetition(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    baseline=baseline,
                    edit_type=edit_type,
                    graph_size=graph_size,
                    repetition=rep,
                    base_spec=base_spec,
                    target_spec=target_spec,
                    profile=profile,
                )
                all_rows.append(row)
                append_reconfiguration_rows(output_csv_path, [row])

    # 2. Compact Graph-Size Sensitivity (5, 25, 100 nodes) for prepare_and_commit
    sensitivity_sizes: tuple[int, ...] = (5, 25, 100)
    for sz in sensitivity_sizes:
        sens_base = generate_layered_dag(sz)
        sens_target = apply_reconfiguration_edit(sens_base, "insert_stateless_node")
        sens_scenario_id = f"graph_size_sensitivity_{sz}"

        for rep in range(1, repetition_count + 1):
            gc.collect()
            row = _run_reconfig_repetition(
                run_id=run_id,
                scenario_id=sens_scenario_id,
                baseline="prepare_and_commit",
                edit_type="insert_stateless_node",
                graph_size=sz,
                repetition=rep,
                base_spec=sens_base,
                target_spec=sens_target,
                profile=profile,
            )
            all_rows.append(row)
            append_reconfiguration_rows(output_csv_path, [row])

    return tuple(all_rows)


def _run_reconfig_repetition(
    run_id: str,
    scenario_id: str,
    baseline: str,
    edit_type: str,
    graph_size: int,
    repetition: int,
    base_spec: WorkflowSpecification,
    target_spec: WorkflowSpecification,
    profile: str,
) -> ReconfigurationSampleRow:
    registry = create_reconfiguration_registry(extra_tracker=True)

    # Initial compilation
    init_compiler = WorkflowCompiler("0.1.0")
    init_cand = init_compiler.compile(base_spec, registry)
    if not isinstance(init_cand, CompiledCandidate):
        raise RuntimeError("Initial compilation failed.")

    initial_plan = init_cand.plan
    executor = PipelineExecutor(initial_plan=initial_plan)

    state_directive = StateDirective()

    if baseline == "stop_rebuild_restart":
        executor.admit_frame(admitted_at_ns=0, frame_id=0)
        t_last_old = time.perf_counter_ns()

        t_request = time.perf_counter_ns()

        t_val_start = time.perf_counter_ns()
        compiler = WorkflowCompiler("0.1.0")
        t_prep_start = time.perf_counter_ns()
        cand = compiler.compile(target_spec, registry, previous_plan=None, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()

        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        t_commit_start = time.perf_counter_ns()
        new_executor = PipelineExecutor(initial_plan=cand.plan)
        t_commit_end = time.perf_counter_ns()

        _res = new_executor.admit_frame(admitted_at_ns=0, frame_id=1)
        t_first_new = time.perf_counter_ns()

        val_ns = t_prep_start - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start
        req_effect_ns = t_first_new - t_request
        max_output_gap_ns = t_first_new - t_last_old

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=initial_plan.version,
            new_plan_version=cand.plan.version,
            terminal_status=ReconfigurationStatus.COMMITTED.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=0,
            maximum_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=0,
            frames_dropped=0,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    elif baseline == "pause_compile_resume":
        executor.admit_frame(admitted_at_ns=0, frame_id=0)
        t_last_old = time.perf_counter_ns()

        t_request = time.perf_counter_ns()

        t_val_start = time.perf_counter_ns()
        t_prep_start = time.perf_counter_ns()
        cand = init_compiler.compile(target_spec, registry, previous_plan=initial_plan, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()

        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        t_commit_start = time.perf_counter_ns()
        executor.commit(cand.plan)
        t_commit_end = time.perf_counter_ns()

        _res = executor.admit_frame(admitted_at_ns=0, frame_id=1)
        t_first_new = time.perf_counter_ns()

        val_ns = t_prep_start - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start
        req_effect_ns = t_first_new - t_request
        max_output_gap_ns = t_first_new - t_last_old

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=initial_plan.version,
            new_plan_version=cand.plan.version,
            terminal_status=ReconfigurationStatus.COMMITTED.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=0,
            maximum_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=0,
            frames_dropped=0,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    elif baseline == "prepare_and_commit":
        controller = ReconfigurationController(executor=executor, compiler=init_compiler, registry=registry)

        stop_bg = Event()
        completed_timestamps: list[int] = []
        committed_version: list[int] = []

        def bg_admit() -> None:
            fid = 0
            while not stop_bg.is_set():
                res = controller.admit_frame(admitted_at_ns=0, frame_id=fid)
                now_ns = time.perf_counter_ns()
                completed_timestamps.append(now_ns)
                if res.plan_version > initial_plan.version and not committed_version:
                    committed_version.append(res.plan_version)
                fid += 1
                time.sleep(0.0005)

        t_worker = Thread(target=bg_admit, daemon=True)
        t_worker.start()

        time.sleep(0.002)

        t_request = time.perf_counter_ns()

        req = ReconfigurationRequest(
            request_id=f"req_{scenario_id}_{repetition}",
            base_version=initial_plan.version,
            target_specification=target_spec,
            state_directive=state_directive,
            submitted_at_ns=t_request,
        )

        controller.submit(req)
        record = controller.wait_for_terminal(req.request_id, timeout_seconds=10.0)

        time.sleep(0.005)
        stop_bg.set()
        t_worker.join()
        controller.close()

        if record is None or record.status != ReconfigurationStatus.COMMITTED:
            status_val = record.status.value if record else "TIMEOUT"
            raise RuntimeError(f"Reconfiguration failed in benchmark: {status_val}")

        val_ns = (
            record.validation_completed_ns - record.validation_started_ns
            if record.validation_completed_ns and record.validation_started_ns
            else 0
        )
        prep_ns = (
            record.preparation_completed_ns - record.preparation_started_ns
            if record.preparation_completed_ns and record.preparation_started_ns
            else 0
        )
        req_ready_ns = record.ready_ns - record.request_received_ns if record.ready_ns else 0
        boundary_wait_ns = (
            record.commit_started_ns - record.ready_ns
            if record.commit_started_ns and record.ready_ns
            else 0
        )
        commit_ns = (
            record.commit_ns - record.commit_started_ns
            if record.commit_ns and record.commit_started_ns
            else 0
        )
        req_effect_ns = (
            record.first_new_frame_completed_ns - record.request_received_ns
            if record.first_new_frame_completed_ns
            else 0
        )
        retire_ns = (
            record.retirement_completed_ns - record.commit_ns
            if record.retirement_completed_ns and record.commit_ns
            else 0
        )

        max_gap = 0
        if len(completed_timestamps) > 1:
            gaps = [
                completed_timestamps[i] - completed_timestamps[i - 1]
                for i in range(1, len(completed_timestamps))
            ]
            max_gap = max(gaps)

        cand_ver = record.candidate_version if record.candidate_version else 2

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=initial_plan.version,
            new_plan_version=cand_ver,
            terminal_status=record.status.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=boundary_wait_ns,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=retire_ns,
            maximum_output_gap_ns=max_gap,
            frames_completed_during_request=len(completed_timestamps),
            frames_dropped=0,
            frames_duplicated=0,
            reused_processor_count=0,
            staged_processor_count=0,
            retired_processor_count=0,
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    else:
        raise ValueError(f"Unknown baseline {baseline!r}")
