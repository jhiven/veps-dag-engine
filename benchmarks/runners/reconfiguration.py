"""RQ2 structural reconfiguration performance and graph-size sensitivity runner."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from queue import Full, Queue
from threading import Event, Lock, Thread

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
from nedo_vision_dag_engine.validation import validate_workflow

__all__ = ["run_reconfiguration_suite"]


@dataclass(frozen=True, slots=True)
class FrameLogEntry:
    frame_id: int
    plan_version: int
    arrival_ns: int
    admission_ns: int
    completion_ns: int


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

    for edit_type in edit_types:
        base_spec = make_reconfiguration_base_spec(with_tracker=(edit_type == "compatible_edit_preserving_tracker"))
        graph_size = len(base_spec.nodes)
        target_spec = apply_reconfiguration_edit(base_spec, edit_type)
        scenario_id = f"reconfig_{edit_type}_size{graph_size}"

        for rep in range(1, repetition_count + 1):
            # Counterbalanced baseline ordering per repetition
            rot_idx = (rep - 1) % len(baselines)
            rep_baselines = baselines[rot_idx:] + baselines[:rot_idx]

            for baseline in rep_baselines:
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
    state_directive = StateDirective()

    # Shared harness components
    inter_arrival_s = 0.0005
    queue_capacity = 100
    frame_queue: Queue[tuple[int, int]] = Queue(maxsize=queue_capacity)
    frame_log: list[FrameLogEntry] = []
    log_lock = Lock()

    stop_producer = Event()
    stop_worker = Event()
    worker_paused = Event()

    dropped_frame_count = 0
    drop_lock = Lock()

    def producer_loop() -> None:
        nonlocal dropped_frame_count
        fid = 0
        while not stop_producer.is_set():
            arr_ns = time.perf_counter_ns()
            try:
                frame_queue.put_nowait((fid, arr_ns))
                fid += 1
            except Full:
                with drop_lock:
                    dropped_frame_count += 1
            time.sleep(inter_arrival_s)

    t_producer = Thread(target=producer_loop, daemon=True)
    t_producer.start()

    if baseline == "stop_rebuild_restart":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_restart() -> None:
            nonlocal current_executor
            while not stop_worker.is_set():
                if worker_paused.is_set():
                    time.sleep(0.0001)
                    continue
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                t_adm = time.perf_counter_ns()
                res = current_executor.admit_frame(admitted_at_ns=arr_ns, frame_id=fid)
                t_comp = time.perf_counter_ns()
                with log_lock:
                    frame_log.append(
                        FrameLogEntry(
                            frame_id=fid,
                            plan_version=res.plan_version,
                            arrival_ns=arr_ns,
                            admission_ns=t_adm,
                            completion_ns=t_comp,
                        )
                    )

        t_worker = Thread(target=worker_loop_restart, daemon=True)
        t_worker.start()

        # Warmup: wait for 5 frames
        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()

        # Pause worker during synchronous rebuild
        worker_paused.set()

        t_val_start = time.perf_counter_ns()
        val_res, _ = validate_workflow(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if not val_res.is_valid:
            raise RuntimeError("Validation failed.")

        t_prep_start = time.perf_counter_ns()
        compiler = WorkflowCompiler("0.1.0")
        cand = compiler.compile(target_spec, registry, previous_plan=None, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()

        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        new_version = initial_plan.version + 1
        new_plan = replace(cand.plan, version=new_version)
        cand = replace(cand, plan=new_plan)

        t_commit_start = time.perf_counter_ns()
        new_executor = PipelineExecutor(initial_plan=cand.plan)
        current_executor = new_executor
        t_commit_end = time.perf_counter_ns()

        # Unpause worker
        worker_paused.clear()

        # Wait for first new frame under new plan to complete
        old_plan_version = initial_plan.version
        new_plan_version = initial_plan.version + 1

        while True:
            with log_lock:
                if any(e.plan_version == cand.plan.version for e in frame_log):
                    break
            time.sleep(0.0001)

        # Allow 5 more frames under new plan
        target_total = len(frame_log) + 5
        while True:
            with log_lock:
                if len(frame_log) >= target_total:
                    break
            time.sleep(0.0001)

        stop_producer.set()
        stop_worker.set()
        t_producer.join()
        t_worker.join()

        val_ns = t_val_end - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start

        first_new_frame = next(e for e in frame_log if e.plan_version == cand.plan.version)
        req_effect_ns = first_new_frame.completion_ns - t_request

        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new_frame.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )
        max_output_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns
        )

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            terminal_status=ReconfigurationStatus.COMMITTED.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=0,
            maximum_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    elif baseline == "pause_compile_resume":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_pause() -> None:
            while not stop_worker.is_set():
                if worker_paused.is_set():
                    time.sleep(0.0001)
                    continue
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                t_adm = time.perf_counter_ns()
                res = current_executor.admit_frame(admitted_at_ns=arr_ns, frame_id=fid)
                t_comp = time.perf_counter_ns()
                with log_lock:
                    frame_log.append(
                        FrameLogEntry(
                            frame_id=fid,
                            plan_version=res.plan_version,
                            arrival_ns=arr_ns,
                            admission_ns=t_adm,
                            completion_ns=t_comp,
                        )
                    )

        t_worker = Thread(target=worker_loop_pause, daemon=True)
        t_worker.start()

        # Warmup: wait for 5 frames
        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()

        # Pause worker during synchronous compilation
        worker_paused.set()

        t_val_start = time.perf_counter_ns()
        val_res, _ = validate_workflow(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if not val_res.is_valid:
            raise RuntimeError("Validation failed.")

        t_prep_start = time.perf_counter_ns()
        cand = init_compiler.compile(target_spec, registry, previous_plan=initial_plan, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()

        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        t_commit_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_commit_end = time.perf_counter_ns()

        # Unpause worker
        worker_paused.clear()

        # Wait for first new frame under new plan to complete
        old_plan_version = initial_plan.version
        new_plan_version = cand.plan.version

        while True:
            with log_lock:
                if any(e.plan_version == cand.plan.version for e in frame_log):
                    break
            time.sleep(0.0001)

        # Allow 5 more frames under new plan
        target_total = len(frame_log) + 5
        while True:
            with log_lock:
                if len(frame_log) >= target_total:
                    break
            time.sleep(0.0001)

        stop_producer.set()
        stop_worker.set()
        t_producer.join()
        t_worker.join()

        val_ns = t_val_end - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start

        first_new_frame = next(e for e in frame_log if e.plan_version == cand.plan.version)
        req_effect_ns = first_new_frame.completion_ns - t_request

        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new_frame.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )
        max_output_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns
        )

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            terminal_status=ReconfigurationStatus.COMMITTED.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=0,
            maximum_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    elif baseline == "prepare_and_commit":
        current_executor = PipelineExecutor(initial_plan=initial_plan)
        controller = ReconfigurationController(executor=current_executor, compiler=init_compiler, registry=registry)

        def worker_loop_prepare() -> None:
            while not stop_worker.is_set():
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                t_adm = time.perf_counter_ns()
                res = controller.admit_frame(admitted_at_ns=arr_ns, frame_id=fid)
                t_comp = time.perf_counter_ns()
                with log_lock:
                    frame_log.append(
                        FrameLogEntry(
                            frame_id=fid,
                            plan_version=res.plan_version,
                            arrival_ns=arr_ns,
                            admission_ns=t_adm,
                            completion_ns=t_comp,
                        )
                    )

        t_worker = Thread(target=worker_loop_prepare, daemon=True)
        t_worker.start()

        # Warmup: wait for 5 frames
        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()

        req = ReconfigurationRequest(
            request_id=f"req_{scenario_id}_{repetition}",
            base_version=initial_plan.version,
            target_specification=target_spec,
            state_directive=state_directive,
            submitted_at_ns=t_request,
        )

        controller.submit(req)
        record = controller.wait_for_effect(req.request_id, timeout_seconds=10.0)

        if record is None or record.status != ReconfigurationStatus.COMMITTED:
            status_val = record.status.value if record else "TIMEOUT"
            raise RuntimeError(f"Reconfiguration failed in benchmark: {status_val}")

        old_plan_version = initial_plan.version
        new_plan_version = record.candidate_version or (initial_plan.version + 1)

        # Wait for 5 more frames after effect
        target_total = len(frame_log) + 5
        while True:
            with log_lock:
                if len(frame_log) >= target_total:
                    break
            time.sleep(0.0001)

        stop_producer.set()
        stop_worker.set()
        t_producer.join()
        t_worker.join()
        controller.close()

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

        first_new_frame = next(e for e in frame_log if e.plan_version == new_plan_version)
        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new_frame.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )
        max_output_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns
        )

        return ReconfigurationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            baseline=baseline,
            edit_type=edit_type,
            graph_size=graph_size,
            repetition=repetition,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            terminal_status=record.status.value,
            validation_ns=val_ns,
            preparation_ns=prep_ns,
            request_to_ready_ns=req_ready_ns,
            boundary_wait_ns=boundary_wait_ns,
            commit_ns=commit_ns,
            request_to_effect_ns=req_effect_ns,
            retirement_ns=retire_ns,
            maximum_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=record.reused_processor_count,
            staged_processor_count=record.staged_processor_count,
            retired_processor_count=record.retired_processor_count,
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
        )

    else:
        raise ValueError(f"Unknown baseline {baseline!r}")
