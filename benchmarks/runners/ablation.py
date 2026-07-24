"""E5 VEPS component ablation runner.

Isolates off-path candidate preparation vs. deferred processor retirement:
- Variant A: Synchronous preparation + Synchronous retirement
- Variant B: Off-path preparation + Synchronous retirement
- Variant C: Synchronous preparation + Deferred retirement
- Variant D: Off-path preparation + Deferred retirement (Full VEPS)

Implements randomized complete block design (RCBD) across variants within each block.
"""

from __future__ import annotations

import gc
import os
import random
import time
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Lock, Thread

from benchmarks.model import AblationSampleRow, validate_ablation_sample
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_ablation_rows, write_ablation_header
from nedo_vision_dag_engine.compiler import (
    CompilationFailure,
    CompiledCandidate,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.lifecycle import retire_superseded_processors, shutdown_plan_processors
from nedo_vision_dag_engine.reconfiguration import ReconfigurationController, ReconfigurationRequest
from nedo_vision_dag_engine.specification import specification_hash


@dataclass(frozen=True, slots=True)
class FrameLogEntry:
    frame_id: int
    plan_version: int
    arrival_ns: int
    admission_ns: int
    completion_ns: int


def _run_ablation_repetition(
    run_id: str,
    scenario_id: str,
    block_id: str,
    block_seed: int,
    variant: str,
    variant_order_position: int,
    edit_type: str,
    repetition: int,
    request_phase_offset_ns: int = 0,
    graph_size: int = 10,
    random_seed: int | None = None,
) -> AblationSampleRow:
    gc.collect()
    registry = create_reconfiguration_registry()
    with_tracker = edit_type == "compatible_edit_preserving_tracker"
    initial_spec = make_reconfiguration_base_spec(with_tracker=with_tracker)
    init_compiler = WorkflowCompiler("0.1.0")

    val_base = init_compiler.validate(initial_spec, registry)
    if isinstance(val_base, CompilationFailure):
        raise RuntimeError(f"Base validation failed: {val_base.reason}")
    initial_cand = init_compiler.compile_validated(val_base, previous_plan=None)
    if not isinstance(initial_cand, CompiledCandidate):
        raise RuntimeError("Base compilation failed.")
    initial_plan = initial_cand.plan

    target_spec = apply_reconfiguration_edit(initial_spec, edit_type)
    state_directive = StateDirective(reset_node_ids=frozenset()) if with_tracker else StateDirective()

    candidate_graph_fingerprint = f"fingerprint_{specification_hash(target_spec)[:12]}"
    preparation_workload_id = f"workload_{edit_type}"

    dropped_frame_count = 0
    frame_queue: Queue[tuple[int, int]] = Queue(maxsize=100)
    frame_log: list[FrameLogEntry] = []
    log_lock = Lock()

    stop_producer = Event()
    stop_worker = Event()
    worker_paused = Event()

    def producer_loop() -> None:
        nonlocal dropped_frame_count
        frame_id = 0
        while not stop_producer.is_set():
            arr_ns = time.perf_counter_ns()
            try:
                frame_queue.put_nowait((frame_id, arr_ns))
            except Full:
                try:
                    frame_queue.get_nowait()
                    dropped_frame_count += 1
                    frame_queue.put_nowait((frame_id, arr_ns))
                except Exception:
                    pass
            frame_id += 1
            time.sleep(0.001)

    t_producer = Thread(target=producer_loop, daemon=True)
    t_producer.start()

    if variant == "variant_a":
        # A: Synchronous Prep + Synchronous Retire
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_a() -> None:
            while not stop_worker.is_set():
                if worker_paused.is_set():
                    time.sleep(0.0001)
                    continue
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                if worker_paused.is_set():
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

        t_worker = Thread(target=worker_loop_a, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        if request_phase_offset_ns > 0:
            time.sleep(request_phase_offset_ns / 1e9)

        t_request = time.perf_counter_ns()
        worker_paused.set()

        t_val_start = time.perf_counter_ns()
        compiler = init_compiler
        validated = compiler.validate(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")

        t_prep_start = time.perf_counter_ns()
        cand = compiler.compile_validated(validated, previous_plan=initial_plan, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Compilation failed.")

        val_ns = t_val_end - t_val_start
        sync_prep_ns = t_prep_end - t_prep_start

        t_pub_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_pub_end = time.perf_counter_ns()
        pub_ns = t_pub_end - t_pub_start

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError("Retirement failed.")
        sync_ret_ns = t_ret_end - t_ret_start

        worker_paused.clear()
        old_plan_version = initial_plan.version
        new_plan_version = cand.plan.version

        while True:
            with log_lock:
                if any(e.plan_version == new_plan_version for e in frame_log):
                    break
            time.sleep(0.0001)

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

        first_new = next(e for e in frame_log if e.plan_version == new_plan_version)
        last_old = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )

        req_effect_ns = first_new.completion_ns - t_request
        gap_ns = first_new.completion_ns - last_old.completion_ns if last_old else 0
        total_sync_ns = val_ns + sync_prep_ns + pub_ns + sync_ret_ns
        ret_duration_ns = sync_ret_ns

        shutdown_plan_processors(current_executor.active_plan)
        row = AblationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            block_id=block_id,
            block_seed=block_seed,
            variant=variant,
            variant_name="sync_prepare_sync_retire",
            preparation_placement="synchronous",
            retirement_policy="synchronous",
            edit_type=edit_type,
            repetition=repetition,
            variant_order_position=variant_order_position,
            candidate_graph_fingerprint=candidate_graph_fingerprint,
            preparation_workload_id=preparation_workload_id,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            validation_ns=val_ns,
            synchronous_preparation_ns=sync_prep_ns,
            offpath_preparation_ns=None,
            boundary_wait_ns=0,
            publication_ns=pub_ns,
            synchronous_retirement_ns=sync_ret_ns,
            deferred_retirement_ns=None,
            first_effect_wait_ns=first_new.completion_ns - t_pub_end,
            request_to_effect_ns=req_effect_ns,
            transition_output_gap_ns=gap_ns,
            total_synchronous_ns=total_sync_ns,
            synchronous_accounting_residual_ns=0,
            synchronous_accounting_valid=True,
            synchronous_accounting_invalid_reason=None,
            retirement_duration_ns=ret_duration_ns,
            old_plan_frames_admitted_after_request_before_commit=0,
            peak_live_processors=len(initial_plan.steps) + len(cand.staged_node_ids),
            random_seed=random_seed,
            request_phase_offset_ns=request_phase_offset_ns,
        )
        validate_ablation_sample(row)
        return row

    elif variant == "variant_b":
        # B: Off-path Prep + Synchronous Retire
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_b() -> None:
            while not stop_worker.is_set():
                if worker_paused.is_set():
                    time.sleep(0.0001)
                    continue
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                if worker_paused.is_set():
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

        t_worker = Thread(target=worker_loop_b, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        if request_phase_offset_ns > 0:
            time.sleep(request_phase_offset_ns / 1e9)

        t_request = time.perf_counter_ns()
        t_offpath_start = time.perf_counter_ns()
        compiler = init_compiler
        validated = compiler.validate(target_spec, registry)
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")
        cand = compiler.compile_validated(validated, previous_plan=initial_plan, state_directive=state_directive)
        t_offpath_end = time.perf_counter_ns()
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Compilation failed.")
        offpath_prep_ns = t_offpath_end - t_offpath_start

        worker_paused.set()
        t_pub_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_pub_end = time.perf_counter_ns()
        pub_ns = t_pub_end - t_pub_start

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError("Retirement failed.")
        sync_ret_ns = t_ret_end - t_ret_start

        worker_paused.clear()
        old_plan_version = initial_plan.version
        new_plan_version = cand.plan.version

        while True:
            with log_lock:
                if any(e.plan_version == new_plan_version for e in frame_log):
                    break
            time.sleep(0.0001)

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

        first_new = next(e for e in frame_log if e.plan_version == new_plan_version)
        last_old = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )

        old_frames = sum(
            1 for e in frame_log
            if e.plan_version == old_plan_version
            and e.admission_ns >= t_request
            and e.completion_ns < t_pub_start
        )

        req_effect_ns = first_new.completion_ns - t_request
        gap_ns = first_new.completion_ns - last_old.completion_ns if last_old else 0
        total_sync_ns = pub_ns + sync_ret_ns
        ret_duration_ns = sync_ret_ns

        shutdown_plan_processors(current_executor.active_plan)
        row = AblationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            block_id=block_id,
            block_seed=block_seed,
            variant=variant,
            variant_name="offpath_prepare_sync_retire",
            preparation_placement="off_path",
            retirement_policy="synchronous",
            edit_type=edit_type,
            repetition=repetition,
            variant_order_position=variant_order_position,
            candidate_graph_fingerprint=candidate_graph_fingerprint,
            preparation_workload_id=preparation_workload_id,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            validation_ns=None,
            synchronous_preparation_ns=None,
            offpath_preparation_ns=offpath_prep_ns,
            boundary_wait_ns=0,
            publication_ns=pub_ns,
            synchronous_retirement_ns=sync_ret_ns,
            deferred_retirement_ns=None,
            first_effect_wait_ns=first_new.completion_ns - t_pub_end,
            request_to_effect_ns=req_effect_ns,
            transition_output_gap_ns=gap_ns,
            total_synchronous_ns=total_sync_ns,
            synchronous_accounting_residual_ns=0,
            synchronous_accounting_valid=True,
            synchronous_accounting_invalid_reason=None,
            retirement_duration_ns=ret_duration_ns,
            old_plan_frames_admitted_after_request_before_commit=old_frames,
            peak_live_processors=len(initial_plan.steps) + len(cand.staged_node_ids),
            random_seed=random_seed,
            request_phase_offset_ns=request_phase_offset_ns,
        )
        validate_ablation_sample(row)
        return row

    elif variant == "variant_c":
        # C: Synchronous Prep + Deferred Retire
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_c() -> None:
            while not stop_worker.is_set():
                if worker_paused.is_set():
                    time.sleep(0.0001)
                    continue
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                if worker_paused.is_set():
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

        t_worker = Thread(target=worker_loop_c, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        if request_phase_offset_ns > 0:
            time.sleep(request_phase_offset_ns / 1e9)

        t_request = time.perf_counter_ns()
        worker_paused.set()

        t_val_start = time.perf_counter_ns()
        compiler = init_compiler
        validated = compiler.validate(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")

        t_prep_start = time.perf_counter_ns()
        cand = compiler.compile_validated(validated, previous_plan=initial_plan, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Compilation failed.")

        val_ns = t_val_end - t_val_start
        sync_prep_ns = t_prep_end - t_prep_start

        t_pub_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_pub_end = time.perf_counter_ns()
        pub_ns = t_pub_end - t_pub_start

        worker_paused.clear()

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        def_ret_ns = t_ret_end - t_ret_start

        old_plan_version = initial_plan.version
        new_plan_version = cand.plan.version

        while True:
            with log_lock:
                if any(e.plan_version == new_plan_version for e in frame_log):
                    break
            time.sleep(0.0001)

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

        first_new = next(e for e in frame_log if e.plan_version == new_plan_version)
        last_old = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )

        req_effect_ns = first_new.completion_ns - t_request
        gap_ns = first_new.completion_ns - last_old.completion_ns if last_old else 0
        total_sync_ns = val_ns + sync_prep_ns + pub_ns
        ret_duration_ns = def_ret_ns

        shutdown_plan_processors(current_executor.active_plan)
        row = AblationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            block_id=block_id,
            block_seed=block_seed,
            variant=variant,
            variant_name="sync_prepare_deferred_retire",
            preparation_placement="synchronous",
            retirement_policy="deferred",
            edit_type=edit_type,
            repetition=repetition,
            variant_order_position=variant_order_position,
            candidate_graph_fingerprint=candidate_graph_fingerprint,
            preparation_workload_id=preparation_workload_id,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            validation_ns=val_ns,
            synchronous_preparation_ns=sync_prep_ns,
            offpath_preparation_ns=None,
            boundary_wait_ns=0,
            publication_ns=pub_ns,
            synchronous_retirement_ns=None,
            deferred_retirement_ns=def_ret_ns,
            first_effect_wait_ns=first_new.completion_ns - t_pub_end,
            request_to_effect_ns=req_effect_ns,
            transition_output_gap_ns=gap_ns,
            total_synchronous_ns=total_sync_ns,
            synchronous_accounting_residual_ns=0,
            synchronous_accounting_valid=True,
            synchronous_accounting_invalid_reason=None,
            retirement_duration_ns=ret_duration_ns,
            old_plan_frames_admitted_after_request_before_commit=0,
            peak_live_processors=len(initial_plan.steps) + len(cand.staged_node_ids),
            random_seed=random_seed,
            request_phase_offset_ns=request_phase_offset_ns,
        )
        validate_ablation_sample(row)
        return row

    elif variant == "variant_d":
        # D: Off-path Prep + Deferred Retire (Full VEPS)
        current_executor = PipelineExecutor(initial_plan=initial_plan)
        controller = ReconfigurationController(executor=current_executor, compiler=init_compiler, registry=registry)

        def worker_loop_d() -> None:
            while not stop_worker.is_set():
                try:
                    fid, arr_ns = frame_queue.get(timeout=0.005)
                except Exception:
                    continue
                t_adm = time.perf_counter_ns()
                res = controller.admit_frame(admitted_at_ns=t_adm, frame_id=fid)
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

        t_worker = Thread(target=worker_loop_d, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        if request_phase_offset_ns > 0:
            time.sleep(request_phase_offset_ns / 1e9)

        t_request = time.perf_counter_ns()
        req = ReconfigurationRequest(
            request_id="ablation_d",
            base_version=initial_plan.version,
            target_specification=target_spec,
            state_directive=state_directive,
            submitted_at_ns=t_request,
        )
        controller.submit(req)

        rec_eff = controller.wait_for_effect(req.request_id, timeout_seconds=10.0)
        if rec_eff is None:
            raise RuntimeError("wait_for_effect timed out.")

        ret_rec = controller.wait_for_retirement(req.request_id, timeout_seconds=10.0)
        if ret_rec is None:
            raise RuntimeError("wait_for_retirement timed out.")

        record = controller.record(req.request_id)
        old_plan_version = initial_plan.version
        new_plan_version = record.candidate_version or (initial_plan.version + 1)

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

        first_new = next(e for e in frame_log if e.plan_version == new_plan_version)
        last_old = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )

        req_effect_ns = (
            record.first_new_frame_completed_ns - t_request
            if record.first_new_frame_completed_ns
            else first_new.completion_ns - t_request
        )
        gap_ns = (
            first_new.completion_ns - last_old.completion_ns if last_old else 0
        )
        pub_ns = (
            record.commit_ns - record.commit_started_ns
            if record.commit_ns and record.commit_started_ns
            else 0
        )
        ret_duration_ns = (
            record.retirement_completed_ns - record.retirement_started_ns
            if record.retirement_completed_ns and record.retirement_started_ns
            else 0
        )
        offpath_prep_ns = (
            record.preparation_completed_ns - record.preparation_started_ns
            if record.preparation_completed_ns and record.preparation_started_ns
            else 0
        )

        row = AblationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            block_id=block_id,
            block_seed=block_seed,
            variant=variant,
            variant_name="offpath_prepare_deferred_retire",
            preparation_placement="off_path",
            retirement_policy="deferred",
            edit_type=edit_type,
            repetition=repetition,
            variant_order_position=variant_order_position,
            candidate_graph_fingerprint=candidate_graph_fingerprint,
            preparation_workload_id=preparation_workload_id,
            old_plan_version=old_plan_version,
            new_plan_version=new_plan_version,
            validation_ns=None,
            synchronous_preparation_ns=None,
            offpath_preparation_ns=offpath_prep_ns,
            boundary_wait_ns=0,
            publication_ns=pub_ns,
            synchronous_retirement_ns=None,
            deferred_retirement_ns=ret_duration_ns,
            first_effect_wait_ns=first_new.completion_ns - (record.commit_ns or t_request),
            request_to_effect_ns=req_effect_ns,
            transition_output_gap_ns=gap_ns,
            total_synchronous_ns=pub_ns,
            synchronous_accounting_residual_ns=0,
            synchronous_accounting_valid=True,
            synchronous_accounting_invalid_reason=None,
            retirement_duration_ns=ret_duration_ns,
            old_plan_frames_admitted_after_request_before_commit=record.old_plan_frames_admitted_after_request_before_commit,
            peak_live_processors=len(initial_plan.steps) + record.staged_processor_count,
            random_seed=random_seed,
            request_phase_offset_ns=request_phase_offset_ns,
        )
        validate_ablation_sample(row)
        return row

    else:
        raise ValueError(f"Unknown variant: {variant!r}")


def run_ablation_suite(
    output_dir: str,
    run_id: str,
    repetitions: int = 5,
    random_seed: int | None = None,
) -> tuple[AblationSampleRow, ...]:
    """Run E5 VEPS component ablation benchmark campaign using a Randomized Complete Block Design (RCBD)."""
    csv_path = os.path.join(output_dir, "ablation-samples.csv")
    write_ablation_header(csv_path)

    base_seed = random_seed if random_seed is not None else 42
    edit_types: list[str] = ["remove_stateless_node", "compatible_edit_preserving_tracker"]
    rows: list[AblationSampleRow] = []

    for edit_type in edit_types:
        scenario_id = f"ablation_{edit_type}"
        for rep in range(1, repetitions + 1):
            block_id = f"{scenario_id}_block_{rep}"
            block_seed = (base_seed * 100003 + hash((scenario_id, edit_type, rep))) & 0xFFFFFFFF
            block_rng = random.Random(block_seed)

            # Generate identical request phase offset for all variants in block
            request_phase_offset_ns = block_rng.randint(0, 500_000)

            # Randomize execution order of Variants A, B, C, D
            variants = ["variant_a", "variant_b", "variant_c", "variant_d"]
            block_rng.shuffle(variants)

            for pos, variant in enumerate(variants, start=1):
                row = _run_ablation_repetition(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    block_id=block_id,
                    block_seed=block_seed,
                    variant=variant,
                    variant_order_position=pos,
                    edit_type=edit_type,
                    repetition=rep,
                    request_phase_offset_ns=request_phase_offset_ns,
                    random_seed=base_seed,
                )
                append_ablation_rows(csv_path, [row])
                rows.append(row)

    return tuple(rows)
