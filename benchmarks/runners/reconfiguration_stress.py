"""RQ2 stress experiment: bounded queue + staging delay.

This runner demonstrates the key advantage of ``prepare_and_commit`` over
pause-based baselines: when candidate preparation involves non-trivial
staging latency (model loading, resource allocation, warmup), the off-path
engine continues processing frames under the old plan while the pause
baselines stop entirely, causing queue overflow and frame drops.

Design
------
- **Bounded queue** (capacity 2–4 frames) with drop-oldest policy.
- **Independent fixed-rate producer** that never pauses — it keeps pushing
  frames at a calibrated rate regardless of whether the consumer is
  blocked.
- **Synthetic staging delay** injected into ``workload_node_99.setup()``
  (the node type used for the inserted/rewired node).  This simulates
  non-trivial model loading without requiring real GPU resources.
- **Calibrated arrival rate** set to ~90–95 % of steady-state capacity so
  the system is stable *before* reconfiguration but saturates the small
  queue almost immediately when the consumer pauses.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from queue import Full, Queue
from threading import Event, Lock, Thread

from benchmarks.model import ReconfigurationSampleRow
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_stress_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_reconfiguration_rows, write_reconfiguration_header
from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    CompilationFailure,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import RetirementStatus
from nedo_vision_dag_engine.lifecycle import retire_superseded_processors
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.registry import RegistrySnapshot
from nedo_vision_dag_engine.specification import WorkflowSpecification

__all__ = ["run_reconfiguration_stress_suite"]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

_CALIBRATION_FRAMES = 200
_CALIBRATION_WARMUP = 20


def _measure_steady_state_service_time_ns(
    registry: RegistrySnapshot,
    base_spec: WorkflowSpecification,
) -> float:
    """Return the median per-frame service time (admission→completion) under
    steady-state, uncompeted conditions using ``prepare_and_commit``."""
    init_compiler = WorkflowCompiler("0.1.0")
    init_cand = init_compiler.compile(base_spec, registry)
    if not isinstance(init_cand, CompiledCandidate):
        raise RuntimeError("Calibration compilation failed.")
    executor = PipelineExecutor(initial_plan=init_cand.plan)
    controller = ReconfigurationController(executor=executor, compiler=init_compiler, registry=registry)

    times_ns: list[int] = []
    for i in range(_CALIBRATION_FRAMES + _CALIBRATION_WARMUP):
        t0 = time.perf_counter_ns()
        controller.admit_frame(admitted_at_ns=t0, frame_id=i)
        t1 = time.perf_counter_ns()
        if i >= _CALIBRATION_WARMUP:
            times_ns.append(t1 - t0)
    controller.close()

    times_ns.sort()
    return float(times_ns[len(times_ns) // 2])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameLogEntry:
    frame_id: int
    plan_version: int
    arrival_ns: int
    admission_ns: int
    completion_ns: int


def _compute_maximum_output_gap_ns(frame_log: list[FrameLogEntry]) -> int | None:
    if len(frame_log) < 2:
        return None
    completions = sorted(entry.completion_ns for entry in frame_log)
    gaps = tuple(
        current - previous
        for previous, current in zip(completions, completions[1:], strict=False)
        if current >= previous
    )
    return max(gaps, default=None)


# ---------------------------------------------------------------------------
# Stress repetition
# ---------------------------------------------------------------------------


def _run_stress_repetition(
    *,
    run_id: str,
    scenario_id: str,
    baseline: str,
    edit_type: str,
    graph_size: int,
    repetition: int,
    base_spec: WorkflowSpecification,
    target_spec: WorkflowSpecification,
    registry: RegistrySnapshot,
    inter_arrival_s: float,
    queue_capacity: int,
    staging_delay_s: float,
) -> ReconfigurationSampleRow:
    """Run one stress repetition with a bounded queue and independent producer."""

    init_compiler = WorkflowCompiler("0.1.0")
    init_cand = init_compiler.compile(base_spec, registry)
    if not isinstance(init_cand, CompiledCandidate):
        raise RuntimeError("Initial compilation failed.")
    initial_plan = init_cand.plan
    state_directive = StateDirective()

    # ---- bounded frame queue ------------------------------------------------
    frame_queue: Queue[tuple[int, int]] = Queue(maxsize=queue_capacity)
    frame_log: list[FrameLogEntry] = []
    log_lock = Lock()

    stop_producer = Event()
    stop_worker = Event()
    worker_paused = Event()

    # Counters (protected by drop_lock for brevity; fine for a benchmark)
    drop_lock = Lock()
    dropped_frame_count = 0
    total_produced = 0

    # ---- independent fixed-rate producer ------------------------------------
    def producer_loop() -> None:
        nonlocal dropped_frame_count, total_produced
        fid = 0
        while not stop_producer.is_set():
            arr_ns = time.perf_counter_ns()
            try:
                frame_queue.put_nowait((fid, arr_ns))
                fid += 1
            except Full:
                # drop-oldest: drain one, then enqueue
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    frame_queue.put_nowait((fid, arr_ns))
                    fid += 1
                except Full:
                    pass
                with drop_lock:
                    dropped_frame_count += 1
            total_produced = fid
            # fixed-rate sleep (busy-wait tuned)
            target = arr_ns + int(inter_arrival_s * 1e9)
            while time.perf_counter_ns() < target:
                time.sleep(0)  # yield GIL

    t_producer = Thread(target=producer_loop, daemon=True)
    t_producer.start()

    # ---- baseline-specific worker + reconfig --------------------------------
    if baseline == "stop_rebuild_restart":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop() -> None:
            nonlocal current_executor
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

        t_worker = Thread(target=worker_loop, daemon=True)
        t_worker.start()

        # warmup
        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()
        worker_paused.set()
        # drain residual
        while True:
            try:
                frame_queue.get_nowait()
            except Exception:
                break

        compiler = WorkflowCompiler("0.1.0")
        t_val_start = time.perf_counter_ns()
        validated = compiler.validate(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")

        t_prep_start = time.perf_counter_ns()
        cand = compiler.compile_validated(validated, previous_plan=None, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        new_version = initial_plan.version + 1
        new_plan = replace(cand.plan, version=new_version)
        cand = replace(cand, plan=new_plan)

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError(f"Retirement failed: {retire_report.failures}")

        t_commit_start = time.perf_counter_ns()
        new_executor = PipelineExecutor(initial_plan=cand.plan)
        current_executor = new_executor
        t_commit_end = time.perf_counter_ns()

        worker_paused.clear()

        old_plan_version = initial_plan.version
        new_plan_version = new_version

        # wait for first new-plan frame
        while True:
            with log_lock:
                if any(e.plan_version == new_plan_version for e in frame_log):
                    break
            time.sleep(0.0001)

        # let a few more frames through
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
        request_to_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start
        retire_duration_ns = t_ret_end - t_ret_start

        first_new_frame = next(e for e in frame_log if e.plan_version == new_plan_version)
        request_to_effect_ns = first_new_frame.completion_ns - t_request

        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new_frame.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )
        transition_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        global_max_gap = _compute_maximum_output_gap_ns(frame_log)
        frames_during = sum(1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns)

        old_frames = sum(
            1 for e in frame_log
            if e.plan_version == old_plan_version
            and e.admission_ns >= t_request
            and e.completion_ns < t_commit_start
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
            request_to_ready_ns=request_to_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=request_to_effect_ns,
            retirement_queue_delay_ns=0,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=None,
            maximum_output_gap_ns=global_max_gap,
            transition_output_gap_ns=transition_gap_ns,
            frames_completed_during_request=frames_during,
            old_plan_frames_admitted_after_request_before_commit=old_frames,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="AUTO",
        )

    elif baseline == "pause_compile_resume":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop() -> None:
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

        t_worker = Thread(target=worker_loop, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()
        worker_paused.set()
        while True:
            try:
                frame_queue.get_nowait()
            except Exception:
                break

        t_val_start = time.perf_counter_ns()
        validated = init_compiler.validate(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")

        t_prep_start = time.perf_counter_ns()
        cand = init_compiler.compile_validated(validated, previous_plan=initial_plan, state_directive=state_directive)
        t_prep_end = time.perf_counter_ns()
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        t_commit_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_commit_end = time.perf_counter_ns()

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError(f"Retirement failed: {retire_report.failures}")

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

        val_ns = t_val_end - t_val_start
        prep_ns = t_prep_end - t_prep_start
        request_to_ready_ns = t_prep_end - t_request
        commit_ns = t_commit_end - t_commit_start
        retire_duration_ns = t_ret_end - t_ret_start

        first_new_frame = next(e for e in frame_log if e.plan_version == new_plan_version)
        request_to_effect_ns = first_new_frame.completion_ns - t_request

        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= first_new_frame.completion_ns),
            key=lambda e: e.completion_ns,
            default=None,
        )
        transition_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        global_max_gap = _compute_maximum_output_gap_ns(frame_log)
        frames_during = sum(1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns)

        old_frames = sum(
            1 for e in frame_log
            if e.plan_version == old_plan_version
            and e.admission_ns >= t_request
            and e.completion_ns < t_commit_start
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
            request_to_ready_ns=request_to_ready_ns,
            boundary_wait_ns=0,
            commit_ns=commit_ns,
            request_to_effect_ns=request_to_effect_ns,
            retirement_queue_delay_ns=0,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=None,
            maximum_output_gap_ns=global_max_gap,
            transition_output_gap_ns=transition_gap_ns,
            frames_completed_during_request=frames_during,
            old_plan_frames_admitted_after_request_before_commit=old_frames,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="AUTO",
        )

    elif baseline == "prepare_and_commit":
        current_executor = PipelineExecutor(initial_plan=initial_plan)
        controller = ReconfigurationController(executor=current_executor, compiler=init_compiler, registry=registry)

        def worker_loop() -> None:
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

        t_worker = Thread(target=worker_loop, daemon=True)
        t_worker.start()

        while True:
            with log_lock:
                if len(frame_log) >= 5:
                    break
            time.sleep(0.0001)

        t_request = time.perf_counter_ns()
        req = ReconfigurationRequest(
            request_id="stress_1",
            base_version=initial_plan.version,
            target_specification=target_spec,
            state_directive=state_directive,
            submitted_at_ns=t_request,
        )
        controller.submit(req)

        rec_eff = controller.wait_for_effect(req.request_id, timeout_seconds=120.0)
        if rec_eff is None:
            raise RuntimeError("wait_for_effect timed out.")

        ret_rec = controller.wait_for_retirement(req.request_id, timeout_seconds=120.0)
        if ret_rec is None:
            raise RuntimeError("wait_for_retirement timed out.")
        if ret_rec.retirement_status is RetirementStatus.FAILED:
            raise RuntimeError(f"Retirement failed: {ret_rec.retirement_failure_reason}")

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
        request_to_ready_ns = record.ready_ns - record.request_received_ns if record.ready_ns else 0
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
        request_to_effect_ns = (
            record.first_new_frame_completed_ns - t_request
            if record.first_new_frame_completed_ns
            else 0
        )
        retire_queue_delay_ns = (
            record.retirement_started_ns - record.commit_ns
            if record.retirement_started_ns and record.commit_ns
            else 0
        )
        retire_duration_ns = (
            record.retirement_completed_ns - record.retirement_started_ns
            if record.retirement_completed_ns and record.retirement_started_ns
            else 0
        )
        commit_to_retire_complete_ns = (
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
        transition_gap_ns = (
            first_new_frame.completion_ns - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        global_max_gap = _compute_maximum_output_gap_ns(frame_log)
        frames_during = sum(1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns)

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
            request_to_ready_ns=request_to_ready_ns,
            boundary_wait_ns=boundary_wait_ns,
            commit_ns=commit_ns,
            request_to_effect_ns=request_to_effect_ns,
            retirement_queue_delay_ns=retire_queue_delay_ns,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=commit_to_retire_complete_ns,
            maximum_output_gap_ns=global_max_gap,
            transition_output_gap_ns=transition_gap_ns,
            frames_completed_during_request=frames_during,
            old_plan_frames_admitted_after_request_before_commit=record.old_plan_frames_admitted_after_request_before_commit,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=record.reused_processor_count,
            staged_processor_count=record.staged_processor_count,
            retired_processor_count=record.retired_processor_count,
            state_transition_policy="AUTO",
        )

    else:
        raise ValueError(f"Unknown baseline {baseline!r}")


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------


def run_reconfiguration_stress_suite(
    run_id: str,
    output_csv_path: str,
    profile: str,
    repetition_count: int,
    seed: int = 42,
) -> tuple[ReconfigurationSampleRow, ...]:
    """Run the focused RQ2 stress experiment.

    Parameters
    ----------
    staging_delays_ms:
        Synthetic staging delays injected into the inserted-node processor.
    queue_capacity:
        Bounded frame-queue capacity (drop-oldest policy).
    target_load:
        Target arrival load as a fraction of steady-state capacity (0–1).
        The inter-arrival time is set to ``service_time / target_load``.
    """
    write_reconfiguration_header(output_csv_path)

    edit_type = "insert_stateless_node"
    baselines: tuple[str, ...] = (
        "stop_rebuild_restart",
        "pause_compile_resume",
        "prepare_and_commit",
    )
    staging_delays_ms: tuple[float, ...] = (0.0, 20.0, 50.0)
    queue_capacity: int = 4
    target_load: float = 0.90

    all_rows: list[ReconfigurationSampleRow] = []

    # ---- calibrate service time once (without staging delay) -----------------
    base_spec_cal = make_reconfiguration_base_spec(with_tracker=False)
    registry_cal = create_reconfiguration_stress_registry(staging_delay_s=0.0, extra_tracker=False)
    service_time_ns = _measure_steady_state_service_time_ns(registry_cal, base_spec_cal)
    inter_arrival_s = (service_time_ns / target_load) / 1e9

    # print(f"Stress calibration: service_time={service_time_ns:.0f} ns, "
    #       f"target_load={target_load}, inter_arrival={inter_arrival_s*1e6:.1f} µs, "
    #       f"queue_capacity={queue_capacity}")

    for delay_ms in staging_delays_ms:
        delay_s = delay_ms / 1000.0
        registry = create_reconfiguration_stress_registry(staging_delay_s=delay_s, extra_tracker=False)
        base_spec = make_reconfiguration_base_spec(with_tracker=False)
        target_spec = apply_reconfiguration_edit(base_spec, edit_type)
        graph_size = len(base_spec.nodes)
        scenario_id = f"stress_delay{int(delay_ms)}ms_size{graph_size}_q{queue_capacity}_load{int(target_load*100)}"

        for rep in range(1, repetition_count + 1):
            rot_idx = (rep - 1) % len(baselines)
            rep_baselines = baselines[rot_idx:] + baselines[:rot_idx]

            for baseline in rep_baselines:
                gc.collect()
                row = _run_stress_repetition(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    baseline=baseline,
                    edit_type=edit_type,
                    graph_size=graph_size,
                    repetition=rep,
                    base_spec=base_spec,
                    target_spec=target_spec,
                    registry=registry,
                    inter_arrival_s=inter_arrival_s,
                    queue_capacity=queue_capacity,
                    staging_delay_s=delay_s,
                )
                all_rows.append(row)
                append_reconfiguration_rows(output_csv_path, [row])
                # print(f"  [{scenario_id}] {baseline} rep={rep} "
                #       f"dropped={row.frames_dropped} "
                #       f"req2effect={(row.request_to_effect_ns or 0)/1e6:.2f}ms "
                #       f"trans_gap={(row.transition_output_gap_ns or 0)/1e6:.2f}ms "
                #       f"old_frames={row.old_plan_frames_admitted_after_request_before_commit}")

    return tuple(all_rows)
