"""RQ2 structural reconfiguration performance and graph-size sensitivity runner."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Lock, Thread

from benchmarks.admission import (
    AdmissionGate,
    FrameAccountingSnapshot,
    OfferedFrameAccounting,
    WorkerErrors,
    drain_queue,
    wait_until,
)
from benchmarks.model import (
    ReconfigurationSampleRow,
    compute_critical_path_decomposition,
    validate_reconfiguration_sample,
)
from benchmarks.ordering import balanced_order
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    generate_layered_dag,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_reconfiguration_rows, write_reconfiguration_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, CompilationFailure, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import FrameStatus, RetirementStatus
from nedo_vision_dag_engine.lifecycle import retire_superseded_processors
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.specification import WorkflowSpecification

__all__ = ["run_reconfiguration_suite"]


@dataclass(frozen=True, slots=True)
class FrameLogEntry:
    frame_id: int
    plan_version: int
    arrival_ns: int
    admission_ns: int
    completion_ns: int
    status: FrameStatus
    error: str | None


def _require_all_frames_completed(
    frame_log: list[FrameLogEntry],
    scenario_id: str,
    baseline: str,
) -> None:
    """Abort the run if any frame errored instead of producing an output.

    A mis-wired scenario graph makes every frame fail while the timing
    instrumentation keeps reporting plausible numbers, so the failure is
    invisible unless it is checked explicitly.
    """
    failed = [entry for entry in frame_log if entry.status is not FrameStatus.COMPLETED]
    if not failed:
        return
    raise RuntimeError(
        f"{len(failed)} of {len(frame_log)} frames failed in scenario "
        f"{scenario_id!r} under baseline {baseline!r}; first error: {failed[0].error}"
    )


def _compute_maximum_output_gap_ns(frame_log: list[FrameLogEntry]) -> int | None:
    """Maximum gap between consecutive frame completions across the entire log."""
    if len(frame_log) < 2:
        return None
    completions = sorted(entry.completion_ns for entry in frame_log)
    gaps = tuple(
        current - previous
        for previous, current in zip(completions, completions[1:], strict=False)
        if current >= previous
    )
    return max(gaps, default=None)


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
            rep_baselines = balanced_order(baselines, seed, scenario_id, rep)

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

    # 2. Graph-Size Sensitivity (5, 10, 25, 50, 100 nodes)
    # All baselines tested at every size for complete scaling trends
    # required by reviewer feedback (P1.4).
    sensitivity_sizes: tuple[int, ...] = (5, 10, 25, 50, 100)
    for sz in sensitivity_sizes:
        sens_base = generate_layered_dag(sz)
        sens_target = apply_reconfiguration_edit(sens_base, "insert_stateless_node")
        sens_scenario_id = f"graph_size_sensitivity_{sz}"

        for rep in range(1, repetition_count + 1):
            rep_sens_baselines = balanced_order(baselines, seed, sens_scenario_id, rep)

            for baseline in rep_sens_baselines:
                gc.collect()
                row = _run_reconfig_repetition(
                    run_id=run_id,
                    scenario_id=sens_scenario_id,
                    baseline=baseline,
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
    admission_gate = AdmissionGate()
    worker_errors = WorkerErrors()
    accounting = OfferedFrameAccounting()
    operation_timeout_seconds = 10.0
    intentional_cancellation_count = 0

    dropped_frame_count = 0
    drop_lock = Lock()

    def producer_loop() -> None:
        nonlocal dropped_frame_count
        try:
            fid = 0
            while not stop_producer.is_set():
                arr_ns = time.perf_counter_ns()
                offered_id = fid
                fid += 1
                accounting.offer(offered_id)
                try:
                    frame_queue.put_nowait((offered_id, arr_ns))
                except Full:
                    accounting.classify(offered_id, "ingress_overflow")
                    with drop_lock:
                        dropped_frame_count += 1
                time.sleep(inter_arrival_s)
        except BaseException as error:
            worker_errors.capture(error)
            stop_worker.set()

    t_producer = Thread(target=producer_loop, daemon=True)
    t_producer.start()

    def finish_accounting() -> FrameAccountingSnapshot:
        pending = drain_queue(frame_queue)
        return accounting.snapshot({frame_id for frame_id, _ in pending})

    if baseline == "stop_rebuild_restart":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_restart() -> None:
            nonlocal current_executor, intentional_cancellation_count
            try:
                while not stop_worker.is_set():
                    try:
                        fid, arr_ns = frame_queue.get(timeout=0.005)
                    except Exception:
                        continue
                    if not admission_gate.try_enter():
                        # Dequeued after the boundary closed, so admission refused it.
                        accounting.classify(fid, "admission_rejection")
                        continue
                    try:
                        t_adm = time.perf_counter_ns()
                        res = current_executor.admit_frame(admitted_at_ns=arr_ns, frame_id=fid)
                        t_comp = time.perf_counter_ns()
                    finally:
                        admission_gate.leave()
                    with log_lock:
                        frame_log.append(
                            FrameLogEntry(
                                frame_id=fid,
                                plan_version=res.plan_version,
                                arrival_ns=arr_ns,
                                admission_ns=t_adm,
                                completion_ns=t_comp,
                                status=res.status,
                                error=res.error,
                            )
                        )
                    accounting.classify(
                        fid,
                        "completed" if res.status is FrameStatus.COMPLETED else "failed_execution",
                    )
            except BaseException as error:
                worker_errors.capture(error)
                stop_producer.set()

        t_worker = Thread(target=worker_loop_restart, daemon=True)
        t_worker.start()

        wait_until(lambda: len(frame_log) >= 5, timeout_seconds=operation_timeout_seconds,
                   description="Stop warm-up frames", errors=worker_errors)

        t_request = time.perf_counter_ns()

        t_adm_stop_start = time.perf_counter_ns()
        admission_gate.close_and_drain(operation_timeout_seconds)
        cancelled = drain_queue(frame_queue)
        for cancelled_id, _ in cancelled:
            accounting.classify(cancelled_id, "intentional_queue_cancellation")
        with drop_lock:
            intentional_cancellation_count += len(cancelled)
        t_adm_stop_end = time.perf_counter_ns()

        # The same compiler instance that produced the base plan, compiled
        # against that plan: all three mechanisms must apply one compatibility
        # and state policy, so Stop differs only by where the work happens.
        t_val_start = time.perf_counter_ns()
        validated = init_compiler.validate(target_spec, registry)
        t_val_end = time.perf_counter_ns()
        if isinstance(validated, CompilationFailure):
            raise RuntimeError(f"Validation failed: {validated.reason}")

        t_prep_start = time.perf_counter_ns()
        cand = init_compiler.compile_validated(
            validated, previous_plan=initial_plan, state_directive=state_directive
        )
        t_prep_end = time.perf_counter_ns()

        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Candidate compilation failed.")

        # Executor teardown: the synthetic benchmark has no external resources
        # (GPU, file descriptors) to release; teardown is the cost of abandoning
        # the old executor's processor references to the GC.  Recorded as a
        # lower bound.
        t_teardown_start = time.perf_counter_ns()
        # Explicitly drop the reference to old executor's processors to force
        # any pending reference cleanup before reconstruction begins.
        del current_executor
        t_teardown_end = time.perf_counter_ns()

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError(f"stop_rebuild_restart processor retirement failed: {retire_report.failures}")

        t_reconstruct_start = time.perf_counter_ns()
        new_executor = PipelineExecutor(initial_plan=cand.plan)
        t_reconstruct_end = time.perf_counter_ns()

        t_pub_start = time.perf_counter_ns()
        current_executor = new_executor
        t_pub_end = time.perf_counter_ns()

        t_restart_start = time.perf_counter_ns()
        admission_gate.open()
        t_restart_end = time.perf_counter_ns()

        old_plan_version = initial_plan.version
        new_plan_version = initial_plan.version + 1

        wait_until(lambda: any(e.plan_version == cand.plan.version for e in frame_log),
                   timeout_seconds=operation_timeout_seconds,
                   description="Stop first new-plan frame", errors=worker_errors)

        target_total = len(frame_log) + 5
        wait_until(lambda: len(frame_log) >= target_total,
                   timeout_seconds=operation_timeout_seconds,
                   description="Stop target frames", errors=worker_errors)

        stop_producer.set()
        stop_worker.set()
        t_producer.join(timeout=operation_timeout_seconds)
        t_worker.join(timeout=operation_timeout_seconds)
        if t_producer.is_alive() or t_worker.is_alive():
            raise TimeoutError("Stop benchmark threads did not terminate")
        worker_errors.raise_if_any()
        accounting_snapshot = finish_accounting()
        _require_all_frames_completed(frame_log, scenario_id, baseline)

        adm_stop_ns = t_adm_stop_end - t_adm_stop_start
        val_ns = t_val_end - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        teardown_ns = t_teardown_end - t_teardown_start
        retirement_ns = t_ret_end - t_ret_start
        reconstruct_ns = t_reconstruct_end - t_reconstruct_start
        pub_ns = t_pub_end - t_pub_start
        restart_ns = t_restart_end - t_restart_start
        commit_ns = reconstruct_ns + pub_ns
        retire_duration_ns = retirement_ns

        first_new_frame = next(e for e in frame_log if e.plan_version == cand.plan.version)
        req_effect_ns = first_new_frame.completion_ns - t_request

        first_adm_wait_ns = max(0, first_new_frame.admission_ns - t_restart_end)
        first_comp_wait_ns = max(0, first_new_frame.completion_ns - first_new_frame.admission_ns)

        phase_intervals: list[tuple[int, int]] = [
            (t_adm_stop_start, t_adm_stop_end),
            (t_val_start, t_val_end),
            (t_prep_start, t_prep_end),
            (t_teardown_start, t_teardown_end),
            (t_ret_start, t_ret_end),
            (t_reconstruct_start, t_reconstruct_end),
            (t_pub_start, t_pub_end),
            (t_restart_start, t_restart_end),
        ]
        if first_new_frame.admission_ns >= t_restart_end:
            phase_intervals.append((t_restart_end, first_new_frame.admission_ns))
        phase_intervals.append((first_new_frame.admission_ns, first_new_frame.completion_ns))
        decomp = compute_critical_path_decomposition(
            t_request=t_request,
            t_effect=first_new_frame.completion_ns,
            phase_intervals=phase_intervals,
        )
        total_sync_ns = (
            adm_stop_ns
            + val_ns
            + prep_ns
            + teardown_ns
            + retirement_ns
            + reconstruct_ns
            + pub_ns
            + restart_ns
        )
        phases_may_overlap = False

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
        global_max_output_gap_ns = _compute_maximum_output_gap_ns(frame_log)
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns
        )

        old_frames_before_commit = sum(
            1
            for event in frame_log
            if event.plan_version == old_plan_version
            and event.admission_ns >= t_adm_stop_start
            and event.completion_ns < t_pub_start
        )
        # NOTE: a small non-zero count (typically ≤ 1) is expected due to
        # the TOCTOU window between the worker's double-check of the pause
        # flag and the actual admit_frame() call.  The value is recorded as
        # old_plan_frames_admitted_after_request_before_commit for auditing.

        row = ReconfigurationSampleRow(
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
            retirement_queue_delay_ns=0,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=None,
            maximum_output_gap_ns=global_max_output_gap_ns,
            transition_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            old_plan_frames_admitted_after_request_before_commit=old_frames_before_commit,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
            admission_stop_ns=adm_stop_ns,
            executor_teardown_ns=teardown_ns,
            executor_reconstruction_ns=reconstruct_ns,
            publication_ns=pub_ns,
            executor_restart_ns=restart_ns,
            first_admission_wait_ns=first_adm_wait_ns,
            first_completion_wait_ns=first_comp_wait_ns,
            retirement_ns=retirement_ns,
            total_synchronous_ns=total_sync_ns,
            phases_may_overlap=phases_may_overlap,
            sum_of_instrumented_phase_durations_ns=decomp.sum_of_instrumented_phase_durations_ns,
            instrumented_phase_sum_ns=decomp.sum_of_instrumented_phase_durations_ns,
            critical_path_instrumented_ns=decomp.critical_path_instrumented_ns,
            unattributed_critical_path_ns=decomp.unattributed_critical_path_ns,
            unattributed_request_time_ns=decomp.unattributed_critical_path_ns,
            instrumented_duration_overlap_ns=decomp.instrumented_duration_overlap_ns,
            instrumented_duration_outside_effect_window_ns=decomp.instrumented_duration_outside_effect_window_ns,
            frames_offered=accounting_snapshot.offered,
            frames_completed=accounting_snapshot.completed,
            frames_failed_execution=accounting_snapshot.failed_execution,
            frames_ingress_overflow=accounting_snapshot.ingress_overflow,
            frames_intentionally_cancelled=accounting_snapshot.intentional_queue_cancellation,
            frames_admission_rejected=accounting_snapshot.admission_rejection,
            frames_still_queued_or_in_flight=accounting_snapshot.still_queued_or_in_flight,
            frame_accounting_residual=accounting_snapshot.residual,
        )
        validate_reconfiguration_sample(row)
        return row

    elif baseline == "pause_compile_resume":
        current_executor = PipelineExecutor(initial_plan=initial_plan)

        def worker_loop_pause() -> None:
            nonlocal intentional_cancellation_count
            try:
                while not stop_worker.is_set():
                    try:
                        fid, arr_ns = frame_queue.get(timeout=0.005)
                    except Exception:
                        continue
                    if not admission_gate.try_enter():
                        # Dequeued after the boundary closed, so admission refused it.
                        accounting.classify(fid, "admission_rejection")
                        continue
                    try:
                        t_adm = time.perf_counter_ns()
                        res = current_executor.admit_frame(admitted_at_ns=arr_ns, frame_id=fid)
                        t_comp = time.perf_counter_ns()
                    finally:
                        admission_gate.leave()
                    with log_lock:
                        frame_log.append(
                            FrameLogEntry(
                                frame_id=fid,
                                plan_version=res.plan_version,
                                arrival_ns=arr_ns,
                                admission_ns=t_adm,
                                completion_ns=t_comp,
                                status=res.status,
                                error=res.error,
                            )
                        )
                    accounting.classify(
                        fid,
                        "completed" if res.status is FrameStatus.COMPLETED else "failed_execution",
                    )
            except BaseException as error:
                worker_errors.capture(error)
                stop_producer.set()

        t_worker = Thread(target=worker_loop_pause, daemon=True)
        t_worker.start()

        wait_until(lambda: len(frame_log) >= 5, timeout_seconds=operation_timeout_seconds,
                   description="Pause warm-up frames", errors=worker_errors)

        t_request = time.perf_counter_ns()

        t_adm_stop_start = time.perf_counter_ns()
        admission_gate.close_and_drain(operation_timeout_seconds)
        cancelled = drain_queue(frame_queue)
        for cancelled_id, _ in cancelled:
            accounting.classify(cancelled_id, "intentional_queue_cancellation")
        with drop_lock:
            intentional_cancellation_count += len(cancelled)
        t_adm_stop_end = time.perf_counter_ns()

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

        t_pub_start = time.perf_counter_ns()
        current_executor.commit(cand.plan)
        t_pub_end = time.perf_counter_ns()

        t_ret_start = time.perf_counter_ns()
        retire_report = retire_superseded_processors(previous_plan=initial_plan, active_plan=cand.plan)
        t_ret_end = time.perf_counter_ns()
        if not retire_report.succeeded:
            raise RuntimeError(f"pause_compile_resume processor retirement failed: {retire_report.failures}")

        t_restart_start = time.perf_counter_ns()
        admission_gate.open()
        t_restart_end = time.perf_counter_ns()

        old_plan_version = initial_plan.version
        new_plan_version = cand.plan.version

        wait_until(lambda: any(e.plan_version == cand.plan.version for e in frame_log),
                   timeout_seconds=operation_timeout_seconds,
                   description="Pause first new-plan frame", errors=worker_errors)

        target_total = len(frame_log) + 5
        wait_until(lambda: len(frame_log) >= target_total,
                   timeout_seconds=operation_timeout_seconds,
                   description="Pause target frames", errors=worker_errors)

        stop_producer.set()
        stop_worker.set()
        t_producer.join(timeout=operation_timeout_seconds)
        t_worker.join(timeout=operation_timeout_seconds)
        if t_producer.is_alive() or t_worker.is_alive():
            raise TimeoutError("Pause benchmark threads did not terminate")
        worker_errors.raise_if_any()
        accounting_snapshot = finish_accounting()
        _require_all_frames_completed(frame_log, scenario_id, baseline)

        adm_stop_ns = t_adm_stop_end - t_adm_stop_start
        val_ns = t_val_end - t_val_start
        prep_ns = t_prep_end - t_prep_start
        req_ready_ns = t_prep_end - t_request
        pub_ns = t_pub_end - t_pub_start
        retirement_ns = t_ret_end - t_ret_start
        teardown_ns = 0
        reconstruct_ns = 0
        restart_ns = t_restart_end - t_restart_start
        commit_ns = pub_ns
        retire_duration_ns = retirement_ns

        first_new_frame = next(e for e in frame_log if e.plan_version == cand.plan.version)
        req_effect_ns = first_new_frame.completion_ns - t_request

        first_adm_wait_ns = max(0, first_new_frame.admission_ns - t_restart_end)
        first_comp_wait_ns = max(0, first_new_frame.completion_ns - first_new_frame.admission_ns)

        phase_intervals: list[tuple[int, int]] = [
            (t_adm_stop_start, t_adm_stop_end),
            (t_val_start, t_val_end),
            (t_prep_start, t_prep_end),
            (t_pub_start, t_pub_end),
            (t_ret_start, t_ret_end),
            (t_restart_start, t_restart_end),
        ]
        if first_new_frame.admission_ns >= t_restart_end:
            phase_intervals.append((t_restart_end, first_new_frame.admission_ns))
        phase_intervals.append((first_new_frame.admission_ns, first_new_frame.completion_ns))
        decomp = compute_critical_path_decomposition(
            t_request=t_request,
            t_effect=first_new_frame.completion_ns,
            phase_intervals=phase_intervals,
        )
        total_sync_ns = adm_stop_ns + val_ns + prep_ns + pub_ns + retirement_ns + restart_ns
        phases_may_overlap = False

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
        global_max_output_gap_ns = _compute_maximum_output_gap_ns(frame_log)
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= first_new_frame.completion_ns
        )

        old_frames_before_commit = sum(
            1
            for event in frame_log
            if event.plan_version == old_plan_version
            and event.admission_ns >= t_adm_stop_start
            and event.completion_ns < t_pub_start
        )
        # NOTE: a small non-zero count (typically ≤ 1) is expected due to
        # the TOCTOU window between the worker's double-check of the pause
        # flag and the actual admit_frame() call.  The value is recorded as
        # old_plan_frames_admitted_after_request_before_commit for auditing.

        row = ReconfigurationSampleRow(
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
            retirement_queue_delay_ns=0,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=None,
            maximum_output_gap_ns=global_max_output_gap_ns,
            transition_output_gap_ns=max_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            old_plan_frames_admitted_after_request_before_commit=old_frames_before_commit,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=len(cand.reused_node_ids),
            staged_processor_count=len(cand.staged_node_ids),
            retired_processor_count=len(initial_plan.steps) - len(cand.reused_node_ids),
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
            admission_stop_ns=adm_stop_ns,
            executor_teardown_ns=teardown_ns,
            executor_reconstruction_ns=reconstruct_ns,
            publication_ns=pub_ns,
            executor_restart_ns=restart_ns,
            first_admission_wait_ns=first_adm_wait_ns,
            first_completion_wait_ns=first_comp_wait_ns,
            retirement_ns=retirement_ns,
            total_synchronous_ns=total_sync_ns,
            phases_may_overlap=phases_may_overlap,
            sum_of_instrumented_phase_durations_ns=decomp.sum_of_instrumented_phase_durations_ns,
            instrumented_phase_sum_ns=decomp.sum_of_instrumented_phase_durations_ns,
            critical_path_instrumented_ns=decomp.critical_path_instrumented_ns,
            unattributed_critical_path_ns=decomp.unattributed_critical_path_ns,
            unattributed_request_time_ns=decomp.unattributed_critical_path_ns,
            instrumented_duration_overlap_ns=decomp.instrumented_duration_overlap_ns,
            instrumented_duration_outside_effect_window_ns=decomp.instrumented_duration_outside_effect_window_ns,
            frames_offered=accounting_snapshot.offered,
            frames_completed=accounting_snapshot.completed,
            frames_failed_execution=accounting_snapshot.failed_execution,
            frames_ingress_overflow=accounting_snapshot.ingress_overflow,
            frames_intentionally_cancelled=accounting_snapshot.intentional_queue_cancellation,
            frames_admission_rejected=accounting_snapshot.admission_rejection,
            frames_still_queued_or_in_flight=accounting_snapshot.still_queued_or_in_flight,
            frame_accounting_residual=accounting_snapshot.residual,
        )
        validate_reconfiguration_sample(row)
        return row

    elif baseline == "prepare_and_commit":
        current_executor = PipelineExecutor(initial_plan=initial_plan)
        controller = ReconfigurationController(executor=current_executor, compiler=init_compiler, registry=registry)

        def worker_loop_prepare() -> None:
            try:
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
                                status=res.status,
                                error=res.error,
                            )
                        )
                    accounting.classify(
                        fid,
                        "completed" if res.status is FrameStatus.COMPLETED else "failed_execution",
                    )
            except BaseException as error:
                worker_errors.capture(error)
                stop_producer.set()

        t_worker = Thread(target=worker_loop_prepare, daemon=True)
        t_worker.start()

        wait_until(lambda: len(frame_log) >= 5, timeout_seconds=operation_timeout_seconds,
                   description="VEPS warm-up frames", errors=worker_errors)

        t_request = time.perf_counter_ns()

        req = ReconfigurationRequest(
            request_id="reconfig_1",
            base_version=initial_plan.version,
            target_specification=target_spec,
            state_directive=state_directive,
            submitted_at_ns=t_request,
        )

        controller.submit(req)

        # Wait for effect
        rec_eff = controller.wait_for_effect(req.request_id, timeout_seconds=10.0)
        if rec_eff is None:
            raise RuntimeError("prepare_and_commit wait_for_effect timed out.")

        # Wait for retirement
        ret_rec = controller.wait_for_retirement(req.request_id, timeout_seconds=10.0)
        if ret_rec is None:
            raise RuntimeError("prepare_and_commit wait_for_retirement timed out.")
        if ret_rec.retirement_status is RetirementStatus.FAILED:
            raise RuntimeError(f"processor retirement failed during benchmark: {ret_rec.retirement_failure_reason}")

        record = controller.record(req.request_id)

        old_plan_version = initial_plan.version
        new_plan_version = record.candidate_version or (initial_plan.version + 1)

        # Wait for 5 more frames after effect
        target_total = len(frame_log) + 5
        wait_until(lambda: len(frame_log) >= target_total,
                   timeout_seconds=operation_timeout_seconds,
                   description="VEPS target frames", errors=worker_errors)

        stop_producer.set()
        stop_worker.set()
        t_producer.join(timeout=operation_timeout_seconds)
        t_worker.join(timeout=operation_timeout_seconds)
        if t_producer.is_alive() or t_worker.is_alive():
            raise TimeoutError("VEPS benchmark threads did not terminate")
        worker_errors.raise_if_any()
        accounting_snapshot = finish_accounting()
        _require_all_frames_completed(frame_log, scenario_id, baseline)
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
        handoff_wait_ns_val: int | None = None
        if record.handoff_wait_start_ns is not None and record.handoff_wait_complete_ns is not None:
            handoff_wait_ns_val = max(0, record.handoff_wait_complete_ns - record.handoff_wait_start_ns)
        # commit_started_ns precedes the conditional handoff drain, so measuring
        # publication from it would report the drain twice once the drain is
        # added back as its own synchronized phase.
        publication_start_ns = (
            record.handoff_wait_complete_ns
            if record.handoff_wait_complete_ns is not None
            else record.commit_started_ns
        )
        pub_ns = (
            max(0, record.commit_ns - publication_start_ns)
            if record.commit_ns and publication_start_ns
            else 0
        )
        commit_ns = pub_ns
        req_effect_ns = (
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
        retirement_ns = retire_duration_ns
        commit_to_retire_complete_ns = (
            record.retirement_completed_ns - record.commit_ns
            if record.retirement_completed_ns and record.commit_ns
            else 0
        )

        first_new_frame = next(e for e in frame_log if e.plan_version == new_plan_version)
        t_effect = (
            record.first_new_frame_completed_ns
            if record.first_new_frame_completed_ns
            else first_new_frame.completion_ns
        )
        req_effect_ns = max(0, t_effect - t_request)
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
        retirement_ns = retire_duration_ns
        commit_to_retire_complete_ns = (
            record.retirement_completed_ns - record.commit_ns
            if record.retirement_completed_ns and record.commit_ns
            else 0
        )

        last_old_frame = max(
            (e for e in frame_log if e.plan_version == old_plan_version and e.completion_ns <= t_effect),
            key=lambda e: e.completion_ns,
            default=None,
        )
        transition_output_gap_ns = (
            t_effect - last_old_frame.completion_ns
            if last_old_frame is not None
            else 0
        )
        global_max_output_gap_ns = _compute_maximum_output_gap_ns(frame_log)
        frames_during_request = sum(
            1 for e in frame_log if t_request <= e.completion_ns <= t_effect
        )

        first_adm_wait_ns = (
            first_new_frame.admission_ns - record.commit_ns
            if record.commit_ns and first_new_frame.admission_ns >= record.commit_ns
            else 0
        )
        first_comp_wait_ns = max(0, first_new_frame.completion_ns - first_new_frame.admission_ns)

        adm_stop_ns = 0
        teardown_ns = 0
        reconstruct_ns = 0
        restart_ns = 0

        phase_intervals_prep: list[tuple[int, int]] = []
        if record.validation_started_ns and record.validation_completed_ns:
            phase_intervals_prep.append((record.validation_started_ns, record.validation_completed_ns))
        if record.preparation_started_ns and record.preparation_completed_ns:
            phase_intervals_prep.append((record.preparation_started_ns, record.preparation_completed_ns))
        if record.ready_ns and record.commit_started_ns:
            phase_intervals_prep.append((record.ready_ns, record.commit_started_ns))
        if record.handoff_wait_start_ns and record.handoff_wait_complete_ns:
            phase_intervals_prep.append((record.handoff_wait_start_ns, record.handoff_wait_complete_ns))
        if publication_start_ns and record.commit_ns:
            phase_intervals_prep.append((publication_start_ns, record.commit_ns))
        if record.commit_ns and first_new_frame.admission_ns >= record.commit_ns:
            phase_intervals_prep.append((record.commit_ns, first_new_frame.admission_ns))
        phase_intervals_prep.append((first_new_frame.admission_ns, first_new_frame.completion_ns))
        if record.retirement_started_ns and record.retirement_completed_ns:
            phase_intervals_prep.append((record.retirement_started_ns, record.retirement_completed_ns))

        decomp_prep = compute_critical_path_decomposition(
            t_request=t_request,
            t_effect=t_effect,
            phase_intervals=phase_intervals_prep,
        )

        # Synchronized admission work: the drain (when a mutable processor is
        # preserved) plus publication. The two intervals are disjoint.
        total_sync_ns = pub_ns + (handoff_wait_ns_val or 0)
        phases_may_overlap = True

        # Extract new grace-period and handoff engine instrumentation
        grace_period_ns_val: int | None = None
        if record.grace_period_start_ns is not None and record.grace_period_complete_ns is not None:
            grace_period_ns_val = max(0, record.grace_period_complete_ns - record.grace_period_start_ns)
        cleanup_duration_ns_val: int | None = None
        if record.cleanup_start_ns is not None and record.cleanup_end_ns is not None:
            cleanup_duration_ns_val = max(0, record.cleanup_end_ns - record.cleanup_start_ns)
        last_old_frame_ns_val: int | None = record.last_old_frame_completed_ns

        row = ReconfigurationSampleRow(
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
            retirement_queue_delay_ns=retire_queue_delay_ns,
            retirement_duration_ns=retire_duration_ns,
            commit_to_retirement_complete_ns=commit_to_retire_complete_ns,
            maximum_output_gap_ns=global_max_output_gap_ns,
            transition_output_gap_ns=transition_output_gap_ns,
            frames_completed_during_request=frames_during_request,
            old_plan_frames_admitted_after_request_before_commit=record.old_plan_frames_admitted_after_request_before_commit,
            frames_dropped=dropped_frame_count,
            frames_duplicated=0,
            reused_processor_count=record.reused_processor_count,
            staged_processor_count=record.staged_processor_count,
            retired_processor_count=record.retired_processor_count,
            state_transition_policy="PRESERVE" if edit_type == "compatible_edit_preserving_tracker" else "AUTO",
            admission_stop_ns=adm_stop_ns,
            executor_teardown_ns=teardown_ns,
            executor_reconstruction_ns=reconstruct_ns,
            publication_ns=pub_ns,
            executor_restart_ns=restart_ns,
            first_admission_wait_ns=first_adm_wait_ns,
            first_completion_wait_ns=first_comp_wait_ns,
            retirement_ns=retirement_ns,
            total_synchronous_ns=total_sync_ns,
            phases_may_overlap=phases_may_overlap,
            sum_of_instrumented_phase_durations_ns=decomp_prep.sum_of_instrumented_phase_durations_ns,
            instrumented_phase_sum_ns=decomp_prep.sum_of_instrumented_phase_durations_ns,
            critical_path_instrumented_ns=decomp_prep.critical_path_instrumented_ns,
            unattributed_critical_path_ns=decomp_prep.unattributed_critical_path_ns,
            unattributed_request_time_ns=decomp_prep.unattributed_critical_path_ns,
            instrumented_duration_overlap_ns=decomp_prep.instrumented_duration_overlap_ns,
            instrumented_duration_outside_effect_window_ns=decomp_prep.instrumented_duration_outside_effect_window_ns,
            frames_offered=accounting_snapshot.offered,
            frames_completed=accounting_snapshot.completed,
            frames_failed_execution=accounting_snapshot.failed_execution,
            frames_ingress_overflow=accounting_snapshot.ingress_overflow,
            frames_intentionally_cancelled=accounting_snapshot.intentional_queue_cancellation,
            frames_admission_rejected=accounting_snapshot.admission_rejection,
            frames_still_queued_or_in_flight=accounting_snapshot.still_queued_or_in_flight,
            frame_accounting_residual=accounting_snapshot.residual,
            grace_period_ns=grace_period_ns_val,
            handoff_wait_ns=handoff_wait_ns_val,
            cleanup_duration_ns=cleanup_duration_ns_val,
            last_old_frame_completed_ns=last_old_frame_ns_val,
        )
        validate_reconfiguration_sample(row)
        return row

    else:
        raise ValueError(f"Unknown baseline {baseline!r}")
