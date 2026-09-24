"""Controller-based 2x2 preparation and retirement placement experiment."""

from __future__ import annotations

import gc
import hashlib
import os
import random
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock, Thread

from benchmarks.model import AblationSampleRow, validate_ablation_sample
from benchmarks.ordering import balanced_order
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_ablation_rows, write_ablation_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import FrameStatus, RetirementStatus
from nedo_vision_dag_engine.lifecycle import shutdown_plan_processors
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.specification import specification_hash


@dataclass(frozen=True, slots=True)
class FrameLogEntry:
    frame_id: int
    plan_version: int
    arrival_ns: int
    admission_ns: int
    completion_ns: int


_VARIANTS: dict[str, tuple[str, str, str]] = {
    "variant_a": ("sync_prepare_sync_retire", "synchronous", "synchronous"),
    "variant_b": ("offpath_prepare_sync_retire", "off_path", "synchronous"),
    "variant_c": ("sync_prepare_deferred_retire", "synchronous", "deferred"),
    "variant_d": ("offpath_prepare_deferred_retire", "off_path", "deferred"),
}
_OPERATION_TIMEOUT_SECONDS = 10.0


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
    """Run each variant on the same managed executor and controller path.

    Only the time at which admission closes for preparation and retirement
    differs. Publication, frame admission, state policy, and cleanup are shared.
    """
    del graph_size
    variant_name, preparation_placement, retirement_policy = _VARIANTS[variant]
    gc.collect()
    registry = create_reconfiguration_registry()
    with_tracker = edit_type == "compatible_edit_preserving_tracker"
    initial_spec = make_reconfiguration_base_spec(with_tracker=with_tracker)
    target_spec = apply_reconfiguration_edit(initial_spec, edit_type)
    compiler = WorkflowCompiler("0.1.0")
    initial = compiler.compile(initial_spec, registry)
    if not isinstance(initial, CompiledCandidate):
        raise RuntimeError(f"ablation initial compilation failed: {initial.reason}")
    executor = PipelineExecutor(initial.plan)
    controller = ReconfigurationController(
        executor, compiler, registry, auto_commit_on_admission=False
    )

    frame_queue: Queue[tuple[int, int]] = Queue(maxsize=100)
    log_lock = Lock()
    gate = Condition(Lock())
    frame_log: list[FrameLogEntry] = []
    worker_errors: list[BaseException] = []
    stop_producer = Event()
    stop_worker = Event()
    admission_closed = False
    active_frames = 0
    offered_frames = 0
    overflow_frames = 0

    def producer_loop() -> None:
        nonlocal offered_frames, overflow_frames
        frame_id = 0
        try:
            while not stop_producer.is_set():
                arrival_ns = time.monotonic_ns()
                with log_lock:
                    offered_frames += 1
                try:
                    frame_queue.put_nowait((frame_id, arrival_ns))
                except Full:
                    # Each discarded offer is recorded exactly once. The
                    # worker may concurrently dequeue, so retry is bounded.
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                        with log_lock:
                            overflow_frames += 1
                        frame_queue.put_nowait((frame_id, arrival_ns))
                    except Empty:
                        frame_queue.put_nowait((frame_id, arrival_ns))
                    except Full:
                        with log_lock:
                            overflow_frames += 1
                frame_id += 1
                stop_producer.wait(0.001)
        except BaseException as error:
            with log_lock:
                worker_errors.append(error)
            stop_producer.set()

    def worker_loop() -> None:
        nonlocal active_frames
        try:
            while not stop_worker.is_set() or not frame_queue.empty():
                try:
                    frame_id, arrival_ns = frame_queue.get(timeout=0.01)
                except Empty:
                    continue
                try:
                    with gate:
                        while admission_closed and not stop_worker.is_set():
                            gate.wait(timeout=0.1)
                        active_frames += 1
                    admission_ns = time.monotonic_ns()
                    result = controller.admit_frame(admission_ns, frame_id)
                    completion_ns = time.monotonic_ns()
                    if result.status is not FrameStatus.COMPLETED:
                        raise RuntimeError(f"ablation frame {frame_id} failed: {result.error!r}")
                    with log_lock:
                        frame_log.append(FrameLogEntry(
                            frame_id, result.plan_version, arrival_ns, admission_ns, completion_ns
                        ))
                finally:
                    with gate:
                        if active_frames:
                            active_frames -= 1
                        gate.notify_all()
                    frame_queue.task_done()
        except BaseException as error:
            with log_lock:
                worker_errors.append(error)
            stop_worker.set()
            with gate:
                gate.notify_all()

    producer = Thread(target=producer_loop, daemon=True)
    worker = Thread(target=worker_loop, daemon=True)
    producer.start()
    worker.start()

    def check_errors() -> None:
        with log_lock:
            if worker_errors:
                raise RuntimeError("ablation worker failed") from worker_errors[0]

    def completed_count() -> int:
        with log_lock:
            return len(frame_log)

    def has_completed_plan(version: int) -> bool:
        with log_lock:
            return any(entry.plan_version == version for entry in frame_log)

    def wait_until(predicate: object, stage: str) -> None:
        if not callable(predicate):
            raise TypeError("wait predicate must be callable")
        deadline = time.monotonic() + _OPERATION_TIMEOUT_SECONDS
        while not predicate():
            check_errors()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ablation {variant} {stage} timed out")
            time.sleep(0.001)

    def close_gate() -> tuple[int, int]:
        nonlocal admission_closed
        closed_at = time.monotonic_ns()
        with gate:
            admission_closed = True
            drained = gate.wait_for(lambda: active_frames == 0, timeout=_OPERATION_TIMEOUT_SECONDS)
        if not drained:
            raise TimeoutError("ablation admission gate did not drain")
        return closed_at, time.monotonic_ns() - closed_at

    def open_gate() -> int:
        nonlocal admission_closed
        with gate:
            admission_closed = False
            gate.notify_all()
        return time.monotonic_ns()

    try:
        wait_until(lambda: completed_count() >= 5, "warm-up")
        if request_phase_offset_ns:
            time.sleep(request_phase_offset_ns / 1e9)
        request_at_ns = time.monotonic_ns()
        gate_closed_at_ns: int | None = None
        boundary_wait_ns = 0
        if preparation_placement == "synchronous":
            gate_closed_at_ns, boundary_wait_ns = close_gate()

        request = ReconfigurationRequest(
            request_id=f"ablation-{variant}-{repetition}",
            base_version=initial.plan.version,
            target_specification=target_spec,
            state_directive=StateDirective(),
            submitted_at_ns=request_at_ns,
        )
        controller.submit(request)
        ready = controller.wait_for_status(
            request.request_id,
            frozenset({ReconfigurationStatus.READY, ReconfigurationStatus.REJECTED,
                        ReconfigurationStatus.FAILED, ReconfigurationStatus.ABORTED}),
            timeout_seconds=_OPERATION_TIMEOUT_SECONDS,
        )
        if ready is None or ready.status is not ReconfigurationStatus.READY:
            raise RuntimeError(f"ablation candidate did not become READY: {ready!r}")

        if preparation_placement == "off_path":
            gate_closed_at_ns, boundary_wait_ns = close_gate()
        committed = controller.commit_ready()
        if committed is None or committed.status is not ReconfigurationStatus.COMMITTED:
            raise RuntimeError(f"ablation publication failed: {committed!r}")
        commit_record = controller.record(request.request_id)
        if commit_record.commit_ns is None:
            raise RuntimeError("ablation commit timestamp is missing")

        retirement_record = None
        if retirement_policy == "synchronous":
            retirement_record = controller.wait_for_retirement(
                request.request_id, _OPERATION_TIMEOUT_SECONDS
            )
            if retirement_record is None or retirement_record.retirement_status not in {
                RetirementStatus.COMPLETED, RetirementStatus.NOT_REQUIRED,
            }:
                raise RuntimeError(f"ablation synchronous retirement failed: {retirement_record!r}")
        gate_opened_at_ns = open_gate()
        assert gate_closed_at_ns is not None

        new_version = committed.candidate_version
        wait_until(lambda: has_completed_plan(new_version), "first effect")
        with log_lock:
            target_completed = len(frame_log) + 5
        wait_until(lambda: completed_count() >= target_completed, "target frames")

        if retirement_record is None:
            retirement_record = controller.wait_for_retirement(
                request.request_id, _OPERATION_TIMEOUT_SECONDS
            )
            if retirement_record is None or retirement_record.retirement_status not in {
                RetirementStatus.COMPLETED, RetirementStatus.NOT_REQUIRED,
            }:
                raise RuntimeError(f"ablation deferred retirement failed: {retirement_record!r}")

        stop_producer.set()
        producer.join(_OPERATION_TIMEOUT_SECONDS)
        if producer.is_alive():
            raise TimeoutError("ablation producer did not stop")
        stop_worker.set()
        with gate:
            gate.notify_all()
        worker.join(_OPERATION_TIMEOUT_SECONDS)
        if worker.is_alive():
            raise TimeoutError("ablation worker did not drain")
        check_errors()
        with log_lock:
            entries = tuple(frame_log)
            if offered_frames != len(entries) + overflow_frames:
                raise RuntimeError("ablation offered-frame accounting has a residual")

        first_new = min(
            (entry for entry in entries if entry.plan_version == new_version),
            key=lambda entry: entry.completion_ns,
        )
        last_old = max(
            (entry for entry in entries if entry.plan_version == initial.plan.version),
            key=lambda entry: entry.completion_ns,
            default=None,
        )
        if last_old is None or first_new.completion_ns <= last_old.completion_ns:
            raise RuntimeError("ablation transition output ordering is invalid")

        record = controller.record(request.request_id)
        required_times = (
            record.validation_started_ns, record.validation_completed_ns,
            record.preparation_started_ns, record.preparation_completed_ns,
            record.commit_started_ns, record.commit_ns,
        )
        if any(value is None for value in required_times):
            raise RuntimeError("ablation phase timestamps are incomplete")
        assert record.validation_started_ns is not None
        assert record.validation_completed_ns is not None
        assert record.preparation_started_ns is not None
        assert record.preparation_completed_ns is not None
        assert record.commit_started_ns is not None
        assert record.commit_ns is not None
        validation_ns = record.validation_completed_ns - record.validation_started_ns
        preparation_ns = record.preparation_completed_ns - record.preparation_started_ns
        publication_ns = record.commit_ns - record.commit_started_ns
        retirement_ns = (
            retirement_record.retirement_completed_ns - retirement_record.retirement_started_ns
            if retirement_record.retirement_completed_ns is not None
            and retirement_record.retirement_started_ns is not None else 0
        )
        synchronous_preparation_ns = preparation_ns if preparation_placement == "synchronous" else None
        offpath_preparation_ns = (
            record.preparation_completed_ns - record.validation_started_ns
            if preparation_placement == "off_path" else None
        )
        synchronous_retirement_ns = retirement_ns if retirement_policy == "synchronous" else None
        deferred_retirement_ns = retirement_ns if retirement_policy == "deferred" else None
        # The gate interval starts before in-flight frames drain, so the
        # boundary wait is part of the synchronized work being decomposed.
        expected_sync_ns = (
            boundary_wait_ns
            + (validation_ns + preparation_ns if preparation_placement == "synchronous" else 0)
            + publication_ns
            + (retirement_ns if retirement_policy == "synchronous" else 0)
        )
        total_sync_ns = gate_opened_at_ns - gate_closed_at_ns
        residual_ns = total_sync_ns - expected_sync_ns
        row = AblationSampleRow(
            run_id=run_id,
            scenario_id=scenario_id,
            block_id=block_id,
            block_seed=block_seed,
            variant=variant,
            variant_name=variant_name,
            preparation_placement=preparation_placement,
            retirement_policy=retirement_policy,
            edit_type=edit_type,
            repetition=repetition,
            variant_order_position=variant_order_position,
            candidate_graph_fingerprint=f"fingerprint_{specification_hash(target_spec)[:12]}",
            preparation_workload_id=f"workload_{edit_type}",
            old_plan_version=initial.plan.version,
            new_plan_version=new_version,
            validation_ns=validation_ns if preparation_placement == "synchronous" else None,
            synchronous_preparation_ns=synchronous_preparation_ns,
            offpath_preparation_ns=offpath_preparation_ns,
            boundary_wait_ns=boundary_wait_ns,
            publication_ns=publication_ns,
            synchronous_retirement_ns=synchronous_retirement_ns,
            deferred_retirement_ns=deferred_retirement_ns,
            first_effect_wait_ns=first_new.completion_ns - record.commit_ns,
            request_to_effect_ns=first_new.completion_ns - request_at_ns,
            transition_output_gap_ns=first_new.completion_ns - last_old.completion_ns,
            total_synchronous_ns=total_sync_ns,
            synchronous_accounting_residual_ns=residual_ns,
            synchronous_accounting_valid=residual_ns >= 0,
            synchronous_accounting_invalid_reason=(
                None if residual_ns >= 0 else "phase durations exceed the closed-gate interval"
            ),
            retirement_duration_ns=retirement_ns,
            old_plan_frames_admitted_after_request_before_commit=(
                record.old_plan_frames_admitted_after_request_before_commit
            ),
            peak_live_processors=len(initial.plan.steps) + record.staged_processor_count,
            random_seed=random_seed,
            request_phase_offset_ns=request_phase_offset_ns,
        )
        validate_ablation_sample(row)
        return row
    finally:
        stop_producer.set()
        stop_worker.set()
        with gate:
            admission_closed = False
            gate.notify_all()
        producer.join(_OPERATION_TIMEOUT_SECONDS)
        worker.join(_OPERATION_TIMEOUT_SECONDS)
        if worker.is_alive() or producer.is_alive():
            raise TimeoutError("ablation thread did not stop during cleanup")
        controller.close()
        shutdown_report = shutdown_plan_processors(executor.active_plan)
        if not shutdown_report.succeeded:
            raise RuntimeError("ablation active-plan shutdown failed")


def run_ablation_suite(
    output_dir: str,
    run_id: str,
    repetitions: int = 5,
    random_seed: int | None = None,
) -> tuple[AblationSampleRow, ...]:
    """Run four variants in stable, position-balanced blocks per edit."""
    csv_path = os.path.join(output_dir, "ablation-samples.csv")
    write_ablation_header(csv_path)
    base_seed = random_seed if random_seed is not None else 42
    rows: list[AblationSampleRow] = []
    for edit_type in ("remove_stateless_node", "compatible_edit_preserving_tracker"):
        scenario_id = f"ablation_{edit_type}"
        for repetition in range(1, repetitions + 1):
            block_id = f"{scenario_id}_block_{repetition}"
            block_seed = int.from_bytes(
                hashlib.sha256(f"{base_seed}:{block_id}".encode()).digest()[:4], "big"
            )
            block_rng = random.Random(block_seed)
            request_phase_offset_ns = block_rng.randint(0, 500_000)
            order = balanced_order(tuple(_VARIANTS), base_seed, scenario_id, repetition)
            for position, variant in enumerate(order, 1):
                row = _run_ablation_repetition(
                    run_id, scenario_id, block_id, block_seed, variant, position,
                    edit_type, repetition, request_phase_offset_ns,
                    random_seed=base_seed,
                )
                append_ablation_rows(csv_path, [row])
                rows.append(row)
    return tuple(rows)
