"""Active-path interference benchmark runner (Redesigned Sections 6, 7, 8, 9, 10).

Evaluates active-plan frame latency and throughput degradation under background candidate preparation:
- Absolute-deadline producer scheduler with overdue deadline skipping.
- Explicit phase window timestamp classification for admission vs completion throughput.
- Confounding flags for producer-inhibited latency measurements.
- Raw per-frame sampling stored to interference-frame-samples.csv.
- Repetition-level summaries stored to interference-samples.csv.
"""

from __future__ import annotations

import gc
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Lock, Thread

import numpy as np

from benchmarks.model import (
    InterferenceFrameSampleRow,
    InterferenceSampleRow,
)
from benchmarks.scenarios import (
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import (
    append_interference_frame_rows,
    append_interference_rows,
    write_interference_frame_header,
    write_interference_header,
)
from nedo_vision_dag_engine.compiler import (
    CompilationFailure,
    CompiledCandidate,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import ReconfigurationController

__all__ = [
    "run_cpu_bound_workload",
    "run_interference_repetition",
    "run_interference_suite",
]


@dataclass(frozen=True, slots=True)
class RawFrameRecord:
    frame_id: int
    plan_version: int
    admission_ns: int
    completion_ns: int
    latency_ns: int
    queue_occupancy: float
    dropped: bool
    duplicated: bool


def run_cpu_bound_workload(target_duration_ns: int, initial_checksum: int = 42) -> tuple[int, int]:
    """Execute deterministic CPU-bound work until target duration deadline is reached."""
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + target_duration_ns
    checksum = initial_checksum

    while time.perf_counter_ns() < deadline_ns:
        checksum = (checksum * 31 + 17) & 0xFFFFFFFF

    end_ns = time.perf_counter_ns()
    return end_ns - start_ns, checksum


def run_interference_repetition(
    run_id: str,
    scenario_id: str,
    preparation_category: str,
    target_window_ns: int,
    repetition: int,
    execution_order_position: int = 1,
    random_seed: int = 42,
    offered_rate_fps: float = 1000.0,
    before_duration_s: float = 1.0,
    after_duration_s: float = 1.0,
    minimum_during_frames_for_p95: int = 100,
    frame_csv_path: str | None = None,
) -> InterferenceSampleRow:
    """Execute a single interference repetition using absolute-deadline producer scheduler."""
    gc.collect()

    base_spec = make_reconfiguration_base_spec(with_tracker=False)
    registry = create_reconfiguration_registry()
    init_compiler = WorkflowCompiler("0.1.0")

    val_init = init_compiler.validate(base_spec, registry)
    if isinstance(val_init, CompilationFailure):
        raise RuntimeError(f"Base validation failed: {val_init.reason}")
    init_cand = init_compiler.compile_validated(val_init, previous_plan=None)
    if not isinstance(init_cand, CompiledCandidate):
        raise RuntimeError("Initial compilation failed")

    executor = PipelineExecutor(initial_plan=init_cand.plan)
    controller = ReconfigurationController(executor=executor, compiler=init_compiler, registry=registry)

    frame_queue: Queue[tuple[int, int]] = Queue(maxsize=100)
    all_raw_records: list[RawFrameRecord] = []
    records_lock = Lock()

    stop_producer = Event()
    stop_worker = Event()
    dropped_frames_count = 0

    interval_ns = int(round(1e9 / offered_rate_fps))

    # Phase-level producer tracking
    current_phase_name = ["warmup"]

    phase_producer_attempted = {"before": 0, "during": 0, "after": 0}
    phase_producer_emitted = {"before": 0, "during": 0, "after": 0}
    phase_producer_skipped = {"before": 0, "during": 0, "after": 0}
    phase_producer_lateness = {"before": list[float](), "during": list[float](), "after": list[float]()}

    def producer_loop() -> None:
        nonlocal dropped_frames_count
        fid = 0
        start_ns = time.perf_counter_ns()
        next_deadline_ns = start_ns

        while not stop_producer.is_set():
            now_ns = time.perf_counter_ns()
            if now_ns < next_deadline_ns:
                sleep_s = (next_deadline_ns - now_ns) / 1e9
                if sleep_s > 0.0002:
                    time.sleep(sleep_s - 0.0001)
                while time.perf_counter_ns() < next_deadline_ns:
                    pass

            emission_now_ns = time.perf_counter_ns()
            lateness_ns = emission_now_ns - next_deadline_ns

            phase = current_phase_name[0]
            if phase in phase_producer_attempted:
                phase_producer_attempted[phase] += 1
                phase_producer_lateness[phase].append(float(lateness_ns))

            # If overdue by more than one full interval, skip overdue emission
            if lateness_ns > interval_ns:
                skipped = int(lateness_ns // interval_ns)
                if phase in phase_producer_skipped:
                    phase_producer_skipped[phase] += skipped
                next_deadline_ns += skipped * interval_ns

            arr_ns = time.perf_counter_ns()
            try:
                frame_queue.put_nowait((fid, arr_ns))
                if phase in phase_producer_emitted:
                    phase_producer_emitted[phase] += 1
            except Full:
                try:
                    frame_queue.get_nowait()
                    dropped_frames_count += 1
                    frame_queue.put_nowait((fid, arr_ns))
                    if phase in phase_producer_emitted:
                        phase_producer_emitted[phase] += 1
                except Exception:
                    pass

            fid += 1
            next_deadline_ns += interval_ns

    t_producer = Thread(target=producer_loop, daemon=True)
    t_producer.start()

    def worker_loop() -> None:
        while not stop_worker.is_set():
            try:
                fid, _ = frame_queue.get(timeout=0.005)
            except Exception:
                continue

            t_adm = time.perf_counter_ns()
            res = controller.admit_frame(admitted_at_ns=t_adm, frame_id=fid)
            t_comp = time.perf_counter_ns()
            lat_ns = t_comp - t_adm
            occ = frame_queue.qsize() / 100.0

            rec = RawFrameRecord(
                frame_id=fid,
                plan_version=res.plan_version,
                admission_ns=t_adm,
                completion_ns=t_comp,
                latency_ns=lat_ns,
                queue_occupancy=occ,
                dropped=False,
                duplicated=False,
            )

            with records_lock:
                all_raw_records.append(rec)

    t_worker = Thread(target=worker_loop, daemon=True)
    t_worker.start()

    # 1. Warm-up
    time.sleep(0.5)

    # 2. Window 1: Before measurement (fixed wall-clock duration)
    with records_lock:
        current_phase_name[0] = "before"
    t_before_start = time.perf_counter_ns()
    time.sleep(before_duration_s)
    t_before_end = time.perf_counter_ns()
    measured_before_ns = t_before_end - t_before_start

    # 3. Window 2: During candidate preparation / control window
    target_duration_s = target_window_ns / 1e9
    actual_prep_ns = 0
    prep_checksum: int | None = None
    active_prep_end_ns: int | None = None

    with records_lock:
        current_phase_name[0] = "during"
    t_during_start = time.perf_counter_ns()

    if preparation_category == "no_candidate_preparation":
        actual_prep_ns = 0
        active_prep_end_ns = None
        time.sleep(target_duration_s)
        t_during_end = time.perf_counter_ns()
    elif preparation_category == "sleep_preparation":
        t_s_start = time.perf_counter_ns()
        time.sleep(target_duration_s)
        t_s_end = time.perf_counter_ns()
        actual_prep_ns = t_s_end - t_s_start
        active_prep_end_ns = t_s_end
        t_during_end = t_s_end
    elif preparation_category == "cpu_bound_preparation":
        actual_prep_ns, prep_checksum = run_cpu_bound_workload(target_window_ns)
        t_during_end = time.perf_counter_ns()
        active_prep_end_ns = t_during_end
    else:
        raise ValueError(f"Unknown preparation category: {preparation_category!r}")

    measured_during_ns = t_during_end - t_during_start

    # 4. Window 3: After measurement (fixed wall-clock duration)
    with records_lock:
        current_phase_name[0] = "after"
    t_after_start = time.perf_counter_ns()
    time.sleep(after_duration_s)
    t_after_end = time.perf_counter_ns()
    measured_after_ns = t_after_end - t_after_start

    stop_producer.set()
    stop_worker.set()
    t_producer.join()
    t_worker.join()
    controller.close()

    # SECTION 7: Classify frames using explicit timestamps
    with records_lock:
        raw_recs = list(all_raw_records)

    admitted_before = [r for r in raw_recs if t_before_start <= r.admission_ns < t_before_end]
    admitted_during = [r for r in raw_recs if t_during_start <= r.admission_ns < t_during_end]
    admitted_after = [r for r in raw_recs if t_after_start <= r.admission_ns < t_after_end]

    completed_before = [r for r in raw_recs if t_before_start <= r.completion_ns < t_before_end]
    completed_during = [r for r in raw_recs if t_during_start <= r.completion_ns < t_during_end]
    completed_after = [r for r in raw_recs if t_after_start <= r.completion_ns < t_after_end]

    if active_prep_end_ns is not None:
        completed_while_active = sum(1 for r in raw_recs if t_during_start <= r.completion_ns < active_prep_end_ns)
    else:
        completed_while_active = None

    # Write raw per-frame CSV if path provided
    if frame_csv_path is not None:
        frame_rows: list[InterferenceFrameSampleRow] = []
        for r in raw_recs:
            if t_before_start <= r.admission_ns < t_before_end:
                ph = "before"
            elif t_during_start <= r.admission_ns < t_during_end:
                ph = "during"
            elif t_after_start <= r.admission_ns < t_after_end:
                ph = "after"
            else:
                ph = "warmup"

            frame_rows.append(
                InterferenceFrameSampleRow(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    preparation_category=preparation_category,
                    target_window_ns=target_window_ns,
                    repetition=repetition,
                    phase=ph,
                    frame_id=r.frame_id,
                    plan_version=r.plan_version,
                    admission_timestamp_ns=r.admission_ns,
                    completion_timestamp_ns=r.completion_ns,
                    frame_latency_ns=r.latency_ns,
                    queue_occupancy=r.queue_occupancy,
                    dropped=r.dropped,
                    duplicated=r.duplicated,
                )
            )
        append_interference_frame_rows(frame_csv_path, frame_rows)

    # Compute Latency Metrics (using admitted frames per phase)
    bef_lats = [float(r.latency_ns) for r in admitted_before]
    dur_lats = [float(r.latency_ns) for r in admitted_during]
    aft_lats = [float(r.latency_ns) for r in admitted_after]

    bef_med = float(np.median(bef_lats)) if len(bef_lats) > 0 else None
    bef_p95 = float(np.percentile(bef_lats, 95.0)) if len(bef_lats) > 0 else None
    aft_med = float(np.median(aft_lats)) if len(aft_lats) > 0 else None
    aft_p95 = float(np.percentile(aft_lats, 95.0)) if len(aft_lats) > 0 else None

    # Validity checking
    if preparation_category == "no_candidate_preparation":
        duration_valid = True
        duration_rel_error = 0.0
        target_prep_duration_ns = None
    else:
        duration_valid = actual_prep_ns >= target_window_ns
        duration_rel_error = (actual_prep_ns - target_window_ns) / float(target_window_ns)
        target_prep_duration_ns = target_window_ns

    valid_for_tail = True
    invalid_reason: str | None = None

    if len(dur_lats) < minimum_during_frames_for_p95:
        valid_for_tail = False
        invalid_reason = f"Insufficient during-window frame samples ({len(dur_lats)} < minimum {minimum_during_frames_for_p95})"

    if not duration_valid:
        valid_for_tail = False
        invalid_reason = (
            f"Preparation under-ran target ({actual_prep_ns} ns < {target_window_ns} ns)"
            if invalid_reason is None
            else f"{invalid_reason}; prep under-ran target"
        )

    if valid_for_tail and len(dur_lats) > 0:
        dur_med: float | None = float(np.median(dur_lats))
        dur_p95: float | None = float(np.percentile(dur_lats, 95.0))
    else:
        dur_med = None
        dur_p95 = None

    if valid_for_tail and dur_p95 is not None and bef_p95 is not None and bef_p95 > 0:
        p95_deg: float | None = 100.0 * (dur_p95 / bef_p95 - 1.0)
    else:
        p95_deg = None

    if valid_for_tail and dur_med is not None and bef_med is not None and bef_med > 0:
        med_deg: float | None = 100.0 * (dur_med / bef_med - 1.0)
    else:
        med_deg = None

    # Throughput metrics
    bef_adm_fps = len(admitted_before) / (measured_before_ns / 1e9) if measured_before_ns > 0 else 0.0
    dur_adm_fps = len(admitted_during) / (measured_during_ns / 1e9) if measured_during_ns > 0 else 0.0
    aft_adm_fps = len(admitted_after) / (measured_after_ns / 1e9) if measured_after_ns > 0 else 0.0

    bef_comp_fps = len(completed_before) / (measured_before_ns / 1e9) if measured_before_ns > 0 else 0.0
    dur_comp_fps = len(completed_during) / (measured_during_ns / 1e9) if measured_during_ns > 0 else 0.0
    aft_comp_fps = len(completed_after) / (measured_after_ns / 1e9) if measured_after_ns > 0 else 0.0

    adm_ratio_vs_before = (dur_adm_fps / bef_adm_fps) if bef_adm_fps > 0 else None
    comp_ratio_vs_before = (dur_comp_fps / bef_comp_fps) if bef_comp_fps > 0 else None

    # Confounding check: if admission rate during is materially altered (< 0.95 or > 1.05 vs before)
    is_confounded = False
    confounded_reason: str | None = None
    if adm_ratio_vs_before is not None and (adm_ratio_vs_before < 0.95 or adm_ratio_vs_before > 1.05):
        is_confounded = True
        confounded_reason = f"Admission rate during preparation ({dur_adm_fps:.1f} FPS) deviates from before ({bef_adm_fps:.1f} FPS, ratio {adm_ratio_vs_before:.3f})"

    # Producer schedule lateness statistics
    def _calc_stats(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        return float(np.median(vals)), float(np.percentile(vals, 95.0))

    lat_med_b, lat_p95_b = _calc_stats(phase_producer_lateness["before"])
    lat_med_d, lat_p95_d = _calc_stats(phase_producer_lateness["during"])
    lat_med_a, lat_p95_a = _calc_stats(phase_producer_lateness["after"])

    dur_occs = [r.queue_occupancy for r in admitted_during]
    occ_med = float(np.median(dur_occs)) if dur_occs else 0.0
    occ_p95 = float(np.percentile(dur_occs, 95.0)) if dur_occs else 0.0

    # Invariant checks for bounds & counts
    bounds_valid = (
        t_before_start < t_before_end <= t_during_start < t_during_end <= t_after_start < t_after_end
    )
    raw_matches = (len(admitted_before) + len(admitted_during) + len(admitted_after)) <= len(raw_recs)

    # Invariant: producer emitted frames <= expected offered frames + 2
    exp_before_frames = int(round(offered_rate_fps * (measured_before_ns / 1e9)))
    exp_during_frames = int(round(offered_rate_fps * (measured_during_ns / 1e9)))
    exp_after_frames = int(round(offered_rate_fps * (measured_after_ns / 1e9)))

    tp_valid = (
        phase_producer_emitted["before"] <= exp_before_frames + 2
        and phase_producer_emitted["during"] <= exp_during_frames + 2
        and phase_producer_emitted["after"] <= exp_after_frames + 2
    )
    tp_invalid_reason = None if tp_valid else "Emitted frames exceeded expected offered frames + rounding tolerance"

    # Section 8: Flow accounting & Backlog tracking
    backlog_start = sum(1 for r in raw_recs if r.admission_ns < t_during_start and r.completion_ns >= t_during_start)
    backlog_end = sum(1 for r in raw_recs if r.admission_ns < t_during_end and r.completion_ns >= t_during_end)
    backlog_drained_after = sum(1 for r in raw_recs if r.admission_ns < t_during_end and r.completion_ns >= t_during_end)
    emitted_during_admitted_after = sum(1 for r in raw_recs if t_during_start <= r.admission_ns < t_during_end and r.completion_ns >= t_during_end)

    flow_residual = (backlog_start + phase_producer_emitted["during"]) - (
        len(completed_during) + dropped_frames_count + backlog_end
    )
    flow_valid = abs(flow_residual) <= 1
    flow_invalid_reason = None if flow_valid else f"Flow accounting residual non-zero ({flow_residual} frames)"

    row = InterferenceSampleRow(
        run_id=run_id,
        scenario_id=scenario_id,
        preparation_category=preparation_category,
        target_window_ns=target_window_ns,
        repetition=repetition,
        execution_order_position=execution_order_position,
        random_seed=random_seed,
        configured_offered_rate_fps=offered_rate_fps,
        before_window_start_ns=t_before_start,
        before_window_end_ns=t_before_end,
        during_window_start_ns=t_during_start,
        during_window_end_ns=t_during_end,
        after_window_start_ns=t_after_start,
        after_window_end_ns=t_after_end,
        measured_before_window_duration_ns=measured_before_ns,
        measured_during_window_duration_ns=measured_during_ns,
        measured_after_window_duration_ns=measured_after_ns,
        producer_attempted_frames_before=phase_producer_attempted["before"],
        producer_attempted_frames_during=phase_producer_attempted["during"],
        producer_attempted_frames_after=phase_producer_attempted["after"],
        producer_emitted_frames_before=phase_producer_emitted["before"],
        producer_emitted_frames_during=phase_producer_emitted["during"],
        producer_emitted_frames_after=phase_producer_emitted["after"],
        producer_skipped_deadlines_before=phase_producer_skipped["before"],
        producer_skipped_deadlines_during=phase_producer_skipped["during"],
        producer_skipped_deadlines_after=phase_producer_skipped["after"],
        producer_schedule_lateness_median_ns_before=lat_med_b,
        producer_schedule_lateness_p95_ns_before=lat_p95_b,
        producer_schedule_lateness_median_ns_during=lat_med_d,
        producer_schedule_lateness_p95_ns_during=lat_p95_d,
        producer_schedule_lateness_median_ns_after=lat_med_a,
        producer_schedule_lateness_p95_ns_after=lat_p95_a,
        admitted_frames_before=len(admitted_before),
        admitted_frames_during=len(admitted_during),
        admitted_frames_after=len(admitted_after),
        completed_frames_before=len(completed_before),
        completed_frames_during=len(completed_during),
        completed_frames_after=len(completed_after),
        frame_latency_before_median_ns=bef_med,
        frame_latency_before_p95_ns=bef_p95,
        frame_latency_during_median_ns=dur_med,
        frame_latency_during_p95_ns=dur_p95,
        frame_latency_after_median_ns=aft_med,
        frame_latency_after_p95_ns=aft_p95,
        p95_degradation_vs_before_pct=p95_deg,
        median_degradation_vs_before_pct=med_deg,
        frames_completed_during_window=len(completed_during),
        frames_completed_while_preparation_active=completed_while_active,
        old_plan_frames_completed=len(completed_during),
        admission_throughput_before_fps=bef_adm_fps,
        admission_throughput_during_fps=dur_adm_fps,
        admission_throughput_after_fps=aft_adm_fps,
        completion_throughput_before_fps=bef_comp_fps,
        completion_throughput_during_fps=dur_comp_fps,
        completion_throughput_after_fps=aft_comp_fps,
        admission_rate_ratio_vs_before=adm_ratio_vs_before,
        completion_rate_ratio_vs_before=comp_ratio_vs_before,
        latency_interpretation_confounded=is_confounded,
        latency_interpretation_confounded_reason=confounded_reason,
        within_run_admission_shift_flag=is_confounded,
        within_run_admission_shift_reason=confounded_reason,
        backlog_frames_at_during_start=backlog_start,
        backlog_frames_at_during_end=backlog_end,
        backlog_frames_drained_after_window=backlog_drained_after,
        frames_emitted_during_but_admitted_after=emitted_during_admitted_after,
        frames_admitted_during_but_emitted_before=0,
        flow_accounting_valid=flow_valid,
        flow_accounting_residual_frames=flow_residual,
        flow_accounting_invalid_reason=flow_invalid_reason,
        queue_occupancy_during_median=occ_med,
        queue_occupancy_during_p95=occ_p95,
        dropped_frames=dropped_frames_count,
        duplicated_frames=0,
        valid_for_tail_latency=valid_for_tail,
        invalid_reason=invalid_reason,
        target_preparation_duration_ns=target_prep_duration_ns,
        actual_preparation_duration_ns=actual_prep_ns,
        preparation_checksum=prep_checksum,
        duration_valid=duration_valid,
        duration_relative_error=duration_rel_error,
        raw_frame_count_matches_summary=raw_matches,
        phase_timestamp_bounds_valid=bounds_valid,
        throughput_accounting_valid=tp_valid,
        throughput_accounting_invalid_reason=tp_invalid_reason,
    )

    return row


def run_interference_suite(
    run_id: str,
    output_csv_path: str,
    repetition_count: int = 5,
    target_durations_ns: Sequence[int] = (200000000, 500000000),
    random_seed: int = 42,
) -> tuple[InterferenceSampleRow, ...]:
    """Execute E6 active-path interference benchmark campaign with category order randomization."""
    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    write_interference_header(output_csv_path)
    frame_csv_path = os.path.join(output_dir, "interference-frame-samples.csv")
    write_interference_frame_header(frame_csv_path)

    categories = ("no_candidate_preparation", "sleep_preparation", "cpu_bound_preparation")

    # Construct all treatment combinations
    combos: list[tuple[str, int]] = []
    for target_ns in target_durations_ns:
        for cat in categories:
            combos.append((cat, target_ns))

    rng = random.Random(random_seed)
    all_rows: list[InterferenceSampleRow] = []

    for rep in range(1, repetition_count + 1):
        rep_combos = list(combos)
        rng.shuffle(rep_combos)

        for pos, (cat, target_ns) in enumerate(rep_combos, start=1):
            scenario_id = f"interference_{cat}_{target_ns // 1000000}ms"
            row = run_interference_repetition(
                run_id=run_id,
                scenario_id=scenario_id,
                preparation_category=cat,
                target_window_ns=target_ns,
                repetition=rep,
                execution_order_position=pos,
                random_seed=random_seed,
                frame_csv_path=frame_csv_path,
            )
            all_rows.append(row)
            append_interference_rows(output_csv_path, [row])

    return tuple(all_rows)
