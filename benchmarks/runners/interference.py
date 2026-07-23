"""Active-path interference instrumentation runner for RQ2/RQ3 evaluation.

Measures frame-service metrics across three distinct phases of off-path candidate preparation:
- Window 1: Before preparation (steady state)
- Window 2: During candidate preparation (off-path compilation worker running)
- Window 3: After publication (steady state under new plan)

Supports 4 preparation categories:
- no_candidate_preparation
- non_contentious_wait
- cpu_bound_candidate_preparation
- real_model_preparation_callback
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, Thread

import numpy as np

from benchmarks.model import InterferenceSampleRow
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    do_cpu_work,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_interference_rows, write_interference_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
    StateDirective,
)

__all__ = [
    "InterferenceMarkers",
    "run_interference_suite",
]


@dataclass(frozen=True, slots=True)
class InterferenceMarkers:
    preparation_start_ns: int
    preparation_ready_ns: int
    publication_start_ns: int
    publication_end_ns: int
    retirement_complete_ns: int


@dataclass(frozen=True, slots=True)
class FrameMeasurement:
    frame_id: int
    admission_ns: int
    completion_ns: int
    latency_ns: int
    phase: str


def run_interference_suite(
    run_id: str,
    output_csv_path: str,
    repetition_count: int = 5,
    custom_callback: Callable[[], None] | None = None,
) -> tuple[InterferenceSampleRow, ...]:
    write_interference_header(output_csv_path)
    categories = (
        "no_candidate_preparation",
        "non_contentious_wait",
        "cpu_bound_candidate_preparation",
        "real_model_preparation_callback",
    )
    all_rows: list[InterferenceSampleRow] = []

    base_spec = make_reconfiguration_base_spec(with_tracker=False)
    target_spec = apply_reconfiguration_edit(base_spec, "insert_stateless_node")
    registry = create_reconfiguration_registry()

    for category in categories:
        scenario_id = f"interference_{category}"

        for rep in range(1, repetition_count + 1):
            gc.collect()
            init_compiler = WorkflowCompiler("0.1.0")
            init_cand = init_compiler.compile(base_spec, registry)
            if not isinstance(init_cand, CompiledCandidate):
                raise RuntimeError("Initial compilation failed.")

            executor = PipelineExecutor(initial_plan=init_cand.plan)
            controller = ReconfigurationController(executor=executor, compiler=init_compiler, registry=registry)

            frame_measurements: list[FrameMeasurement] = []
            meas_lock = Lock()
            stop_worker = [False]

            def worker_loop() -> None:
                fid = 0
                while not stop_worker[0]:
                    t_adm = time.perf_counter_ns()
                    _ = controller.admit_frame(admitted_at_ns=t_adm, frame_id=fid)
                    t_comp = time.perf_counter_ns()

                    try:
                        rec = controller.record("reconfig_1")
                    except KeyError:
                        rec = None

                    if rec is None or rec.preparation_started_ns is None:
                        phase = "before"
                    elif rec.ready_ns is None:
                        phase = "during"
                    else:
                        phase = "after"

                    with meas_lock:
                        frame_measurements.append(
                            FrameMeasurement(
                                frame_id=fid,
                                admission_ns=t_adm,
                                completion_ns=t_comp,
                                latency_ns=t_comp - t_adm,
                                phase=phase,
                            )
                        )
                    fid += 1
                    time.sleep(0.001)

            t_worker = Thread(target=worker_loop, daemon=True)
            t_worker.start()

            # Warmup: wait for 10 frames
            while True:
                with meas_lock:
                    if len(frame_measurements) >= 10:
                        break
                time.sleep(0.001)

            # Window 1: Before preparation
            t_before_start = time.perf_counter_ns()
            time.sleep(0.05)
            t_before_end = time.perf_counter_ns()

            # Prepare request with payload delay corresponding to category
            t_prep_start = time.perf_counter_ns()
            if category == "non_contentious_wait":
                time.sleep(0.02)
            elif category == "cpu_bound_candidate_preparation":
                do_cpu_work(1_000_000)
            elif category == "real_model_preparation_callback" and custom_callback is not None:
                custom_callback()

            req = ReconfigurationRequest(
                request_id="reconfig_1",
                base_version=1,
                target_specification=target_spec,
                state_directive=StateDirective(),
                submitted_at_ns=t_prep_start,
            )
            controller.submit(req)

            # Window 2: During candidate preparation
            t_during_start = time.perf_counter_ns()
            ready_rec = controller.wait_for_status("reconfig_1", frozenset({ReconfigurationStatus.READY}), timeout_seconds=5.0)
            assert ready_rec is not None
            t_during_end = time.perf_counter_ns()

            _ = controller.wait_for_effect("reconfig_1", timeout_seconds=5.0)
            _ = controller.wait_for_retirement("reconfig_1", timeout_seconds=5.0)

            # Window 3: After publication
            t_after_start = time.perf_counter_ns()
            time.sleep(0.05)
            t_after_end = time.perf_counter_ns()

            stop_worker[0] = True
            t_worker.join()
            controller.close()

            with meas_lock:
                before_latencies = [m.latency_ns for m in frame_measurements if m.phase == "before"]
                during_latencies = [m.latency_ns for m in frame_measurements if m.phase == "during"]
                after_latencies = [m.latency_ns for m in frame_measurements if m.phase == "after"]

            bef_med = float(np.median(before_latencies)) if before_latencies else 0.0
            dur_med = float(np.median(during_latencies)) if during_latencies else 0.0
            dur_p95 = float(np.percentile(during_latencies, 95)) if during_latencies else 0.0
            aft_med = float(np.median(after_latencies)) if after_latencies else 0.0

            dur_frames_cnt = len(during_latencies)
            bef_fps = (len(before_latencies) / ((t_before_end - t_before_start) / 1e9)) if t_before_end > t_before_start else 0.0
            dur_fps = (dur_frames_cnt / ((t_during_end - t_during_start) / 1e9)) if t_during_end > t_during_start else 0.0
            aft_fps = (len(after_latencies) / ((t_after_end - t_after_start) / 1e9)) if t_after_end > t_after_start else 0.0

            row = InterferenceSampleRow(
                run_id=run_id,
                scenario_id=scenario_id,
                preparation_category=category,
                repetition=rep,
                frame_latency_before_median_ns=bef_med,
                frame_latency_during_median_ns=dur_med,
                frame_latency_during_p95_ns=dur_p95,
                frame_latency_after_median_ns=aft_med,
                frames_completed_during_preparation=dur_frames_cnt,
                throughput_before_fps=bef_fps,
                throughput_during_fps=dur_fps,
                throughput_after_fps=aft_fps,
            )
            all_rows.append(row)
            append_interference_rows(output_csv_path, [row])

    return tuple(all_rows)
