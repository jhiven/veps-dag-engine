"""RQ1 steady-state execution overhead runner."""

from __future__ import annotations

import gc
import time

from benchmarks.model import SteadyStateSampleRow
from benchmarks.scenarios import (
    WorkloadProcessor,
    build_branch_merge_9_spec,
    build_linear_5_spec,
    create_workload_registry,
    execute_hard_coded_branch_merge_9,
    execute_hard_coded_linear_5,
)
from benchmarks.storage import append_steady_state_rows, write_steady_state_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor, execute_frame
from nedo_vision_dag_engine.processor import FrameContext
from nedo_vision_dag_engine.workspace import WorkspacePool

__all__ = ["run_steady_state_suite"]


def _get_execution_order(repetition: int) -> tuple[str, str, str]:
    orders = (
        ("hard_coded", "static_compiled", "versioned_compiled"),
        ("versioned_compiled", "static_compiled", "hard_coded"),
        ("static_compiled", "hard_coded", "versioned_compiled"),
    )
    return orders[(repetition - 1) % len(orders)]


def run_steady_state_suite(
    run_id: str,
    output_csv_path: str,
    profile: str,
    repetition_count: int,
    calibrated_iterations: dict[str, int],
    seed: int = 42,
) -> tuple[SteadyStateSampleRow, ...]:
    write_steady_state_header(output_csv_path)

    topologies: tuple[str, ...] = ("linear_5", "branch_merge_9")
    workload_ids: tuple[str, ...] = ("minimal", "approximately_1_ms", "approximately_5_ms")

    all_rows: list[SteadyStateSampleRow] = []

    for topology in topologies:
        spec_builder = build_linear_5_spec if topology == "linear_5" else build_branch_merge_9_spec
        spec = spec_builder()

        for workload_id in workload_ids:
            iters = calibrated_iterations[workload_id]

            if profile == "smoke":
                ops_per_rep = 20 if workload_id == "approximately_5_ms" else 50
                warmup_ops = 5
            else:
                if workload_id == "approximately_5_ms":
                    ops_per_rep = 100
                elif workload_id == "approximately_1_ms":
                    ops_per_rep = 500
                else:
                    ops_per_rep = 2000
                warmup_ops = 50

            scenario_id = f"steady_state_{topology}_{workload_id}"

            for rep in range(1, repetition_count + 1):
                order = _get_execution_order(rep)

                registry = create_workload_registry(iters)
                compiler = WorkflowCompiler("0.1.0")
                candidate = compiler.compile(spec, registry)
                if not isinstance(candidate, CompiledCandidate):
                    raise RuntimeError(f"Failed to compile spec for {scenario_id}")

                plan = candidate.plan
                workspace_pool = WorkspacePool()

                for pos, impl in enumerate(order, start=1):
                    gc.collect()

                    if impl == "hard_coded":
                        processors = [step.processor_ref for step in plan.steps]
                        procs_typed = [p for p in processors if isinstance(p, WorkloadProcessor)]

                        for f in range(warmup_ops):
                            ctx = FrameContext(frame_id=f, plan_version=1, admitted_at_ns=0)
                            if topology == "linear_5":
                                execute_hard_coded_linear_5(tuple(procs_typed), ctx)
                            else:
                                procs_map = {step.node_id: p for step, p in zip(plan.steps, procs_typed)}
                                execute_hard_coded_branch_merge_9(procs_map, ctx)

                        t0 = time.perf_counter_ns()
                        for f in range(ops_per_rep):
                            ctx = FrameContext(frame_id=f, plan_version=1, admitted_at_ns=0)
                            if topology == "linear_5":
                                execute_hard_coded_linear_5(tuple(procs_typed), ctx)
                            else:
                                procs_map = {step.node_id: p for step, p in zip(plan.steps, procs_typed)}
                                execute_hard_coded_branch_merge_9(procs_map, ctx)
                        t1 = time.perf_counter_ns()

                    elif impl == "static_compiled":
                        ws = workspace_pool.acquire(plan.output_slot_count)
                        try:
                            for f in range(warmup_ops):
                                execute_frame(plan, f, 0, ws)

                            t0 = time.perf_counter_ns()
                            for f in range(ops_per_rep):
                                execute_frame(plan, f, 0, ws)
                            t1 = time.perf_counter_ns()
                        finally:
                            workspace_pool.release(ws)

                    elif impl == "versioned_compiled":
                        executor = PipelineExecutor(initial_plan=plan, workspace_pool=workspace_pool)

                        for f in range(warmup_ops):
                            executor.admit_frame(admitted_at_ns=0, frame_id=f)

                        t0 = time.perf_counter_ns()
                        for f in range(ops_per_rep):
                            executor.admit_frame(admitted_at_ns=0, frame_id=warmup_ops + f)
                        t1 = time.perf_counter_ns()

                    else:
                        raise ValueError(f"Unknown implementation {impl!r}")

                    elapsed_ns = t1 - t0
                    ns_per_frame = elapsed_ns / ops_per_rep if ops_per_rep > 0 else 0.0

                    row = SteadyStateSampleRow(
                        run_id=run_id,
                        scenario_id=scenario_id,
                        topology=topology,
                        workload_id=workload_id,
                        implementation=impl,
                        repetition=rep,
                        execution_order_position=pos,
                        operations=ops_per_rep,
                        elapsed_ns=elapsed_ns,
                        normalized_ns_per_frame=ns_per_frame,
                        frames_completed=ops_per_rep,
                        plan_version=1,
                    )
                    all_rows.append(row)

                append_steady_state_rows(output_csv_path, all_rows[-3:])

    return tuple(all_rows)
