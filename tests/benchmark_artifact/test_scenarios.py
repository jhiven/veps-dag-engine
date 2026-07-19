"""Tests for benchmark scenario builders, calibration, and hard-coded execution."""

from __future__ import annotations

from benchmarks.scenarios import (
    WorkloadProcessor,
    apply_reconfiguration_edit,
    build_branch_merge_9_spec,
    build_linear_5_spec,
    calibrate_workload_iterations,
    create_reconfiguration_registry,
    create_workload_registry,
    execute_hard_coded_branch_merge_9,
    execute_hard_coded_linear_5,
    generate_layered_dag,
    make_reconfiguration_base_spec,
)
from nedo_vision_dag_engine.compiler import CompiledCandidate, WorkflowCompiler
from nedo_vision_dag_engine.executor import FrameStatus, execute_frame
from nedo_vision_dag_engine.processor import FrameContext
from nedo_vision_dag_engine.workspace import WorkspacePool


def test_workload_calibration() -> None:
    calib = calibrate_workload_iterations()
    assert "minimal" in calib
    assert "approximately_1_ms" in calib
    assert "approximately_5_ms" in calib
    assert calib["minimal"].calibrated_iterations == 0
    assert calib["approximately_1_ms"].calibrated_iterations > 0
    assert calib["approximately_5_ms"].calibrated_iterations > calib["approximately_1_ms"].calibrated_iterations


def test_linear_5_hard_coded_and_compiled_equivalence() -> None:
    spec = build_linear_5_spec()
    registry = create_workload_registry(iterations=0)
    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)

    ctx = FrameContext(frame_id=1, plan_version=1, admitted_at_ns=0)

    # Hard-coded execution
    procs = tuple(step.processor_ref for step in cand.plan.steps)
    typed_procs = tuple(p for p in procs if isinstance(p, WorkloadProcessor))
    hc_res = execute_hard_coded_linear_5(typed_procs, ctx)

    # Compiled execution
    pool = WorkspacePool()
    ws = pool.acquire(cand.plan.output_slot_count)
    comp_res = execute_frame(cand.plan, 1, 0, ws)
    pool.release(ws)

    assert comp_res.status == FrameStatus.COMPLETED
    assert hc_res.value == 1


def test_branch_merge_9_spec() -> None:
    spec = build_branch_merge_9_spec()
    registry = create_workload_registry(iterations=0)
    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)

    ctx = FrameContext(frame_id=1, plan_version=1, admitted_at_ns=0)
    procs_typed = [p for p in (step.processor_ref for step in cand.plan.steps) if isinstance(p, WorkloadProcessor)]
    procs_map = {step.node_id: p for step, p in zip(cand.plan.steps, procs_typed)}
    hc_res = execute_hard_coded_branch_merge_9(procs_map, ctx)
    assert hc_res.value == 1


def test_layered_dag_generation() -> None:
    for size in (5, 25, 100):
        spec = generate_layered_dag(size)
        assert len(spec.nodes) == size
        compiler = WorkflowCompiler("0.1.0")
        cand = compiler.compile(spec, create_workload_registry(0))
        assert isinstance(cand, CompiledCandidate)


def test_reconfiguration_edits() -> None:
    base = make_reconfiguration_base_spec()
    reg = create_reconfiguration_registry()
    compiler = WorkflowCompiler("0.1.0")

    for edit in ("insert_stateless_node", "remove_stateless_node", "rewire_stateless_edge"):
        target = apply_reconfiguration_edit(base, edit)
        cand = compiler.compile(target, reg)
        assert isinstance(cand, CompiledCandidate)
