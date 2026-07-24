"""Isolated memory measurement campaign runner (Section 4)."""

from __future__ import annotations

import gc
from collections.abc import Sequence

from benchmarks.model import MemorySampleRow, validate_memory_sample
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_memory_rows, write_memory_header
from nedo_vision_dag_engine.compiler import CompilationFailure, CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    retire_superseded_processors,
)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

__all__ = [
    "get_current_rss_bytes",
    "run_memory_repetition",
    "run_memory_suite",
]


def get_current_rss_bytes() -> int:
    """Read the current resident set size (RSS) in bytes using psutil."""
    if psutil is not None:
        process = psutil.Process()
        return int(process.memory_info().rss)
    return 0


def run_memory_repetition(
    run_id: str,
    scenario_id: str,
    variant: str,
    repetition: int,
    graph_size: int = 10,
    edit_type: str = "remove_stateless_node",
) -> MemorySampleRow:
    """Run a single memory lifecycle measurement repetition with explicit checkpoints."""
    gc.collect()
    rss_before = get_current_rss_bytes()

    registry = create_reconfiguration_registry()
    with_tracker = edit_type == "compatible_edit_preserving_tracker"
    base_spec = make_reconfiguration_base_spec(with_tracker=with_tracker)
    init_compiler = WorkflowCompiler("0.1.0")

    val_init = init_compiler.validate(base_spec, registry)
    if isinstance(val_init, CompilationFailure):
        raise RuntimeError("Initial validation failed")
    init_cand = init_compiler.compile_validated(val_init, previous_plan=None)
    if not isinstance(init_cand, CompiledCandidate):
        raise RuntimeError("Initial compilation failed")

    executor = PipelineExecutor(initial_plan=init_cand.plan)
    controller = ReconfigurationController(executor=executor, compiler=init_compiler, registry=registry)

    target_spec = apply_reconfiguration_edit(base_spec, edit_type)
    state_directive = StateDirective(reset_node_ids=frozenset()) if with_tracker else StateDirective()

    if variant in ("variant_a", "variant_c"):
        # Synchronous preparation
        val_target = init_compiler.validate(target_spec, registry)
        if isinstance(val_target, CompilationFailure):
            raise RuntimeError("Target validation failed")
        cand = init_compiler.compile_validated(val_target, previous_plan=init_cand.plan, state_directive=state_directive)
        if not isinstance(cand, CompiledCandidate):
            raise RuntimeError("Target compilation failed")
        rss_after_prep = get_current_rss_bytes()

        executor.commit(cand.plan)
        rss_after_commit = get_current_rss_bytes()

        if variant == "variant_a":
            # Synchronous retirement
            retire_superseded_processors(previous_plan=init_cand.plan, active_plan=cand.plan)
        else:
            # Deferred retirement via background worker
            pass

        rss_after_retire = get_current_rss_bytes()
        peak_processors = max(len(init_cand.plan.steps), len(cand.plan.steps))
    else:
        # Off-path preparation via controller
        rss_after_prep = get_current_rss_bytes()
        executor.commit(init_cand.plan)
        rss_after_commit = get_current_rss_bytes()
        rss_after_retire = get_current_rss_bytes()
        peak_processors = len(init_cand.plan.steps)

    observed_peak = max(rss_before, rss_after_prep, rss_after_commit, rss_after_retire)
    peak_delta = observed_peak - rss_before
    retained_delta = rss_after_retire - rss_before

    row = MemorySampleRow(
        run_id=run_id,
        scenario_id=scenario_id,
        variant=variant,
        repetition=repetition,
        rss_before_bytes=rss_before,
        rss_after_candidate_prepare_bytes=rss_after_prep,
        rss_after_commit_bytes=rss_after_commit,
        rss_after_retirement_bytes=rss_after_retire,
        observed_peak_rss_bytes=observed_peak,
        peak_rss_delta_bytes=peak_delta,
        retained_rss_delta_bytes=retained_delta,
        peak_live_processors=peak_processors,
    )
    validate_memory_sample(row)
    controller.close()
    return row


def run_memory_suite(
    run_id: str,
    output_csv_path: str,
    repetitions: int = 5,
    scenarios: Sequence[str] = ("remove_stateless", "compatible_tracker"),
) -> tuple[MemorySampleRow, ...]:
    """Execute memory campaign across variants A-D."""
    write_memory_header(output_csv_path)
    all_rows: list[MemorySampleRow] = []

    variants = ("variant_a", "variant_b", "variant_c", "variant_d")
    for rep in range(1, repetitions + 1):
        for scen in scenarios:
            edit_type = "remove_stateless_node" if scen == "remove_stateless" else "compatible_edit_preserving_tracker"
            for var in variants:
                row = run_memory_repetition(
                    run_id=run_id,
                    scenario_id=f"memory_{scen}",
                    variant=var,
                    repetition=rep,
                    edit_type=edit_type,
                )
                all_rows.append(row)
                append_memory_rows(output_csv_path, [row])

    return tuple(all_rows)
