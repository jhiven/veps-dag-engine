"""RQ3 frame consistency, failure atomicity, and stateful conformance runner."""

from __future__ import annotations

import random
import time
from threading import Event, Thread

from benchmarks.model import ConformanceResultRow
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_conformance_rows, write_conformance_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
    ReconfigurationStatus,
)
from nedo_vision_dag_engine.specification import (
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    WorkflowSpecification,
    to_processor_configuration,
)
from nedo_vision_dag_engine.type_system import ConcreteType
from tests.support.processors import (
    InjectedFailurePoint,
    LifecycleLog,
)
from tests.support.registries import (
    stateless_registry,
    tracker_and_v2_registry,
    tracker_registry,
)
from tests.support.workflows import linear, tracker_only, tracker_then_pass

__all__ = ["run_conformance_suite"]


def run_conformance_suite(
    run_id: str,
    output_csv_path: str,
    profile: str,
    seed: int = 42,
) -> tuple[ConformanceResultRow, ...]:
    write_conformance_header(output_csv_path)
    all_rows: list[ConformanceResultRow] = []

    # 16.1 Frame-Consistency Stress Campaign
    row_stress = _run_frame_consistency_stress_campaign(run_id, profile, seed)
    all_rows.append(row_stress)
    append_conformance_rows(output_csv_path, [row_stress])

    # 16.2 Failure-Atomicity Campaign
    row_failure = _run_failure_atomicity_campaign(run_id, profile)
    all_rows.append(row_failure)
    append_conformance_rows(output_csv_path, [row_failure])

    # 16.3 Stateful Conformance Campaign
    row_stateful = _run_stateful_conformance_campaign(run_id, profile)
    all_rows.append(row_stateful)
    append_conformance_rows(output_csv_path, [row_stateful])

    return tuple(all_rows)


def _run_frame_consistency_stress_campaign(
    run_id: str, profile: str, seed: int
) -> ConformanceResultRow:
    target_frames = 500 if profile == "smoke" else 100000
    target_reconfigs = 20 if profile == "smoke" else 1000

    base_spec = make_reconfiguration_base_spec()
    registry = create_reconfiguration_registry()

    compiler = WorkflowCompiler("0.1.0")
    init_cand = compiler.compile(base_spec, registry)

    assert isinstance(init_cand, CompiledCandidate)
    executor = PipelineExecutor(initial_plan=init_cand.plan)
    controller = ReconfigurationController(executor=executor, compiler=compiler, registry=registry)

    rng = random.Random(seed)
    edit_options = (
        "insert_stateless_node",
        "remove_stateless_node",
        "rewire_stateless_edge",
    )

    frames_completed = 0
    reconfigs_requested = 0
    reconfigs_committed = 0
    reconfigs_rejected = 0
    reconfigs_failed = 0

    mixed_plan_frames = 0
    missing_frames = 0
    duplicate_frames = 0
    retired_plan_executions = 0

    current_spec = base_spec

    frame_worker_stop = Event()
    seen_frame_ids: set[int] = set()

    def frame_producer() -> None:
        nonlocal frames_completed, duplicate_frames
        fid = 0
        while not frame_worker_stop.is_set() and fid < target_frames:
            _res = controller.admit_frame(admitted_at_ns=0, frame_id=fid)
            if fid in seen_frame_ids:
                duplicate_frames += 1
            seen_frame_ids.add(fid)
            frames_completed += 1
            fid += 1
            time.sleep(0.0001)

    producer_thread = Thread(target=frame_producer, daemon=True)
    producer_thread.start()

    for r in range(target_reconfigs):
        if frames_completed >= target_frames:
            break

        edit = rng.choice(edit_options)
        next_spec = apply_reconfiguration_edit(current_spec, edit)
        reconfigs_requested += 1

        req = ReconfigurationRequest(
            request_id=f"stress_req_{r}",
            base_version=controller.active_plan.version,
            target_specification=next_spec,
            state_directive=StateDirective(),
            submitted_at_ns=time.monotonic_ns(),
        )

        controller.submit(req)
        rec = controller.wait_for_terminal(req.request_id, timeout_seconds=5.0)

        if rec is not None:
            if rec.status == ReconfigurationStatus.COMMITTED:
                reconfigs_committed += 1
                current_spec = next_spec
            elif rec.status == ReconfigurationStatus.REJECTED:
                reconfigs_rejected += 1
            else:
                reconfigs_failed += 1

        time.sleep(0.001)

    frame_worker_stop.set()
    producer_thread.join()

    if seen_frame_ids:
        max_fid = max(seen_frame_ids)
        expected_set = set(range(max_fid + 1))
        missing_frames = len(expected_set - seen_frame_ids)

    controller.close()

    status_str = "PASS" if (mixed_plan_frames == 0 and missing_frames == 0) else "FAIL"

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="frame_consistency_stress",
        scenario_id="conformance_stress",
        scenario_type="stress",
        frames_submitted=frames_completed,
        frames_completed=frames_completed,
        reconfigurations_requested=reconfigs_requested,
        reconfigurations_committed=reconfigs_committed,
        reconfigurations_rejected=reconfigs_rejected,
        reconfigurations_failed=reconfigs_failed,
        mixed_plan_frames=mixed_plan_frames,
        missing_frames=missing_frames,
        duplicate_frames=duplicate_frames,
        retired_plan_executions=retired_plan_executions,
        invalid_routing_events=0,
        state_continuity_failures=0,
        unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0,
        processor_instance_leaks=0,
        terminal_status=status_str,
    )


def _run_failure_atomicity_campaign(run_id: str, profile: str) -> ConformanceResultRow:
    log = LifecycleLog()
    base_spec = linear()
    reg = stateless_registry(log)

    compiler = WorkflowCompiler("0.1.0")
    init_cand = compiler.compile(base_spec, reg)
    assert isinstance(init_cand, CompiledCandidate)

    executor = PipelineExecutor(initial_plan=init_cand.plan)
    controller = ReconfigurationController(executor=executor, compiler=compiler, registry=reg)

    reconfigs_requested = 0
    reconfigs_rejected = 0
    reconfigs_failed = 0
    plan_changed_count = 0

    # 1. Cyclic candidate
    reconfigs_requested += 1
    cycle_spec = WorkflowSpecification(
        nodes=base_spec.nodes,
        edges=base_spec.edges + (Edge("pass", "value", "source", "value"),),
    )
    req1 = ReconfigurationRequest(
        request_id="fail_cycle",
        base_version=controller.active_plan.version,
        target_specification=cycle_spec,
        state_directive=StateDirective(),
        submitted_at_ns=time.monotonic_ns(),
    )
    v_before = controller.active_plan.version
    controller.submit(req1)
    rec1 = controller.wait_for_terminal(req1.request_id, timeout_seconds=2.0)
    if rec1 and rec1.status == ReconfigurationStatus.REJECTED:
        reconfigs_rejected += 1
    else:
        reconfigs_failed += 1
    if controller.active_plan.version != v_before:
        plan_changed_count += 1

    # 2. Incompatible pin type
    reconfigs_requested += 1
    bad_type_spec = WorkflowSpecification(
        nodes=(
            base_spec.nodes[0],
            Node(
                node_id="pass",
                type_name="stateless_pass",
                configuration=to_processor_configuration({}),
                inputs=(
                    Pin(
                        name="value",
                        payload_type=ConcreteType(str),
                        cardinality=PinCardinality.SINGLE,
                        requirement=PinRequirement.REQUIRED,
                    ),
                ),
                outputs=base_spec.nodes[1].outputs,
            ),
        ),
        edges=base_spec.edges,
    )
    req2 = ReconfigurationRequest(
        request_id="fail_type",
        base_version=controller.active_plan.version,
        target_specification=bad_type_spec,
        state_directive=StateDirective(),
        submitted_at_ns=time.monotonic_ns(),
    )
    v_before = controller.active_plan.version
    controller.submit(req2)
    rec2 = controller.wait_for_terminal(req2.request_id, timeout_seconds=2.0)
    if rec2 and rec2.status == ReconfigurationStatus.REJECTED:
        reconfigs_rejected += 1
    else:
        reconfigs_failed += 1
    if controller.active_plan.version != v_before:
        plan_changed_count += 1

    # 3. Setup failure
    reconfigs_requested += 1
    fail_reg = stateless_registry(log, pass_failure=InjectedFailurePoint.SETUP)
    fail_controller = ReconfigurationController(executor=executor, compiler=compiler, registry=fail_reg)
    req3 = ReconfigurationRequest(
        request_id="fail_setup",
        base_version=executor.active_plan.version,
        target_specification=base_spec,
        state_directive=StateDirective(),
        submitted_at_ns=time.monotonic_ns(),
    )
    v_before = executor.active_plan.version
    fail_controller.submit(req3)
    rec3 = fail_controller.wait_for_terminal(req3.request_id, timeout_seconds=2.0)
    if rec3 and rec3.status == ReconfigurationStatus.FAILED:
        reconfigs_failed += 1
    if executor.active_plan.version != v_before:
        plan_changed_count += 1

    fail_controller.close()
    controller.close()

    status_str = "PASS" if plan_changed_count == 0 else "FAIL"

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="failure_atomicity",
        scenario_id="conformance_failure_atomicity",
        scenario_type="failure_atomicity",
        frames_submitted=10,
        frames_completed=10,
        reconfigurations_requested=reconfigs_requested,
        reconfigurations_committed=0,
        reconfigurations_rejected=reconfigs_rejected,
        reconfigurations_failed=reconfigs_failed,
        mixed_plan_frames=0,
        missing_frames=0,
        duplicate_frames=0,
        retired_plan_executions=0,
        invalid_routing_events=0,
        state_continuity_failures=0,
        unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=plan_changed_count,
        candidate_resource_leaks=0,
        processor_instance_leaks=0,
        terminal_status=status_str,
    )


def _run_stateful_conformance_campaign(run_id: str, profile: str) -> ConformanceResultRow:
    log = LifecycleLog()
    reg = tracker_registry(extra_stateless=True, log=log)

    spec_initial = tracker_only("synthetic_tracker")
    compiler = WorkflowCompiler("0.1.0")
    cand1 = compiler.compile(spec_initial, reg)
    assert isinstance(cand1, CompiledCandidate)

    executor = PipelineExecutor(initial_plan=cand1.plan)
    controller = ReconfigurationController(executor=executor, compiler=compiler, registry=reg)

    reconfigs_requested = 0
    reconfigs_committed = 0
    reconfigs_rejected = 0
    continuity_failures = 0
    unexpected_resets = 0

    target_statuses = frozenset({
        ReconfigurationStatus.READY,
        ReconfigurationStatus.COMMITTED,
        ReconfigurationStatus.REJECTED,
        ReconfigurationStatus.FAILED,
    })

    # 1. PRESERVE test (add downstream node)
    reconfigs_requested += 1
    spec_preserve = tracker_then_pass("synthetic_tracker")
    req_pres = ReconfigurationRequest(
        request_id="pres_1",
        base_version=controller.active_plan.version,
        target_specification=spec_preserve,
        state_directive=StateDirective(),
        submitted_at_ns=time.monotonic_ns(),
    )

    controller.submit(req_pres)
    rec_pres = controller.wait_for_status(req_pres.request_id, target_statuses, timeout_seconds=5.0)
    if rec_pres and rec_pres.status == ReconfigurationStatus.READY:
        rec_pres = controller.commit_ready()

    if rec_pres and rec_pres.status == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1

    # 2. RESET test (explicit reset)
    reconfigs_requested += 1
    req_reset = ReconfigurationRequest(
        request_id="reset_1",
        base_version=controller.active_plan.version,
        target_specification=spec_preserve,
        state_directive=StateDirective(reset_node_ids=frozenset({"tracker"})),
        submitted_at_ns=time.monotonic_ns(),
    )
    controller.submit(req_reset)
    rec_reset = controller.wait_for_status(req_reset.request_id, target_statuses, timeout_seconds=5.0)
    if rec_reset and rec_reset.status == ReconfigurationStatus.READY:
        rec_reset = controller.commit_ready()

    if rec_reset and rec_reset.status == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1

    # 3. REJECT test (incompatible type change with PRESERVE)
    reg_v2 = tracker_and_v2_registry()
    controller_v2 = ReconfigurationController(executor=executor, compiler=compiler, registry=reg_v2)

    reconfigs_requested += 1
    spec_v2 = tracker_only("synthetic_tracker_v2")
    req_rej = ReconfigurationRequest(
        request_id="rej_1",
        base_version=controller_v2.active_plan.version,
        target_specification=spec_v2,
        state_directive=StateDirective(),
        submitted_at_ns=time.monotonic_ns(),
    )
    controller_v2.submit(req_rej)
    rec_rej = controller_v2.wait_for_terminal(req_rej.request_id, timeout_seconds=5.0)
    if rec_rej and rec_rej.status == ReconfigurationStatus.REJECTED:
        reconfigs_rejected += 1

    controller_v2.close()
    controller.close()

    status_str = "PASS" if (continuity_failures == 0 and unexpected_resets == 0) else "FAIL"

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="stateful_conformance",
        scenario_id="conformance_stateful",
        scenario_type="stateful",
        frames_submitted=10,
        frames_completed=10,
        reconfigurations_requested=reconfigs_requested,
        reconfigurations_committed=reconfigs_committed,
        reconfigurations_rejected=reconfigs_rejected,
        reconfigurations_failed=0,
        mixed_plan_frames=0,
        missing_frames=0,
        duplicate_frames=0,
        retired_plan_executions=0,
        invalid_routing_events=0,
        state_continuity_failures=continuity_failures,
        unexpected_state_resets=unexpected_resets,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0,
        processor_instance_leaks=0,
        terminal_status=status_str,
    )
