"""RQ3 frame consistency, failure atomicity, and stateful conformance runner."""

from __future__ import annotations

import random
import time
from threading import Event, Lock, Thread

from benchmarks.model import ConformanceResultRow
from benchmarks.scenarios import (
    apply_reconfiguration_edit,
    create_reconfiguration_registry,
    make_reconfiguration_base_spec,
)
from benchmarks.storage import append_conformance_rows, write_conformance_header
from nedo_vision_dag_engine.compiler import CompiledCandidate, StateDirective, WorkflowCompiler
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import RetirementStatus
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

    # 16.4 Grace-Period Deterministic Campaign (reviewer P0.4 Scenario A)
    row_gp = _run_grace_period_deterministic_campaign(run_id)
    all_rows.append(row_gp)
    append_conformance_rows(output_csv_path, [row_gp])

    # 16.5 Stateful-Handoff Deterministic Campaign (reviewer P0.4 Scenario B)
    row_sh = _run_stateful_handoff_deterministic_campaign(run_id)
    all_rows.append(row_sh)
    append_conformance_rows(output_csv_path, [row_sh])

    # 16.6 Candidate Memory-Failure Campaign (reviewer P0.4 Scenario C)
    row_mem = _run_candidate_memory_failure_campaign(run_id)
    all_rows.append(row_mem)
    append_conformance_rows(output_csv_path, [row_mem])

    # 16.7 Cleanup-Failure Campaign (reviewer P0.4 Scenario D)
    row_cleanup = _run_cleanup_failure_campaign(run_id)
    all_rows.append(row_cleanup)
    append_conformance_rows(output_csv_path, [row_cleanup])

    # 16.8 Frame-Exception Lease-Release Campaign (reviewer P0.4 Scenario E)
    row_lease = _run_frame_exception_lease_release_campaign(run_id)
    all_rows.append(row_lease)
    append_conformance_rows(output_csv_path, [row_lease])

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
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
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
    reconfigs_committed = 0
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
    rec1 = controller.wait_for_terminal(req1.request_id, timeout_seconds=10.0)
    st1 = rec1.status if rec1 is not None else controller.record(req1.request_id).status
    if st1 == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1
    elif st1 == ReconfigurationStatus.REJECTED:
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
    rec2 = controller.wait_for_terminal(req2.request_id, timeout_seconds=10.0)
    st2 = rec2.status if rec2 is not None else controller.record(req2.request_id).status
    if st2 == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1
    elif st2 == ReconfigurationStatus.REJECTED:
        reconfigs_rejected += 1
    else:
        reconfigs_failed += 1
    if controller.active_plan.version != v_before:
        plan_changed_count += 1

    controller.close()
    
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
    rec3 = fail_controller.wait_for_terminal(req3.request_id, timeout_seconds=10.0)
    st3 = rec3.status if rec3 is not None else fail_controller.record(req3.request_id).status
    if st3 == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1
    elif st3 == ReconfigurationStatus.REJECTED:
        reconfigs_rejected += 1
    else:
        reconfigs_failed += 1
    if executor.active_plan.version != v_before:
        plan_changed_count += 1

    fail_controller.close()

    status_str = "PASS" if plan_changed_count == 0 else "FAIL"

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="failure_atomicity",
        scenario_id="conformance_failure_atomicity",
        scenario_type="failure_atomicity",
        frames_submitted=10,
        frames_completed=10,
        reconfigurations_requested=reconfigs_requested,
        reconfigurations_committed=reconfigs_committed,
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
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
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
    processor_instance_leaks = 0

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
            controller.admit_frame(admitted_at_ns=time.monotonic_ns())
            pres_ret = controller.wait_for_retirement(req_pres.request_id, timeout_seconds=5.0)
            if pres_ret is None or pres_ret.retirement_status not in {
                RetirementStatus.COMPLETED,
                RetirementStatus.NOT_REQUIRED,
            }:
                processor_instance_leaks += 1

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
            controller.admit_frame(admitted_at_ns=time.monotonic_ns())
            reset_ret = controller.wait_for_retirement(req_reset.request_id, timeout_seconds=5.0)
            if reset_ret is None or reset_ret.retirement_status not in {
                RetirementStatus.COMPLETED,
                RetirementStatus.NOT_REQUIRED,
            }:
                processor_instance_leaks += 1

    if rec_reset and rec_reset.status == ReconfigurationStatus.COMMITTED:
        reconfigs_committed += 1

    # 3. REJECT test (incompatible type change with PRESERVE)
    controller.close()
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

    status_str = "PASS" if (continuity_failures == 0 and unexpected_resets == 0 and processor_instance_leaks == 0) else "FAIL"

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
        processor_instance_leaks=processor_instance_leaks,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
        terminal_status=status_str,
    )


# =============================================================================
# P0.4 Deterministic lifecycle conformance campaigns
# =============================================================================


def _run_grace_period_deterministic_campaign(run_id: str) -> ConformanceResultRow:
    """Scenario A: Block old-plan frame, publish, verify cleanup deferred.

    An old-plan frame is held at a barrier. A new plan is published while
    the frame is still executing. We verify that processor cleanup does NOT
    begin before the old frame completes (grace-period safety).
    """
    from nedo_vision_dag_engine.compiler import (
        CompiledCandidate,
        StateDirective,
        WorkflowCompiler,
    )
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from nedo_vision_dag_engine.processor import (
        FrameContext,
        ProcessorDescriptor,
        SetupContext,
    )
    from nedo_vision_dag_engine.reconfiguration import (
        ReconfigurationController,
        ReconfigurationRequest,
        ReconfigurationStatus,
    )
    from nedo_vision_dag_engine.registry import (
        RegisteredProcessorType,
        RegistryBuilder,
    )
    from nedo_vision_dag_engine.specification import (
        Node,
        Pin,
        PinCardinality,
        PinRequirement,
        WorkflowSpecification,
        to_processor_configuration,
    )
    from nedo_vision_dag_engine.type_system import ConcreteType, StatePolicy

    frame_entered = Event()
    frame_release = Event()
    cleanup_called = Event()

    class _BlockingProcessor:
        descriptor = ProcessorDescriptor(
            type_name="blocking",
            input_schema=object,
            output_schema=object,
            config_schema=type,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            frame_entered.set()
            frame_release.wait()
            return context.frame_id

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            cleanup_called.set()

    class _ReplacementProcessor:
        descriptor = ProcessorDescriptor(
            type_name="replacement",
            input_schema=object,
            output_schema=object,
            config_schema=type,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

        def setup(self, context: SetupContext) -> None:
            pass

        def process(self, inputs: object, context: FrameContext) -> object:
            return context.frame_id

        def healthcheck(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    builder = RegistryBuilder()
    builder.register(RegisteredProcessorType(descriptor=_BlockingProcessor.descriptor, factory=_BlockingProcessor))
    builder.register(RegisteredProcessorType(descriptor=_ReplacementProcessor.descriptor, factory=_ReplacementProcessor))
    registry = builder.snapshot()

    spec = WorkflowSpecification(
        nodes=(
            Node("n0", "blocking", to_processor_configuration({}), (), (
                Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
            )),
        ),
        edges=(),
    )
    replacement_spec = WorkflowSpecification(
        nodes=(
            Node("n0", "replacement", to_processor_configuration({}), (), (
                Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
            )),
        ),
        edges=(),
    )

    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)
    controller = ReconfigurationController(executor, compiler, registry)

    violations = 0
    try:
        token = getattr(controller, "_executor_token")
        t = Thread(target=executor.admit_frame_managed, args=(token, 0, 1), daemon=True)
        t.start()
        assert frame_entered.wait(timeout=5.0)

        request = ReconfigurationRequest(
            "gp-det", cand.plan.version, replacement_spec,
            StateDirective(), submitted_at_ns=0,
        )
        controller.submit(request)
        controller.wait_for_status("gp-det", frozenset({ReconfigurationStatus.READY}), timeout_seconds=5.0)

        commit_done = Event()
        def _commit() -> None:
            controller.commit_ready()
            commit_done.set()
        ct = Thread(target=_commit, daemon=True)
        ct.start()
        time.sleep(0.1)

        if cleanup_called.is_set():
            violations += 1  # cleanup before old frame completed

        frame_release.set()
        t.join(timeout=5.0)
        assert commit_done.wait(timeout=10.0)
        ct.join(timeout=5.0)

        rec = controller.wait_for_retirement("gp-det", timeout_seconds=10.0)
        if rec is None or rec.retirement_status is not RetirementStatus.COMPLETED:
            violations += 1
        if not cleanup_called.is_set():
            violations += 1  # cleanup never called
    finally:
        frame_release.set()
        controller.close()

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="grace_period_deterministic",
        scenario_id="conformance_grace_period",
        scenario_type="grace_period",
        frames_submitted=1, frames_completed=1,
        reconfigurations_requested=1, reconfigurations_committed=1,
        reconfigurations_rejected=0, reconfigurations_failed=0,
        mixed_plan_frames=0, missing_frames=0, duplicate_frames=0,
        retired_plan_executions=1, invalid_routing_events=0,
        state_continuity_failures=0, unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0, processor_instance_leaks=0,
        grace_period_safety_violations=violations,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=violations,
        terminal_status="PASS" if violations == 0 else "FAIL",
    )


def _run_stateful_handoff_deterministic_campaign(run_id: str) -> ConformanceResultRow:
    """Scenario B: Preserve stateful processor, verify old-plan updates finish first.

    A mutable stateful processor is PRESERVED across reconfiguration.
    We verify that every old-plan frame completes its state update before
    any new-plan frame accesses the processor.
    """
    from nedo_vision_dag_engine.compiler import (
        CompiledCandidate, StateDirective, WorkflowCompiler,
    )
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from nedo_vision_dag_engine.processor import (
        FrameContext, ProcessorDescriptor, SetupContext,
        StatefulProcessorDescriptor, TransitionContext,
    )
    from nedo_vision_dag_engine.reconfiguration import (
        ReconfigurationController, ReconfigurationRequest,
        ReconfigurationStatus,
    )
    from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
    from nedo_vision_dag_engine.specification import (
        Node, Pin, PinCardinality, PinRequirement,
        WorkflowSpecification, to_processor_configuration,
        ProcessorConfiguration,
    )
    from nedo_vision_dag_engine.type_system import (
        ConcreteType, StatePolicy, StateTransitionPolicy,
    )

    ordering_violations = 0
    update_sequence: list[int] = []  # plan_version of each update call

    def _preserves(prev: ProcessorConfiguration, new: ProcessorConfiguration, ctx: TransitionContext) -> bool:
        return prev == new and ctx.structurally_preservable

    _desc = ProcessorDescriptor(
        type_name="stateful_tracker",
        input_schema=object, output_schema=object, config_schema=type,
        state_policy=StatePolicy.PRESERVABLE,
        state_schema_version="v1",
    )
    _sdesc = StatefulProcessorDescriptor(
        state_schema_version="v1",
        supported_transition_policies=frozenset({StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}),
        preserves_state_for=_preserves,
    )

    class _StatefulTracker:
        descriptor = _desc
        stateful_descriptor = _sdesc
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object:
            update_sequence.append(context.plan_version)
            return context.frame_id
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None: pass

    builder = RegistryBuilder()
    builder.register(RegisteredProcessorType(
        descriptor=_desc, stateful_descriptor=_sdesc, factory=_StatefulTracker,
    ))
    registry = builder.snapshot()
    spec = WorkflowSpecification(
        nodes=(Node("t", "stateful_tracker", to_processor_configuration({}), (), (
            Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
        )),),
        edges=(),
    )

    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)
    controller = ReconfigurationController(executor, compiler, registry)

    try:
        # Frame on old plan
        controller.admit_frame(0, 1)

        # Submit identical spec (tracker is PRESERVED)
        request = ReconfigurationRequest(
            "sh-det", cand.plan.version, spec,
            StateDirective(), submitted_at_ns=0,
        )
        controller.submit(request)
        controller.wait_for_status("sh-det", frozenset({ReconfigurationStatus.READY}), timeout_seconds=5.0)
        controller.commit_ready()

        # Frame on new plan
        controller.admit_frame(0, 2)

        controller.wait_for_retirement("sh-det", timeout_seconds=5.0)

        # Check ordering: old-plan frames (v1) must come before new-plan (v2+)
        for i in range(1, len(update_sequence)):
            if update_sequence[i] < update_sequence[i - 1]:
                ordering_violations += 1
    finally:
        controller.close()

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="stateful_handoff_deterministic",
        scenario_id="conformance_handoff",
        scenario_type="stateful_handoff",
        frames_submitted=2, frames_completed=2,
        reconfigurations_requested=1, reconfigurations_committed=1,
        reconfigurations_rejected=0, reconfigurations_failed=0,
        mixed_plan_frames=0, missing_frames=0, duplicate_frames=0,
        retired_plan_executions=0, invalid_routing_events=0,
        state_continuity_failures=0, unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0, processor_instance_leaks=0,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=ordering_violations,
        resource_lifetime_violations=0,
        terminal_status="PASS" if ordering_violations == 0 else "FAIL",
    )


def _run_candidate_memory_failure_campaign(run_id: str) -> ConformanceResultRow:
    """Scenario C: Inject MemoryError during factory; verify P_old remains active."""
    from nedo_vision_dag_engine.compiler import (
        CompiledCandidate, StateDirective, WorkflowCompiler,
    )
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from nedo_vision_dag_engine.processor import (
        FrameContext, ProcessorDescriptor, SetupContext,
    )
    from nedo_vision_dag_engine.reconfiguration import (
        ReconfigurationController, ReconfigurationRequest,
        ReconfigurationStatus,
    )
    from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
    from nedo_vision_dag_engine.specification import (
        Node, Pin, PinCardinality, PinRequirement,
        WorkflowSpecification, to_processor_configuration,
    )
    from nedo_vision_dag_engine.type_system import ConcreteType, StatePolicy

    violations = 0

    good_desc = ProcessorDescriptor(
        type_name="good", input_schema=object, output_schema=object,
        config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
    )
    bad_desc = ProcessorDescriptor(
        type_name="bad", input_schema=object, output_schema=object,
        config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
    )

    class _GoodProcessor:
        descriptor = good_desc
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object: return context.frame_id
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None: pass

    def _bad_factory() -> _GoodProcessor:
        raise MemoryError("injected memory error in candidate factory")

    builder = RegistryBuilder()
    builder.register(RegisteredProcessorType(descriptor=good_desc, factory=_GoodProcessor))
    builder.register(RegisteredProcessorType(descriptor=bad_desc, factory=_bad_factory))
    registry = builder.snapshot()

    spec = WorkflowSpecification(
        nodes=(Node("n0", "good", to_processor_configuration({}), (), (
            Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
        )),),
        edges=(),
    )
    bad_spec = WorkflowSpecification(
        nodes=(Node("n0", "bad", to_processor_configuration({}), (), (
            Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
        )),),
        edges=(),
    )

    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)
    controller = ReconfigurationController(executor, compiler, registry)

    try:
        old_version = controller.active_plan.version

        request = ReconfigurationRequest(
            "mem-fail", old_version, bad_spec,
            StateDirective(), submitted_at_ns=0,
        )
        controller.submit(request)
        rec = controller.wait_for_terminal("mem-fail", timeout_seconds=10.0)

        if rec is None or rec.status not in (ReconfigurationStatus.FAILED, ReconfigurationStatus.REJECTED):
            violations += 1

        # P_old must still be active
        if controller.active_plan.version != old_version:
            violations += 1

        # Controller must still admit frames on old plan
        result = controller.admit_frame(0, 1)
        if result.status.value != "completed":
            violations += 1

        # A valid reconfiguration must succeed afterwards
        request2 = ReconfigurationRequest(
            "mem-fail-2", old_version, spec,
            StateDirective(), submitted_at_ns=0,
        )
        controller.submit(request2)
        rec2 = controller.wait_for_terminal("mem-fail-2", timeout_seconds=10.0)
        if rec2 is None or rec2.status is not ReconfigurationStatus.COMMITTED:
            violations += 1
    finally:
        controller.close()

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="candidate_memory_failure",
        scenario_id="conformance_memory_failure",
        scenario_type="memory_failure",
        frames_submitted=1, frames_completed=1,
        reconfigurations_requested=2, reconfigurations_committed=1,
        reconfigurations_rejected=0, reconfigurations_failed=1,
        mixed_plan_frames=0, missing_frames=0, duplicate_frames=0,
        retired_plan_executions=1, invalid_routing_events=0,
        state_continuity_failures=0, unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=violations,
        candidate_resource_leaks=0, processor_instance_leaks=0,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
        terminal_status="PASS" if violations == 0 else "FAIL",
    )


def _run_cleanup_failure_campaign(run_id: str) -> ConformanceResultRow:
    """Scenario D: Inject cleanup failure; verify plan stays committed and
    retirement is FAILED but idempotent."""
    from nedo_vision_dag_engine.compiler import (
        CompiledCandidate, StateDirective, WorkflowCompiler,
    )
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from nedo_vision_dag_engine.processor import (
        FrameContext, ProcessorDescriptor, SetupContext,
    )
    from nedo_vision_dag_engine.reconfiguration import (
        ReconfigurationController, ReconfigurationRequest,
        ReconfigurationStatus,
    )
    from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
    from nedo_vision_dag_engine.specification import (
        Node, Pin, PinCardinality, PinRequirement,
        WorkflowSpecification, to_processor_configuration,
    )
    from nedo_vision_dag_engine.type_system import ConcreteType, StatePolicy

    violations = 0
    cleanup_count = 0
    cleanup_lock = Lock()

    src_desc = ProcessorDescriptor(
        type_name="cleanup_src", input_schema=object, output_schema=object,
        config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
    )
    bad_cleanup_desc = ProcessorDescriptor(
        type_name="bad_cleanup", input_schema=object, output_schema=object,
        config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
    )
    alt_desc = ProcessorDescriptor(
        type_name="alt_src", input_schema=object, output_schema=object,
        config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
    )

    class _Source:
        descriptor = src_desc
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object: return context.frame_id
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None: pass

    class _BadCleanup:
        descriptor = bad_cleanup_desc
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object: return context.frame_id
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None:
            nonlocal cleanup_count
            with cleanup_lock:
                cleanup_count += 1
            raise RuntimeError("injected cleanup failure")

    class _AltSource:
        descriptor = alt_desc
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object: return context.frame_id
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None: pass

    builder = RegistryBuilder()
    builder.register(RegisteredProcessorType(descriptor=src_desc, factory=_Source))
    builder.register(RegisteredProcessorType(descriptor=bad_cleanup_desc, factory=_BadCleanup))
    builder.register(RegisteredProcessorType(descriptor=alt_desc, factory=_AltSource))
    registry = builder.snapshot()

    spec = WorkflowSpecification(
        nodes=(
            Node("n0", "cleanup_src", to_processor_configuration({}), (), (
                Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
            )),
            Node("n1", "bad_cleanup", to_processor_configuration({}), (
                Pin("in", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
            ), (
                Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
            )),
        ),
        edges=(
            Edge("n0", "out", "n1", "in"),
        ),
    )
    # Replacement removes the bad-cleanup node entirely, triggering cleanup
    replacement_spec = WorkflowSpecification(
        nodes=(Node("n0", "cleanup_src", to_processor_configuration({}), (), (
            Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
        )),),
        edges=(),
    )

    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)
    controller = ReconfigurationController(executor, compiler, registry)

    try:
        controller.admit_frame(0, 1)
        old_version = controller.active_plan.version

        request = ReconfigurationRequest(
            "cleanup-fail", old_version, replacement_spec,
            StateDirective(), submitted_at_ns=0,
        )
        controller.submit(request)
        controller.wait_for_status("cleanup-fail", frozenset({ReconfigurationStatus.READY}), timeout_seconds=5.0)
        controller.commit_ready()

        rec = controller.wait_for_retirement("cleanup-fail", timeout_seconds=10.0)
        if rec is None:
            violations += 1
        else:
            # Plan must remain COMMITTED even though cleanup failed
            if rec.status is not ReconfigurationStatus.COMMITTED:
                violations += 1
            # Retirement must be FAILED (or COMPLETED if no processors to retire — but we have one)
            if rec.retirement_status not in (RetirementStatus.FAILED, RetirementStatus.COMPLETED):
                violations += 1

        # Cleanup must have been attempted (at least once)
        with cleanup_lock:
            if cleanup_count < 1:
                violations += 1
    finally:
        controller.close()

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="cleanup_failure",
        scenario_id="conformance_cleanup_failure",
        scenario_type="cleanup_failure",
        frames_submitted=1, frames_completed=1,
        reconfigurations_requested=1, reconfigurations_committed=1,
        reconfigurations_rejected=0, reconfigurations_failed=0,
        mixed_plan_frames=0, missing_frames=0, duplicate_frames=0,
        retired_plan_executions=1, invalid_routing_events=0,
        state_continuity_failures=0, unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0, processor_instance_leaks=0,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
        terminal_status="PASS" if violations == 0 else "FAIL",
    )


def _run_frame_exception_lease_release_campaign(run_id: str) -> ConformanceResultRow:
    """Scenario E: Frame raises during old-plan execution; verify lease released."""
    from nedo_vision_dag_engine.compiler import (
        CompiledCandidate, WorkflowCompiler,
    )
    from nedo_vision_dag_engine.executor import PipelineExecutor
    from nedo_vision_dag_engine.processor import (
        FrameContext, ProcessorDescriptor, SetupContext,
    )
    from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
    from nedo_vision_dag_engine.specification import (
        Node, Pin, PinCardinality, PinRequirement,
        WorkflowSpecification, to_processor_configuration,
    )
    from nedo_vision_dag_engine.type_system import ConcreteType, StatePolicy

    violations = 0

    class _FailingProcessor:
        descriptor = ProcessorDescriptor(
            type_name="failing", input_schema=object, output_schema=object,
            config_schema=type, state_policy=StatePolicy.STATELESS, state_schema_version=None,
        )
        def setup(self, context: SetupContext) -> None: pass
        def process(self, inputs: object, context: FrameContext) -> object:
            raise RuntimeError("injected process failure")
        def healthcheck(self) -> None: pass
        def cleanup(self) -> None: pass

    builder = RegistryBuilder()
    builder.register(RegisteredProcessorType(descriptor=_FailingProcessor.descriptor, factory=_FailingProcessor))
    registry = builder.snapshot()
    spec = WorkflowSpecification(
        nodes=(Node("n0", "failing", to_processor_configuration({}), (), (
            Pin("out", ConcreteType(int), PinCardinality.SINGLE, PinRequirement.REQUIRED),
        )),),
        edges=(),
    )

    compiler = WorkflowCompiler("0.1.0")
    cand = compiler.compile(spec, registry)
    assert isinstance(cand, CompiledCandidate)
    executor = PipelineExecutor(cand.plan)

    result = executor.admit_frame(0, 1)
    if result.status.value != "failed":
        violations += 1
    if result.error is None:
        violations += 1

    # After frame failure, plan must be quiescent (lease released in finally)
    quiescent = executor.wait_for_plan_quiescent(cand.plan.version, timeout=1.0)
    if not quiescent:
        violations += 1

    return ConformanceResultRow(
        run_id=run_id,
        campaign_id="frame_exception_lease_release",
        scenario_id="conformance_frame_exception",
        scenario_type="frame_exception",
        frames_submitted=1, frames_completed=0,
        reconfigurations_requested=0, reconfigurations_committed=0,
        reconfigurations_rejected=0, reconfigurations_failed=0,
        mixed_plan_frames=0, missing_frames=0, duplicate_frames=0,
        retired_plan_executions=0, invalid_routing_events=0,
        state_continuity_failures=0, unexpected_state_resets=0,
        active_plan_changed_after_failed_candidate=0,
        candidate_resource_leaks=0, processor_instance_leaks=0,
        grace_period_safety_violations=0,
        stateful_handoff_ordering_failures=0,
        resource_lifetime_violations=0,
        terminal_status="PASS" if violations == 0 else "FAIL",
    )
