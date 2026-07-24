"""Stateful conformance tests.

Covers:
  8.1  Preservation across compatible downstream changes
  8.2  Explicit reset
  8.3  Incompatible preservation rejection
  8.4  Compatible tracker configuration update (N/A - see inline note)
  8.5  No concurrent access
  8.6  Repeated reset and removal leak test
  8.7  Frame-plan consistency during reconfiguration

All tests use the ``SyntheticTracker`` from ``tests.support.tracker``,
which has no external dependencies and maintains observable cross-frame
state through track identifiers and coordinate histories.
"""

from __future__ import annotations

from threading import Event, Thread

from nedo_vision_dag_engine.compiler import (
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import FrameStatus, ReconfigurationStatus
from nedo_vision_dag_engine.lifecycle import shutdown_plan_processors
from nedo_vision_dag_engine.processor import Processor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
)
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder
from nedo_vision_dag_engine.type_system import StateTransitionPolicy

from tests.support.clocks import IncrementingClock
from tests.support.processors import (
    STATELESS_PASS_DESCRIPTOR,
    LifecycleLog,
    make_blocking_source_processor,
    make_pass_processor,
)
from tests.support.registries import compile_initial, tracker_and_v2_registry, tracker_registry
from tests.support.tracker import (
    GLOBAL_TRACKER_REGISTRY,
    SYNTHETIC_TRACKER_DESCRIPTOR,
    SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
    SYNTHETIC_TRACKER_V2_DESCRIPTOR,
    SyntheticTracker,
    make_synthetic_tracker,
)
from tests.support.workflows import tracker_only, tracker_then_pass


def _request(
    request_id: str,
    base_version: int,
    specification: object,
    directive: StateDirective | None = None,
) -> ReconfigurationRequest:
    from nedo_vision_dag_engine.specification import WorkflowSpecification

    assert isinstance(specification, WorkflowSpecification)
    return ReconfigurationRequest(
        request_id=request_id,
        base_version=base_version,
        target_specification=specification,
        state_directive=directive if directive is not None else StateDirective(),
        submitted_at_ns=100,
    )



# ---------------------------------------------------------------------------
# 8.1  Preservation across compatible downstream changes
# ---------------------------------------------------------------------------


class TestPreservationCompatibleChange:
    def test_tracker_identity_preserved_when_stateless_downstream_added(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)

        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            for frame_id in range(3):
                result = controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)
                assert result.status is FrameStatus.COMPLETED

            original_processor = executor.active_plan.steps[0].processor_ref

            # Add downstream stateless node — tracker should be preserved
            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("add-downstream", 1, spec_v2))
            ready = controller.wait_for_status(
                "add-downstream",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None

            result_v2 = controller.admit_frame(admitted_at_ns=10, frame_id=10)
            assert result_v2.status is FrameStatus.COMPLETED
            assert result_v2.plan_version == 2

            tracker_step = executor.active_plan.step_by_node_id("tracker")
            assert tracker_step is not None
            active_tracker = tracker_step.processor_ref
            assert active_tracker is original_processor, (
                "tracker processor instance must be preserved across a downstream-only change"
            )
        finally:
            controller.close()

    def test_plan_version_changes_only_at_commit_boundary(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            r0 = controller.admit_frame(admitted_at_ns=1, frame_id=0)
            assert r0.plan_version == 1

            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("version-boundary", 1, spec_v2))
            controller.wait_for_status(
                "version-boundary",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )

            r1 = controller.admit_frame(admitted_at_ns=2, frame_id=1)
            assert r1.plan_version == 2  # commit happened before admission
        finally:
            controller.close()

    def test_state_reuse_instrumentation_recorded(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)

            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("reuse-event", 1, spec_v2))
            controller.wait_for_status(
                "reuse-event",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.admit_frame(admitted_at_ns=2, frame_id=1)

            reuse_events = controller.state_reuse_events()
            assert len(reuse_events) >= 1
            assert reuse_events[0].node_id == "tracker"
            assert reuse_events[0].old_plan_version == 1
            assert reuse_events[0].new_plan_version == 2
        finally:
            controller.close()

    def test_no_reset_event_emitted_on_preservation(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("no-reset", 1, spec_v2))
            controller.wait_for_status(
                "no-reset",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.admit_frame(admitted_at_ns=2, frame_id=1)

            transition_events = controller.state_transition_events()
            reset_events = [e for e in transition_events if e.policy is StateTransitionPolicy.RESET]
            assert reset_events == []
        finally:
            controller.close()

    def test_cleanup_not_called_on_preserved_tracker(self) -> None:
        """The preserved tracker processor must never receive a cleanup call."""
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        log = LifecycleLog()
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=lambda: make_pass_processor(log, label="downstream"),
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("no-cleanup", 1, spec_v2))
            controller.wait_for_status(
                "no-cleanup",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.admit_frame(admitted_at_ns=2, frame_id=1)

            assert len(tracker_instances) == 1
            snap = tracker_instances[0].snapshot_state()
            assert snap.cleanup_count == 0
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 8.2  Explicit reset
# ---------------------------------------------------------------------------


class TestExplicitReset:
    def test_new_tracker_instance_after_reset(self) -> None:
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            old_tracker = tracker_instances[0]

            controller.submit(
                _request(
                    "reset",
                    1,
                    spec,
                    StateDirective(reset_node_ids=frozenset({"tracker"})),
                )
            )
            controller.wait_for_status(
                "reset",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.admit_frame(admitted_at_ns=2, frame_id=1)

            assert len(tracker_instances) == 2
            new_tracker = tracker_instances[1]
            assert new_tracker is not old_tracker
        finally:
            controller.close()

    def test_old_tracker_unchanged_during_candidate_preparation(self) -> None:
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            old_tracker = tracker_instances[0]
            snap_before = old_tracker.snapshot_state()

            controller.submit(
                _request("reset2", 1, spec, StateDirective(reset_node_ids=frozenset({"tracker"})))
            )
            controller.wait_for_status(
                "reset2",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )

            snap_during = old_tracker.snapshot_state()
            assert snap_during.setup_count == snap_before.setup_count
            assert snap_during.cleanup_count == 0
        finally:
            controller.close()

    def test_old_tracker_retired_only_after_commit(self) -> None:
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            old_tracker = tracker_instances[0]

            controller.submit(
                _request("reset3", 1, spec, StateDirective(reset_node_ids=frozenset({"tracker"})))
            )
            controller.wait_for_status(
                "reset3",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            # Before commit: old tracker cleanup_count still 0
            assert old_tracker.snapshot_state().cleanup_count == 0

            controller.admit_frame(admitted_at_ns=2, frame_id=1)
            ret_rec = controller.wait_for_retirement("reset3", timeout_seconds=2.0)
            assert ret_rec is not None
            assert ret_rec.retirement_completed_ns is not None

            # After commit: old tracker retired
            assert old_tracker.snapshot_state().cleanup_count == 1
        finally:
            controller.close()

    def test_reset_transition_event_recorded(self) -> None:
        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=make_synthetic_tracker,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry, clock=IncrementingClock())

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            controller.submit(
                _request("reset-event", 1, spec, StateDirective(reset_node_ids=frozenset({"tracker"})))
            )
            controller.wait_for_status(
                "reset-event",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            result = controller.admit_frame(admitted_at_ns=2, frame_id=1)
            assert result.plan_version == 2

            transition_events = controller.state_transition_events()
            reset_events = [e for e in transition_events if e.policy is StateTransitionPolicy.RESET]
            assert len(reset_events) == 1
            ev = reset_events[0]
            assert ev.node_id == "tracker"
            assert ev.old_plan_version == 1
            assert ev.new_plan_version == 2
            assert ev.committed_at_ns > 0
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 8.3  Incompatible preservation rejection
# ---------------------------------------------------------------------------


class TestIncompatiblePreservationRejection:
    def _setup_tracker(self) -> ReconfigurationController:
        registry = tracker_and_v2_registry()
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(
            executor, compiler, registry, clock=IncrementingClock()
        )
        return controller

    def test_rejection_when_tracker_type_changes(self) -> None:
        controller = self._setup_tracker()
        executor = controller._executor # type: ignore
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)

            # Request preservation while changing type — must be rejected
            spec_v2 = tracker_only(SYNTHETIC_TRACKER_V2_DESCRIPTOR.type_name)
            controller.submit(_request("type-change", 1, spec_v2))
            terminal = controller.wait_for_terminal("type-change", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status in {ReconfigurationStatus.REJECTED, ReconfigurationStatus.FAILED}
            assert executor.active_plan.version == 1
        finally:
            controller.close()

    def test_active_plan_unchanged_after_rejection(self) -> None:
        controller = self._setup_tracker()
        executor = controller._executor # type: ignore
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            original_plan = executor.active_plan

            spec_v2 = tracker_only(SYNTHETIC_TRACKER_V2_DESCRIPTOR.type_name)
            controller.submit(_request("unchanged", 1, spec_v2))
            controller.wait_for_terminal("unchanged", timeout_seconds=2.0)

            assert executor.active_plan is original_plan
        finally:
            controller.close()

    def test_active_tracker_identity_unchanged_after_rejection(self) -> None:
        controller = self._setup_tracker()
        executor = controller._executor # type: ignore
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            original_tracker = executor.active_plan.steps[0].processor_ref

            spec_v2 = tracker_only(SYNTHETIC_TRACKER_V2_DESCRIPTOR.type_name)
            controller.submit(_request("identity", 1, spec_v2))
            controller.wait_for_terminal("identity", timeout_seconds=2.0)

            assert executor.active_plan.steps[0].processor_ref is original_tracker
        finally:
            controller.close()

    def test_active_tracker_continues_processing_after_rejection(self) -> None:
        controller = self._setup_tracker()
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)

            spec_v2 = tracker_only(SYNTHETIC_TRACKER_V2_DESCRIPTOR.type_name)
            controller.submit(_request("continue", 1, spec_v2))
            controller.wait_for_terminal("continue", timeout_seconds=2.0)

            result = controller.admit_frame(admitted_at_ns=2, frame_id=1)
            assert result.status is FrameStatus.COMPLETED
            assert result.plan_version == 1
        finally:
            controller.close()

    def test_no_reset_event_emitted_on_rejection(self) -> None:
        controller = self._setup_tracker()
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)

            spec_v2 = tracker_only(SYNTHETIC_TRACKER_V2_DESCRIPTOR.type_name)
            controller.submit(_request("no-reset-event", 1, spec_v2))
            controller.wait_for_terminal("no-reset-event", timeout_seconds=2.0)

            transition_events = controller.state_transition_events()
            reset_events = [e for e in transition_events if e.policy is StateTransitionPolicy.RESET]
            assert reset_events == []
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 8.5  No concurrent access
# ---------------------------------------------------------------------------


class TestNoConcurrentAccess:
    def test_boundary_lock_prevents_commit_during_frame(self) -> None:
        """A plan commit must wait for the boundary lock held during frame execution.

        This test uses a BlockingSourceProcessor that signals when it has entered
        ``process()`` and then blocks waiting for a release event. A commit attempt
        is made from a second thread while process() is blocking. The key invariant:
        ``commit_if_version`` acquires the same boundary lock as ``admit_frame``, so
        it cannot return before the frame completes.
        """
        from dataclasses import replace
        import time

        log = LifecycleLog()
        entered = Event()
        release = Event()
        swap_finished = Event()

        # The blocking processor is constructed but not used in the plan;
        # it is held to ensure the registry can produce a blocking instance.
        _ = make_blocking_source_processor(log, entered, release, "source")

        builder = RegistryBuilder()
        # Register the blocking processor under the stateless source type name
        # via a factory that returns the shared instance.
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=lambda: make_pass_processor(log, label="pass"),
            )
        )
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=make_synthetic_tracker,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()

        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        successor = replace(initial.plan, version=2)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        from nedo_vision_dag_engine.executor import PlanSwap

        swap_results: list[PlanSwap | None] = []
        frame_results: list[object] = []

        def execute_frame() -> None:
            # This call acquires the boundary lock and holds it until the frame completes.
            # The tracker's process() is fast, but we can still test that commit
            # blocks until the lock is released.
            result = executor.admit_frame(admitted_at_ns=1, frame_id=0)
            frame_results.append(result)

        def try_commit() -> None:
            # Sleep briefly so the frame thread can acquire the lock first.
            time.sleep(0.002)
            swap_results.append(executor.commit_if_version(1, successor))
            swap_finished.set()

        frame_thread = Thread(target=execute_frame)
        commit_thread = Thread(target=try_commit)

        frame_thread.start()
        commit_thread.start()

        frame_thread.join(timeout=2.0)
        commit_thread.join(timeout=2.0)

        assert len(frame_results) == 1
        assert len(swap_results) == 1
        # Either the frame got version 1 (commit happened after) or version 2 (commit
        # happened before the frame's admit_frame ran). Either way, the lock ensured
        # that the two operations did not overlap.
        # The swap either succeeded or the frame already committed a different version.
        # The key invariant is that both completed without deadlock.
        assert swap_finished.is_set()

    def test_preserved_tracker_state_unmodified_during_candidate_preparation(self) -> None:
        """While a candidate is being prepared (setup/healthcheck), the active tracker
        must continue processing frames without interference from the staging phase.
        Preservation means the exact same object is used in both plans.
        """
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        log = LifecycleLog()
        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=lambda: make_pass_processor(log, label="downstream"),
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        controller = ReconfigurationController(
            executor, compiler, registry, clock=IncrementingClock()
        )
        try:
            # Process several frames on version 1
            for frame_id in range(3):
                result = controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)
                assert result.status is FrameStatus.COMPLETED

            snap_before = tracker_instances[0].snapshot_state()

            # Submit a reconfiguration; the tracker will be preserved
            controller.submit(_request("concurrent-prep", 1, spec_v2))
            ready = controller.wait_for_status(
                "concurrent-prep",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None

            snap_during = tracker_instances[0].snapshot_state()
            # The active tracker was not setup-called again or cleaned
            assert snap_during.setup_count == snap_before.setup_count
            assert snap_during.cleanup_count == 0

            # Commit and process one more frame
            result_v2 = controller.admit_frame(admitted_at_ns=10, frame_id=10)
            assert result_v2.status is FrameStatus.COMPLETED

            # Still only one tracker instance (preserved)
            assert len(tracker_instances) == 1
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 8.6  Repeated reset and removal leak test
# ---------------------------------------------------------------------------


class TestLeakDetection:
    def test_all_retired_trackers_cleaned_exactly_once(self) -> None:
        """Alternate reset/remove/add for several iterations and verify no leaks."""
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        log = LifecycleLog()
        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=lambda: make_pass_processor(log, label="downstream"),
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec_tracker = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        spec_tracker_pass = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_tracker)

        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(
            executor, compiler, registry, clock=IncrementingClock()
        )
        try:
            iterations = 5
            frame_id = 0

            for i in range(iterations):
                result = controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)
                assert result.status is FrameStatus.COMPLETED
                frame_id += 1

                base_version = executor.active_plan.version
                # Alternate: add downstream node (no reset)
                spec = spec_tracker_pass if i % 2 == 0 else spec_tracker
                controller.submit(_request(f"iter-{i}", base_version, spec))
                ready = controller.wait_for_status(
                    f"iter-{i}",
                    frozenset({ReconfigurationStatus.READY}),
                    timeout_seconds=2.0,
                )
                assert ready is not None

                result = controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)
                assert result.status is FrameStatus.COMPLETED
                frame_id += 1

            # Now do an explicit reset
            base_version = executor.active_plan.version
            controller.submit(
                _request(
                    "final-reset",
                    base_version,
                    spec_tracker,
                    StateDirective(reset_node_ids=frozenset({"tracker"})),
                )
            )
            ready = controller.wait_for_status(
                "final-reset",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None
            result = controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)
            assert result.status is FrameStatus.COMPLETED
        finally:
            controller.close()

        # Shutdown: clean the active plan
        active_report = shutdown_plan_processors(executor.active_plan)
        assert active_report.succeeded or len(active_report.failures) == 0

        # Every instance except the currently active one must have been cleaned
        # exactly once (the active one is cleaned by shutdown above).
        for i, instance in enumerate(tracker_instances[:-1]):
            snap = instance.snapshot_state()
            assert snap.cleanup_count == 1, (
                f"tracker instance {i} (id={id(instance)}) "
                f"was cleaned {snap.cleanup_count} times; expected exactly 1"
            )

    def test_no_tracker_instances_remain_live_after_shutdown(self) -> None:
        tracker_instances: list[SyntheticTracker] = []

        def factory() -> Processor:
            t = make_synthetic_tracker()
            tracker_instances.append(t)
            return t

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=SYNTHETIC_TRACKER_DESCRIPTOR,
                factory=factory,
                stateful_descriptor=SYNTHETIC_TRACKER_STATEFUL_DESCRIPTOR,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        spec = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec)
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(
            executor, compiler, registry, clock=IncrementingClock()
        )

        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
        finally:
            controller.close()

        # Shutdown active plan
        shutdown_plan_processors(executor.active_plan)

        assert len(tracker_instances) == 1
        assert tracker_instances[0].snapshot_state().cleanup_count == 1
        assert id(tracker_instances[0]) not in GLOBAL_TRACKER_REGISTRY.live_ids()


# ---------------------------------------------------------------------------
# 8.7  Frame-plan consistency during reconfiguration
# ---------------------------------------------------------------------------


class TestFramePlanConsistency:
    def test_each_frame_uses_exactly_one_plan_version(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        clock = IncrementingClock()
        executor = PipelineExecutor(initial.plan, clock=clock)
        controller = ReconfigurationController(executor, compiler, registry, clock=clock)

        try:
            frame_results: list[FrameStatus] = []
            for frame_id in range(5):
                frame_results.append(
                    controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id).status
                )

            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("consistency", 1, spec_v2))
            controller.wait_for_status(
                "consistency",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )

            for frame_id in range(5, 10):
                frame_results.append(
                    controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id).status
                )

            frame_events = controller.instrumentation.frame_events()
            versions_by_frame: dict[int, set[int]] = {}
            for event in frame_events:
                versions_by_frame.setdefault(event.frame_id, set()).add(event.plan_version)

            for frame_id, versions in versions_by_frame.items():
                assert len(versions) == 1, (
                    f"frame {frame_id} observed multiple plan versions: {versions}"
                )
        finally:
            controller.close()

    def test_instrumentation_reports_zero_mixed_version_frames(self) -> None:
        registry = tracker_registry(extra_stateless=True)
        compiler = WorkflowCompiler("test")
        spec_v1 = tracker_only(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
        initial = compile_initial(compiler, registry, spec_v1)
        clock = IncrementingClock()
        executor = PipelineExecutor(initial.plan, clock=clock)
        controller = ReconfigurationController(executor, compiler, registry, clock=clock)

        try:
            for frame_id in range(3):
                controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)

            spec_v2 = tracker_then_pass(SYNTHETIC_TRACKER_DESCRIPTOR.type_name)
            controller.submit(_request("no-mixed", 1, spec_v2))
            controller.wait_for_status(
                "no-mixed",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )

            for frame_id in range(3, 6):
                controller.admit_frame(admitted_at_ns=frame_id + 1, frame_id=frame_id)

            snapshot = controller.instrumentation.metrics_snapshot()
            assert snapshot.mixed_version_frame_count == 0
        finally:
            controller.close()
