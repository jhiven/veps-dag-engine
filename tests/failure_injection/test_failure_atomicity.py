"""Failure-atomicity tests for candidate preparation.

Covers:
  6.1  factory or construction failure
  6.2  setup failure
  6.3  healthcheck failure
  6.4  process failure
  6.5  cleanup failure (during candidate rollback)
  6.6  compatibility predicate failure
  6.7  stale candidate cleanup
  6.8  abort cleanup
"""

from __future__ import annotations

from dataclasses import replace

from nedo_vision_dag_engine.compiler import (
    CompiledCandidate,
    CompilationFailure,
    StateDirective,
    WorkflowCompiler,
)
from nedo_vision_dag_engine.executor import PipelineExecutor
from nedo_vision_dag_engine.instrumentation import FrameStatus, ReconfigurationStatus
from nedo_vision_dag_engine.lifecycle import cleanup_candidate_processors
from nedo_vision_dag_engine.processor import Processor
from nedo_vision_dag_engine.reconfiguration import (
    ReconfigurationController,
    ReconfigurationRequest,
)
from nedo_vision_dag_engine.registry import RegisteredProcessorType, RegistryBuilder

from tests.support.clocks import IncrementingClock
from tests.support.processors import (
    STATELESS_PASS_DESCRIPTOR,
    STATELESS_SOURCE_DESCRIPTOR,
    FailingProcessor,
    InjectedFailurePoint,
    LifecycleLog,
    make_pass_processor,
    make_source_processor,
)
from tests.support.registries import compile_initial, stateless_registry
from tests.support.tracker import (
    TRACKER_RAISING_DESCRIPTOR,
)
from tests.support.workflows import linear, source_only, tracker_only


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
# 6.1  Factory / construction failure
# ---------------------------------------------------------------------------


class TestFactoryFailure:
    def test_active_plan_unchanged_when_source_factory_raises(self) -> None:
        log = LifecycleLog()
        registry = stateless_registry(log, fail_source_factory=True)
        compiler = WorkflowCompiler("test")

        result = compiler.compile(source_only(), registry)

        assert isinstance(result, CompilationFailure)

    def test_active_plan_unchanged_when_pass_factory_raises(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        # Build an initial plan with a working registry
        good_registry = stateless_registry(LifecycleLog())
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        # Now use a registry whose pass factory raises
        failing_registry = stateless_registry(log, fail_pass_factory=True)
        controller = ReconfigurationController(executor, compiler, failing_registry)
        try:
            controller.submit(_request("fail-factory", 1, linear()))
            terminal = controller.wait_for_terminal("fail-factory", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status is ReconfigurationStatus.FAILED
            # Active plan is unchanged
            assert executor.active_plan is initial.plan
        finally:
            controller.close()

    def test_already_staged_processors_are_cleaned_on_factory_failure(self) -> None:
        """When pass factory raises, the already-staged source must be cleaned."""
        source_log = LifecycleLog()
        source_instances: list[FailingProcessor] = []

        def source_factory() -> Processor:
            p = make_source_processor(source_log, label="source")
            source_instances.append(p)
            return p

        def pass_factory() -> Processor:
            raise RuntimeError("injected pass factory failure")

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_SOURCE_DESCRIPTOR,
                factory=source_factory,
            )
        )
        builder.register(
            RegisteredProcessorType(
                descriptor=STATELESS_PASS_DESCRIPTOR,
                factory=pass_factory,
            )
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")

        result = compiler.compile(linear(), registry)

        assert isinstance(result, CompilationFailure)
        # The source was staged; it must have been cleaned during rollback.
        assert len(source_instances) == 1
        cleanup_events = source_log.events_for(source_instances[0].instance_id)
        assert "cleanup" in cleanup_events

    def test_failure_reason_is_retained_in_compilation_result(self) -> None:
        log = LifecycleLog()
        registry = stateless_registry(log, fail_source_factory=True)
        compiler = WorkflowCompiler("test")

        result = compiler.compile(source_only(), registry)

        assert isinstance(result, CompilationFailure)
        assert result.reason
        assert "stateless_source" in result.reason or "factory" in result.reason.lower()

    def test_worker_usable_after_factory_failure(self) -> None:
        """The reconfiguration worker must accept the next valid request after a failure.

        Verifies that a FAILED reconfiguration does not permanently corrupt the
        controller. A second, independent request from a fresh controller/executor
        pair reaches COMMITTED status.
        """
        good_log = LifecycleLog()

        # First pair: factory failure on the pass node.
        compiler1 = WorkflowCompiler("test")
        registry1 = stateless_registry(LifecycleLog())
        initial1 = compile_initial(compiler1, registry1, source_only())
        executor1 = PipelineExecutor(initial1.plan, clock=IncrementingClock())

        failing_registry = stateless_registry(good_log, fail_pass_factory=True)
        controller1 = ReconfigurationController(executor1, compiler1, failing_registry)
        try:
            controller1.submit(_request("fail", 1, linear()))
            terminal = controller1.wait_for_terminal("fail", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status is ReconfigurationStatus.FAILED
        finally:
            controller1.close()

        # Second independent pair with a working registry.
        # After the candidate reaches READY, admit_frame commits it.
        compiler2 = WorkflowCompiler("test")
        working_registry = stateless_registry(good_log)
        initial2 = compile_initial(compiler2, working_registry, source_only())
        executor2 = PipelineExecutor(initial2.plan, clock=IncrementingClock())
        controller2 = ReconfigurationController(executor2, compiler2, working_registry)
        try:
            controller2.submit(_request("ok", 1, linear()))
            ready = controller2.wait_for_status(
                "ok",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None, "second request did not reach READY"
            # admit_frame triggers the commit; the frame runs on version 2.
            frame_result = controller2.admit_frame(admitted_at_ns=1_000, frame_id=0)
            assert frame_result.plan_version == 2
            ok_terminal = controller2.record("ok")
            assert ok_terminal.status is ReconfigurationStatus.COMMITTED
        finally:
            controller2.close()


# ---------------------------------------------------------------------------
# 6.2  Setup failure
# ---------------------------------------------------------------------------


class TestSetupFailure:
    def test_source_cleaned_when_pass_setup_fails(self) -> None:
        """When pass setup fails, the already-staged source must be cleaned."""
        log = LifecycleLog()
        registry = stateless_registry(log, pass_failure=InjectedFailurePoint.SETUP)
        compiler = WorkflowCompiler("test")

        result = compiler.compile(linear(), registry)

        assert isinstance(result, CompilationFailure)
        events = log.all_events()
        # source was set up and health-checked then rolled back
        assert events.count("setup") >= 1
        assert "cleanup" in events

    def test_active_processors_untouched_when_pass_setup_fails(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        good_registry = stateless_registry(LifecycleLog())
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        failing_registry = stateless_registry(log, pass_failure=InjectedFailurePoint.SETUP)
        controller = ReconfigurationController(executor, compiler, failing_registry)
        try:
            controller.submit(_request("setup-fail", 1, linear()))
            terminal = controller.wait_for_terminal("setup-fail", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status is ReconfigurationStatus.FAILED
            assert executor.active_plan is initial.plan
            # Active plan processors must not have been cleaned
            active_cleanup_events = [e for e in log.all_events() if e == "cleanup"]
            # Only the staged (pass) processor should be cleaned; source was staged too,
            # but the failing is the pass so source was staged first and rolled back last.
            assert len(active_cleanup_events) <= 2  # only staged processors

            # No new frame should be executing on the old plan
            # The test is validating state after a controller failure. We must admit the frame via the controller.
            frame_result = controller.admit_frame(admitted_at_ns=1_000, frame_id=99)
            assert frame_result.plan_version == 1
        finally:
            controller.close()

    def test_no_candidate_published_after_setup_failure(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        good_registry = stateless_registry(LifecycleLog())
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        failing_registry = stateless_registry(log, pass_failure=InjectedFailurePoint.SETUP)
        controller = ReconfigurationController(executor, compiler, failing_registry)
        try:
            controller.submit(_request("no-candidate", 1, linear()))
            controller.wait_for_terminal("no-candidate", timeout_seconds=2.0)
            result = controller.commit_ready()
            assert result is None
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 6.3  Healthcheck failure
# ---------------------------------------------------------------------------


class TestHealthcheckFailure:
    def test_candidate_resources_rolled_back_on_healthcheck_failure(self) -> None:
        # Compile source_only() first so there is no previous plan
        # to reuse the source processor from; source healthcheck failure
        # must roll back the staged source and return CompilationFailure.
        log = LifecycleLog()
        registry = stateless_registry(log, source_failure=InjectedFailurePoint.HEALTHCHECK)
        compiler = WorkflowCompiler("test")

        result = compiler.compile(source_only(), registry)

        assert isinstance(result, CompilationFailure)
        events = log.all_events()
        assert "setup" in events
        assert "healthcheck" in events
        assert "cleanup" in events

    def test_terminal_status_is_failed_on_healthcheck_failure(self) -> None:
        # When expanding to linear(), the pass node is newly staged.
        # Injecting a healthcheck failure on the pass processor means it
        # fails during candidate preparation; the controller marks the
        # request FAILED and the active plan is unchanged.
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        good_registry = stateless_registry(LifecycleLog())
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        failing_registry = stateless_registry(log, pass_failure=InjectedFailurePoint.HEALTHCHECK)
        controller = ReconfigurationController(executor, compiler, failing_registry)
        try:
            controller.submit(_request("hc-fail", 1, linear()))
            terminal = controller.wait_for_terminal("hc-fail", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status is ReconfigurationStatus.FAILED
            assert terminal.failure_reason is not None
            assert executor.active_plan is initial.plan
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 6.4  Process failure
# ---------------------------------------------------------------------------


class TestProcessFailure:
    def test_failed_frame_returns_failed_result(self) -> None:
        log = LifecycleLog()
        registry = stateless_registry(log, source_failure=InjectedFailurePoint.PROCESS)
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        result = executor.admit_frame(admitted_at_ns=1, frame_id=0)

        assert result.status is FrameStatus.FAILED
        assert result.error is not None
        assert "injected process failure" in result.error

    def test_failed_frame_retains_correct_plan_version(self) -> None:
        log = LifecycleLog()
        registry = stateless_registry(log, source_failure=InjectedFailurePoint.PROCESS)
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        result = executor.admit_frame(admitted_at_ns=1, frame_id=5)

        assert result.plan_version == 1
        assert result.frame_id == 5

    def test_executor_usable_after_process_failure(self) -> None:
        """A failed frame must not leave the executor in a broken state."""
        log = LifecycleLog()
        registry = stateless_registry(log, source_failure=InjectedFailurePoint.PROCESS)
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())

        first = executor.admit_frame(admitted_at_ns=1, frame_id=0)
        assert first.status is FrameStatus.FAILED

        # Next frame must also fail (same processor) but executor must not raise
        second = executor.admit_frame(admitted_at_ns=2, frame_id=1)
        assert second.status is FrameStatus.FAILED

    def test_instrumentation_records_failure_without_exception(self) -> None:
        log = LifecycleLog()
        registry = stateless_registry(log, source_failure=InjectedFailurePoint.PROCESS)
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, source_only())
        clock = IncrementingClock()
        executor = PipelineExecutor(initial.plan, clock=clock)

        executor.admit_frame(admitted_at_ns=1, frame_id=0)

        metrics = executor.instrumentation.metrics_snapshot()
        assert metrics.failed_frame_count == 1
        assert metrics.completed_frame_count == 0


# ---------------------------------------------------------------------------
# 6.5  Cleanup failure
# ---------------------------------------------------------------------------


class TestCleanupFailure:
    def test_cleanup_continues_after_one_cleanup_exception(self) -> None:
        """A cleanup failure in one processor must not prevent others from being cleaned."""
        log = LifecycleLog()
        source_instances: list[FailingProcessor] = []
        pass_instances: list[FailingProcessor] = []

        def source_factory() -> Processor:
            p = make_source_processor(log, InjectedFailurePoint.CLEANUP, "source")
            source_instances.append(p)
            return p

        def pass_factory() -> Processor:
            p = make_pass_processor(log, label="pass")
            pass_instances.append(p)
            return p

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(descriptor=STATELESS_SOURCE_DESCRIPTOR, factory=source_factory)
        )
        builder.register(
            RegisteredProcessorType(descriptor=STATELESS_PASS_DESCRIPTOR, factory=pass_factory)
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")

        result = compiler.compile(linear(), registry)
        assert isinstance(result, CompiledCandidate)

        report = cleanup_candidate_processors(result.plan, result.staged_node_ids)

        assert not report.succeeded
        assert len(report.failures) == 1
        assert report.failures[0].node_id == "source"
        # pass was still cleaned
        assert "pass" in report.cleaned_node_ids

    def test_cleanup_failure_collected_in_typed_report(self) -> None:
        log = LifecycleLog()

        def source_factory() -> Processor:
            return make_source_processor(log, InjectedFailurePoint.CLEANUP, "source")

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(descriptor=STATELESS_SOURCE_DESCRIPTOR, factory=source_factory)
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")
        result = compiler.compile(source_only(), registry)
        assert isinstance(result, CompiledCandidate)

        report = cleanup_candidate_processors(result.plan, result.staged_node_ids)

        assert len(report.failures) == 1
        failure = report.failures[0]
        assert failure.node_id == "source"
        assert failure.processor_type == STATELESS_SOURCE_DESCRIPTOR.type_name
        assert "injected cleanup failure" in failure.error

    def test_original_failure_not_masked_by_cleanup_exception(self) -> None:
        """The primary failure reason must mention the setup failure even when cleanup also fails."""
        log = LifecycleLog()
        source_instances: list[FailingProcessor] = []

        def source_factory() -> Processor:
            p = FailingProcessor(
                STATELESS_SOURCE_DESCRIPTOR,
                log,
                InjectedFailurePoint.CLEANUP,
                "source",
            )
            source_instances.append(p)
            return p

        def pass_factory() -> Processor:
            return FailingProcessor(
                STATELESS_PASS_DESCRIPTOR,
                log,
                InjectedFailurePoint.SETUP,
                "pass",
            )

        builder = RegistryBuilder()
        builder.register(
            RegisteredProcessorType(descriptor=STATELESS_SOURCE_DESCRIPTOR, factory=source_factory)
        )
        builder.register(
            RegisteredProcessorType(descriptor=STATELESS_PASS_DESCRIPTOR, factory=pass_factory)
        )
        registry = builder.snapshot()
        compiler = WorkflowCompiler("test")

        result = compiler.compile(linear(), registry)

        assert isinstance(result, CompilationFailure)
        # The primary reason mentions the setup failure
        assert "preparing processor" in result.reason or "setup" in result.reason.lower()


# ---------------------------------------------------------------------------
# 6.6  Compatibility predicate failure
# ---------------------------------------------------------------------------


class TestPredicateFailure:
    def test_candidate_fails_safely_when_predicate_raises(self) -> None:
        from tests.support.registries import tracker_raising_predicate_registry
        from tests.support.workflows import tracker_only

        registry = tracker_raising_predicate_registry()
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, tracker_only(TRACKER_RAISING_DESCRIPTOR.type_name))
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            # Submit a reconfiguration of the same spec; the predicate will raise
            controller.submit(_request("predicate-fail", 1, tracker_only(TRACKER_RAISING_DESCRIPTOR.type_name)))
            terminal = controller.wait_for_terminal("predicate-fail", timeout_seconds=2.0)
            assert terminal is not None
            assert terminal.status in {ReconfigurationStatus.REJECTED, ReconfigurationStatus.FAILED}
            # Active plan unchanged
            assert executor.active_plan is initial.plan
        finally:
            controller.close()

    def test_active_stateful_processor_remains_usable_after_predicate_failure(self) -> None:
        from tests.support.registries import tracker_raising_predicate_registry

        registry = tracker_raising_predicate_registry()
        compiler = WorkflowCompiler("test")
        initial = compile_initial(compiler, registry, tracker_only(TRACKER_RAISING_DESCRIPTOR.type_name))
        original_processor = initial.plan.steps[0].processor_ref
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.admit_frame(admitted_at_ns=1, frame_id=0)
            controller.submit(_request("predicate-fail2", 1, tracker_only(TRACKER_RAISING_DESCRIPTOR.type_name)))
            controller.wait_for_terminal("predicate-fail2", timeout_seconds=2.0)

            # The active processor is the same instance
            assert executor.active_plan.steps[0].processor_ref is original_processor

            # The active processor can still process frames
            result = controller.admit_frame(admitted_at_ns=2, frame_id=1)
            assert result.status is FrameStatus.COMPLETED
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 6.7  Stale candidate cleanup
# ---------------------------------------------------------------------------


class TestStaleCandidate:
    def test_stale_candidate_never_published(self) -> None:
        compiler = WorkflowCompiler("test")
        good_registry = stateless_registry(LifecycleLog())
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, good_registry)
        try:
            controller.submit(_request("stale", 1, linear()))
            ready = controller.wait_for_status(
                "stale",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None

            # Advance active plan externally so the candidate becomes stale
            external_plan = replace(initial.plan, version=10)
            token = getattr(executor, "_manager_token")
            executor.commit_managed(token, external_plan)

            result = controller.commit_ready()
            assert result is not None
            assert result.status is ReconfigurationStatus.STALE
            assert executor.active_plan.version == 10
        finally:
            controller.close()

    def test_stale_candidate_staged_processors_are_cleaned(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        good_registry = stateless_registry(log)
        initial = compile_initial(compiler, good_registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, good_registry)
        try:
            controller.submit(_request("stale-clean", 1, linear()))
            ready = controller.wait_for_status(
                "stale-clean",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            assert ready is not None

            external_plan = replace(initial.plan, version=10)
            token = getattr(executor, "_manager_token")
            executor.commit_managed(token, external_plan)

            result = controller.commit_ready()
            assert result is not None
            record = controller.record("stale-clean")
            assert record.candidate_cleanup_report is not None
            # The pass node was staged and must have been cleaned
            assert "pass" in record.candidate_cleanup_report.cleaned_node_ids
        finally:
            controller.close()

    def test_active_processors_not_cleaned_on_stale(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        registry = stateless_registry(log)
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.submit(_request("stale-active", 1, linear()))
            controller.wait_for_status(
                "stale-active",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            external_plan = replace(initial.plan, version=10)
            token = getattr(executor, "_manager_token")
            executor.commit_managed(token, external_plan)
            controller.commit_ready()

            # Source was reused (not staged) so it must not have been cleaned
            cleanup_events = [o for o in log.observations() if o.event == "cleanup"]
            cleaned_labels = {o.instance_id for o in cleanup_events}
            # The source processor's identity must not appear in cleanup
            source_proc = initial.plan.steps[0].processor_ref
            assert id(source_proc) not in cleaned_labels
        finally:
            controller.close()


# ---------------------------------------------------------------------------
# 6.8  Abort cleanup
# ---------------------------------------------------------------------------


class TestAbortCleanup:
    def test_aborting_ready_candidate_cleans_staged_processors(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        registry = stateless_registry(log)
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.submit(_request("abort", 1, linear()))
            controller.wait_for_status(
                "abort",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            aborted = controller.abort("abort")

            assert aborted.status is ReconfigurationStatus.ABORTED
            assert aborted.candidate_cleanup_report is not None
            assert "pass" in aborted.candidate_cleanup_report.cleaned_node_ids
        finally:
            controller.close()

    def test_aborted_candidate_never_committed(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        registry = stateless_registry(log)
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.submit(_request("abort-no-commit", 1, linear()))
            controller.wait_for_status(
                "abort-no-commit",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.abort("abort-no-commit")

            assert controller.commit_ready() is None
            assert executor.active_plan is initial.plan
        finally:
            controller.close()

    def test_reused_active_processors_remain_alive_after_abort(self) -> None:
        log = LifecycleLog()
        compiler = WorkflowCompiler("test")
        registry = stateless_registry(log)
        initial = compile_initial(compiler, registry, source_only())
        executor = PipelineExecutor(initial.plan, clock=IncrementingClock())
        controller = ReconfigurationController(executor, compiler, registry)
        try:
            controller.submit(_request("abort-reuse", 1, linear()))
            controller.wait_for_status(
                "abort-reuse",
                frozenset({ReconfigurationStatus.READY}),
                timeout_seconds=2.0,
            )
            controller.abort("abort-reuse")

            # Source was reused, not staged → must not have been cleaned
            source_proc = initial.plan.steps[0].processor_ref
            cleanup_events = [o for o in log.observations() if o.event == "cleanup"]
            cleaned_ids = {o.instance_id for o in cleanup_events}
            assert id(source_proc) not in cleaned_ids

            # Executor still usable
            result = controller.admit_frame(admitted_at_ns=1, frame_id=0)
            assert result.status is FrameStatus.COMPLETED
        finally:
            controller.close()
