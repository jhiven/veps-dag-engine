"""Controlled schedules must fail replay when their decisive event is lost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.conformance_trace import replay_trace
from benchmarks.runners.conformance import run_conformance_suite


@pytest.fixture(scope="module")
def trace_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("conformance-oracles")
    run_conformance_suite("oracle-test", str(directory), "smoke", (42,))
    return directory


@pytest.mark.parametrize(
    ("campaign", "missing_kind", "expected_error"),
    (
        ("stateless_publication", "plan_published", "old lease, publication"),
        ("mutable_handoff", "handoff_gate_closed", "gate close"),
        ("failure_atomicity", "request_rejected", "wrong terminal outcome"),
        ("candidate_failure", "request_committed", "valid publication"),
        ("cleanup_failure", "processor_cleanup_failed", "cleanup"),
        ("frame_exception", "frame_failed", "frame failure"),
        ("state_policy", "processor_reset", "preserve/reset"),
        ("successive_generation_ownership", "plan_published", "successive generation"),
        ("randomized_consistency", "frame_completed", "randomized frame target"),
    ),
)
def test_decisive_event_is_required(
    trace_directory: Path,
    tmp_path: Path,
    campaign: str,
    missing_kind: str,
    expected_error: str,
) -> None:
    source = trace_directory / f"{campaign}-seed-42.jsonl"
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    removed = False
    damaged: list[dict[str, object]] = []
    for event in events:
        if not removed and event["event_kind"] == missing_kind:
            removed = True
            continue
        event["event_sequence"] = len(damaged) + 1
        damaged.append(event)
    assert removed
    path = tmp_path / source.name
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in damaged),
        encoding="utf-8",
    )
    report = replay_trace(str(path))
    assert not report.passed
    assert any(expected_error in error for error in report.errors), report.errors


def test_node_execution_must_use_its_plans_bound_instance(
    trace_directory: Path, tmp_path: Path
) -> None:
    """A frame that runs another plan's processor instance is a mixed-plan frame."""
    source = trace_directory / "stateless_publication-seed-42.jsonl"
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    bound_to_plan_one = {
        event["processor_instance_id"]
        for event in events
        if event["event_kind"] == "processor_staged" and event.get("plan_id") == 1
    }
    foreign = "0xdeadbeef"
    assert foreign not in bound_to_plan_one
    corrupted = False
    original: object = None
    for event in events:
        if (not corrupted and event["event_kind"] == "node_entered"
                and event.get("plan_id") == 1):
            original = event["processor_instance_id"]
            event["processor_instance_id"] = foreign
            corrupted = True
        elif corrupted and event["event_kind"] == "node_exited" and event.get("plan_id") == 1 \
                and event.get("processor_instance_id") == original:
            event["processor_instance_id"] = foreign
            break
    assert corrupted
    path = tmp_path / source.name
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    report = replay_trace(str(path))
    assert not report.passed
    assert any("does not bind" in error for error in report.errors), report.errors
