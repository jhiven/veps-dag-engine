"""Schema-2 raw conformance traces and offline replay validation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

from nedo_vision_dag_engine.instrumentation import JsonValue

SCHEMA_VERSION = "2.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "ConformanceTrace",
    "ReplayReport",
    "replay_trace",
    "validate_conformance_directory",
]


class ConformanceTrace:
    """Thread-safe append-only writer for one campaign and seed."""

    __slots__ = ("_campaign_id", "_file", "_lock", "_run_id", "_seed", "_sequence")

    def __init__(self, path: str, run_id: str, campaign_id: str, seed: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "x", encoding="utf-8")
        self._run_id = run_id
        self._campaign_id = campaign_id
        self._seed = seed
        self._sequence = 1
        self._lock = Lock()

    def emit(self, kind: str, fields: Mapping[str, JsonValue]) -> None:
        with self._lock:
            event: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self._run_id,
                "campaign_id": self._campaign_id,
                "seed": self._seed,
                "event_sequence": self._sequence,
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "event_kind": kind,
            }
            event.update(fields)
            self._file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            self._file.flush()
            self._sequence += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> ConformanceTrace:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ReplayReport:
    path: str
    campaign_id: str
    seed: int
    event_count: int
    counts: dict[str, int]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def replay_trace(path: str) -> ReplayReport:
    """Derive counts and ordering invariants exclusively from a JSONL trace."""
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                raw: Any = json.loads(line)
            except Exception as error:
                errors.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(raw, dict):
                errors.append(f"line {line_number}: event must be a JSON object")
                continue
            events.append(cast(dict[str, Any], raw))

    campaign_id = str(events[0].get("campaign_id", "")) if events else ""
    seed = int(events[0].get("seed", -1)) if events else -1
    previous_timestamp = -1
    counts: dict[str, int] = {}
    offered: dict[int, int] = {}
    admitted: dict[int, int] = {}
    terminal: dict[int, str] = {}
    leases: set[tuple[int, int]] = set()
    node_entries: set[tuple[int, str, str]] = set()
    campaign_completed = False

    for index, event in enumerate(events, 1):
        if event.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"event {index}: unsupported schema_version")
        if event.get("event_sequence") != index:
            errors.append(f"event {index}: missing or non-contiguous event_sequence")
        timestamp = event.get("monotonic_timestamp_ns")
        if not isinstance(timestamp, int) or timestamp < previous_timestamp:
            errors.append(f"event {index}: non-monotonic timestamp")
        else:
            previous_timestamp = timestamp
        if event.get("campaign_id") != campaign_id or event.get("seed") != seed:
            errors.append(f"event {index}: campaign/seed envelope changed")

        kind = str(event.get("event_kind", ""))
        counts[kind] = counts.get(kind, 0) + 1
        frame_raw = event.get("frame_id")
        frame_id = frame_raw if isinstance(frame_raw, int) else None
        plan_raw = event.get("plan_id")
        plan_id = plan_raw if isinstance(plan_raw, int) else None

        if kind == "frame_offered" and frame_id is not None:
            if frame_id in offered:
                errors.append(f"frame {frame_id}: duplicate offer")
            offered[frame_id] = index
        elif kind == "frame_admitted" and frame_id is not None:
            if frame_id in admitted:
                errors.append(f"frame {frame_id}: duplicate admission")
            admitted[frame_id] = index
        elif kind in {"frame_completed", "frame_failed", "frame_cancelled"} and frame_id is not None:
            if frame_id in terminal:
                errors.append(f"frame {frame_id}: duplicate terminal outcome")
            terminal[frame_id] = kind
        elif kind == "lease_acquired" and frame_id is not None and plan_id is not None:
            key = (frame_id, plan_id)
            if key in leases:
                errors.append(f"frame {frame_id}: duplicate lease acquisition")
            leases.add(key)
        elif kind == "lease_released" and frame_id is not None and plan_id is not None:
            key = (frame_id, plan_id)
            if key not in leases:
                errors.append(f"frame {frame_id}: lease release without acquisition")
            else:
                leases.remove(key)
        elif kind == "node_entered" and frame_id is not None:
            key = (frame_id, str(event.get("node_id", "")), str(event.get("processor_instance_id", "")))
            if key in node_entries:
                errors.append(f"frame {frame_id}: duplicate node entry {key[1]!r}")
            node_entries.add(key)
        elif kind == "node_exited" and frame_id is not None:
            key = (frame_id, str(event.get("node_id", "")), str(event.get("processor_instance_id", "")))
            if key not in node_entries:
                errors.append(f"frame {frame_id}: node exit without entry {key[1]!r}")
            else:
                node_entries.remove(key)
        elif kind == "campaign_completed":
            campaign_completed = True
            if event.get("outcome") != "passed":
                errors.append("campaign terminal outcome is not passed")

    for frame_id, admission_index in admitted.items():
        offer_index = offered.get(frame_id)
        if offer_index is None or offer_index > admission_index:
            errors.append(f"frame {frame_id}: admission is not preceded by an offer")
        if frame_id not in terminal:
            errors.append(f"frame {frame_id}: admitted without terminal outcome")
    if leases:
        errors.append(f"unreleased leases: {sorted(leases)!r}")
    if node_entries:
        errors.append(f"unclosed node executions: {sorted(node_entries)!r}")
    if not campaign_completed:
        errors.append("campaign_completed event is missing")

    return ReplayReport(path, campaign_id, seed, len(events), counts, tuple(errors))


def validate_conformance_directory(
    path: str,
    required_campaigns: frozenset[str],
    randomized_seeds: tuple[int, ...],
) -> tuple[ReplayReport, ...]:
    trace_paths = sorted(
        os.path.join(path, name) for name in os.listdir(path) if name.endswith(".jsonl")
    )
    reports = tuple(replay_trace(trace_path) for trace_path in trace_paths)
    failures = [report for report in reports if not report.passed]
    if failures:
        details = "; ".join(f"{r.path}: {r.errors}" for r in failures)
        raise ValueError(f"conformance replay failed: {details}")
    present = {report.campaign_id for report in reports}
    missing = required_campaigns.difference(present)
    if missing:
        raise ValueError(f"required conformance campaigns are missing: {sorted(missing)!r}")
    observed_random_seeds = {
        report.seed for report in reports if report.campaign_id == "randomized_consistency"
    }
    if observed_random_seeds != set(randomized_seeds):
        raise ValueError(
            "randomized conformance seeds mismatch: "
            f"expected {list(randomized_seeds)!r}, got {sorted(observed_random_seeds)!r}"
        )
    return reports
