"""Benchmark validation gate — fails before reporting if any requirement is violated.

Run as: python -m benchmarks.validation_gate <run_directory>
Exit code 0 = PASS, exit code 1 = FAIL (with JSON report).
"""

from __future__ import annotations

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    detail: str = ""
    count_expected: int = 0
    count_actual: int = 0


@dataclass
class ValidationReport:
    run_directory: str
    results: list[ValidationResult] = field(default_factory=list[ValidationResult])  # type: ignore[arg-type]
    errors: list[str] = field(default_factory=list[str])  # type: ignore[arg-type]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_directory": self.run_directory,
            "all_passed": self.all_passed,
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "errors": self.errors,
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "expected": r.count_expected,
                    "actual": r.count_actual,
                }
                for r in self.results
            ],
        }


def _add(report: ValidationReport, name: str, passed: bool, detail: str = "",
         expected: int = 0, actual: int = 0) -> None:
    report.results.append(ValidationResult(name, passed, detail, expected, actual))


# ---------------------------------------------------------------------------
# Per-CSV validators
# ---------------------------------------------------------------------------


def _validate_steady_state(report: ValidationReport, run_dir: str) -> None:
    path = os.path.join(run_dir, "steady-state-samples.csv")
    if not os.path.exists(path):
        _add(report, "steady_state_exists", False, f"Missing: {path}")
        return
    _add(report, "steady_state_exists", True, path)

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        report.errors.append(f"steady-state CSV read error: {e}")
        return

    _add(report, "steady_state_not_empty", len(rows) > 0, f"{len(rows)} rows")

    # Check for duplicate keys (execution_order_position disambiguates same-rep rows)
    keys: set[tuple[str, str, str, str, int, int]] = set()
    dup_count = 0
    for r in rows:
        key = (r["run_id"], r["scenario_id"], r["topology"], r["workload_id"],
               int(r["repetition"]), int(r["execution_order_position"]))
        if key in keys:
            dup_count += 1
        keys.add(key)
    _add(report, "steady_state_no_duplicate_keys", dup_count == 0,
         f"{dup_count} duplicate composite keys", 0, dup_count)

    # Check run_id consistency
    run_ids = {r["run_id"] for r in rows}
    _add(report, "steady_state_single_run_id", len(run_ids) <= 1,
         f"Run IDs: {run_ids}", 1, len(run_ids))


def _validate_reconfiguration(report: ValidationReport, run_dir: str) -> None:
    path = os.path.join(run_dir, "reconfiguration-samples.csv")
    if not os.path.exists(path):
        _add(report, "reconfiguration_exists", False, f"Missing: {path}")
        return
    _add(report, "reconfiguration_exists", True, path)

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        report.errors.append(f"reconfiguration CSV read error: {e}")
        return

    _add(report, "reconfiguration_not_empty", len(rows) > 0, f"{len(rows)} rows")

    # Check for duplicate keys
    keys: set[tuple[str, str, str, str, int, int]] = set()
    dup_count = 0
    for r in rows:
        key = (r["run_id"], r["scenario_id"], r["baseline"], r["edit_type"],
               int(r["graph_size"]), int(r["repetition"]))
        if key in keys:
            dup_count += 1
        keys.add(key)
    _add(report, "reconfiguration_no_duplicate_keys", dup_count == 0,
         f"{dup_count} duplicate composite keys", 0, dup_count)

    # Check that all baselines are present for each scenario+size+rep+edit
    from collections import defaultdict
    baseline_groups: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
    for r in rows:
        gkey: tuple[str, int, int, str] = (
            r["scenario_id"], int(r["graph_size"]),
            int(r["repetition"]), r["edit_type"],
        )
        baseline_groups[gkey].add(r["baseline"])
    incomplete = sum(
        1 for g, bs in baseline_groups.items()
        if len(bs) < 3 and not str(g[0]).startswith("graph_size_sensitivity_5")
    )
    _add(report, "reconfiguration_all_baselines_present", incomplete == 0,
         f"{incomplete} groups missing baselines", 0, incomplete)

    # Check frame accounting: no duplicated frames
    dup_frames = sum(int(r.get("frames_duplicated", 0)) for r in rows)
    _add(report, "reconfiguration_no_duplicated_frames", dup_frames == 0,
         f"Total duplicated frames: {dup_frames}", 0, dup_frames)

    # Check synchronous accounting residuals for variants that require zero
    residual_violations = 0
    for r in rows:
        baseline = r.get("baseline", "")
        if baseline == "prepare_and_commit":
            # phases_may_overlap must be True
            if r.get("phases_may_overlap", "").lower() != "true":
                residual_violations += 1
        elif baseline in ("stop_rebuild_restart", "pause_compile_resume"):
            if r.get("phases_may_overlap", "").lower() != "false":
                residual_violations += 1
    _add(report, "reconfiguration_phases_may_overlap_valid", residual_violations == 0,
         f"{residual_violations} rows with invalid phases_may_overlap", 0, residual_violations)


def _validate_conformance(report: ValidationReport, run_dir: str) -> None:
    path = os.path.join(run_dir, "conformance-results.csv")
    if not os.path.exists(path):
        _add(report, "conformance_exists", False, f"Missing: {path}")
        return
    _add(report, "conformance_exists", True, path)

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        report.errors.append(f"conformance CSV read error: {e}")
        return

    _add(report, "conformance_not_empty", len(rows) > 0, f"{len(rows)} rows")

    # Check terminal status
    non_pass = [r for r in rows if r.get("terminal_status", "") != "PASS"]
    _add(report, "conformance_all_pass", len(non_pass) == 0,
         f"{len(non_pass)} non-PASS rows: {[r['campaign_id'] for r in non_pass]}", 0, len(non_pass))

    # Required campaigns must be present
    required_campaigns = [
        "frame_consistency_stress",
        "failure_atomicity",
        "stateful_conformance",
        "grace_period_deterministic",
        "stateful_handoff_deterministic",
        "candidate_memory_failure",
        "cleanup_failure",
        "frame_exception_lease_release",
    ]
    present_campaigns = {r["campaign_id"] for r in rows}
    missing_campaigns = [c for c in required_campaigns if c not in present_campaigns]
    _add(report, "conformance_all_campaigns_present", len(missing_campaigns) == 0,
         f"Missing: {missing_campaigns}", len(required_campaigns),
         len(required_campaigns) - len(missing_campaigns))

    # Check specific counters
    for r in rows:
        cid = r.get("campaign_id", "")
        # grace_period_safety_violations must be 0
        gp = int(r.get("grace_period_safety_violations", 0))
        if gp > 0:
            _add(report, f"conformance_{cid}_grace_period_violations", False,
                 f"{gp} grace period safety violations in {cid}", 0, gp)
        # stateful_handoff_ordering_failures must be 0
        sh = int(r.get("stateful_handoff_ordering_failures", 0))
        if sh > 0:
            _add(report, f"conformance_{cid}_handoff_failures", False,
                 f"{sh} handoff ordering failures in {cid}", 0, sh)


def _validate_realworld_video(report: ValidationReport, run_dir: str) -> None:
    path = os.path.join(run_dir, "realworld-video-samples.csv")
    if not os.path.exists(path):
        _add(report, "realworld_video_exists", False, f"Missing: {path}")
        return
    _add(report, "realworld_video_exists", True, path)

    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        report.errors.append(f"realworld-video CSV read error: {e}")
        return

    _add(report, "realworld_video_not_empty", len(rows) > 0, f"{len(rows)} rows")

    # Check all mechanisms present per repetition
    from collections import defaultdict
    mech_groups: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        mech_groups[int(r["repetition"])].add(r["mechanism"])
    missing_mech = sum(1 for mechs in mech_groups.values()
                       if mechs != {"Stop", "Pause", "VEPS"})
    _add(report, "realworld_video_all_mechanisms_per_rep", missing_mech == 0,
         f"{missing_mech} repetitions with missing mechanisms", 0, missing_mech)

    # Check fixed-window accounting validity (if the new fields exist)
    if "fixed_window_accounting_valid" in rows[0] if rows else False:
        invalid_fw = sum(1 for r in rows
                         if r.get("fixed_window_accounting_valid", "True").lower() != "true")
        _add(report, "realworld_video_fixed_window_accounting_valid", invalid_fw == 0,
             f"{invalid_fw} rows with invalid fixed-window accounting", 0, invalid_fw)

    # Check no duplicated frames
    total_dup = sum(int(r.get("duplicated_frame_count", 0)) for r in rows)
    _add(report, "realworld_video_no_duplicated_frames", total_dup == 0,
         f"Total duplicated frames: {total_dup}", 0, total_dup)

    # Check all source frames and admitted/completed/dropped accounting
    accounting_issues = 0
    for r in rows:
        source = int(r.get("source_frames_received", 0))
        admitted = int(r.get("frames_admitted", 0))
        dropped = int(r.get("total_dropped_frame_count", 0))
        completed = int(r.get("frames_completed", 0))
        in_flight = admitted - completed
        if source != admitted + dropped + in_flight:
            accounting_issues += 1
    _add(report, "realworld_video_frame_accounting", accounting_issues == 0,
         f"{accounting_issues} rows with frame accounting mismatch", 0, accounting_issues)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_benchmark_directory(run_dir: str) -> ValidationReport:
    """Run all validation checks and return a structured report."""
    report = ValidationReport(run_directory=run_dir)

    if not os.path.isdir(run_dir):
        report.errors.append(f"Not a directory: {run_dir}")
        return report

    # Check required files exist
    for fname in ("run.json", "completion.json"):
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            report.errors.append(f"Missing: {fname}")
        else:
            _add(report, f"metadata_{fname.replace('.json', '')}_exists", True, path)

    # Check no failure.json
    fail_path = os.path.join(run_dir, "failure.json")
    if os.path.exists(fail_path):
        _add(report, "no_failure_json", False, "failure.json present — run failed")

    # Per-CSV validation
    _validate_steady_state(report, run_dir)
    _validate_reconfiguration(report, run_dir)
    _validate_conformance(report, run_dir)
    _validate_realworld_video(report, run_dir)

    return report


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python -m benchmarks.validation_gate <run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]
    report = validate_benchmark_directory(run_dir)

    # Write JSON report
    json_path = os.path.join(run_dir, "validation-gate.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Print summary
    d: dict[str, Any] = report.to_dict()
    print(f"Validation Gate: {run_dir}")
    print(f"  All passed: {d['all_passed']}")
    print(f"  Checks: {d['passed_checks']}/{d['total_checks']} passed, "
          f"{d['failed_checks']} failed, {len(d['errors'])} errors")
    if d["errors"]:
        print("  Errors:")
        for e in d["errors"]:
            print(f"    - {e}")
    checks: list[dict[str, Any]] = d.get("checks", [])
    for c in checks:
        if not c.get("passed", True):
            print(f"  FAIL: {c.get('name', '?')} — {c.get('detail', '?')}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
