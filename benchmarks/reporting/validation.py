"""Machine-readable experiment validation and loud failure enforcement (Section 11)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

from benchmarks.model import AblationSampleRow, InterferenceSampleRow
from benchmarks.reporting.ablation import calculate_ablation_summaries
from benchmarks.reporting.interference import calculate_interference_contrasts

__all__ = [
    "ExperimentValidationReport",
    "validate_experiment_artifacts",
    "write_experiment_validation_json",
]


@dataclass(frozen=True, slots=True)
class ExperimentValidationReport:
    ablation_block_completeness: bool
    ablation_pairing_completeness: bool
    ablation_accounting_valid_count: int
    interference_matched_control_completeness: bool
    raw_frame_summary_consistency_count: int
    valid_tail_latency_count: int
    duration_valid_count: int
    flow_accounting_valid_count: int
    confounded_interference_pair_count: int
    missing_derived_field_count: int
    warnings: tuple[str, ...]


def validate_experiment_artifacts(
    ablation_rows: Sequence[AblationSampleRow] = (),
    interference_rows: Sequence[InterferenceSampleRow] = (),
    expected_ablation_repetitions: int = 5,
    expected_interference_repetitions: int = 5,
) -> ExperimentValidationReport:
    """Validate ablation and interference artifact rows and enforce strict completeness."""
    warnings: list[str] = []

    # 1. Ablation Validation
    block_map: dict[str, set[str]] = defaultdict(set)
    block_rep_map: dict[str, set[int]] = defaultdict(set)

    ablation_accounting_valid_count = 0
    missing_derived_field_count = 0

    for r in ablation_rows:
        block_map[r.block_id].add(r.variant)
        block_rep_map[r.block_id].add(r.repetition)

        if r.synchronous_accounting_valid and r.synchronous_accounting_residual_ns == 0:
            ablation_accounting_valid_count += 1
        else:
            raise ValueError(
                f"Ablation row in block {r.block_id} has invalid synchronous accounting residual: {r.synchronous_accounting_residual_ns}"
            )

        if r.offpath_preparation_ns is None and r.variant in ("variant_b", "variant_d"):
            missing_derived_field_count += 1

    ablation_block_completeness = True
    for block_id, variants in block_map.items():
        if len(variants) != 4:
            ablation_block_completeness = False
            raise ValueError(
                f"Ablation block {block_id} incomplete: contains {len(variants)} variants ({variants}), expected 4"
            )

    ablation_pairing_completeness = True
    if ablation_rows:
        _, contrast_results, _ = calculate_ablation_summaries(ablation_rows)
        if not contrast_results:
            ablation_pairing_completeness = False
            raise ValueError("No ablation contrasts were successfully paired")

    # 2. Interference Validation
    interference_matched_control_completeness = True
    raw_frame_summary_consistency_count = 0
    valid_tail_latency_count = 0
    duration_valid_count = 0
    flow_accounting_valid_count = 0
    confounded_interference_pair_count = 0

    if interference_rows:
        contrast_rows, _ = calculate_interference_contrasts(interference_rows)
        if not contrast_rows:
            interference_matched_control_completeness = False
            raise ValueError("No interference matched-control contrast pairs were found")

        # Verify duplicate repetition detection
        seen_pairs: set[tuple[str, int, int]] = set()
        for r in interference_rows:
            key = (r.preparation_category, r.target_window_ns, r.repetition)
            if key in seen_pairs:
                raise ValueError(f"Duplicate interference repetition detected: {key}")
            seen_pairs.add(key)

        for c in contrast_rows:
            if c.matched_control_confounded:
                confounded_interference_pair_count += 1
                warnings.append(
                    f"Interference pair rep {c.repetition} target {c.target_window_ns}ns confounded: {c.matched_control_confounded_reason}"
                )

        for r in interference_rows:
            if r.raw_frame_count_matches_summary:
                raw_frame_summary_consistency_count += 1
            if r.valid_for_tail_latency:
                valid_tail_latency_count += 1
            if r.duration_valid:
                duration_valid_count += 1
            if r.flow_accounting_valid:
                flow_accounting_valid_count += 1

            if r.producer_skipped_deadlines_during > 0:
                warnings.append(
                    f"Repetition {r.repetition} category {r.preparation_category} skipped {r.producer_skipped_deadlines_during} deadlines"
                )

    report = ExperimentValidationReport(
        ablation_block_completeness=ablation_block_completeness,
        ablation_pairing_completeness=ablation_pairing_completeness,
        ablation_accounting_valid_count=ablation_accounting_valid_count,
        interference_matched_control_completeness=interference_matched_control_completeness,
        raw_frame_summary_consistency_count=raw_frame_summary_consistency_count,
        valid_tail_latency_count=valid_tail_latency_count,
        duration_valid_count=duration_valid_count,
        flow_accounting_valid_count=flow_accounting_valid_count,
        confounded_interference_pair_count=confounded_interference_pair_count,
        missing_derived_field_count=missing_derived_field_count,
        warnings=tuple(warnings),
    )
    return report


def write_experiment_validation_json(report: ExperimentValidationReport, output_path: str) -> None:
    """Write ExperimentValidationReport to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = asdict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
