"""Execution-level checks for the controller-based factorial ablation."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from benchmarks.runners.ablation import _run_ablation_repetition, run_ablation_suite  # pyright: ignore[reportPrivateUsage]
from nedo_vision_dag_engine.executor import PipelineExecutor


@pytest.mark.parametrize(
    "edit_type",
    ("remove_stateless_node", "compatible_edit_preserving_tracker"),
)
@pytest.mark.parametrize("variant", ("variant_a", "variant_b", "variant_c", "variant_d"))
def test_every_ablation_variant_uses_managed_publication(
    edit_type: str, variant: str,
) -> None:
    # The uncorrected A-C variants used the bare executor.commit() path.
    with patch.object(PipelineExecutor, "commit", side_effect=AssertionError("bare commit")):
        row = _run_ablation_repetition(
            "probe", "scenario", "block", 42, variant, 1, edit_type, 1
        )
    assert row.old_plan_version == 1
    assert row.new_plan_version == 2
    assert row.synchronous_accounting_valid
    assert row.request_to_effect_ns > 0
    assert row.transition_output_gap_ns > 0


def test_ablation_order_is_seeded_and_position_balanced() -> None:
    with TemporaryDirectory() as left, TemporaryDirectory() as right:
        rows_left = run_ablation_suite(left, "probe", repetitions=4, random_seed=42)
        rows_right = run_ablation_suite(right, "probe", repetitions=4, random_seed=42)
    labels_left = tuple(
        (row.scenario_id, row.repetition, row.variant_order_position, row.variant, row.block_seed)
        for row in rows_left
    )
    labels_right = tuple(
        (row.scenario_id, row.repetition, row.variant_order_position, row.variant, row.block_seed)
        for row in rows_right
    )
    assert labels_left == labels_right
    for scenario_id in {row.scenario_id for row in rows_left}:
        for variant in ("variant_a", "variant_b", "variant_c", "variant_d"):
            positions = {
                row.variant_order_position for row in rows_left
                if row.scenario_id == scenario_id and row.variant == variant
            }
            assert positions == {1, 2, 3, 4}
