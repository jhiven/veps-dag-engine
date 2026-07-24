"""Unit and contract tests for E5 VEPS component ablation (Sections 1, 2, 3, 4, 5, 12)."""

from __future__ import annotations

import os
import tempfile
import pytest

from benchmarks.model import (
    AblationSampleRow,
    validate_ablation_sample,
)
from benchmarks.reporting.ablation import (
    apply_holm_correction,
    calculate_ablation_summaries,
    generate_ablation_summary_files,
)
from benchmarks.storage import (
    append_ablation_rows,
    read_ablation_rows,
    write_ablation_header,
)


def test_ablation_variant_mapping() -> None:
    """A. Verify 2x2 variant mapping semantics, RCBD block fields, and validation."""
    row_a = AblationSampleRow(
        run_id="run_1",
        scenario_id="ablation_test",
        block_id="ablation_test_block_1",
        block_seed=12345,
        variant="variant_a",
        variant_name="sync_prepare_sync_retire",
        preparation_placement="synchronous",
        retirement_policy="synchronous",
        edit_type="remove_stateless_node",
        repetition=1,
        variant_order_position=1,
        candidate_graph_fingerprint="fp_123",
        preparation_workload_id="wl_remove_stateless",
        old_plan_version=1,
        new_plan_version=2,
        validation_ns=100,
        synchronous_preparation_ns=400,
        offpath_preparation_ns=None,
        boundary_wait_ns=0,
        publication_ns=100,
        synchronous_retirement_ns=200,
        deferred_retirement_ns=None,
        first_effect_wait_ns=200,
        request_to_effect_ns=1000,
        transition_output_gap_ns=500,
        total_synchronous_ns=800,  # 100 + 400 + 100 + 200 = 800
        synchronous_accounting_residual_ns=0,
        synchronous_accounting_valid=True,
        synchronous_accounting_invalid_reason=None,
        retirement_duration_ns=200,
        old_plan_frames_admitted_after_request_before_commit=0,
        peak_live_processors=10,
        random_seed=42,
        request_phase_offset_ns=150000,
    )
    validate_ablation_sample(row_a)
    assert row_a.variant_name == "sync_prepare_sync_retire"
    assert row_a.preparation_placement == "synchronous"
    assert row_a.retirement_policy == "synchronous"
    assert row_a.total_synchronous_ns == 800

    row_b = AblationSampleRow(
        run_id="run_1",
        scenario_id="ablation_test",
        block_id="ablation_test_block_1",
        block_seed=12345,
        variant="variant_b",
        variant_name="offpath_prepare_sync_retire",
        preparation_placement="off_path",
        retirement_policy="synchronous",
        edit_type="remove_stateless_node",
        repetition=1,
        variant_order_position=2,
        candidate_graph_fingerprint="fp_123",
        preparation_workload_id="wl_remove_stateless",
        old_plan_version=1,
        new_plan_version=2,
        validation_ns=None,
        synchronous_preparation_ns=None,
        offpath_preparation_ns=500,
        boundary_wait_ns=0,
        publication_ns=100,
        synchronous_retirement_ns=200,
        deferred_retirement_ns=None,
        first_effect_wait_ns=200,
        request_to_effect_ns=1000,
        transition_output_gap_ns=500,
        total_synchronous_ns=300,  # 100 + 200 = 300
        synchronous_accounting_residual_ns=0,
        synchronous_accounting_valid=True,
        synchronous_accounting_invalid_reason=None,
        retirement_duration_ns=200,
        old_plan_frames_admitted_after_request_before_commit=1,
        peak_live_processors=10,
        random_seed=42,
        request_phase_offset_ns=150000,
    )
    validate_ablation_sample(row_b)
    assert row_b.preparation_placement == "off_path"
    assert row_b.retirement_policy == "synchronous"
    assert row_b.total_synchronous_ns == 300


def test_ablation_block_integrity() -> None:
    """1, 2, 12. Verify RCBD block integrity and identical preparation work parameters."""
    rows: list[AblationSampleRow] = []
    block_seed = 999
    phase_offset = 250000
    fingerprint = "fp_test_block"
    workload_id = "wl_test_block"

    for pos, var in enumerate(["variant_a", "variant_b", "variant_c", "variant_d"], start=1):
        vname = {
            "variant_a": "sync_prepare_sync_retire",
            "variant_b": "offpath_prepare_sync_retire",
            "variant_c": "sync_prepare_deferred_retire",
            "variant_d": "offpath_prepare_deferred_retire",
        }[var]
        prep = "synchronous" if var in ("variant_a", "variant_c") else "off_path"
        ret = "synchronous" if var in ("variant_a", "variant_b") else "deferred"
        pub_ns = 100
        val_ns = 100 if prep == "synchronous" else None
        sync_prep_ns = 400 if prep == "synchronous" else None
        sync_ret_ns = 200 if ret == "synchronous" else None
        tot_sync = (val_ns or 0) + (sync_prep_ns or 0) + pub_ns + (sync_ret_ns or 0)

        rows.append(
            AblationSampleRow(
                run_id="run_block",
                scenario_id="scen_block",
                block_id="scen_block_block_1",
                block_seed=block_seed,
                variant=var,
                variant_name=vname,
                preparation_placement=prep,
                retirement_policy=ret,
                edit_type="remove_stateless",
                repetition=1,
                variant_order_position=pos,
                candidate_graph_fingerprint=fingerprint,
                preparation_workload_id=workload_id,
                old_plan_version=1,
                new_plan_version=2,
                validation_ns=val_ns,
                synchronous_preparation_ns=sync_prep_ns,
                publication_ns=pub_ns,
                synchronous_retirement_ns=sync_ret_ns,
                total_synchronous_ns=tot_sync,
                synchronous_accounting_residual_ns=0,
                random_seed=42,
                request_phase_offset_ns=phase_offset,
            )
        )

    # All 4 variants in block must share block_seed, phase_offset, fingerprint, workload_id
    assert len({r.block_seed for r in rows}) == 1
    assert len({r.request_phase_offset_ns for r in rows}) == 1
    assert len({r.candidate_graph_fingerprint for r in rows}) == 1
    assert len({r.preparation_workload_id for r in rows}) == 1

    # Order positions must be unique and span 1..4
    order_positions = [r.variant_order_position for r in rows]
    assert sorted(order_positions) == [1, 2, 3, 4]


def test_synchronous_work_accounting_and_residual() -> None:
    """3. Verify phase-level accounting residual and invalid accounting handling."""
    # Correct accounting row
    row_valid = AblationSampleRow(
        run_id="run_1",
        scenario_id="ablation_test",
        block_id="b1",
        block_seed=1,
        variant="variant_a",
        variant_name="sync_prepare_sync_retire",
        preparation_placement="synchronous",
        retirement_policy="synchronous",
        edit_type="remove_stateless",
        repetition=1,
        variant_order_position=1,
        candidate_graph_fingerprint="fp1",
        preparation_workload_id="wl1",
        old_plan_version=1,
        new_plan_version=2,
        validation_ns=100,
        synchronous_preparation_ns=400,
        publication_ns=100,
        synchronous_retirement_ns=200,
        total_synchronous_ns=800,  # sum = 800
        synchronous_accounting_residual_ns=0,
        synchronous_accounting_valid=True,
    )
    validate_ablation_sample(row_valid)

    # Mismatched residual must raise ValueError
    with pytest.raises(ValueError, match="synchronous_accounting_residual_ns"):
        validate_ablation_sample(
            AblationSampleRow(
                run_id="run_1",
                scenario_id="ablation_test",
                block_id="b1",
                block_seed=1,
                variant="variant_a",
                variant_name="sync_prepare_sync_retire",
                preparation_placement="synchronous",
                retirement_policy="synchronous",
                edit_type="remove_stateless",
                repetition=1,
                variant_order_position=1,
                candidate_graph_fingerprint="fp1",
                preparation_workload_id="wl1",
                old_plan_version=1,
                new_plan_version=2,
                validation_ns=100,
                synchronous_preparation_ns=400,
                publication_ns=100,
                synchronous_retirement_ns=200,
                total_synchronous_ns=800,
                synchronous_accounting_residual_ns=50,  # Invalid! Residual must be 0
                synchronous_accounting_valid=False,
            )
        )


def test_planned_ablation_contrasts_pairing() -> None:
    """4, 12. Verify contrast pairing enforces identical request offset and workload ID."""
    rows: list[AblationSampleRow] = []
    variants = [
        ("variant_a", "sync_prepare_sync_retire", "synchronous", "synchronous", 800),
        ("variant_b", "offpath_prepare_sync_retire", "off_path", "synchronous", 300),
        ("variant_c", "sync_prepare_deferred_retire", "synchronous", "deferred", 600),
        ("variant_d", "offpath_prepare_deferred_retire", "off_path", "deferred", 100),
    ]

    for rep in range(1, 11):
        for var, vname, prep, ret, base_sync in variants:
            rows.append(
                AblationSampleRow(
                    run_id="run_test",
                    scenario_id="scen_1",
                    block_id=f"scen_1_block_{rep}",
                    block_seed=100 + rep,
                    variant=var,
                    variant_name=vname,
                    preparation_placement=prep,
                    retirement_policy=ret,
                    edit_type="remove_stateless",
                    repetition=rep,
                    variant_order_position=1,
                    candidate_graph_fingerprint="fp_same",
                    preparation_workload_id="wl_same",
                    old_plan_version=1,
                    new_plan_version=2,
                    request_to_effect_ns=1000 + rep * 10,
                    transition_output_gap_ns=500,
                    total_synchronous_ns=base_sync + rep,
                    publication_ns=100,
                    synchronous_preparation_ns=400 if prep == "synchronous" else None,
                    synchronous_retirement_ns=200 if ret == "synchronous" else None,
                    synchronous_accounting_residual_ns=0,
                    old_plan_frames_admitted_after_request_before_commit=1 if var in ("variant_b", "variant_d") else 0,
                    peak_live_processors=10,
                    random_seed=42,
                    request_phase_offset_ns=150000,
                )
            )

    _, contrasts, _ = calculate_ablation_summaries(rows)
    contrast_ids = {c.contrast_id for c in contrasts}
    assert contrast_ids == {"A_vs_B", "C_vs_D", "A_vs_C", "B_vs_D"}

    ab_contrast = next(c for c in contrasts if c.contrast_id == "A_vs_B" and c.metric == "total_synchronous_ns")
    assert ab_contrast.count == 10
    assert ab_contrast.paired_median_difference < 0
    assert ab_contrast.family in ("preparation_placement", "retirement_policy")
    assert ab_contrast.holm_p_value >= ab_contrast.raw_p_value


def test_holm_correction() -> None:
    """Verify Holm-Bonferroni p-value adjustment logic."""
    raw_p = [0.01, 0.04, 0.03, 0.001]
    adj = apply_holm_correction(raw_p)
    assert len(adj) == 4
    assert adj[3] <= adj[0] <= adj[2] <= adj[1]
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_stratified_variant_d_summary() -> None:
    """5. Verify stratified request-to-effect summary for Variant D (0 vs >= 1 frames)."""
    rows: list[AblationSampleRow] = []
    for rep in range(1, 6):
        rows.append(
            AblationSampleRow(
                run_id="run_d",
                scenario_id="scen_d",
                block_id=f"b_{rep}",
                block_seed=rep,
                variant="variant_d",
                variant_name="offpath_prepare_deferred_retire",
                preparation_placement="off_path",
                retirement_policy="deferred",
                edit_type="remove_stateless",
                repetition=rep,
                variant_order_position=1,
                candidate_graph_fingerprint="fp_d",
                preparation_workload_id="wl_d",
                old_plan_version=1,
                new_plan_version=2,
                request_to_effect_ns=100,
                transition_output_gap_ns=50,
                total_synchronous_ns=20,
                publication_ns=20,
                old_plan_frames_admitted_after_request_before_commit=0,
            )
        )
    for rep in range(6, 11):
        rows.append(
            AblationSampleRow(
                run_id="run_d",
                scenario_id="scen_d",
                block_id=f"b_{rep}",
                block_seed=rep,
                variant="variant_d",
                variant_name="offpath_prepare_deferred_retire",
                preparation_placement="off_path",
                retirement_policy="deferred",
                edit_type="remove_stateless",
                repetition=rep,
                variant_order_position=1,
                candidate_graph_fingerprint="fp_d",
                preparation_workload_id="wl_d",
                old_plan_version=1,
                new_plan_version=2,
                request_to_effect_ns=1000,
                transition_output_gap_ns=500,
                total_synchronous_ns=20,
                publication_ns=20,
                old_plan_frames_admitted_after_request_before_commit=2,
            )
        )

    _, _, bimodal = calculate_ablation_summaries(rows)
    assert len(bimodal) == 2
    b0 = next(b for b in bimodal if b.frame_group == "zero_frames")
    b1 = next(b for b in bimodal if b.frame_group == "one_or_more_frames")

    assert b0.count == 5
    assert b0.request_to_effect_median_ns == 100.0
    assert b1.count == 5
    assert b1.request_to_effect_median_ns == 1000.0


def test_ablation_csv_roundtrip() -> None:
    """Roundtrip CSV storage for AblationSampleRow and summary file generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = f"{tmp_dir}/ablation-samples.csv"
        write_ablation_header(csv_path)

        row = AblationSampleRow(
            run_id="run_rt",
            scenario_id="scen_rt",
            block_id="block_rt_1",
            block_seed=54321,
            variant="variant_a",
            variant_name="sync_prepare_sync_retire",
            preparation_placement="synchronous",
            retirement_policy="synchronous",
            edit_type="edit_rt",
            repetition=1,
            variant_order_position=2,
            candidate_graph_fingerprint="fp_rt",
            preparation_workload_id="wl_rt",
            old_plan_version=1,
            new_plan_version=2,
            validation_ns=50,
            synchronous_preparation_ns=350,
            publication_ns=100,
            synchronous_retirement_ns=200,
            total_synchronous_ns=700,
            synchronous_accounting_residual_ns=0,
            synchronous_accounting_valid=True,
            random_seed=42,
            request_phase_offset_ns=150000,
        )
        append_ablation_rows(csv_path, [row])
        read_rows = read_ablation_rows(csv_path)
        assert len(read_rows) == 1
        r = read_rows[0]
        assert r.block_id == "block_rt_1"
        assert r.block_seed == 54321
        assert r.variant_order_position == 2
        assert r.candidate_graph_fingerprint == "fp_rt"
        assert r.preparation_workload_id == "wl_rt"
        assert r.total_synchronous_ns == 700

        generate_ablation_summary_files(csv_path, tmp_dir)
        assert os.path.exists(f"{tmp_dir}/ablation-summary.csv")
        assert os.path.exists(f"{tmp_dir}/ablation-contrasts.csv")
        assert os.path.exists(f"{tmp_dir}/ablation-contrast-summary.csv")
        assert os.path.exists(f"{tmp_dir}/ablation-variant-d-stratified-summary.csv")
