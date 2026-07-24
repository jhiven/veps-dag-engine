"""CSV schema contract tests and metadata verification (Section 12H)."""

from __future__ import annotations

from benchmarks.model import (
    ABLATION_CONTRAST_HEADERS,
    ABLATION_HEADERS,
    ABLATION_STRATIFIED_HEADERS,
    CONFORMANCE_HEADERS,
    INTERFERENCE_AGGREGATE_CONTRAST_HEADERS,
    INTERFERENCE_CONTRAST_HEADERS,
    INTERFERENCE_FRAME_HEADERS,
    INTERFERENCE_HEADERS,
    MEMORY_HEADERS,
    RECONFIGURATION_HEADERS,
    STEADY_STATE_HEADERS,
)
from benchmarks.system import (
    SystemEnvironment,
    collect_system_environment,
    is_free_threaded_build,
    is_gil_enabled,
)


def test_system_metadata_collection() -> None:
    """Verify runtime metadata records interpreter configuration without treatment columns."""
    env = collect_system_environment()
    assert isinstance(env, SystemEnvironment)
    assert isinstance(env.python_version, str)
    assert isinstance(env.python_implementation, str)
    assert isinstance(env.free_threaded_build, bool)
    assert isinstance(env.gil_enabled, bool)
    assert isinstance(env.operating_system, str)
    assert isinstance(env.cpu_model, str)

    assert env.free_threaded_build == is_free_threaded_build()
    assert env.gil_enabled == is_gil_enabled()

    # Verify gil_enabled is NOT an experimental column in sample headers
    assert "gil_enabled" not in ABLATION_HEADERS
    assert "gil_enabled" not in INTERFERENCE_HEADERS
    assert "gil_enabled" not in RECONFIGURATION_HEADERS


def test_csv_schema_headers_contract() -> None:
    """Verify required CSV headers, order, and column names across all benchmark outputs."""
    # Steady State
    assert "normalized_ns_per_frame" in STEADY_STATE_HEADERS
    assert "plan_version" in STEADY_STATE_HEADERS

    # Reconfiguration
    assert "request_to_effect_ns" in RECONFIGURATION_HEADERS
    assert "total_synchronous_ns" in RECONFIGURATION_HEADERS

    # Ablation 2x2 RCBD design
    assert "block_id" in ABLATION_HEADERS
    assert "block_seed" in ABLATION_HEADERS
    assert "variant" in ABLATION_HEADERS
    assert "variant_name" in ABLATION_HEADERS
    assert "preparation_placement" in ABLATION_HEADERS
    assert "retirement_policy" in ABLATION_HEADERS
    assert "variant_order_position" in ABLATION_HEADERS
    assert "candidate_graph_fingerprint" in ABLATION_HEADERS
    assert "preparation_workload_id" in ABLATION_HEADERS
    assert "validation_ns" in ABLATION_HEADERS
    assert "synchronous_preparation_ns" in ABLATION_HEADERS
    assert "publication_ns" in ABLATION_HEADERS
    assert "synchronous_retirement_ns" in ABLATION_HEADERS
    assert "deferred_retirement_ns" in ABLATION_HEADERS
    assert "synchronous_accounting_residual_ns" in ABLATION_HEADERS
    assert "synchronous_accounting_valid" in ABLATION_HEADERS

    # Ablation Summaries
    assert "contrast" in ABLATION_CONTRAST_HEADERS
    assert "holm_adjusted_p_value" in ABLATION_CONTRAST_HEADERS
    assert "frame_group" in ABLATION_STRATIFIED_HEADERS

    # Memory Campaign
    assert "rss_before_bytes" in MEMORY_HEADERS
    assert "rss_after_candidate_prepare_bytes" in MEMORY_HEADERS
    assert "rss_after_commit_bytes" in MEMORY_HEADERS
    assert "rss_after_retirement_bytes" in MEMORY_HEADERS
    assert "observed_peak_rss_bytes" in MEMORY_HEADERS
    assert "peak_rss_delta_bytes" in MEMORY_HEADERS
    assert "retained_rss_delta_bytes" in MEMORY_HEADERS

    # Interference Repetition & Pacing
    assert "before_window_start_ns" in INTERFERENCE_HEADERS
    assert "producer_attempted_frames_before" in INTERFERENCE_HEADERS
    assert "producer_skipped_deadlines_before" in INTERFERENCE_HEADERS
    assert "producer_schedule_lateness_median_ns_before" in INTERFERENCE_HEADERS
    assert "admitted_frames_during" in INTERFERENCE_HEADERS
    assert "completed_frames_during" in INTERFERENCE_HEADERS
    assert "admission_throughput_during_fps" in INTERFERENCE_HEADERS
    assert "completion_throughput_during_fps" in INTERFERENCE_HEADERS
    assert "admission_rate_ratio_vs_before" in INTERFERENCE_HEADERS
    assert "completion_rate_ratio_vs_before" in INTERFERENCE_HEADERS
    assert "latency_interpretation_confounded" in INTERFERENCE_HEADERS
    assert "frames_completed_during_window" in INTERFERENCE_HEADERS
    assert "frames_completed_while_preparation_active" in INTERFERENCE_HEADERS
    assert "raw_frame_count_matches_summary" in INTERFERENCE_HEADERS
    assert "throughput_accounting_valid" in INTERFERENCE_HEADERS

    # Interference Contrasts
    assert "treatment_category" in INTERFERENCE_CONTRAST_HEADERS
    assert "control_category" in INTERFERENCE_CONTRAST_HEADERS
    assert "paired_p95_ratio" in INTERFERENCE_CONTRAST_HEADERS
    assert "paired_admission_throughput_ratio" in INTERFERENCE_CONTRAST_HEADERS
    assert "confounded_pair_count" in INTERFERENCE_AGGREGATE_CONTRAST_HEADERS

    # Interference Raw Frame
    assert "frame_id" in INTERFERENCE_FRAME_HEADERS
    assert "phase" in INTERFERENCE_FRAME_HEADERS
    assert "admission_timestamp_ns" in INTERFERENCE_FRAME_HEADERS
    assert "completion_timestamp_ns" in INTERFERENCE_FRAME_HEADERS
    assert "frame_latency_ns" in INTERFERENCE_FRAME_HEADERS

    # Conformance
    assert "mixed_plan_frames" in CONFORMANCE_HEADERS
    assert "terminal_status" in CONFORMANCE_HEADERS
