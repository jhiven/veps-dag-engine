"""Unit tests for memory measurement campaign and RSS helpers (Section 4, 10)."""

from __future__ import annotations

import tempfile
import pytest

from benchmarks.model import (
    MemorySampleRow,
    validate_memory_sample,
)
from benchmarks.runners.memory import get_current_rss_bytes
from benchmarks.storage import (
    append_memory_rows,
    read_memory_rows,
    write_memory_header,
)


def test_get_current_rss_bytes() -> None:
    """Verify get_current_rss_bytes returns a positive integer via psutil."""
    rss = get_current_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0


def test_memory_sample_row_validation() -> None:
    """Verify MemorySampleRow checkpoint peak calculations and validation."""
    row = MemorySampleRow(
        run_id="run_mem",
        scenario_id="scen_mem",
        variant="variant_a",
        repetition=1,
        rss_before_bytes=100_000_000,
        rss_after_candidate_prepare_bytes=120_000_000,
        rss_after_commit_bytes=115_000_000,
        rss_after_retirement_bytes=105_000_000,
        observed_peak_rss_bytes=120_000_000,
        peak_rss_delta_bytes=20_000_000,
        retained_rss_delta_bytes=5_000_000,
        peak_live_processors=10,
    )
    validate_memory_sample(row)
    assert row.observed_peak_rss_bytes == 120_000_000
    assert row.peak_rss_delta_bytes == 20_000_000
    assert row.retained_rss_delta_bytes == 5_000_000


def test_memory_sample_row_invalid_peak() -> None:
    """Verify mismatched peak or delta raises ValueError."""
    with pytest.raises(ValueError, match="observed_peak_rss_bytes"):
        validate_memory_sample(
            MemorySampleRow(
                run_id="run_mem",
                scenario_id="scen_mem",
                variant="variant_a",
                repetition=1,
                rss_before_bytes=100_000_000,
                rss_after_candidate_prepare_bytes=120_000_000,
                rss_after_commit_bytes=115_000_000,
                rss_after_retirement_bytes=105_000_000,
                observed_peak_rss_bytes=110_000_000,  # Invalid! Max is 120MB
                peak_rss_delta_bytes=10_000_000,
                retained_rss_delta_bytes=5_000_000,
                peak_live_processors=10,
            )
        )


def test_memory_csv_roundtrip() -> None:
    """Verify CSV storage and header serialization for MemorySampleRow."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = f"{tmp_dir}/memory-samples.csv"
        write_memory_header(csv_path)

        row = MemorySampleRow(
            run_id="run_mem",
            scenario_id="scen_mem",
            variant="variant_b",
            repetition=2,
            rss_before_bytes=200_000_000,
            rss_after_candidate_prepare_bytes=210_000_000,
            rss_after_commit_bytes=250_000_000,
            rss_after_retirement_bytes=205_000_000,
            observed_peak_rss_bytes=250_000_000,
            peak_rss_delta_bytes=50_000_000,
            retained_rss_delta_bytes=5_000_000,
            peak_live_processors=12,
        )
        append_memory_rows(csv_path, [row])
        read_rows = read_memory_rows(csv_path)

        assert len(read_rows) == 1
        r = read_rows[0]
        assert r.run_id == "run_mem"
        assert r.variant == "variant_b"
        assert r.observed_peak_rss_bytes == 250_000_000
        assert r.peak_rss_delta_bytes == 50_000_000
        assert r.retained_rss_delta_bytes == 5_000_000
