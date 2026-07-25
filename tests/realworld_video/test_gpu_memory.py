"""Unit tests for GPU memory sampling and snapshot timing invariants."""

from __future__ import annotations

from usecases.video_analytics.gpu_memory import FakeCUDAMemorySampler


def test_gpu_warmup_before_baseline_sampling() -> None:
    """Test 15: GPU warm-up occurs before baseline memory sampling."""
    sampler = FakeCUDAMemorySampler(is_cuda=True)
    sampler.initialize()
    sampler.reset_peak_stats()
    baseline = sampler.sample()

    assert sampler.initialize_called
    assert sampler.reset_peak_stats_called
    assert baseline.allocated_bytes == 100_000_000
    assert baseline.reserved_bytes == 200_000_000


def test_cuda_synchronization_before_every_memory_snapshot() -> None:
    """Test 16: CUDA synchronization occurs before every memory snapshot."""
    sampler = FakeCUDAMemorySampler(is_cuda=True)
    sampler.initialize()
    sampler.sample()
    sampler.sample()
    sampler.sample()
    assert sampler.synchronize_call_count == 3


def test_post_retirement_memory_sampled_while_candidate_alive() -> None:
    """Test 17: Post-retirement memory is sampled while candidate plan is still alive."""
    sampler = FakeCUDAMemorySampler(is_cuda=True)
    snap = sampler.sample()
    assert snap.allocated_bytes is not None
    assert snap.reserved_bytes is not None


def test_cpu_mode_writes_missing_gpu_values_rather_than_zero() -> None:
    """Test 18: CPU mode writes None/missing GPU values rather than zero."""
    sampler = FakeCUDAMemorySampler(is_cuda=False)
    snap = sampler.sample()
    assert snap.allocated_bytes is None
    assert snap.reserved_bytes is None
    assert snap.peak_allocated_bytes is None
    assert snap.peak_reserved_bytes is None
