"""PyTorch CUDA memory allocator metrics sampler and snapshot recorder."""

from __future__ import annotations

from usecases.video_analytics.contracts import CUDAMemorySamplerProtocol, CUDAMemorySnapshot

__all__ = [
    "PyTorchCUDAMemorySampler",
    "FakeCUDAMemorySampler",
    "sample_gpu_memory",
]


class PyTorchCUDAMemorySampler(CUDAMemorySamplerProtocol):
    """Production PyTorch CUDA memory allocator metric sampler."""

    def __init__(self, device: str = "cuda:0") -> None:
        self._device_str: str = device
        self._is_cuda: bool = device.startswith("cuda")
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize CUDA device and verify availability."""
        if not self._is_cuda:
            return

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.init()
                self._initialized = True
        except Exception:
            self._initialized = False

    def reset_peak_stats(self) -> None:
        """Reset PyTorch CUDA peak memory tracking statistics."""
        if not self._is_cuda or not self._initialized:
            return

        try:
            import torch

            torch.cuda.reset_peak_memory_stats(self._device_str)
        except Exception:
            pass

    def synchronize(self) -> None:
        """Synchronize CUDA stream before taking a memory snapshot."""
        if not self._is_cuda or not self._initialized:
            return

        try:
            import torch

            torch.cuda.synchronize(self._device_str)
        except Exception:
            pass

    def sample(self) -> CUDAMemorySnapshot:
        """Capture snapshot of PyTorch CUDA memory allocator metrics."""
        if not self._is_cuda or not self._initialized:
            return CUDAMemorySnapshot(
                allocated_bytes=None,
                reserved_bytes=None,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
            )

        try:
            import torch

            self.synchronize()
            allocated: int = torch.cuda.memory_allocated(self._device_str)
            reserved: int = torch.cuda.memory_reserved(self._device_str)
            peak_allocated: int = torch.cuda.max_memory_allocated(self._device_str)
            peak_reserved: int = torch.cuda.max_memory_reserved(self._device_str)

            return CUDAMemorySnapshot(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
            )
        except Exception:
            return CUDAMemorySnapshot(
                allocated_bytes=None,
                reserved_bytes=None,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
            )


class FakeCUDAMemorySampler(CUDAMemorySamplerProtocol):
    """Synthetic CUDA memory sampler for deterministic unit testing."""

    def __init__(
        self,
        is_cuda: bool = True,
        allocated_bytes: int = 100_000_000,
        reserved_bytes: int = 200_000_000,
    ) -> None:
        self._is_cuda: bool = is_cuda
        self._allocated_bytes: int = allocated_bytes
        self._reserved_bytes: int = reserved_bytes
        self.initialize_called: bool = False
        self.reset_peak_stats_called: bool = False
        self.synchronize_call_count: int = 0
        self.sample_call_count: int = 0

    def initialize(self) -> None:
        self.initialize_called = True

    def reset_peak_stats(self) -> None:
        self.reset_peak_stats_called = True

    def synchronize(self) -> None:
        self.synchronize_call_count += 1

    def sample(self) -> CUDAMemorySnapshot:
        self.sample_call_count += 1
        self.synchronize()
        if not self._is_cuda:
            return CUDAMemorySnapshot(
                allocated_bytes=None,
                reserved_bytes=None,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
            )
        return CUDAMemorySnapshot(
            allocated_bytes=self._allocated_bytes,
            reserved_bytes=self._reserved_bytes,
            peak_allocated_bytes=self._allocated_bytes + 50_000_000,
            peak_reserved_bytes=self._reserved_bytes + 50_000_000,
        )


def sample_gpu_memory(device: str) -> CUDAMemorySnapshot:
    """Helper function to sample GPU memory metrics on a target device."""
    sampler = PyTorchCUDAMemorySampler(device=device)
    sampler.initialize()
    return sampler.sample()
