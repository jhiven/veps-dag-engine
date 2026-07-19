"""System environment and hardware metadata collection using standard library."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

__all__ = [
    "SystemEnvironment",
    "collect_system_environment",
]


@dataclass(frozen=True, slots=True)
class SystemEnvironment:
    git_commit: str
    dirty_working_tree: bool
    python_implementation: str
    python_version: str
    uv_lock_sha256: str
    operating_system: str
    kernel_version: str
    hostname: str
    cpu_model: str
    logical_cpu_count: int
    available_cpu_affinity: tuple[int, ...]
    applied_cpu_affinity: tuple[int, ...]
    timer_implementation: str
    timer_resolution_ns: float


def _get_git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output if output else "unknown"
    except Exception:
        return "unknown"


def _get_dirty_working_tree() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return len(output) > 0
    except Exception:
        return False


def _get_uv_lock_sha256() -> str:
    path = "uv.lock"
    if not os.path.exists(path):
        return "none"
    try:
        with open(path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except Exception:
        return "unknown"


def _get_cpu_model() -> str:
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("model name"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _get_cpu_affinity() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
            return tuple(affinity)
        except Exception:
            pass
    count = os.cpu_count() or 1
    return tuple(range(count))


def collect_system_environment() -> SystemEnvironment:
    timer_info = time.get_clock_info("perf_counter")
    affinity = _get_cpu_affinity()
    return SystemEnvironment(
        git_commit=_get_git_commit(),
        dirty_working_tree=_get_dirty_working_tree(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        uv_lock_sha256=_get_uv_lock_sha256(),
        operating_system=sys.platform,
        kernel_version=platform.release(),
        hostname=socket.gethostname(),
        cpu_model=_get_cpu_model(),
        logical_cpu_count=os.cpu_count() or 1,
        available_cpu_affinity=affinity,
        applied_cpu_affinity=affinity,
        timer_implementation=f"time.perf_counter_ns ({timer_info.implementation})",
        timer_resolution_ns=timer_info.resolution * 1e9,
    )
