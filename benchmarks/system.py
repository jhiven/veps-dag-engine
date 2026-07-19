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
    physical_cpu_count: int
    available_cpu_affinity: tuple[int, ...]
    applied_cpu_affinity: tuple[int, ...]
    executor_cpu: int
    compiler_cpu: int
    physical_core_id: int | None
    smt_sibling_ids: tuple[int, ...]
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


def _get_physical_core_id(cpu_id: int) -> int | None:
    path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return int(file.read().strip())
        except Exception:
            pass
    return None


def _parse_cpu_list(val: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in val.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) == 2:
                try:
                    start, end = int(bounds[0]), int(bounds[1])
                    cpus.extend(range(start, end + 1))
                except ValueError:
                    pass
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                pass
    return tuple(sorted(cpus))


def _get_smt_siblings(cpu_id: int) -> tuple[int, ...]:
    path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/thread_siblings_list"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return _parse_cpu_list(file.read().strip())
        except Exception:
            pass
    return (cpu_id,)


def collect_system_environment(pin_cpu: int | None = None) -> SystemEnvironment:
    timer_info = time.get_clock_info("perf_counter")
    available_affinity = _get_cpu_affinity()

    if pin_cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {pin_cpu})
            applied_affinity = (pin_cpu,)
        except Exception:
            applied_affinity = available_affinity
    else:
        applied_affinity = available_affinity

    primary_cpu = applied_affinity[0] if applied_affinity else 0
    physical_core_id = _get_physical_core_id(primary_cpu)
    smt_siblings = _get_smt_siblings(primary_cpu)

    # Estimate physical CPU count from sysfs or os.cpu_count()
    physical_cores: set[int] = set()
    for cpu in available_affinity:
        cid = _get_physical_core_id(cpu)
        if cid is not None:
            physical_cores.add(cid)
    physical_cpu_count = len(physical_cores) if physical_cores else (os.cpu_count() or 1)

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
        physical_cpu_count=physical_cpu_count,
        available_cpu_affinity=available_affinity,
        applied_cpu_affinity=applied_affinity,
        executor_cpu=primary_cpu,
        compiler_cpu=primary_cpu,
        physical_core_id=physical_core_id,
        smt_sibling_ids=smt_siblings,
        timer_implementation=f"time.perf_counter_ns ({timer_info.implementation})",
        timer_resolution_ns=timer_info.resolution * 1e9,
    )
