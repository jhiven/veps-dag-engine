"""The per-frame workspace.

A workspace is where one frame's intermediate and final step outputs
live while that frame is being processed: index i holds whatever the
step whose ``output_index`` equals i produced, or ``MISSING`` if that
step has not run (or was skipped) for this frame.

A workspace's size is tied to a specific plan version's
``output_slot_count`` -- different plan versions can have a different
number of steps, so a workspace built for one version cannot simply be
reused as-is for another. ``WorkspacePool`` still lets us avoid
reallocating a fresh Python list on every single frame: it keeps
same-sized workspaces around between frames and hands one back out on
request, but only after clearing every slot back to ``MISSING`` first.
Reuse must never let a value from a previous frame leak into the next
one; clearing on acquire, rather than on release, makes that the one
place this guarantee has to hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from nedo_vision_dag_engine.type_system import MISSING

__all__ = [
    "Workspace",
    "WorkspacePool",
]


class Workspace(Sequence[object]):
    """A fixed-size, mutable array of workspace slots for one frame.

    Implements ``Sequence[object]`` (read-only iteration and indexing) so
    it can be passed anywhere a plan's readiness rules or input
    construction expect a ``WorkspaceView``, while still exposing ``set``
    and ``clear`` for the executor, which is the only thing allowed to
    mutate it.
    """

    __slots__ = ("_slots",)

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"workspace size must be non-negative, got {size}.")
        self._slots: list[object] = [MISSING] * size

    def __len__(self) -> int:
        return len(self._slots)

    @overload
    def __getitem__(self, index: int) -> object: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[object, ...]: ...

    def __getitem__(self, index: int | slice) -> object | tuple[object, ...]:
        if isinstance(index, slice):
            return tuple(self._slots[index])
        return self._slots[index]

    def set(self, index: int, value: object) -> None:
        if not (0 <= index < len(self._slots)):
            raise IndexError(f"workspace slot index {index} out of range for size {len(self._slots)}.")
        self._slots[index] = value

    def clear(self) -> None:
        for index in range(len(self._slots)):
            self._slots[index] = MISSING


class WorkspacePool:
    """Keeps previously used, now-idle workspaces around for reuse,
    grouped by size, so most frames do not need to allocate a new one.

    This pool is **not** internally synchronized.  The executor always
    calls ``acquire`` and ``release`` while holding its frame-execution
    lock, serializing all access.  If the pool is ever
    shared across callers that do not already hold a common lock, wrap it
    or add internal locking.
    """

    def __init__(self) -> None:
        self._idle_by_size: dict[int, list[Workspace]] = {}

    def acquire(self, size: int) -> Workspace:
        idle = self._idle_by_size.get(size)
        if idle:
            workspace = idle.pop()
            workspace.clear()
            return workspace
        return Workspace(size)

    def release(self, workspace: Workspace) -> None:
        self._idle_by_size.setdefault(len(workspace), []).append(workspace)

    def idle_count(self, size: int) -> int:
        return len(self._idle_by_size.get(size, ()))
