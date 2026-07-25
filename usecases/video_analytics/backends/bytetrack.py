"""ByteTrack stateful tracking backend adapter."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from usecases.video_analytics.config import ByteTrackConfig
from usecases.video_analytics.contracts import (
    BackendKind,
    BoundingBox,
    Detection,
    Track,
    TrackerBackend,
    TrackerStateSnapshot,
)

__all__ = [
    "ByteTrackerBackend",
    "FakeTrackerBackend",
    "calculate_box_iou",
]


def calculate_box_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA.x2 - boxA.x1) * max(0.0, boxA.y2 - boxA.y1)
    boxBArea = max(0.0, boxB.x2 - boxB.x1) * max(0.0, boxB.y2 - boxB.y1)

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return float(interArea / unionArea)


@dataclass(slots=True)
class _ActiveTrackState:
    track_id: int
    class_id: int
    class_name: str
    box: BoundingBox
    score: float
    time_since_update: int = 0
    hits: int = 1


class ByteTrackerBackend(TrackerBackend):
    """Production stateful ByteTrack tracker implementation using IoU & Hungarian matching."""

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self._config: ByteTrackConfig = config or ByteTrackConfig()
        self._instance_id: str = f"bytetrack_{uuid.uuid4().hex[:8]}"
        self._reset_count: int = 0
        self._processed_frame_count: int = 0
        self._next_track_id: int = 1
        self._tracks: dict[int, _ActiveTrackState] = {}
        self._is_closed: bool = False

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.PRODUCTION

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def processed_frame_count(self) -> int:
        return self._processed_frame_count

    def snapshot(self) -> TrackerStateSnapshot:
        return TrackerStateSnapshot(
            instance_id=self._instance_id,
            reset_count=self._reset_count,
            processed_frame_count=self._processed_frame_count,
            live_track_ids=tuple(sorted(self._tracks.keys())),
        )

    def reset_state(self) -> None:
        """Reset internal tracking state."""
        self._tracks.clear()
        self._reset_count += 1
        self._next_track_id = 1

    def update(
        self,
        frame_id: int,
        source_timestamp_ns: int,
        detections: tuple[Detection, ...],
    ) -> tuple[Track, ...]:
        """Update tracker state with new frame detections using ByteTrack 2-stage association."""
        if self._is_closed:
            raise RuntimeError("Cannot update a closed ByteTrackerBackend")

        self._processed_frame_count += 1

        # Predict track state (increment time since update)
        for track in self._tracks.values():
            track.time_since_update += 1

        # Separate detections into high-score and low-score
        high_dets = [d for d in detections if d.score >= self._config.track_thresh]
        low_dets = [d for d in detections if d.score < self._config.track_thresh]

        existing_track_ids = list(self._tracks.keys())
        existing_tracks = [self._tracks[tid] for tid in existing_track_ids]

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # Stage 1: Match high-confidence detections with existing tracks
        if existing_tracks and high_dets:
            cost_matrix = np.zeros((len(existing_tracks), len(high_dets)), dtype=np.float64)
            for i, t in enumerate(existing_tracks):
                for j, d in enumerate(high_dets):
                    cost_matrix[i, j] = 1.0 - calculate_box_iou(t.box, d.box)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind, strict=False):
                r_i = int(r)
                c_i = int(c)
                if cost_matrix[r_i, c_i] <= (1.0 - self._config.match_thresh):
                    t = existing_tracks[r_i]
                    d = high_dets[c_i]
                    t.box = d.box
                    t.score = d.score
                    t.class_id = d.class_id
                    t.class_name = d.class_name
                    t.time_since_update = 0
                    t.hits += 1
                    matched_track_ids.add(t.track_id)
                    matched_det_indices.add(c_i)

        # Stage 2: Match low-confidence detections with remaining tracks
        unmatched_tracks = [t for t in existing_tracks if t.track_id not in matched_track_ids]
        if unmatched_tracks and low_dets:
            cost_matrix_low = np.zeros((len(unmatched_tracks), len(low_dets)), dtype=np.float64)
            for i, t in enumerate(unmatched_tracks):
                for j, d in enumerate(low_dets):
                    cost_matrix_low[i, j] = 1.0 - calculate_box_iou(t.box, d.box)

            r_ind, c_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(r_ind, c_ind, strict=False):
                r_i = int(r)
                c_i = int(c)
                if cost_matrix_low[r_i, c_i] <= 0.5:  # IoU >= 0.5 threshold for low-score match
                    t = unmatched_tracks[r_i]
                    d = low_dets[c_i]
                    t.box = d.box
                    t.score = d.score
                    t.time_since_update = 0
                    t.hits += 1
                    matched_track_ids.add(int(t.track_id))

        # Initialize new tracks for unmatched high-confidence detections
        for j, d in enumerate(high_dets):
            if j not in matched_det_indices:
                new_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[new_id] = _ActiveTrackState(
                    track_id=new_id,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    box=d.box,
                    score=d.score,
                    time_since_update=0,
                    hits=1,
                )

        # Remove lost tracks older than track_buffer
        to_remove = [
            tid for tid, t in self._tracks.items() if t.time_since_update > self._config.track_buffer
        ]
        for tid in to_remove:
            del self._tracks[tid]

        # Emit active tracks
        active_tracks: list[Track] = []
        for tid, t in sorted(self._tracks.items()):
            if t.time_since_update == 0:
                active_tracks.append(
                    Track(
                        track_id=t.track_id,
                        class_id=t.class_id,
                        class_name=t.class_name,
                        box=t.box,
                        score=t.score,
                    )
                )

        return tuple(active_tracks)

    def close(self) -> None:
        """Release tracker state."""
        if self._is_closed:
            return
        self._is_closed = True
        self._tracks.clear()


class FakeTrackerBackend(TrackerBackend):
    """Fake tracker backend for unit testing."""

    def __init__(self, instance_id: str = "fake_tracker_1") -> None:
        self._instance_id: str = instance_id
        self._reset_count: int = 0
        self._processed_frame_count: int = 0
        self.is_closed: bool = False

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.FAKE

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def processed_frame_count(self) -> int:
        return self._processed_frame_count

    def snapshot(self) -> TrackerStateSnapshot:
        return TrackerStateSnapshot(
            instance_id=self._instance_id,
            reset_count=self._reset_count,
            processed_frame_count=self._processed_frame_count,
            live_track_ids=(1,),
        )

    def update(
        self,
        frame_id: int,
        source_timestamp_ns: int,
        detections: tuple[Detection, ...],
    ) -> tuple[Track, ...]:
        self._processed_frame_count += 1
        tracks: list[Track] = []
        for idx, det in enumerate(detections, start=1):
            tracks.append(
                Track(
                    track_id=idx,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    box=det.box,
                    score=det.score,
                )
            )
        return tuple(tracks)

    def close(self) -> None:
        self.is_closed = True
