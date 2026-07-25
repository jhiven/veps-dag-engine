"""Tracker processor node implementation for stateful ByteTrack."""

from __future__ import annotations

from nedo_vision_dag_engine.processor import (
    FrameContext,
    ProcessorDescriptor,
    SetupContext,
    StatefulProcessor,
    StatefulProcessorDescriptor,
    TransitionContext,
)
from nedo_vision_dag_engine.specification import ProcessorConfiguration
from nedo_vision_dag_engine.type_system import StatePolicy, StateTransitionPolicy
from usecases.video_analytics.config import ByteTrackConfig
from usecases.video_analytics.contracts import (
    DetectionBatch,
    TrackBatch,
    TrackerBackend,
)

__all__ = [
    "TrackerProcessor",
    "preserves_tracker_state",
]


def preserves_tracker_state(
    previous_configuration: ProcessorConfiguration,
    new_configuration: ProcessorConfiguration,
    context: TransitionContext,
) -> bool:
    """Predicate evaluating whether tracker state is preserved across a reconfiguration."""
    return context.structurally_preservable


class TrackerProcessor(StatefulProcessor):
    """Stateful processor wrapping a TrackerBackend."""

    def __init__(self, backend: TrackerBackend, config: ByteTrackConfig | None = None) -> None:
        self.backend: TrackerBackend = backend
        self.config: ByteTrackConfig = config or ByteTrackConfig()
        self.descriptor: ProcessorDescriptor = ProcessorDescriptor(
            type_name="bytetrack_tracker",
            input_schema=object,
            output_schema=TrackBatch,
            config_schema=ByteTrackConfig,
            state_policy=StatePolicy.PRESERVABLE,
            state_schema_version="1.0",
        )
        self.stateful_descriptor: StatefulProcessorDescriptor = StatefulProcessorDescriptor(
            state_schema_version="1.0",
            supported_transition_policies=frozenset({StateTransitionPolicy.PRESERVE, StateTransitionPolicy.RESET}),
            preserves_state_for=preserves_tracker_state,
        )

    def setup(self, context: SetupContext) -> None:
        """Setup tracker processor."""
        pass

    def process(self, inputs: object, context: FrameContext) -> object:
        """Process detections into tracked object batch."""
        batch: DetectionBatch | None = None
        if isinstance(inputs, DetectionBatch):
            batch = inputs
        elif isinstance(inputs, dict):
            inp_dict: dict[str, object] = inputs # pyright: ignore[reportUnknownVariableType]
            val = inp_dict.get("input")
            if isinstance(val, DetectionBatch):
                batch = val

        if batch is None:
            raise TypeError(f"TrackerProcessor expects DetectionBatch, got {type(inputs)!r}")  # pyright: ignore[reportUnknownArgumentType]

        tracks = self.backend.update(
            frame_id=batch.frame_id,
            source_timestamp_ns=batch.source_timestamp_ns,
            detections=batch.detections,
        )

        res = TrackBatch(
            frame_id=batch.frame_id,
            source_timestamp_ns=batch.source_timestamp_ns,
            admission_timestamp_ns=batch.admission_timestamp_ns,
            detector_id=batch.detector_id,
            tracker_instance_id=self.backend.instance_id,
            tracks=tracks,
            plan_version=batch.plan_version,
            inside_measurement_window=batch.inside_measurement_window,
            queue_occupancy_before_enqueue=batch.queue_occupancy_before_enqueue,
            queue_occupancy_after_enqueue=batch.queue_occupancy_after_enqueue,
            queue_capacity=batch.queue_capacity,
            media_pts_ns=batch.media_pts_ns,
            receiver_ingress_timestamp_ns=batch.receiver_ingress_timestamp_ns,
            enqueue_decision_timestamp_ns=batch.enqueue_decision_timestamp_ns,
            drop_decision_timestamp_ns=batch.drop_decision_timestamp_ns,
            media_frame_index=batch.media_frame_index,
        )
        return res

    def healthcheck(self) -> None:
        """Check tracker health."""
        if getattr(self.backend, "is_closed", False):
            raise RuntimeError("Tracker backend is closed")

    def cleanup(self) -> None:
        """Clean up tracker backend."""
        self.backend.close()
