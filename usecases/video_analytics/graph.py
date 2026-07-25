"""DAG graph construction and workspace specification factory for video analytics."""

from __future__ import annotations

from typing import Any

from nedo_vision_dag_engine.processor import Processor
from nedo_vision_dag_engine.registry import (
    RegisteredProcessorType,
    RegistryBuilder,
    RegistrySnapshot,
)
from nedo_vision_dag_engine.specification import (
    ConcreteType,
    Edge,
    Node,
    Pin,
    PinCardinality,
    PinRequirement,
    WorkflowSpecification,
    to_processor_configuration,
)

from usecases.video_analytics.config import ByteTrackConfig, NullSinkConfig, RTDETRConfig
from usecases.video_analytics.contracts import (
    DetectionBatch,
    DetectorBackend,
    FrameSource,
    ResultSink,
    TrackBatch,
    TrackerBackend,
)
from usecases.video_analytics.processors.class_filter import (
    ClassFilterConfig,
    PersonFilterProcessor,
)
from usecases.video_analytics.processors.detector import DetectorProcessor
from usecases.video_analytics.processors.tracker import TrackerProcessor
from usecases.video_analytics.sink import NullSink

__all__ = [
    "build_video_analytics_specification",
    "build_video_analytics_registry",
    "SinkProcessor",
]


class SinkProcessor(Processor):
    """Processor wrapper around a ResultSink instance."""

    def __init__(self, sink: ResultSink) -> None:
        from nedo_vision_dag_engine.processor import ProcessorDescriptor
        from nedo_vision_dag_engine.type_system import StatePolicy

        self.sink: ResultSink = sink
        self.descriptor: ProcessorDescriptor = ProcessorDescriptor(
            type_name="null_sink",
            input_schema=object,
            output_schema=TrackBatch,
            config_schema=NullSinkConfig,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

    def setup(self, context: Any) -> None:
        pass

    def process(self, inputs: object, context: Any) -> object:
        tb: TrackBatch | None = None
        if isinstance(inputs, TrackBatch):
            tb = inputs
        elif isinstance(inputs, dict):
            inp_dict: dict[str, object] = inputs # pyright: ignore[reportUnknownVariableType]
            val = inp_dict.get("input")
            if isinstance(val, TrackBatch):
                tb = val
        if tb is not None:
            self.sink.consume(tb)
        return tb

    def healthcheck(self) -> None:
        pass

    def cleanup(self) -> None:
        self.sink.close()


def build_video_analytics_specification(
    *,
    detector_config: RTDETRConfig,
    tracker_config: ByteTrackConfig | None = None,
    sink_config: NullSinkConfig | None = None,
) -> WorkflowSpecification:
    """Build the canonical video analytics WorkflowSpecification."""
    _tracker_config = tracker_config or ByteTrackConfig()
    _sink_config = sink_config or NullSinkConfig()

    detector_node = Node(
        node_id="detector",
        type_name="rt_detr_detector",
        configuration=to_processor_configuration({
            "model_id": detector_config.model_id,
            "device": detector_config.device,
            "dtype": detector_config.dtype,
            "confidence_threshold": detector_config.confidence_threshold,
            "person_class_name": detector_config.person_class_name,
        }),
        inputs=(),
        outputs=(
            Pin(
                name="output",
                payload_type=ConcreteType(DetectionBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )

    filter_node = Node(
        node_id="person_filter",
        type_name="person_class_filter",
        configuration=to_processor_configuration({"target_class_name": detector_config.person_class_name}),
        inputs=(
            Pin(
                name="input",
                payload_type=ConcreteType(DetectionBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
        outputs=(
            Pin(
                name="output",
                payload_type=ConcreteType(DetectionBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )

    tracker_node = Node(
        node_id="tracker",
        type_name="bytetrack_tracker",
        configuration=to_processor_configuration({
            "track_thresh": _tracker_config.track_thresh,
            "track_buffer": _tracker_config.track_buffer,
            "match_thresh": _tracker_config.match_thresh,
            "frame_rate": _tracker_config.frame_rate,
        }),
        inputs=(
            Pin(
                name="input",
                payload_type=ConcreteType(DetectionBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
        outputs=(
            Pin(
                name="output",
                payload_type=ConcreteType(TrackBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )

    sink_node = Node(
        node_id="sink",
        type_name="null_sink",
        configuration=to_processor_configuration({
            "validate_monotonic_frame_ids": _sink_config.validate_monotonic_frame_ids,
        }),
        inputs=(
            Pin(
                name="input",
                payload_type=ConcreteType(TrackBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
        outputs=(
            Pin(
                name="output",
                payload_type=ConcreteType(TrackBatch),
                cardinality=PinCardinality.SINGLE,
                requirement=PinRequirement.REQUIRED,
            ),
        ),
    )

    edges = (
        Edge(
            source_node_id="detector",
            source_pin="output",
            destination_node_id="person_filter",
            destination_pin="input",
        ),
        Edge(
            source_node_id="person_filter",
            source_pin="output",
            destination_node_id="tracker",
            destination_pin="input",
        ),
        Edge(
            source_node_id="tracker",
            source_pin="output",
            destination_node_id="sink",
            destination_pin="input",
        ),
    )

    return WorkflowSpecification(
        nodes=(detector_node, filter_node, tracker_node, sink_node),
        edges=edges,
    )


def build_video_analytics_registry(
    *,
    detector_backend: DetectorBackend,
    tracker_backend: TrackerBackend,
    sink_instance: ResultSink | None = None,
    source_instance: FrameSource | None = None,
    detector_config: RTDETRConfig | None = None,
    tracker_config: ByteTrackConfig | None = None,
) -> RegistrySnapshot:
    """Build a RegistrySnapshot mapping node type names to processor factories."""
    det_config = detector_config or RTDETRConfig()
    trk_config = tracker_config or ByteTrackConfig()
    sink_obj = sink_instance or NullSink()

    det_processor = DetectorProcessor(detector_backend, det_config, source=source_instance)
    flt_processor = PersonFilterProcessor(ClassFilterConfig(target_class_name=det_config.person_class_name))
    trk_processor = TrackerProcessor(tracker_backend, trk_config)
    snk_processor = SinkProcessor(sink_obj)

    builder = RegistryBuilder()
    builder.register(
        RegisteredProcessorType(
            descriptor=det_processor.descriptor,
            factory=lambda: DetectorProcessor(detector_backend, det_config, source=source_instance),
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=flt_processor.descriptor,
            factory=lambda: PersonFilterProcessor(ClassFilterConfig(target_class_name=det_config.person_class_name)),
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=trk_processor.descriptor,
            factory=lambda: TrackerProcessor(tracker_backend, trk_config),
            stateful_descriptor=trk_processor.stateful_descriptor,
        )
    )
    builder.register(
        RegisteredProcessorType(
            descriptor=snk_processor.descriptor,
            factory=lambda: SinkProcessor(sink_obj),
        )
    )

    return builder.snapshot()
