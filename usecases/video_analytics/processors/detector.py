"""Detector processor node implementation for RT-DETR."""

from __future__ import annotations

import numpy as np

from nedo_vision_dag_engine.processor import (
    FrameContext,
    Processor,
    ProcessorDescriptor,
    SetupContext,
)
from nedo_vision_dag_engine.type_system import StatePolicy
from usecases.video_analytics.config import RTDETRConfig
from usecases.video_analytics.contracts import (
    DetectionBatch,
    DetectorBackend,
    FramePacket,
    FrameSource,
)

__all__ = [
    "DetectorProcessor",
]


class DetectorProcessor(Processor):
    """Processor wrapper around a DetectorBackend instance."""

    def __init__(
        self,
        backend: DetectorBackend,
        config: RTDETRConfig,
        source: FrameSource | None = None,
    ) -> None:
        self.backend: DetectorBackend = backend
        self.config: RTDETRConfig = config
        self.source: FrameSource | None = source
        self.descriptor: ProcessorDescriptor = ProcessorDescriptor(
            type_name="rt_detr_detector",
            input_schema=object,
            output_schema=DetectionBatch,
            config_schema=RTDETRConfig,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

    def setup(self, context: SetupContext) -> None:
        """Prepare the detector backend during candidate preparation."""
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        sample_packet = FramePacket(
            frame_id=1,
            source_timestamp_ns=0,
            image_bgr=dummy_bgr,
            width=640,
            height=480,
        )
        self.backend.prepare(sample_packet)

    def process(self, inputs: object, context: FrameContext) -> object:
        """Execute object detection on the input FramePacket."""
        frame_packet: FramePacket | None = None

        if isinstance(inputs, FramePacket):
            frame_packet = inputs
        elif isinstance(inputs, dict):
            inp_dict: dict[str, object] = inputs  # type: ignore[assignment]
            val = inp_dict.get("input")
            if isinstance(val, FramePacket):
                frame_packet = val
        if frame_packet is None and self.source is not None:
            frame_packet = self.source.read()
            if frame_packet is None:
                raise StopIteration("End of video stream reached")

        if frame_packet is None:
            dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_packet = FramePacket(
                frame_id=context.frame_id,
                source_timestamp_ns=context.admitted_at_ns,
                image_bgr=dummy_bgr,
                width=640,
                height=480,
            )

        detections = self.backend.infer(frame_packet)
        batch = DetectionBatch(
            frame_id=frame_packet.frame_id,
            source_timestamp_ns=frame_packet.source_timestamp_ns,
            admission_timestamp_ns=context.admitted_at_ns,
            detector_id=self.backend.model_id,
            plan_version=context.plan_version,
            detections=detections,
        )
        return batch

    def healthcheck(self) -> None:
        """Verify detector health."""
        if getattr(self.backend, "is_closed", False):
            raise RuntimeError("Detector backend is closed")

    def cleanup(self) -> None:
        """Release detector backend resources."""
        self.backend.close()
