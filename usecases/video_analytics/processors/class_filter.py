"""Person-class filter processor node implementation."""

from __future__ import annotations

from dataclasses import dataclass

from nedo_vision_dag_engine.processor import (
    FrameContext,
    Processor,
    ProcessorDescriptor,
    SetupContext,
)
from nedo_vision_dag_engine.type_system import StatePolicy
from usecases.video_analytics.contracts import DetectionBatch

__all__ = [
    "ClassFilterConfig",
    "PersonFilterProcessor",
]


@dataclass(frozen=True, slots=True)
class ClassFilterConfig:
    """Configuration for person class filter processor."""

    target_class_name: str = "person"


class PersonFilterProcessor(Processor):
    """Stateless processor filtering detections to target class (e.g. person)."""

    def __init__(self, config: ClassFilterConfig | None = None) -> None:
        self.config: ClassFilterConfig = config or ClassFilterConfig()
        self.descriptor: ProcessorDescriptor = ProcessorDescriptor(
            type_name="person_class_filter",
            input_schema=object,
            output_schema=DetectionBatch,
            config_schema=ClassFilterConfig,
            state_policy=StatePolicy.STATELESS,
            state_schema_version=None,
        )

    def setup(self, context: SetupContext) -> None:
        """No setup required for stateless filter."""
        pass

    def process(self, inputs: object, context: FrameContext) -> object:
        """Filter input DetectionBatch to target class name."""
        batch: DetectionBatch | None = None
        if isinstance(inputs, DetectionBatch):
            batch = inputs
        elif isinstance(inputs, dict):
            inp_dict: dict[str, object] = inputs  # type: ignore[assignment]
            val = inp_dict.get("input")
            if isinstance(val, DetectionBatch):
                batch = val

        if batch is None:
            raise TypeError(f"PersonFilterProcessor expects DetectionBatch, got {type(inputs)!r}")  # pyright: ignore[reportUnknownArgumentType]

        filtered = tuple(
            d for d in batch.detections if d.class_name.lower() == self.config.target_class_name.lower()
        )
        res = DetectionBatch(
            frame_id=batch.frame_id,
            source_timestamp_ns=batch.source_timestamp_ns,
            detector_id=batch.detector_id,
            plan_version=batch.plan_version,
            detections=filtered,
        )
        return res

    def healthcheck(self) -> None:
        """Stateless healthcheck."""
        pass

    def cleanup(self) -> None:
        """Stateless cleanup."""
        pass
