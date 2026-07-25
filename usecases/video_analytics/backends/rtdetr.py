"""RT-DETR object detector backend adapter using PyTorch and Hugging Face Transformers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from transformers import (
    RTDetrConfig as HFRTDetrConfig,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)

from usecases.video_analytics.config import RTDETRConfig
from usecases.video_analytics.contracts import BackendKind, BoundingBox, Detection, DetectorBackend, FramePacket

__all__ = [
    "RTDETRDetectorBackend",
    "FakeDetectorBackend",
    "preload_rtdetr_models_to_ram",
]


@dataclass(frozen=True, slots=True)
class _RAMModelCacheEntry:
    image_processor: RTDetrImageProcessor
    config: HFRTDetrConfig
    state_dict: dict[str, torch.Tensor]


_RAM_MODEL_CACHE: dict[str, _RAMModelCacheEntry] = {}


def preload_rtdetr_models_to_ram(model_ids: list[str]) -> None:
    """Pre-load model weights and configs from disk into an in-memory CPU RAM state_dict cache ONCE."""
    try:
        from huggingface_hub.utils.tqdm import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    for mid in model_ids:
        if mid in _RAM_MODEL_CACHE:
            continue

        try:
            image_processor = RTDetrImageProcessor.from_pretrained(mid)  # pyright: ignore[reportUnknownMemberType]
            cpu_model = RTDetrForObjectDetection.from_pretrained(mid)  # pyright: ignore[reportUnknownMemberType]
            cpu_model.eval()

            raw_state_dict: dict[str, torch.Tensor] = cpu_model.state_dict()
            state_dict: dict[str, torch.Tensor] = {
                k: v.detach().cpu().clone() for k, v in raw_state_dict.items()
            }
            config: HFRTDetrConfig = cpu_model.config

            _RAM_MODEL_CACHE[mid] = _RAMModelCacheEntry(
                image_processor=image_processor,
                config=config,
                state_dict=state_dict,
            )
        except Exception:
            pass


class RTDETRDetectorBackend(DetectorBackend):
    """Production RT-DETR backend adapter for Hugging Face RTDetrForObjectDetection."""

    def __init__(self, config: RTDETRConfig) -> None:
        self._config: RTDETRConfig = config
        self._model: RTDetrForObjectDetection | None = None
        self._image_processor: RTDetrImageProcessor | None = None
        self._person_class_id: int | None = None
        self._is_prepared: bool = False
        self._is_closed: bool = False

        # Preparation subphase timestamps (ns)
        self.model_construction_start_ns: int = 0
        self.weights_loaded_ns: int = 0
        self.device_transfer_completed_ns: int = 0
        self.warmup_completed_ns: int = 0
        self.backend_ready_ns: int = 0

        from usecases.video_analytics.lifecycle import CoexistenceTracker
        CoexistenceTracker.detector_created(id(self))

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.PRODUCTION

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def prepare(self, sample_frame: FramePacket) -> None:
        """Load weights, move to device, resolve class labels, and run warm-up inference."""
        if self._is_prepared or self._is_closed:
            return

        self.model_construction_start_ns = time.monotonic_ns()

        if self._config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self._config.device!r} was requested, but CUDA is not available on this system."
                )

        cache_entry = _RAM_MODEL_CACHE.get(self._config.model_id)
        model: RTDetrForObjectDetection
        image_processor: RTDetrImageProcessor

        if cache_entry is not None:
            # 1. Instantiate a brand NEW model instance from cached config (in RAM)
            model = RTDetrForObjectDetection(cache_entry.config)
            # 2. Populate weights from RAM state_dict cache (zero disk I/O!)
            ram_state_dict: dict[str, torch.Tensor] = {k: v.clone() for k, v in cache_entry.state_dict.items()}
            model.load_state_dict(ram_state_dict, strict=True)
            image_processor = cache_entry.image_processor
        else:
            image_processor = RTDetrImageProcessor.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                self._config.model_id,
                local_files_only=self._config.local_files_only,
            )
            model = RTDetrForObjectDetection.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                self._config.model_id,
                local_files_only=self._config.local_files_only,
            )

        self.weights_loaded_ns = time.monotonic_ns()

        model.eval()
        device = torch.device(self._config.device)

        torch_dtype = torch.float32
        if self._config.dtype == "float16":
            torch_dtype = torch.float16
        elif self._config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model.to(device, torch_dtype)  # pyright: ignore[reportUnknownMemberType,reportArgumentType,reportCallIssue]
        self.device_transfer_completed_ns = time.monotonic_ns()

        self._model = model
        self._image_processor = image_processor

        # Resolve person class ID from id2label
        config_obj: HFRTDetrConfig = model.config
        id2label_raw: Any = getattr(config_obj, "id2label", None)
        id2label: dict[int, str] | dict[str, str] = id2label_raw or {}
        person_id: int | None = None
        for cid, name in id2label.items():
            if str(name).lower() == self._config.person_class_name.lower():
                person_id = int(cid)
                break
        if person_id is None:
            person_id = 0
        self._person_class_id = person_id

        # Warmup inference
        self.infer(sample_frame)
        if self._config.device.startswith("cuda"):
            torch.cuda.synchronize()

        self.warmup_completed_ns = time.monotonic_ns()
        self.backend_ready_ns = self.warmup_completed_ns
        self._is_prepared = True

    def infer(self, frame: FramePacket) -> tuple[Detection, ...]:
        """Perform object detection on a single frame."""
        if self._model is None or self._image_processor is None:
            raise RuntimeError("RTDETRDetectorBackend must be prepared before calling infer()")

        model = self._model
        image_processor = self._image_processor

        from PIL import Image

        # Convert BGR image to RGB PIL Image
        image_rgb = np.ascontiguousarray(frame.image_bgr[:, :, ::-1])
        image_pil = Image.fromarray(image_rgb)  # pyright: ignore[reportUnknownMemberType]

        inputs = image_processor(images=image_pil, return_tensors="pt")  # pyright: ignore[reportUnknownMemberType]

        device = torch.device(self._config.device)
        torch_dtype = torch.float32
        if self._config.dtype == "float16":
            torch_dtype = torch.float16
        elif self._config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        pixel_values: torch.Tensor = inputs["pixel_values"].to(device, dtype=torch_dtype)

        with torch.no_grad():
            outputs: Any = model(pixel_values=pixel_values)

        results_raw = image_processor.post_process_object_detection(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            outputs,
            threshold=self._config.confidence_threshold,
            target_sizes=[(frame.height, frame.width)],
        )
        results = cast("list[dict[str, torch.Tensor]]", results_raw)

        detections: list[Detection] = []
        if len(results) > 0:
            result = results[0]
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                label_id = int(label)
                if self._person_class_id is not None and label_id != self._person_class_id:
                    continue

                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                detections.append(
                    Detection(
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        score=float(score),
                        class_id=label_id,
                        class_name=self._config.person_class_name,
                    )
                )

        return tuple(detections)

    @property
    def is_prepared(self) -> bool:
        return self._is_prepared

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def close(self) -> None:
        """Release backend resources."""
        if self._is_closed:
            return
        self._is_closed = True
        self._model = None
        self._image_processor = None
        from usecases.video_analytics.lifecycle import CoexistenceTracker
        CoexistenceTracker.detector_closed(id(self))


class FakeDetectorBackend(DetectorBackend):
    """Deterministic synthetic detector backend for testing and smoke execution."""

    def __init__(
        self,
        model_id: str = "fake-rtdetr-v1",
        confidence_threshold: float = 0.5,
        detections: tuple[Detection, ...] | None = None,
    ) -> None:
        self._model_id: str = model_id
        self._confidence_threshold: float = confidence_threshold
        self._custom_detections: tuple[Detection, ...] | None = detections
        self._is_prepared: bool = False
        self._is_closed: bool = False

        self.model_construction_start_ns: int = 0
        self.weights_loaded_ns: int = 0
        self.device_transfer_completed_ns: int = 0
        self.warmup_completed_ns: int = 0
        self.backend_ready_ns: int = 0

        from usecases.video_analytics.lifecycle import CoexistenceTracker
        CoexistenceTracker.detector_created(id(self))

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.FAKE

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_prepared(self) -> bool:
        return self._is_prepared

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def prepare(self, sample_frame: FramePacket) -> None:
        if self._is_prepared or self._is_closed:
            return

        t0 = time.monotonic_ns()
        self.model_construction_start_ns = t0
        self.weights_loaded_ns = t0 + 100_000  # +100us
        self.device_transfer_completed_ns = t0 + 200_000  # +200us
        self.warmup_completed_ns = t0 + 500_000  # +500us
        self.backend_ready_ns = self.warmup_completed_ns
        self._is_prepared = True

    def infer(self, frame: FramePacket) -> tuple[Detection, ...]:
        if self._is_closed:
            raise RuntimeError("Cannot infer on closed FakeDetectorBackend")

        if self._custom_detections is not None:
            return self._custom_detections

        bbox = BoundingBox(
            x1=10.0 + (frame.frame_id % 50),
            y1=20.0 + (frame.frame_id % 50),
            x2=100.0 + (frame.frame_id % 50),
            y2=200.0 + (frame.frame_id % 50),
        )
        det = Detection(
            box=bbox,
            score=0.95,
            class_id=0,
            class_name="person",
        )
        return (det,)

    def close(self) -> None:
        self._is_closed = True
        from usecases.video_analytics.lifecycle import CoexistenceTracker
        CoexistenceTracker.detector_closed(id(self))
