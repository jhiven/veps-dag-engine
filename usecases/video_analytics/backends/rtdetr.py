"""RT-DETR object detector backend adapter using PyTorch and Hugging Face Transformers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from usecases.video_analytics.config import RTDETRConfig
from usecases.video_analytics.contracts import BackendKind, BoundingBox, Detection, DetectorBackend, FramePacket

__all__ = [
    "RTDETRDetectorBackend",
    "FakeDetectorBackend",
]


class RTDETRDetectorBackend(DetectorBackend):
    """Production RT-DETR backend adapter for Hugging Face RTDetrForObjectDetection."""

    def __init__(self, config: RTDETRConfig) -> None:
        self._config: RTDETRConfig = config
        self._model: Any | None = None
        self._image_processor: Any | None = None
        self._person_class_id: int | None = None
        self._is_prepared: bool = False
        self._is_closed: bool = False

        # Preparation subphase timestamps (ns)
        self.model_construction_start_ns: int = 0
        self.weights_loaded_ns: int = 0
        self.device_transfer_completed_ns: int = 0
        self.warmup_completed_ns: int = 0
        self.backend_ready_ns: int = 0

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

        try:
            import torch
            from transformers import (
                RTDetrForObjectDetection, 
                RTDetrImageProcessor,
            )
        except ImportError as err:
            raise RuntimeError(
                "PyTorch and Hugging Face Transformers are required for RTDETRDetectorBackend. "
                "Install with: uv sync --group realworld-video"
            ) from err

        if self._config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self._config.device!r} was requested, but CUDA is not available on this system."
                )

        image_processor = RTDetrImageProcessor.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            self._config.model_id,
            local_files_only=self._config.local_files_only,
        )
        model = RTDetrForObjectDetection.from_pretrained( # pyright: ignore[reportUnknownMemberType]
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

        to_fn: Any = getattr(model, "to")
        to_fn(device, torch_dtype)
        self.device_transfer_completed_ns = time.monotonic_ns()

        self._model = model
        self._image_processor = image_processor

        # Resolve person class ID from id2label
        config_obj = model.config
        id2label: dict[int, str] | dict[str, str] = config_obj.id2label or {}
        person_id: int | None = None
        for cid, name in id2label.items():
            if str(name).lower() == self._config.person_class_name.lower():
                person_id = int(cid)
                break
        if person_id is None:
            # Default fallback to 0 if COCO person class
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

        import torch
        from PIL import Image

        # Convert BGR image to RGB PIL Image
        image_rgb: np.ndarray[Any, Any] = np.ascontiguousarray(frame.image_bgr[:, :, ::-1])
        image_pil = Image.fromarray(image_rgb)

        inputs = self._image_processor(images=image_pil, return_tensors="pt")
        device = torch.device(self._config.device) 
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            results = self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=[(frame.height, frame.width)],
                threshold=self._config.confidence_threshold,
            )[0]

        boxes = results["boxes"].cpu().numpy() 
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cid = int(label)
            cname = self._config.person_class_name if cid == self._person_class_id else f"class_{cid}"
            detections.append(
                Detection(
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(score),
                    class_id=cid,
                    class_name=cname,
                )
            )

        return tuple(detections)

    def close(self) -> None:
        """Release PyTorch model references and CUDA memory."""
        if self._is_closed:
            return
        self._is_closed = True
        self._model = None
        self._image_processor = None


class FakeDetectorBackend(DetectorBackend):
    """Fake detector backend for unit testing without GPU or network dependencies."""

    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_r18vd",
        detections: tuple[Detection, ...] | None = None,
        prepare_delay_ns: int = 0,
    ) -> None:
        self._model_id: str = model_id
        self._detections: tuple[Detection, ...] = detections if detections is not None else (
            Detection(
                box=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
                score=0.92,
                class_id=0,
                class_name="person",
            ),
        )
        self._prepare_delay_ns: int = prepare_delay_ns
        self.is_prepared: bool = False
        self.is_closed: bool = False

    @property
    def backend_kind(self) -> BackendKind:
        return BackendKind.FAKE

    @property
    def model_id(self) -> str:
        return self._model_id

    def prepare(self, sample_frame: FramePacket) -> None:
        if self._prepare_delay_ns > 0:
            time.sleep(self._prepare_delay_ns / 1e9)
        self.is_prepared = True

    def infer(self, frame: FramePacket) -> tuple[Detection, ...]:
        return self._detections

    def close(self) -> None:
        self.is_closed = True

