"""Video analytics use case module."""

from __future__ import annotations

from usecases.video_analytics.application import VideoAnalyticsApplication
from usecases.video_analytics.config import (
    ByteTrackConfig,
    FileVideoSourceConfig,
    NullSinkConfig,
    RTDETRConfig,
    VideoAnalyticsConfig,
)
from usecases.video_analytics.contracts import (
    BackendKind,
    CUDAMemorySamplerProtocol,
    CUDAMemorySnapshot,
    DetectorBackend,
    DropReason,
    ExecutionMode,
    FramePacket,
    FrameSource,
    RTSPPublisherProtocol,
    TerminalStatus,
    TrackerBackend,
)
from usecases.video_analytics.gpu_memory import PyTorchCUDAMemorySampler
from usecases.video_analytics.ingress import BoundedIngressQueue, RTSPCaptureIngress
from usecases.video_analytics.metrics import FlowMetricsSummary, calculate_flow_metrics
from usecases.video_analytics.publisher import FFmpegRTSPPublisher
from usecases.video_analytics.source import FileVideoSource, RTSPVideoSource
from usecases.video_analytics.lifecycle import CoexistenceTracker

__all__ = [
    "VideoAnalyticsApplication",
    "VideoAnalyticsConfig",
    "RTDETRConfig",
    "ByteTrackConfig",
    "FileVideoSourceConfig",
    "NullSinkConfig",
    "ExecutionMode",
    "BackendKind",
    "TerminalStatus",
    "DropReason",
    "FramePacket",
    "FrameSource",
    "DetectorBackend",
    "TrackerBackend",
    "RTSPPublisherProtocol",
    "CUDAMemorySamplerProtocol",
    "CUDAMemorySnapshot",
    "FFmpegRTSPPublisher",
    "BoundedIngressQueue",
    "RTSPCaptureIngress",
    "FileVideoSource",
    "RTSPVideoSource",
    "PyTorchCUDAMemorySampler",
    "FlowMetricsSummary",
    "calculate_flow_metrics",
    "CoexistenceTracker",
]
