"""GPU model infrastructure: device, lease, registry, adapters.

Never imported by ``visionforge.api`` or ``visionforge.application``.
"""

from visionforge.infra.ml.clip_adapter import ClipEmbeddingAnalyzer, embed_text
from visionforge.infra.ml.device import describe, select_device, vram_snapshot
from visionforge.infra.ml.face_adapter import FaceDetectionAnalyzer
from visionforge.infra.ml.lease import (
    GpuBudgetExceededError,
    GpuBusyError,
    GpuLeaseManager,
    get_lease_manager,
)
from visionforge.infra.ml.registry import (
    ModelRegistry,
    VramEstimateError,
    get_registry,
    reset_registry,
)

__all__ = [
    "ClipEmbeddingAnalyzer",
    "FaceDetectionAnalyzer",
    "GpuBudgetExceededError",
    "GpuBusyError",
    "GpuLeaseManager",
    "ModelRegistry",
    "VramEstimateError",
    "describe",
    "embed_text",
    "get_lease_manager",
    "get_registry",
    "reset_registry",
    "select_device",
    "vram_snapshot",
]
