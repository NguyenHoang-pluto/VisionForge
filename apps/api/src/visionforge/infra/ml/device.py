"""Device selection and VRAM measurement.

The only module that imports ``torch`` for device concerns. Everything else asks
here, so switching to CPU on a machine without a GPU is one decision in one
place rather than a conditional in every adapter.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from visionforge.infra.ml.lease import VramSnapshot

logger = logging.getLogger(__name__)

#: Ampere (8.6) supports fp16 tensor cores. Below 7.0, fp16 is emulated and
#: slower than fp32, so precision is chosen from the hardware rather than assumed.
MIN_FP16_CAPABILITY = (7, 0)


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


@lru_cache(maxsize=1)
def select_device() -> str:
    """``"cuda"`` when a GPU is usable, otherwise ``"cpu"``.

    Falling back rather than failing: analysis on CPU is slow but correct, and a
    developer without a GPU should still be able to run the pipeline.
    """
    if not cuda_available():
        logger.warning("no CUDA device; model inference will run on CPU")
        return "cpu"
    return "cuda"


@lru_cache(maxsize=1)
def device_name() -> str:
    return torch.cuda.get_device_name(0) if cuda_available() else "cpu"


@lru_cache(maxsize=1)
def use_fp16() -> bool:
    """Half precision on Ampere: half the VRAM, faster, no accuracy cost here.

    CLIP embeddings and face boxes are not numerically delicate; fp16 is the
    right default on a 4 GB card and the memory saving is what makes a second
    model resident feasible at all.
    """
    if not cuda_available():
        return False
    return torch.cuda.get_device_capability(0) >= MIN_FP16_CAPABILITY


def vram_snapshot() -> VramSnapshot:
    """Current VRAM state. Zeroes on CPU, so callers need no special case."""
    if not cuda_available():
        return VramSnapshot(total_mb=0.0, allocated_mb=0.0, reserved_mb=0.0, free_mb=0.0)

    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    # Free as the driver sees it, not total-minus-allocated: PyTorch's caching
    # allocator holds reserved blocks that are unavailable to anything else.
    free_bytes, _ = torch.cuda.mem_get_info()
    return VramSnapshot(
        total_mb=total,
        allocated_mb=allocated,
        reserved_mb=reserved,
        free_mb=free_bytes / 1024**2,
    )


def reset_peak_stats() -> None:
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def peak_allocated_mb() -> float:
    if not cuda_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024**2)


def empty_cache() -> None:
    """Return cached blocks to the driver.

    Called after unloading a model. Without it, PyTorch keeps the freed memory
    reserved, and the next allocation of a different shape can still OOM on a
    card this small.
    """
    if cuda_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    """Wait for queued kernels. Required before any timing measurement is real."""
    if cuda_available():
        torch.cuda.synchronize()


def describe() -> dict[str, object]:
    """Device summary for health and job metrics."""
    snapshot = vram_snapshot()
    return {
        "device": select_device(),
        "device_name": device_name(),
        "cuda_available": cuda_available(),
        "fp16": use_fp16(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)) if cuda_available() else None,
        **snapshot.as_dict(),
    }


def reset_caches() -> None:
    """Clear the cached device decisions. Tests only."""
    for fn in (cuda_available, select_device, device_name, use_fp16):
        fn.cache_clear()
