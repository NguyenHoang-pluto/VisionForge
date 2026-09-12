"""Model registry: lazy loading, warm caching, LRU eviction.

Two costs shape this module. Loading OpenCLIP ViT-B/32 takes seconds, so a model
must stay warm across jobs rather than reload per task. And the card has 4 GiB,
so a warm model that is not being used must be evictable when something else
needs the room.

The registry owns *residency*. Adapters own *inference*. Keeping those apart is
what lets a model be swapped without touching the memory policy, and the memory
policy tuned without touching any model.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, TypeVar

from visionforge.infra.ml.device import (
    empty_cache,
    peak_allocated_mb,
    reset_peak_stats,
    select_device,
    synchronize,
    vram_snapshot,
)
from visionforge.infra.ml.lease import GpuLeaseManager, LeaseReceipt, get_lease_manager

logger = logging.getLogger(__name__)


class LoadedModel(Protocol):
    """Anything the registry can hold and later throw away."""

    def unload(self) -> None: ...


TModel = TypeVar("TModel", bound=LoadedModel)


class ModelSpec(Protocol):
    """How to identify and build one model.

    Adapters implement this. The registry never knows what PyTorch, OpenCLIP or
    OpenCV is -- only that something can be named, sized, and constructed.
    """

    @property
    def key(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def estimated_vram_mb(self) -> int: ...

    def load(self) -> LoadedModel: ...


class ModelRegistry:
    """Process-wide cache of loaded models.

    One registry per worker process. The Celery ``--pool=solo`` GPU worker means
    one process per GPU, so "process-wide" and "device-wide" coincide -- which is
    exactly why the solo pool was chosen in Phase 0 rather than worked around.
    """

    def __init__(self, lease_manager: GpuLeaseManager | None = None) -> None:
        self._lease = lease_manager or get_lease_manager()
        self._models: dict[str, LoadedModel] = {}
        self._specs: dict[str, ModelSpec] = {}

    @property
    def loaded_keys(self) -> tuple[str, ...]:
        return tuple(self._models)

    def register(self, spec: ModelSpec) -> None:
        """Declare a model. Does not load it -- that happens on first use."""
        self._specs[spec.key] = spec

    def is_loaded(self, key: str) -> bool:
        return key in self._models

    # ------------------------------------------------------------------- load
    def get(self, key: str) -> LoadedModel:
        """Return a loaded model, loading and making room for it if needed."""
        existing = self._models.get(key)
        if existing is not None:
            # Touch it so the LRU order reflects real use.
            self._lease.register(key, self._specs[key].estimated_vram_mb)
            return existing

        spec = self._specs.get(key)
        if spec is None:
            raise KeyError(f"no model registered under {key!r}")

        for victim in self._lease.plan_eviction(spec.estimated_vram_mb):
            self.unload(victim)

        started = time.perf_counter()
        before = vram_snapshot()
        model = spec.load()
        synchronize()
        after = vram_snapshot()

        self._models[key] = model
        self._lease.register(key, spec.estimated_vram_mb)

        logger.info(
            "model loaded",
            extra={
                "model": key,
                "version": spec.version,
                "device": select_device(),
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
                "vram_delta_mb": round(after.allocated_mb - before.allocated_mb, 1),
                "estimated_mb": spec.estimated_vram_mb,
            },
        )
        return model

    def unload(self, key: str) -> None:
        model = self._models.pop(key, None)
        if model is None:
            return
        model.unload()
        self._lease.release(key)
        # Return the blocks to the driver. Without this PyTorch keeps them
        # reserved and the next differently-shaped allocation can still OOM.
        empty_cache()
        logger.info("model unloaded", extra={"model": key})

    def unload_all(self) -> None:
        for key in list(self._models):
            self.unload(key)

    # ------------------------------------------------------------------ leased
    def lease(self, key: str) -> _LeasedModel:
        """Acquire the GPU and the model together.

        A single call site for "I am about to use the GPU" means measurement,
        eviction and the mutex cannot be forgotten independently of each other.
        """
        spec = self._specs.get(key)
        if spec is None:
            raise KeyError(f"no model registered under {key!r}")
        return _LeasedModel(self, self._lease, spec)


class _LeasedModel:
    """Context manager yielding a loaded model while holding the GPU lease.

    Records what it measured on exit, so every GPU step reports real numbers
    rather than estimates.
    """

    def __init__(self, registry: ModelRegistry, lease: GpuLeaseManager, spec: ModelSpec) -> None:
        self._registry = registry
        self._lease = lease
        self._spec = spec
        self._started = 0.0
        self._before = 0.0
        self._evicted: tuple[str, ...] = ()
        self.receipt: LeaseReceipt | None = None

    def __enter__(self) -> LoadedModel:
        self._guard = self._lease.acquire(
            model=self._spec.key, estimated_mb=self._spec.estimated_vram_mb
        )
        self._guard.__enter__()

        self._evicted = self._lease.plan_eviction(self._spec.estimated_vram_mb)
        reset_peak_stats()
        self._before = vram_snapshot().allocated_mb
        self._started = time.perf_counter()
        return self._registry.get(self._spec.key)

    def __exit__(self, *exc: object) -> None:
        synchronize()
        duration_ms = (time.perf_counter() - self._started) * 1000
        after = vram_snapshot()
        self.receipt = LeaseReceipt(
            model=f"{self._spec.key}@{self._spec.version}",
            device=select_device(),
            vram_before_mb=self._before,
            vram_peak_mb=peak_allocated_mb(),
            vram_after_mb=after.allocated_mb,
            duration_ms=duration_ms,
            evicted=self._evicted,
        )
        self._guard.__exit__(None, None, None)


#: Process-wide registry, for the same reason as the lease manager.
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    """Unload everything and drop the singleton. Tests and worker shutdown."""
    global _registry
    if _registry is not None:
        _registry.unload_all()
    _registry = None
