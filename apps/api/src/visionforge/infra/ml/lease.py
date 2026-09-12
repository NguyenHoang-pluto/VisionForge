"""GPU lease management.

The RTX 3050 has 4096 MiB, of which roughly 3.2-3.5 GiB is usable under WDDM --
Windows reserves the rest for the desktop compositor. Two model loads that each
fit individually can therefore still fail together, and a CUDA OOM in the middle
of a batch is an expensive way to discover that.

This module makes VRAM a budget that is checked before allocation rather than
discovered after it:

- one in-process lock, so a worker never runs two inferences concurrently;
- a declared budget per adapter, checked against what is actually free;
- LRU eviction when the next model will not fit.

The process-level mutex is the Celery ``--pool=solo`` GPU worker (ADR-0003).
This lock is the second line: it also protects against a task that internally
tries to parallelise, and it is what makes the budget arithmetic meaningful.

**Known limitation: the lease is per-process, not per-device.**

``threading.RLock`` and the residency table live in one process. Two processes
that both touch the GPU -- a running worker plus a pytest session, or two GPU
workers started by mistake -- each believe they own the whole card, and nothing
here can stop them overcommitting it.

The design relies on there being exactly one GPU process: the solo-pool worker
on the ``gpu`` queue. That holds in production and in normal development; it is
violated whenever GPU tests run while a worker is live, which is why those tests
are marked ``gpu`` and excluded from every default run.

Making this robust needs a cross-process primitive -- a filesystem lock, or a
Redis lease keyed on the device UUID. That is deliberately not built yet: it
would be real machinery in service of a configuration the deployment does not
create. Revisit it when more than one process is ever meant to share a card.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Usable VRAM budget in MiB.
#:
#: Deliberately below the 4096 MiB the card reports. WDDM reserves several
#: hundred MiB for the desktop, PyTorch's caching allocator holds reserved blocks
#: beyond what is allocated, and cuDNN workspaces spike transiently. Budgeting
#: the full card is how you get an OOM at 95% utilisation.
DEFAULT_VRAM_BUDGET_MB = 3000

#: How long a task waits for the lease before giving up. Long enough for a
#: previous inference to finish, short enough that a deadlock surfaces as a
#: failed job rather than a hung worker.
DEFAULT_LEASE_TIMEOUT_S = 300.0


class GpuBusyError(RuntimeError):
    """The lease could not be acquired within the timeout."""


class GpuBudgetExceededError(RuntimeError):
    """The request does not fit the VRAM budget even with everything evicted."""


@dataclass(frozen=True, slots=True)
class VramSnapshot:
    """What the device reported at one instant, in MiB."""

    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float

    def as_dict(self) -> dict[str, float]:
        return {
            "vram_total_mb": round(self.total_mb, 1),
            "vram_allocated_mb": round(self.allocated_mb, 1),
            "vram_reserved_mb": round(self.reserved_mb, 1),
            "vram_free_mb": round(self.free_mb, 1),
        }


@dataclass(frozen=True, slots=True)
class LeaseReceipt:
    """What a completed lease measured. Lands in ``job_steps.metrics``."""

    model: str
    device: str
    vram_before_mb: float
    vram_peak_mb: float
    vram_after_mb: float
    duration_ms: float
    evicted: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "device": self.device,
            "vram_before_mb": round(self.vram_before_mb, 1),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "vram_after_mb": round(self.vram_after_mb, 1),
            "duration_ms": round(self.duration_ms, 1),
            "evicted": list(self.evicted),
        }


class GpuLeaseManager:
    """Serialises GPU work and enforces a VRAM budget.

    Not a general resource scheduler. It does one job -- make sure only one thing
    is on the GPU at a time and that it was known to fit before it started.
    """

    def __init__(
        self,
        *,
        budget_mb: int = DEFAULT_VRAM_BUDGET_MB,
        timeout_s: float = DEFAULT_LEASE_TIMEOUT_S,
    ) -> None:
        self._budget_mb = budget_mb
        self._timeout_s = timeout_s
        self._lock = threading.RLock()
        #: Models currently resident, oldest first. Ordinary dict: insertion
        #: order is the LRU order, and re-inserting moves an entry to the end.
        self._resident: dict[str, int] = {}

    @property
    def budget_mb(self) -> int:
        return self._budget_mb

    @property
    def resident_models(self) -> tuple[str, ...]:
        return tuple(self._resident)

    @property
    def committed_mb(self) -> int:
        return sum(self._resident.values())

    # ------------------------------------------------------------------ budget
    def fits(self, estimated_mb: int) -> bool:
        """Whether a model fits without evicting anything."""
        return self.committed_mb + estimated_mb <= self._budget_mb

    def plan_eviction(self, estimated_mb: int) -> tuple[str, ...]:
        """Which residents must go, oldest first, for ``estimated_mb`` to fit.

        Raises if it cannot fit even on an empty device: better a clear error at
        the boundary than a CUDA OOM three layers down.
        """
        if estimated_mb > self._budget_mb:
            raise GpuBudgetExceededError(
                f"model needs {estimated_mb} MiB but the whole budget is " f"{self._budget_mb} MiB"
            )

        evict: list[str] = []
        freed = 0
        for name, size in self._resident.items():
            if self.committed_mb - freed + estimated_mb <= self._budget_mb:
                break
            evict.append(name)
            freed += size
        return tuple(evict)

    def register(self, name: str, estimated_mb: int) -> None:
        """Record a model as resident, or mark it as most recently used."""
        with self._lock:
            self._resident.pop(name, None)
            self._resident[name] = estimated_mb

    def release(self, name: str) -> None:
        with self._lock:
            self._resident.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._resident.clear()

    # ------------------------------------------------------------------- lease
    @contextmanager
    def acquire(self, *, model: str, estimated_mb: int) -> Iterator[None]:
        """Hold the GPU for one unit of work.

        Blocking with a timeout rather than failing fast: GPU jobs queue behind
        each other by design, and a caller that waits is the expected case.
        """
        if not self._lock.acquire(timeout=self._timeout_s):
            raise GpuBusyError(f"could not acquire the GPU for {model} within {self._timeout_s}s")
        try:
            logger.debug(
                "gpu lease acquired",
                extra={
                    "model": model,
                    "estimated_mb": estimated_mb,
                    "committed_mb": self.committed_mb,
                    "budget_mb": self._budget_mb,
                },
            )
            yield
        finally:
            self._lock.release()


#: Process-wide manager. One GPU, one budget; a second instance would each think
#: it owned the whole card.
_manager: GpuLeaseManager | None = None


def get_lease_manager() -> GpuLeaseManager:
    global _manager
    if _manager is None:
        _manager = GpuLeaseManager()
    return _manager


def reset_lease_manager() -> None:
    """Drop the singleton. Tests only."""
    global _manager
    _manager = None
