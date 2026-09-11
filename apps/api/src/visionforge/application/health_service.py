"""Readiness aggregation.

Probes run concurrently and are individually timed out, so one wedged dependency
cannot make the readiness endpoint hang -- which would take the whole service out
of a load balancer for the wrong reason.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from visionforge.domain.health import ComponentHealth, ComponentStatus, HealthProbe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    components: tuple[ComponentHealth, ...]


class HealthService:
    """Aggregates the registered dependency probes into a readiness verdict."""

    def __init__(self, probes: list[HealthProbe], timeout_s: float = 2.0) -> None:
        self._probes = probes
        self._timeout_s = timeout_s

    async def readiness(self) -> ReadinessReport:
        results = await asyncio.gather(
            *(self._check_one(probe) for probe in self._probes),
        )
        required_ok = all(
            health.is_ready
            for probe, health in zip(self._probes, results, strict=True)
            if probe.required_for_readiness
        )
        return ReadinessReport(ready=required_ok, components=tuple(results))

    async def _check_one(self, probe: HealthProbe) -> ComponentHealth:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await probe.check()
        except TimeoutError:
            return ComponentHealth(
                name=probe.name,
                status=ComponentStatus.FAILED,
                detail=f"probe timed out after {self._timeout_s}s",
            )
        except Exception as exc:  # a probe must never raise into the caller
            logger.warning("health probe failed", extra={"probe": probe.name, "error": str(exc)})
            return ComponentHealth(
                name=probe.name,
                status=ComponentStatus.FAILED,
                detail=type(exc).__name__,
            )
