"""Health/readiness value objects and the port every dependency probe implements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ComponentStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """The outcome of probing one infrastructure dependency."""

    name: str
    status: ComponentStatus
    latency_ms: float | None = None
    detail: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.status is ComponentStatus.OK


@runtime_checkable
class HealthProbe(Protocol):
    """A dependency that can report whether it is reachable.

    Defining this as a port is what lets the readiness endpoint be unit-tested
    without Postgres, Redis or MinIO running.
    """

    @property
    def name(self) -> str: ...

    @property
    def required_for_readiness(self) -> bool: ...

    async def check(self) -> ComponentHealth: ...
