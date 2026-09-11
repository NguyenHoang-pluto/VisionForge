"""Readiness aggregation, exercised entirely with fakes.

This is the payoff of defining ``HealthProbe`` as a port: the readiness rules --
including the timeout and the required/optional distinction -- are testable with
no Postgres, Redis or MinIO anywhere near the test runner.
"""

from __future__ import annotations

import asyncio

import pytest

from visionforge.application.health_service import HealthService
from visionforge.domain.health import ComponentHealth, ComponentStatus


class FakeProbe:
    def __init__(
        self,
        name: str,
        *,
        required: bool = True,
        status: ComponentStatus = ComponentStatus.OK,
        delay_s: float = 0.0,
        raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.required_for_readiness = required
        self._status = status
        self._delay_s = delay_s
        self._raises = raises

    async def check(self) -> ComponentHealth:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._raises is not None:
            raise self._raises
        return ComponentHealth(name=self.name, status=self._status, latency_ms=1.0)


@pytest.mark.asyncio
async def test_all_healthy_is_ready() -> None:
    service = HealthService([FakeProbe("postgres"), FakeProbe("redis")])
    report = await service.readiness()

    assert report.ready is True
    assert len(report.components) == 2


@pytest.mark.asyncio
async def test_failing_required_probe_blocks_readiness() -> None:
    service = HealthService([FakeProbe("postgres", status=ComponentStatus.FAILED)])
    report = await service.readiness()

    assert report.ready is False


@pytest.mark.asyncio
async def test_failing_optional_probe_does_not_block_readiness() -> None:
    service = HealthService(
        [
            FakeProbe("postgres"),
            FakeProbe("storage", required=False, status=ComponentStatus.FAILED),
        ]
    )
    report = await service.readiness()

    assert report.ready is True
    assert any(c.name == "storage" and not c.is_ready for c in report.components)


@pytest.mark.asyncio
async def test_raising_probe_is_reported_not_propagated() -> None:
    service = HealthService([FakeProbe("redis", raises=ConnectionError("refused"))])
    report = await service.readiness()

    assert report.ready is False
    assert report.components[0].detail == "ConnectionError"


@pytest.mark.asyncio
async def test_hanging_probe_is_timed_out() -> None:
    service = HealthService([FakeProbe("postgres", delay_s=5.0)], timeout_s=0.05)
    report = await service.readiness()

    assert report.ready is False
    assert "timed out" in (report.components[0].detail or "")
