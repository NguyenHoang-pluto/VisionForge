"""Readiness probe for Redis."""

from __future__ import annotations

import time

from visionforge.domain.health import ComponentHealth, ComponentStatus
from visionforge.infra.redis.client import get_redis


class RedisProbe:
    name = "redis"
    required_for_readiness = True

    async def check(self) -> ComponentHealth:
        started = time.perf_counter()
        await get_redis().ping()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ComponentHealth(
            name=self.name,
            status=ComponentStatus.OK,
            latency_ms=round(elapsed_ms, 2),
        )
