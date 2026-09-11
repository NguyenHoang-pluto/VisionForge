"""Readiness probe for PostgreSQL."""

from __future__ import annotations

import time

from sqlalchemy import text

from visionforge.domain.health import ComponentHealth, ComponentStatus
from visionforge.infra.db.engine import get_engine


class DatabaseProbe:
    """Opens a connection and runs ``SELECT 1``.

    A real round-trip rather than a pool inspection: a pool can hold handles to a
    database that has since gone away.
    """

    name = "postgres"
    required_for_readiness = True

    async def check(self) -> ComponentHealth:
        started = time.perf_counter()
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ComponentHealth(
            name=self.name,
            status=ComponentStatus.OK,
            latency_ms=round(elapsed_ms, 2),
        )
