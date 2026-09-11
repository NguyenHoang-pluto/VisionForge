"""Readiness probe for object storage."""

from __future__ import annotations

import asyncio
import time

from visionforge.core.config import get_settings
from visionforge.domain.health import ComponentHealth, ComponentStatus
from visionforge.infra.storage.client import build_s3_client


class StorageProbe:
    """Verifies the media bucket is reachable via ``head_bucket``.

    Not required for readiness: the API never touches media bytes (Phase 0 rule
    #1), so it can still serve requests while storage is briefly unavailable.
    Jobs that need storage will fail on their own and report it honestly.
    """

    name = "storage"
    required_for_readiness = False

    async def check(self) -> ComponentHealth:
        settings = get_settings()
        started = time.perf_counter()
        client = build_s3_client(settings)
        await asyncio.to_thread(client.head_bucket, Bucket=settings.s3_bucket_media)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ComponentHealth(
            name=self.name,
            status=ComponentStatus.OK,
            latency_ms=round(elapsed_ms, 2),
            detail=f"bucket={settings.s3_bucket_media}",
        )
