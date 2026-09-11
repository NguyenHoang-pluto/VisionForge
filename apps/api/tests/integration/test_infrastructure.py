"""Integration tests against live infrastructure.

Marked ``integration`` and excluded from the default run, because CI must stay
green on a machine with no Docker. Run them with::

    pytest -m integration
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from visionforge.core.config import get_settings
from visionforge.infra.db import get_engine
from visionforge.infra.redis import get_redis
from visionforge.infra.storage import build_s3_client

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_postgres_connect_query_disconnect() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_pgvector_extension_is_installed() -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        assert result.scalar_one_or_none() is not None, "run: alembic upgrade head"
    await engine.dispose()


@pytest.mark.asyncio
async def test_redis_roundtrip() -> None:
    client = get_redis()
    assert await client.ping() is True
    await client.set("visionforge:smoke", "ok", ex=10)
    assert await client.get("visionforge:smoke") == "ok"
    await client.delete("visionforge:smoke")


def test_storage_buckets_exist() -> None:
    settings = get_settings()
    existing = {b["Name"] for b in build_s3_client().list_buckets()["Buckets"]}
    assert set(settings.buckets).issubset(existing)
