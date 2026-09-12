"""Job progress events over Redis.

Two structures per job, and the split is the point:

- a **snapshot** (``job:{id}:state``) holding the last known state;
- a **pub/sub channel** (``job:{id}:events``) carrying live updates.

SSE clients read the snapshot on connect and then subscribe. That is what makes a
mid-render page refresh cheap: the client is immediately correct, rather than
waiting for the next event to learn anything.

Redis is a delivery and cache mechanism here. PostgreSQL remains the source of
truth (ADR-0003); everything published below is also written to ``jobs`` and
``job_steps``.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

import redis as sync_redis

from visionforge.core.config import get_settings

logger = logging.getLogger(__name__)

#: Long enough to survive a worker restart, short enough not to accumulate.
SNAPSHOT_TTL_S = 24 * 60 * 60


def state_key(job_id: UUID) -> str:
    return f"job:{job_id}:state"


def channel(job_id: UUID) -> str:
    return f"job:{job_id}:events"


def publish_sync(job_id: UUID, event: dict[str, Any]) -> None:
    """Publish from a worker. Never raises: progress must not fail a job.

    A dropped progress event costs a stale UI for a few seconds. A job that fails
    because Redis hiccuped costs a re-render.
    """
    payload = json.dumps(event, default=str)
    try:
        client = sync_redis.Redis.from_url(
            get_settings().redis_url, decode_responses=True, socket_timeout=2
        )
        pipe = client.pipeline()
        pipe.set(state_key(job_id), payload, ex=SNAPSHOT_TTL_S)
        pipe.publish(channel(job_id), payload)
        pipe.execute()  # type: ignore[no-untyped-call]
        client.close()  # type: ignore[no-untyped-call]
    except Exception as exc:  # progress is best-effort by design
        logger.warning(
            "failed to publish job event", extra={"job_id": str(job_id), "error": str(exc)}
        )
