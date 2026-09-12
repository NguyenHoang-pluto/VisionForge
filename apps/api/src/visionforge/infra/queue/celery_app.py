"""Celery application.

Three queues, as decided in Phase 0:

- ``cpu``     probing, thumbnails, deterministic analysis   (threads pool)
- ``gpu``     model inference                               (solo pool = the GPU mutex)
- ``render``  FFmpeg encoding                               (solo pool, long-running)

Results are disabled deliberately. Job state lives in PostgreSQL; reading status
from a Celery result backend is what makes these systems undebuggable.

**Every worker must be started with a unique ``-n`` node name** (``-n cpu@%h``,
``-n gpu@%h``, ``-n render@%h``). Celery derives a node name from the hostname by
default, so two workers on one machine register as the same node and then share
the broker-side bookkeeping that tracks which messages each has reserved. The
result is silently lost tasks: a message is consumed, the job row stays QUEUED
with ``attempts=0``, and nothing is logged. This was latent through Phase 2,
which ran a single worker, and surfaced the moment Phase 3 added a second queue.
"""

from __future__ import annotations

from celery import Celery

from visionforge.core.config import get_settings

QUEUE_CPU = "cpu"
QUEUE_GPU = "gpu"
QUEUE_RENDER = "render"


def build_celery_app() -> Celery:
    settings = get_settings()
    app = Celery("visionforge", broker=settings.redis_url)
    app.conf.update(
        # --- serialization ---
        task_serializer="json",
        accept_content=["json"],
        result_backend=None,
        task_ignore_result=True,
        # --- routing ---
        task_default_queue=QUEUE_CPU,
        task_queues={
            QUEUE_CPU: {"exchange": QUEUE_CPU, "routing_key": QUEUE_CPU},
            QUEUE_GPU: {"exchange": QUEUE_GPU, "routing_key": QUEUE_GPU},
            QUEUE_RENDER: {"exchange": QUEUE_RENDER, "routing_key": QUEUE_RENDER},
        },
        # --- reliability ---
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # long tasks must not be hoarded by one worker
        broker_connection_retry_on_startup=True,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = build_celery_app()
