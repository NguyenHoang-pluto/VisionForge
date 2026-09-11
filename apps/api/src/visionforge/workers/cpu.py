"""CPU worker entrypoint. Queue: ``cpu``. Pool: threads (I/O and subprocess bound)."""

from visionforge.infra.queue.celery_app import QUEUE_CPU, celery_app

app = celery_app
QUEUE = QUEUE_CPU

__all__ = ["QUEUE", "app"]
