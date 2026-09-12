"""GPU worker entrypoint. Queue: ``gpu``. Pool: solo.

The solo pool is the design, not a Windows workaround: with ~3.2 GB of usable
VRAM exactly one model may be resident and one inference may run at a time, so a
single-slot worker *is* the GPU mutex. See docs/adr/0003-job-state-ownership.md.
"""

from visionforge.infra.queue.celery_app import QUEUE_GPU, celery_app
from visionforge.workers import tasks  # noqa: F401  (registers tasks with Celery)

app = celery_app
QUEUE = QUEUE_GPU

__all__ = ["QUEUE", "app"]
