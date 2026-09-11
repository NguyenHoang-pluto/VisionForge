"""Render worker entrypoint. Queue: ``render``. Pool: solo.

Kept separate from ``gpu`` so that long FFmpeg encodes never queue behind a model
load, and vice versa.
"""

from visionforge.infra.queue.celery_app import QUEUE_RENDER, celery_app

app = celery_app
QUEUE = QUEUE_RENDER

__all__ = ["QUEUE", "app"]
