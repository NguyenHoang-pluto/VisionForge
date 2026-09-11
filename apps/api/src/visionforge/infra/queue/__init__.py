"""Celery adapter. Transport only -- PostgreSQL owns job state (Phase 0 rule #3)."""

from visionforge.infra.queue.celery_app import celery_app

__all__ = ["celery_app"]
