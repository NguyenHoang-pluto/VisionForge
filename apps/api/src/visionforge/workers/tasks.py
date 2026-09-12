"""Celery task definitions.

Tasks are deliberately thin. They open a session, load the job, hand control to
``JobRunner``, and translate the two outcomes Celery needs to know about: a retry
request and a duplicate upload.

Celery's own retry machinery does not make the backoff *decision* -- ``JobRunner``
does, and persists it, so the state survives a broker flush. Celery is only asked
to redeliver the message at the time PostgreSQL already recorded.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from celery import Task
from sqlalchemy.orm import Session

from visionforge.domain.errors import DuplicateMediaError
from visionforge.domain.jobs import JobStatus, StepStatus
from visionforge.domain.media import MediaStatus
from visionforge.infra.db.models import Job, MediaAsset
from visionforge.infra.db.sync_session import get_sync_sessionmaker
from visionforge.infra.ffmpeg import assert_ffmpeg_available
from visionforge.infra.queue.celery_app import QUEUE_CPU, celery_app
from visionforge.infra.redis.events import publish_sync
from visionforge.workers import media_ingest
from visionforge.workers.runtime import JobContext, JobRunner

logger = logging.getLogger(__name__)

TERMINAL = (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


@celery_app.task(
    bind=True,
    name="visionforge.media_ingest",
    queue=QUEUE_CPU,
    acks_late=True,
    max_retries=None,  # the retry budget lives in jobs.max_attempts, not here
)
def media_ingest_task(self: Task, job_id: str) -> str:
    """Run one media-ingest job.

    The return value is informational. Nothing reads it to determine status,
    because PostgreSQL is the source of truth (ADR-0003).
    """
    assert_ffmpeg_available()

    with get_sync_sessionmaker()() as session:
        job = session.get(Job, UUID(job_id))
        if job is None:
            logger.warning("job not found; message discarded", extra={"job_id": job_id})
            return "missing"

        if JobStatus(job.status) in TERMINAL:
            # Redelivery of an already-finished job. ``acks_late`` makes this
            # possible when a worker dies between finishing and acking, and
            # doing nothing is the correct response.
            return str(job.status)

        if job.cancel_requested:
            _finish(session, job, JobStatus.CANCELLED)
            return str(JobStatus.CANCELLED)

        ctx = JobContext(session, job)
        try:
            JobRunner(session, job).run(media_ingest.STEPS)
        except DuplicateMediaError as exc:
            _resolve_duplicate(session, job, exc)
            return str(JobStatus.SUCCEEDED)
        finally:
            media_ingest.cleanup(ctx)

        session.refresh(job)
        status = JobStatus(job.status)

        if status is JobStatus.RETRY_WAIT:
            return _schedule_retry(self, session, job)

        if status is JobStatus.FAILED:
            _mark_media_failed(session, job)

        return str(status)


# --------------------------------------------------------------------- outcomes
def _schedule_retry(task: Task, session: Session, job: Job) -> str:
    """Hand the redelivery to Celery at the time the runner already persisted."""
    delay = 0.0
    if job.retry_at is not None:
        delay = max(0.0, (job.retry_at - datetime.now(UTC)).total_seconds())

    job.status = JobStatus.QUEUED
    job.queued_at = datetime.now(UTC)
    session.commit()

    logger.info(
        "scheduling retry",
        extra={"job_id": str(job.id), "attempt": job.attempts, "delay_s": round(delay, 1)},
    )
    raise task.retry(countdown=delay)


def _resolve_duplicate(session: Session, job: Job, exc: DuplicateMediaError) -> None:
    """A duplicate upload is a successful outcome, not a failure.

    The redundant pending row is deleted and the job records which asset the
    caller should use instead. The uploaded bytes are left for the object
    lifecycle rule rather than deleted here, where they could race a reader.
    """
    for step in job.steps:
        if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
            step.status = StepStatus.SKIPPED
            step.finished_at = datetime.now(UTC)

    media = session.get(MediaAsset, job.media_id)
    if media is not None:
        session.delete(media)

    job.media_id = exc.duplicate_of
    job.status = JobStatus.SUCCEEDED
    job.finished_at = datetime.now(UTC)
    job.error = None
    job.result = {"duplicate": True, "duplicate_of": str(exc.duplicate_of)}
    session.commit()

    publish_sync(
        job.id,
        {
            "event": "job.succeeded",
            "job_id": str(job.id),
            "status": str(JobStatus.SUCCEEDED),
            "progress": 1.0,
            "duplicate_of": str(exc.duplicate_of),
            "ts": datetime.now(UTC).isoformat(),
        },
    )


def _finish(session: Session, job: Job, status: JobStatus) -> None:
    job.status = status
    job.finished_at = datetime.now(UTC)
    session.commit()


def _mark_media_failed(session: Session, job: Job) -> None:
    media = session.get(MediaAsset, job.media_id)
    if media is not None:
        media.status = MediaStatus.FAILED
        media.error = job.error
        session.commit()


__all__ = ["media_ingest_task"]
