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
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from celery import Task
from sqlalchemy.orm import Session

from visionforge.domain.errors import DuplicateMediaError
from visionforge.domain.jobs import JobStatus, StepStatus
from visionforge.domain.media import MediaStatus
from visionforge.domain.render import RenderStatus
from visionforge.infra.db.models import Job, MediaAsset, RenderRow
from visionforge.infra.db.sync_session import get_sync_sessionmaker
from visionforge.infra.ffmpeg import FFmpegNotAvailableError, assert_ffmpeg_available
from visionforge.infra.queue.celery_app import (
    QUEUE_CPU,
    QUEUE_GPU,
    QUEUE_RENDER,
    celery_app,
)
from visionforge.infra.redis.events import publish_sync
from visionforge.workers import media_analysis, media_ingest, video_render
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

        # Checked here, after the job row is in hand, rather than at the top of
        # the task. Asserting before loading the job meant a worker without
        # FFmpeg raised, Celery recorded a task failure, and the job was left
        # QUEUED with attempts=0 and nothing on it saying why -- invisible from
        # the API and from the UI. Now the misconfiguration is written to the
        # job it broke.
        try:
            assert_ffmpeg_available()
        except FFmpegNotAvailableError as exc:
            _fail_job(session, job, code="ffmpeg_unavailable", message=str(exc))
            _mark_media_failed(session, job)
            logger.error("worker cannot run media jobs", extra={"error": str(exc)})
            return str(JobStatus.FAILED)

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


def _fail_job(session: Session, job: Job, *, code: str, message: str) -> None:
    """Record a failure that happened before the runner could start.

    Marked non-retryable: a worker missing FFmpeg will still be missing it on
    the next attempt, so burning the retry budget only delays the report.
    """
    job.status = JobStatus.FAILED
    job.finished_at = datetime.now(UTC)
    job.error = {
        "code": code,
        "message": message,
        "hint": "Install FFmpeg and put its bin directory on the worker's PATH.",
        "retryable": False,
    }
    session.commit()
    publish_sync(
        job.id,
        {
            "event": "job.failed",
            "job_id": str(job.id),
            "status": str(JobStatus.FAILED),
            "progress": 0.0,
            "error": job.error,
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


@celery_app.task(
    bind=True,
    name="visionforge.media_analyze_cpu",
    queue=QUEUE_CPU,
    acks_late=True,
    max_retries=None,
)
def media_analyze_cpu_task(self: Task, job_id: str) -> str:
    """Deterministic CPU analysis: quality, scenes, perceptual hash."""
    return _run_analysis(self, job_id, media_analysis.CPU_STEPS)


@celery_app.task(
    bind=True,
    name="visionforge.media_analyze_gpu",
    queue=QUEUE_GPU,
    acks_late=True,
    max_retries=None,
)
def media_analyze_gpu_task(self: Task, job_id: str) -> str:
    """Model inference: CLIP embeddings and face detection.

    Routed to the ``gpu`` queue, whose worker runs ``--pool=solo``. That single
    slot is the process-level GPU mutex (ADR-0003); ``GpuLeaseManager`` enforces
    the VRAM budget within it.
    """
    return _run_analysis(self, job_id, media_analysis.GPU_STEPS)


def _run_analysis(task: Task, job_id: str, steps: Mapping[str, Any]) -> str:
    """Shared body for both analysis job types.

    Identical to the ingest task's shape: the same terminal-status guard, the
    same cancellation check, the same retry hand-off. Analysis needs no FFmpeg
    assertion -- OpenCV decodes its own inputs.
    """
    with get_sync_sessionmaker()() as session:
        job = session.get(Job, UUID(job_id))
        if job is None:
            logger.warning("job not found; message discarded", extra={"job_id": job_id})
            return "missing"

        if JobStatus(job.status) in TERMINAL:
            return str(job.status)

        if job.cancel_requested:
            _finish(session, job, JobStatus.CANCELLED)
            return str(JobStatus.CANCELLED)

        ctx = JobContext(session, job)
        try:
            JobRunner(session, job).run(steps)
        finally:
            media_analysis.cleanup(ctx)

        session.refresh(job)
        status = JobStatus(job.status)

        if status is JobStatus.RETRY_WAIT:
            return _schedule_retry(task, session, job)

        return str(status)


@celery_app.task(
    bind=True,
    name="visionforge.render_video",
    queue=QUEUE_RENDER,
    acks_late=True,
    max_retries=None,
)
def render_video_task(self: Task, job_id: str) -> str:
    """Render one edit plan to an MP4.

    FFmpeg is asserted with the job in hand, for the same reason as ingest: a
    worker without FFmpeg must fail the job it broke with a readable error
    rather than strand it QUEUED with nothing recorded.
    """
    with get_sync_sessionmaker()() as session:
        job = session.get(Job, UUID(job_id))
        if job is None:
            logger.warning("job not found; message discarded", extra={"job_id": job_id})
            return "missing"

        if JobStatus(job.status) in TERMINAL:
            return str(job.status)

        if job.cancel_requested:
            _finish(session, job, JobStatus.CANCELLED)
            _mark_render(session, job, RenderStatus.CANCELLED)
            return str(JobStatus.CANCELLED)

        try:
            assert_ffmpeg_available()
        except FFmpegNotAvailableError as exc:
            _fail_job(session, job, code="ffmpeg_unavailable", message=str(exc))
            _mark_render(session, job, RenderStatus.FAILED, error=job.error)
            logger.error("worker cannot render", extra={"error": str(exc)})
            return str(JobStatus.FAILED)

        ctx = JobContext(session, job)
        try:
            JobRunner(session, job).run(video_render.STEPS)
        finally:
            video_render.cleanup(ctx)

        session.refresh(job)
        status = JobStatus(job.status)

        if status is JobStatus.RETRY_WAIT:
            return _schedule_retry(self, session, job)

        if status is JobStatus.FAILED:
            _mark_render(session, job, RenderStatus.FAILED, error=job.error)
        elif status is JobStatus.CANCELLED:
            _mark_render(session, job, RenderStatus.CANCELLED)

        return str(status)


def _mark_render(
    session: Session,
    job: Job,
    status: RenderStatus,
    *,
    error: dict[str, Any] | None = None,
) -> None:
    """Mirror a terminal job outcome onto its render row.

    The render row is what the UI reads. A job that failed while its render
    still says "rendering" is the kind of inconsistency that makes a dashboard
    untrustworthy.
    """
    render_id = job.params.get("render_id")
    if not render_id:
        return
    render = session.get(RenderRow, UUID(str(render_id)))
    if render is None:
        return
    render.status = status
    if error is not None:
        render.error = error
    session.commit()


__all__ = [
    "media_analyze_cpu_task",
    "media_analyze_gpu_task",
    "media_ingest_task",
    "render_video_task",
]
