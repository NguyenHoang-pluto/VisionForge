"""Job routes, including the SSE progress stream."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from visionforge.api.dependencies import (
    current_user_id,
    get_job_dispatcher,
    get_job_repo,
    get_project_repo,
    get_session,
)
from visionforge.api.schemas.media import JobCreateRequest, JobResponse
from visionforge.api.serializers import job_progress, serialize_job
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import ProjectId, UserId
from visionforge.domain.jobs import TERMINAL_STATUSES, JobStatus, JobType
from visionforge.infra.db.models import Job
from visionforge.infra.db.repositories import JobRepository, ProjectRepository
from visionforge.infra.redis import get_redis
from visionforge.infra.redis.events import channel, state_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Keep-alive cadence. Proxies drop idle connections, and a comment frame is far
#: cheaper than the reconnect it prevents.
SSE_PING_S = 15
SSE_MAX_DURATION_S = 30 * 60


async def _owned_job(
    job_id: UUID,
    jobs: JobRepository,
    projects: ProjectRepository,
    user_id: UserId,
) -> Job:
    """Load a job only if the caller owns its project.

    Job ids are never trusted on their own: ownership always resolves through
    the project, which is the authorization anchor.
    """
    job = await jobs.get(job_id)
    if job is None:
        raise NotFoundError("job not found")
    if await projects.get_owned(job.project_id, user_id) is None:
        raise NotFoundError("job not found")
    return job


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    body: JobCreateRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    projects: ProjectRepository = Depends(get_project_repo),
    dispatcher: JobDispatcher = Depends(get_job_dispatcher),
    user_id: UserId = Depends(current_user_id),
) -> JobResponse:
    """Create a job and dispatch it after the transaction commits.

    Returns 201 for a new job and 200-equivalent semantics for an idempotent
    replay: the same key returns the same job rather than starting a second one.
    """
    if await projects.get_owned(body.project_id, user_id) is None:
        raise NotFoundError("project not found")

    try:
        job_type = JobType(body.type)
    except ValueError:
        supported = ", ".join(t.value for t in JobType)
        raise ValidationError(
            f"unknown job type '{body.type}'", hint=f"Supported types: {supported}"
        ) from None

    if job_type is JobType.MEDIA_INGEST and body.media_id is None:
        raise ValidationError("media_id is required for media_ingest jobs")

    job, _created = await dispatcher.create_and_dispatch(
        project_id=ProjectId(body.project_id),
        job_type=job_type,
        params={"media_id": str(body.media_id)} if body.media_id else {},
        media_id=body.media_id,
        idempotency_key=idempotency_key,
    )
    return serialize_job(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    jobs: JobRepository = Depends(get_job_repo),
    projects: ProjectRepository = Depends(get_project_repo),
    user_id: UserId = Depends(current_user_id),
) -> JobResponse:
    return serialize_job(await _owned_job(job_id, jobs, projects, user_id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    jobs: JobRepository = Depends(get_job_repo),
    projects: ProjectRepository = Depends(get_project_repo),
    user_id: UserId = Depends(current_user_id),
) -> JobResponse:
    """Request cooperative cancellation.

    This sets a flag; it does not kill anything. A running worker observes the
    flag at its next step boundary and stops cleanly, leaving no truncated
    objects behind. A job that has not started yet is cancelled outright.
    """
    job = await _owned_job(job_id, jobs, projects, user_id)
    await jobs.request_cancel(job.id)
    await session.commit()
    refreshed = await jobs.get(job_id)
    assert refreshed is not None
    return serialize_job(refreshed)


@router.get("/{job_id}/events")
async def job_events(
    job_id: UUID,
    request: Request,
    jobs: JobRepository = Depends(get_job_repo),
    projects: ProjectRepository = Depends(get_project_repo),
    user_id: UserId = Depends(current_user_id),
) -> EventSourceResponse:
    """Server-sent events for one job.

    The stream always opens with the current state, assembled from PostgreSQL
    (the source of truth) rather than waiting for the next Redis message. That is
    what makes a mid-render page refresh cheap and correct: the client knows
    where the job stands immediately, even if the next event is minutes away.

    SSE rather than WebSockets because progress is one-directional, SSE
    reconnects on its own, and it needs no sticky sessions behind a load
    balancer. Cancellation travels back over a normal POST.
    """
    job = await _owned_job(job_id, jobs, projects, user_id)
    snapshot = _snapshot_from_job(job)

    async def stream() -> AsyncIterator[dict[str, Any]]:
        yield {"event": "state", "data": json.dumps(snapshot)}

        if JobStatus(snapshot["status"]) in TERMINAL_STATUSES:
            # Nothing further will ever be published; do not hold the connection.
            return

        redis = get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel(job_id))
        try:
            # A terminal event published between the snapshot read and the
            # subscribe would otherwise be missed. Re-check the cached state.
            cached = await redis.get(state_key(job_id))
            if cached:
                payload = json.loads(cached)
                if payload.get("status") != snapshot["status"]:
                    yield {"event": "state", "data": cached}
                    if JobStatus(payload["status"]) in TERMINAL_STATUSES:
                        return

            elapsed = 0.0
            while elapsed < SSE_MAX_DURATION_S:
                if await request.is_disconnected():
                    return

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=SSE_PING_S
                )
                if message is None:
                    elapsed += SSE_PING_S
                    yield {"event": "ping", "data": "{}"}
                    continue

                data = message["data"]
                yield {"event": "state", "data": data}

                try:
                    if JobStatus(json.loads(data)["status"]) in TERMINAL_STATUSES:
                        return
                except (json.JSONDecodeError, KeyError, ValueError):
                    logger.warning("malformed job event", extra={"job_id": str(job_id)})
        finally:
            await pubsub.unsubscribe(channel(job_id))
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    return EventSourceResponse(stream())


def _snapshot_from_job(job: Job) -> dict[str, Any]:
    """Build the initial SSE frame from the database, not from the Redis cache."""
    running = next(
        (s.name for s in sorted(job.steps, key=lambda s: s.seq) if s.status == "running"),
        None,
    )
    return {
        "event": "state",
        "job_id": str(job.id),
        "status": str(job.status),
        "attempt": job.attempts,
        "max_attempts": job.max_attempts,
        "progress": job_progress(job),
        "steps_done": sum(1 for s in job.steps if s.status in ("succeeded", "skipped")),
        "steps_total": len(job.steps),
        "current_step": running,
        "retry_at": job.retry_at.isoformat() if job.retry_at else None,
        "error": job.error,
        "result": job.result,
    }


__all__ = ["router"]
