"""Job creation and dispatch.

The ordering rule (Phase 0, ADR-0003):

    BEGIN -> create job + steps -> COMMIT -> publish to broker

Publishing before the commit produces the classic race where a worker claims a
job whose row is not visible yet, and a rollback leaves a message for a job that
never existed.

Failure behaviour is asymmetric on purpose:

- **Commit fails** -> nothing is published. There is no job and no message.
- **Publish fails after commit** -> the job stays ``PENDING`` with ``queued_at``
  NULL. It is not lost: ``find_undispatched`` locates it and ``redispatch`` sends
  it. That is why ``PENDING`` and ``QUEUED`` are distinct states rather than one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from visionforge.domain.jobs import JOB_STEP_PLANS, JobType
from visionforge.domain.ports import JobRepositoryPort, UnitOfWork

logger = logging.getLogger(__name__)

#: A job still PENDING after this long was almost certainly never published.
UNDISPATCHED_GRACE_S = 60


class TaskPublisher(Protocol):
    """The broker port. Implemented by ``infra.queue.publisher``."""

    def publish(self, job_type: JobType, job_id: UUID) -> str | None: ...


class JobDispatcher:
    """Creates jobs inside a transaction and publishes them after it commits."""

    def __init__(
        self, session: UnitOfWork, jobs: JobRepositoryPort, publisher: TaskPublisher
    ) -> None:
        self._session = session
        self._jobs = jobs
        self._publisher = publisher

    async def create_and_dispatch(
        self,
        *,
        project_id: UUID,
        job_type: JobType,
        params: dict[str, Any],
        media_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[Any, bool]:
        """Return ``(job, created)``.

        ``created`` is False when an idempotency key matched an existing job, in
        which case nothing new is created and nothing is published.
        """
        if idempotency_key:
            existing = await self._jobs.find_by_idempotency_key(
                project_id, job_type, idempotency_key
            )
            if existing is not None:
                logger.info(
                    "idempotent job replay",
                    extra={"job_id": str(existing.id), "idempotency_key": idempotency_key},
                )
                return existing, False

        job = await self._jobs.create(
            project_id=project_id,
            job_type=job_type,
            params=params,
            step_names=JOB_STEP_PLANS[job_type],
            media_id=media_id,
            idempotency_key=idempotency_key,
        )

        # --- the commit boundary ---
        await self._session.commit()

        await self._publish(job.id, job_type)
        return job, True

    async def redispatch(self, jobs_to_send: Sequence[Any]) -> int:
        """Re-publish jobs that committed but were never queued."""
        sent = 0
        for job in jobs_to_send:
            if await self._publish(job.id, JobType(job.type)):
                sent += 1
        await self._session.commit()
        return sent

    async def _publish(self, job_id: UUID, job_type: JobType) -> bool:
        try:
            task_id = self._publisher.publish(job_type, job_id)
        except Exception as exc:  # broker down; the job is safe in Postgres
            logger.error(
                "broker publish failed; job left PENDING for redispatch",
                extra={"job_id": str(job_id), "error": str(exc)},
            )
            return False

        await self._jobs.mark_queued(job_id, task_id)
        await self._session.commit()
        return True


def undispatched_cutoff(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) - timedelta(seconds=UNDISPATCHED_GRACE_S)
