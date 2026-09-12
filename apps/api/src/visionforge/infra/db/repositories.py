"""Repositories: the only place that issues SQL.

Every media/job lookup takes a ``project_id``. There is no ``get_media(media_id)``
that skips ownership -- that absence is the Phase 2 authorization design, and it
is why Phase 11 can add real principals without auditing every call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from visionforge.domain.jobs import JobStatus, JobType, StepStatus
from visionforge.domain.media import DerivativeKind, MediaStatus
from visionforge.infra.db.models import (
    Event,
    Job,
    JobStep,
    MediaAsset,
    MediaDerivative,
    Project,
    User,
)


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(self, *, email: str, display_name: str) -> User:
        user = User(email=email, display_name=display_name)
        self._session.add(user)
        await self._session.flush()
        return user


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, user_id: UUID, title: str, description: str | None) -> Project:
        project = Project(user_id=user_id, title=title, description=description)
        self._session.add(project)
        await self._session.flush()
        return project

    async def get_owned(self, project_id: UUID, user_id: UUID) -> Project | None:
        """Fetch a project only if this user owns it. The authorization primitive."""
        result = await self._session.execute(
            select(Project).where(Project.id == project_id, Project.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: UUID, *, limit: int = 50) -> Sequence[Project]:
        result = await self._session.execute(
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


class MediaRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **kwargs: Any) -> MediaAsset:
        media = MediaAsset(**kwargs)
        self._session.add(media)
        await self._session.flush()
        return media

    async def get_in_project(self, media_id: UUID, project_id: UUID) -> MediaAsset | None:
        result = await self._session.execute(
            select(MediaAsset)
            .options(selectinload(MediaAsset.derivatives))
            .where(MediaAsset.id == media_id, MediaAsset.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def find_by_sha256(self, project_id: UUID, sha256: str) -> MediaAsset | None:
        """Deduplication lookup, scoped to the project by the unique constraint."""
        result = await self._session.execute(
            select(MediaAsset)
            .options(selectinload(MediaAsset.derivatives))
            .where(MediaAsset.project_id == project_id, MediaAsset.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def list_in_project(
        self, project_id: UUID, *, limit: int = 200, offset: int = 0
    ) -> Sequence[MediaAsset]:
        result = await self._session.execute(
            select(MediaAsset)
            .options(selectinload(MediaAsset.derivatives))
            .where(MediaAsset.project_id == project_id)
            .order_by(MediaAsset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_in_project(self, project_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(MediaAsset).where(MediaAsset.project_id == project_id)
        )
        return int(result.scalar_one())

    async def set_status(
        self, media_id: UUID, status: MediaStatus, *, error: dict[str, Any] | None = None
    ) -> None:
        await self._session.execute(
            update(MediaAsset).where(MediaAsset.id == media_id).values(status=status, error=error)
        )

    async def upsert_derivative(
        self,
        *,
        media_id: UUID,
        kind: DerivativeKind,
        variant: str,
        storage_key: str,
        bytes_size: int | None,
        mime_type: str | None,
        width: int | None = None,
        height: int | None = None,
    ) -> MediaDerivative:
        """Idempotent by ``(media_id, kind, variant)`` so a retry cannot duplicate."""
        result = await self._session.execute(
            select(MediaDerivative).where(
                MediaDerivative.media_id == media_id,
                MediaDerivative.kind == kind,
                MediaDerivative.variant == variant,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            existing.storage_key = storage_key
            existing.bytes_size = bytes_size
            existing.mime_type = mime_type
            existing.width = width
            existing.height = height
            await self._session.flush()
            return existing

        derivative = MediaDerivative(
            media_id=media_id,
            kind=kind,
            variant=variant,
            storage_key=storage_key,
            bytes_size=bytes_size,
            mime_type=mime_type,
            width=width,
            height=height,
        )
        self._session.add(derivative)
        await self._session.flush()
        return derivative


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        project_id: UUID,
        job_type: JobType,
        params: dict[str, Any],
        step_names: Sequence[str],
        media_id: UUID | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> Job:
        """Create the job and all of its step rows in one flush.

        The steps exist before the worker starts, so the API can report "step 0
        of 6" honestly instead of inventing progress.
        """
        # Steps are passed to the constructor, not appended afterwards. Appending
        # to an unloaded collection makes SQLAlchemy load it first, which is IO in
        # an async session; building the relationship on a transient object needs
        # no IO and leaves the returned job serializable without a lazy load.
        job = Job(
            project_id=project_id,
            media_id=media_id,
            type=job_type,
            status=JobStatus.PENDING,
            params=params,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            steps=[JobStep(seq=seq, name=name) for seq, name in enumerate(step_names)],
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(self, job_id: UUID) -> Job | None:
        result = await self._session.execute(
            select(Job).options(selectinload(Job.steps)).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()

    async def get_in_project(self, job_id: UUID, project_id: UUID) -> Job | None:
        result = await self._session.execute(
            select(Job)
            .options(selectinload(Job.steps))
            .where(Job.id == job_id, Job.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def find_by_idempotency_key(
        self, project_id: UUID, job_type: JobType, key: str
    ) -> Job | None:
        result = await self._session.execute(
            select(Job)
            .options(selectinload(Job.steps))
            .where(
                Job.project_id == project_id,
                Job.type == job_type,
                Job.idempotency_key == key,
            )
        )
        return result.scalar_one_or_none()

    async def mark_queued(self, job_id: UUID, celery_task_id: str | None) -> None:
        await self._session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status=JobStatus.QUEUED,
                queued_at=datetime.now(UTC),
                celery_task_id=celery_task_id,
                retry_at=None,
            )
        )

    async def request_cancel(self, job_id: UUID) -> bool:
        """Set the cooperative cancellation flag. Returns False if already terminal.

        A job that has not started yet goes straight to CANCELLED: there is no
        worker to observe the flag.
        """
        job = await self._session.get(Job, job_id)
        if job is None or job.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            return False

        job.cancel_requested = True
        if job.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RETRY_WAIT):
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
        else:
            job.status = JobStatus.CANCEL_REQUESTED
        await self._session.flush()
        return True

    async def list_in_project(self, project_id: UUID, *, limit: int = 50) -> Sequence[Job]:
        result = await self._session.execute(
            select(Job)
            .options(selectinload(Job.steps))
            .where(Job.project_id == project_id)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def find_undispatched(self, older_than: datetime, *, limit: int = 50) -> Sequence[Job]:
        """Jobs committed but never queued.

        The enqueue happens after commit, so a crash in that window leaves a
        PENDING row with no broker message. This query is how the sweeper finds
        them -- see ``application.job_dispatch``.
        """
        result = await self._session.execute(
            select(Job)
            .where(Job.status == JobStatus.PENDING, Job.created_at < older_than)
            .order_by(Job.created_at)
            .limit(limit)
        )
        return result.scalars().all()


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self, *, kind: str, actor: str, project_id: UUID | None, payload: dict[str, Any]
    ) -> None:
        self._session.add(Event(kind=kind, actor=actor, project_id=project_id, payload=payload))


__all__ = [
    "EventRepository",
    "JobRepository",
    "MediaRepository",
    "ProjectRepository",
    "StepStatus",
    "UserRepository",
]
