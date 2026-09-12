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

from visionforge.domain.analysis import AnalysisOutcome, AnalyzerName
from visionforge.domain.jobs import JobStatus, JobType, StepStatus
from visionforge.domain.media import DerivativeKind, MediaStatus
from visionforge.infra.db.models import (
    Event,
    Job,
    JobStep,
    MediaAnalysis,
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


class AnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, *, media_id: UUID, project_id: UUID, outcome: AnalysisOutcome) -> Any:
        """Write one analyzer result.

        Keyed on ``(media_id, analyzer, analyzer_version)``: re-running the same
        version overwrites its row, a new version adds one. Upgrading a model
        therefore never destroys what the previous version reported, which is
        what makes a quality regression attributable instead of invisible.
        """
        result = await self._session.execute(
            select(MediaAnalysis).where(
                MediaAnalysis.media_id == media_id,
                MediaAnalysis.analyzer == outcome.analyzer,
                MediaAnalysis.analyzer_version == outcome.version,
            )
        )
        row = result.scalar_one_or_none()
        embedding = list(outcome.embedding) if outcome.embedding is not None else None

        if row is not None:
            row.status = outcome.status
            row.payload = dict(outcome.payload)
            row.metrics = dict(outcome.metrics) or None
            row.embedding = embedding
            await self._session.flush()
            return row

        row = MediaAnalysis(
            media_id=media_id,
            project_id=project_id,
            analyzer=outcome.analyzer,
            analyzer_version=outcome.version,
            status=outcome.status,
            payload=dict(outcome.payload),
            metrics=dict(outcome.metrics) or None,
            embedding=embedding,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(
        self, *, media_id: UUID, analyzer: AnalyzerName, version: str
    ) -> MediaAnalysis | None:
        result = await self._session.execute(
            select(MediaAnalysis).where(
                MediaAnalysis.media_id == media_id,
                MediaAnalysis.analyzer == analyzer,
                MediaAnalysis.analyzer_version == version,
            )
        )
        return result.scalar_one_or_none()

    async def latest_for_media(self, media_id: UUID) -> Sequence[MediaAnalysis]:
        """Every analyzer's newest version for one asset.

        ``DISTINCT ON`` keeps one row per analyzer -- the highest version -- so a
        caller sees current results without also seeing superseded ones.
        """
        result = await self._session.execute(
            select(MediaAnalysis)
            .where(MediaAnalysis.media_id == media_id)
            .distinct(MediaAnalysis.analyzer)
            .order_by(MediaAnalysis.analyzer, MediaAnalysis.analyzer_version.desc())
        )
        return result.scalars().all()

    async def list_for_project(
        self, project_id: UUID, *, analyzer: AnalyzerName | None = None, limit: int = 500
    ) -> Sequence[MediaAnalysis]:
        stmt = select(MediaAnalysis).where(MediaAnalysis.project_id == project_id)
        if analyzer is not None:
            stmt = stmt.where(MediaAnalysis.analyzer == analyzer)
        result = await self._session.execute(
            stmt.order_by(MediaAnalysis.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    async def find_similar(
        self, *, project_id: UUID, embedding: Sequence[float], limit: int = 10
    ) -> Sequence[tuple[MediaAnalysis, float]]:
        """Nearest CLIP embeddings within one project, by cosine distance.

        Project-scoped in the WHERE clause so similarity search can never leak
        media across the authorization boundary.
        """
        distance = MediaAnalysis.embedding.cosine_distance(list(embedding))
        result = await self._session.execute(
            select(MediaAnalysis, distance.label("distance"))
            .where(
                MediaAnalysis.project_id == project_id,
                MediaAnalysis.analyzer == AnalyzerName.CLIP,
                MediaAnalysis.embedding.isnot(None),
            )
            .order_by(distance)
            .limit(limit)
        )
        return [(row, float(dist)) for row, dist in result.all()]


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self, *, kind: str, actor: str, project_id: UUID | None, payload: dict[str, Any]
    ) -> None:
        self._session.add(Event(kind=kind, actor=actor, project_id=project_id, payload=payload))


__all__ = [
    "AnalysisRepository",
    "EventRepository",
    "JobRepository",
    "MediaRepository",
    "ProjectRepository",
    "StepStatus",
    "UserRepository",
]
