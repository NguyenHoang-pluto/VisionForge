"""FastAPI dependency providers.

Assembling wiring here, in one place, is what lets tests override the probe list
or the object store with fakes and exercise the real handlers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.application.health_service import HealthService
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import MediaService
from visionforge.core.config import get_settings
from visionforge.domain.errors import NotFoundError
from visionforge.domain.health import HealthProbe
from visionforge.domain.ids import ProjectId, UserId
from visionforge.domain.storage import ObjectStore
from visionforge.infra.db import DatabaseProbe, get_sessionmaker
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import (
    AnalysisRepository,
    EventRepository,
    JobRepository,
    MediaRepository,
    ProjectRepository,
    UserRepository,
)
from visionforge.infra.queue.publisher import CeleryTaskPublisher
from visionforge.infra.redis import RedisProbe
from visionforge.infra.storage import S3ObjectStore, StorageProbe

#: Phase 2 has no authentication. Every request runs as a single seeded
#: development user whose id is fixed so that data survives restarts.
#:
#: The important part is the *shape*: handlers depend on ``current_user_id`` and
#: resolve access through project ownership. Phase 11 replaces the body of this
#: function with real token validation and nothing else has to change.
DEV_USER_ID = UserId(UUID("00000000-0000-0000-0000-0000000000a1"))
DEV_USER_EMAIL = "dev@visionforge.local"


def current_user_id() -> UserId:
    return DEV_USER_ID


# ------------------------------------------------------------------- resources
async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request, rolled back if the handler raises."""
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_object_store() -> ObjectStore:
    return S3ObjectStore()


# ---------------------------------------------------------------- repositories
def get_project_repo(session: AsyncSession = Depends(get_session)) -> ProjectRepository:
    return ProjectRepository(session)


def get_media_repo(session: AsyncSession = Depends(get_session)) -> MediaRepository:
    return MediaRepository(session)


def get_job_repo(session: AsyncSession = Depends(get_session)) -> JobRepository:
    return JobRepository(session)


def get_analysis_repo(session: AsyncSession = Depends(get_session)) -> AnalysisRepository:
    return AnalysisRepository(session)


def get_event_repo(session: AsyncSession = Depends(get_session)) -> EventRepository:
    return EventRepository(session)


def get_user_repo(session: AsyncSession = Depends(get_session)) -> UserRepository:
    return UserRepository(session)


# -------------------------------------------------------------------- services
def get_media_service(
    session: AsyncSession = Depends(get_session),
    media: MediaRepository = Depends(get_media_repo),
    events: EventRepository = Depends(get_event_repo),
    store: ObjectStore = Depends(get_object_store),
) -> MediaService:
    return MediaService(session, media, events, store)


def get_job_dispatcher(
    session: AsyncSession = Depends(get_session),
    jobs: JobRepository = Depends(get_job_repo),
) -> JobDispatcher:
    return JobDispatcher(session, jobs, CeleryTaskPublisher())


# --------------------------------------------------------------- authorization
async def require_project(
    project_id: UUID,
    projects: ProjectRepository = Depends(get_project_repo),
    user_id: UserId = Depends(current_user_id),
) -> Project:
    """Resolve a project the caller owns, or 404.

    **This is the authorization anchor.** Every media and job route depends on
    it, so there is no route that reaches a media row without first proving
    ownership of its project. 404 rather than 403 so the API does not confirm
    the existence of other people's projects.
    """
    project = await projects.get_owned(project_id, user_id)
    if project is None:
        raise NotFoundError("project not found")
    return project


def project_id_of(project: Project) -> ProjectId:
    return ProjectId(project.id)


# ---------------------------------------------------------------------- health
def get_health_probes() -> list[HealthProbe]:
    return [DatabaseProbe(), RedisProbe(), StorageProbe()]


def get_health_service() -> HealthService:
    return HealthService(
        probes=get_health_probes(),
        timeout_s=get_settings().readiness_timeout_s,
    )
