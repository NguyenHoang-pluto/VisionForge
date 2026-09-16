"""FastAPI dependency providers.

Assembling wiring here, in one place, is what lets tests override the probe list
or the object store with fakes and exercise the real handlers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.application.coedit_service import CoEditService
from visionforge.application.edit_service import EditService
from visionforge.application.health_service import HealthService
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import MediaService
from visionforge.application.planner_factory import PlannerSelection, build_planner
from visionforge.core.config import get_settings
from visionforge.domain.errors import NotFoundError
from visionforge.domain.health import HealthProbe
from visionforge.domain.ids import ProjectId, UserId
from visionforge.domain.llm import LlmProvider
from visionforge.domain.llm_planner import PlannerMode
from visionforge.domain.planner import Planner, RulesEnginePlanner
from visionforge.domain.storage import ObjectStore
from visionforge.domain.style import EditStyle
from visionforge.infra.db import DatabaseProbe, get_sessionmaker
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import (
    AnalysisRepository,
    EditPlanRepository,
    EditVersionRepository,
    EventRepository,
    JobRepository,
    LlmRunRepository,
    MediaRepository,
    ProjectRepository,
    UserRepository,
)
from visionforge.infra.llm import build_provider
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


def get_edit_plan_repo(session: AsyncSession = Depends(get_session)) -> EditPlanRepository:
    return EditPlanRepository(session)


def get_llm_run_repo(session: AsyncSession = Depends(get_session)) -> LlmRunRepository:
    return LlmRunRepository(session)


def get_version_repo(session: AsyncSession = Depends(get_session)) -> EditVersionRepository:
    return EditVersionRepository(session)


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


def get_planner() -> Planner:
    """The deterministic planner.

    Still here, still the default, and still what every route falls back to. The
    Phase 5 note that used to sit on this function said an LLM would be swapped
    in here; what actually happened is better -- the LLM planner is *added*
    beside it through :class:`EditServiceFactory`, and this one remains the
    implementation that always works.
    """
    return RulesEnginePlanner()


def get_llm_provider() -> LlmProvider | None:
    """The configured provider, or ``None`` when AI planning is off.

    ``None`` is the normal state for a developer with no API key, and every
    caller treats it as "plan deterministically" rather than as an error. The
    key is read inside the factory from server settings and never leaves it.
    """
    return build_provider()


class EditServiceFactory:
    """Builds an ``EditService`` once the request body has been read.

    A plain ``Depends`` cannot do this: which planner runs depends on the mode,
    the style and whether the caller wrote anything, and none of that is known
    until the body is parsed. So the dependency supplies the collaborators and
    the route asks for a service configured for the request it actually got.
    """

    def __init__(
        self,
        session: AsyncSession,
        media: MediaRepository,
        plans: EditPlanRepository,
        events: EventRepository,
        llm_runs: LlmRunRepository,
        provider: LlmProvider | None,
        versions: EditVersionRepository,
    ) -> None:
        self._session = session
        self._media = media
        self._plans = plans
        self._events = events
        self._llm_runs = llm_runs
        self._provider = provider
        self._versions = versions

    def for_request(
        self,
        *,
        mode: PlannerMode,
        style: EditStyle | None,
        request_text: str | None,
    ) -> tuple[EditService, PlannerSelection]:
        selection = build_planner(
            mode=mode,
            style=style,
            request_text=request_text,
            provider=self._provider,
        )
        service = EditService(
            self._session,
            self._media,
            self._plans,
            self._events,
            selection.planner,
            llm_runs=self._llm_runs,
            mode_decision=selection.decision,
            versions=self._versions,
        )
        return service, selection

    def plain(self) -> EditService:
        """A service with the rules engine, for routes that never plan."""
        return EditService(
            self._session,
            self._media,
            self._plans,
            self._events,
            RulesEnginePlanner(),
            versions=self._versions,
        )


def get_edit_service_factory(
    session: AsyncSession = Depends(get_session),
    media: MediaRepository = Depends(get_media_repo),
    plans: EditPlanRepository = Depends(get_edit_plan_repo),
    events: EventRepository = Depends(get_event_repo),
    llm_runs: LlmRunRepository = Depends(get_llm_run_repo),
    provider: LlmProvider | None = Depends(get_llm_provider),
    versions: EditVersionRepository = Depends(get_version_repo),
) -> EditServiceFactory:
    return EditServiceFactory(session, media, plans, events, llm_runs, provider, versions)


def get_edit_service(
    factory: EditServiceFactory = Depends(get_edit_service_factory),
) -> EditService:
    """The rules-engine service, for routes that dispatch rather than plan."""
    return factory.plain()


def get_coedit_service(
    session: AsyncSession = Depends(get_session),
    media: MediaRepository = Depends(get_media_repo),
    plans: EditPlanRepository = Depends(get_edit_plan_repo),
    versions: EditVersionRepository = Depends(get_version_repo),
    events: EventRepository = Depends(get_event_repo),
) -> CoEditService:
    """The co-editor (Phase 10).

    Note what it is *not* given: a planner. Co-editing patches the plan that
    exists rather than producing a new one, so there is nothing here that could
    quietly re-plan a project when a change failed to apply.
    """
    return CoEditService(session, media, plans, versions, events)


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
