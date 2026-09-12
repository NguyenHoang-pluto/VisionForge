"""Edit plan and render routes.

The API's whole job here is to take a small set of typed preferences, hand them
to a planner, and dispatch a render job. It never sees a filter graph, a command
or a filesystem path: it deals in plan ids and render ids, and the worker
resolves everything else from the database.

Every route resolves access through ``require_project``.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.api.dependencies import (
    get_edit_plan_repo,
    get_edit_service,
    get_job_dispatcher,
    get_media_service,
    get_session,
    require_project,
)
from visionforge.api.schemas.edit import (
    EditPlanDetail,
    EditPlanListResponse,
    EditPlanSummary,
    PlanCreateRequest,
    RenderCreateRequest,
    RenderListResponse,
    RenderResponse,
)
from visionforge.api.serializers import serialize_edit_plan, serialize_render
from visionforge.application.edit_service import EditService
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import DOWNLOAD_URL_TTL_S, MediaService
from visionforge.domain.editplan import AspectRatio, AudioMode, FitMode, PlanInvalidError
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import ProjectId
from visionforge.domain.jobs import JobType
from visionforge.domain.planner import ClipOrder, PlanRequest
from visionforge.domain.render import RenderStatus
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import EditPlanRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["edit"])

#: Output geometry per aspect ratio. A closed map rather than free width/height:
#: the caller picks a shape, the server picks dimensions that are even (required
#: by H.264 4:2:0) and sane for this hardware. Arbitrary geometry from a client
#: is exactly what the plan validator would then have to defend against.
PRESET_DIMENSIONS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.LANDSCAPE_16_9: (1280, 720),
    AspectRatio.PORTRAIT_9_16: (720, 1280),
    AspectRatio.SQUARE_1_1: (720, 720),
}


@router.post(
    "/projects/{project_id}/edit-plan",
    response_model=EditPlanDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_edit_plan(
    body: PlanCreateRequest,
    project: Project = Depends(require_project),
    service: EditService = Depends(get_edit_service),
) -> EditPlanDetail:
    """Generate and persist an edit plan from the project's analysed media.

    Synchronous: planning is pure arithmetic over rows already in the database
    and takes milliseconds. Making it a job would add a queue round-trip and a
    progress bar to something that finishes before the response is written.
    """
    width, height = PRESET_DIMENSIONS[body.aspect_ratio]
    request = PlanRequest(
        project_id=ProjectId(project.id),
        target_duration_ms=body.target_duration_ms,
        max_clips=body.max_clips,
        min_clips=body.min_clips,
        aspect_ratio=body.aspect_ratio,
        width=width,
        height=height,
        fps=body.fps,
        fit=body.fit,
        audio=body.audio,
        order=body.order,
    )

    try:
        row, _outcome = await service.create_plan(project_id=ProjectId(project.id), request=request)
    except PlanInvalidError as exc:
        # A planner that emits an invalid plan is a bug, not user error. 422
        # with the full violation list, so it is diagnosable rather than a
        # generic failure.
        raise ValidationError(
            "the planner produced an invalid plan",
            hint="; ".join(v.message for v in exc.violations),
        ) from exc

    return serialize_edit_plan(row, include_plan=True)


@router.get("/projects/{project_id}/edit-plan", response_model=EditPlanListResponse)
async def list_edit_plans(
    limit: int = Query(default=20, ge=1, le=100),
    project: Project = Depends(require_project),
    repo: EditPlanRepository = Depends(get_edit_plan_repo),
) -> EditPlanListResponse:
    rows = await repo.list_for_project(project.id, limit=limit)
    return EditPlanListResponse(
        items=[EditPlanSummary.model_validate(serialize_edit_plan(row)) for row in rows],
        total=len(rows),
    )


@router.get("/projects/{project_id}/edit-plan/{edit_plan_id}", response_model=EditPlanDetail)
async def get_edit_plan(
    edit_plan_id: UUID,
    project: Project = Depends(require_project),
    repo: EditPlanRepository = Depends(get_edit_plan_repo),
) -> EditPlanDetail:
    row = await repo.get_in_project(edit_plan_id, project.id)
    if row is None:
        raise NotFoundError("edit plan not found in this project")
    return serialize_edit_plan(row, include_plan=True)


# ---------------------------------------------------------------------- renders
@router.post(
    "/projects/{project_id}/render",
    response_model=RenderResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_render(
    body: RenderCreateRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    session: AsyncSession = Depends(get_session),
    project: Project = Depends(require_project),
    service: EditService = Depends(get_edit_service),
    repo: EditPlanRepository = Depends(get_edit_plan_repo),
    dispatcher: JobDispatcher = Depends(get_job_dispatcher),
) -> RenderResponse:
    """Queue a render of an existing plan.

    202: the render row exists and the job is queued, but the file does not
    exist yet. Progress follows through the existing job endpoints and SSE
    stream -- Phase 4 adds no new progress mechanism.
    """
    render = await service.create_render(
        project_id=ProjectId(project.id), edit_plan_id=body.edit_plan_id
    )

    job, _created = await dispatcher.create_and_dispatch(
        project_id=ProjectId(project.id),
        job_type=JobType.RENDER_VIDEO,
        params={"render_id": str(render.id), "edit_plan_id": str(body.edit_plan_id)},
        # Keyed on the render, so a retried request attaches to the render it
        # already created rather than starting a second encode of the same plan.
        idempotency_key=idempotency_key or f"render:{render.id}",
    )
    await repo.attach_job(render.id, job.id)
    await session.commit()

    refreshed = await repo.get_render_in_project(render.id, project.id)
    assert refreshed is not None
    return serialize_render(refreshed)


@router.get("/projects/{project_id}/renders", response_model=RenderListResponse)
async def list_renders(
    limit: int = Query(default=20, ge=1, le=100),
    project: Project = Depends(require_project),
    repo: EditPlanRepository = Depends(get_edit_plan_repo),
) -> RenderListResponse:
    rows = await repo.list_renders(project.id, limit=limit)
    return RenderListResponse(items=[serialize_render(row) for row in rows], total=len(rows))


@router.get("/projects/{project_id}/renders/{render_id}", response_model=RenderResponse)
async def get_render(
    render_id: UUID,
    project: Project = Depends(require_project),
    repo: EditPlanRepository = Depends(get_edit_plan_repo),
    media_service: MediaService = Depends(get_media_service),
) -> RenderResponse:
    """One render, with a short-lived playback URL once it is ready.

    The URL is presigned and expires in minutes. The API never streams the file
    itself -- the same rule that governs every other byte in the system.
    """
    row = await repo.get_render_in_project(render_id, project.id)
    if row is None:
        raise NotFoundError("render not found in this project")

    response = serialize_render(row)
    if RenderStatus(row.status) is RenderStatus.READY and row.storage_key:
        response.playback_url = media_service.presign_download(row.storage_key)
        response.playback_expires_in_s = DOWNLOAD_URL_TTL_S
    return response


__all__ = ["PRESET_DIMENSIONS", "AudioMode", "ClipOrder", "FitMode", "router"]
