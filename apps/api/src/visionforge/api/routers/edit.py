"""Edit plan and render routes.

The API's whole job here is to take a small set of typed preferences, hand them
to a planner, and dispatch a render job. It never sees a filter graph, a command
or a filesystem path: it deals in plan ids and render ids, and the worker
resolves everything else from the database.

Every route resolves access through ``require_project``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.api.dependencies import (
    EditServiceFactory,
    get_edit_plan_repo,
    get_edit_service,
    get_edit_service_factory,
    get_job_dispatcher,
    get_llm_provider,
    get_llm_run_repo,
    get_media_service,
    get_session,
    require_project,
)
from visionforge.api.schemas.edit import (
    EditPlanDetail,
    EditPlanListResponse,
    EditPlanSummary,
    PlanCreateRequest,
    PlannerCapabilities,
    RenderCreateRequest,
    RenderListResponse,
    RenderResponse,
)
from visionforge.api.serializers import serialize_edit_plan, serialize_render
from visionforge.application.edit_service import EditService
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import DOWNLOAD_URL_TTL_S, MediaService
from visionforge.domain.editbrief import MAX_REQUEST_CHARS
from visionforge.domain.editplan import AspectRatio, AudioMode, FitMode, PlanInvalidError
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import ProjectId
from visionforge.domain.jobs import JobType
from visionforge.domain.llm import LlmProvider
from visionforge.domain.llm_planner import PlannerMode
from visionforge.domain.planner import ClipOrder, PlanRequest
from visionforge.domain.prompts import PROMPT_VERSION
from visionforge.domain.render import RenderStatus
from visionforge.domain.style import FPS_PRESETS, STYLE_PROFILES, QualityPreset, profile_for
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import EditPlanRepository, LlmRunRepository
from visionforge.infra.llm import describe_capabilities

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


@router.get("/planner/capabilities", response_model=PlannerCapabilities)
async def planner_capabilities(
    provider: LlmProvider | None = Depends(get_llm_provider),
) -> PlannerCapabilities:
    """What this server can plan with, so the UI can be honest about it.

    Not project-scoped, and deliberately so: it describes the server, not any
    user's data, and it contains nothing a caller could not infer by attempting
    a plan. **It never contains an API key.** The factory reports whether one is
    configured; the key itself has no path out of the process.
    """
    capabilities = describe_capabilities(provider)
    return PlannerCapabilities(
        ai_available=bool(capabilities["ai_available"]),
        provider=_as_str(capabilities["provider"]),
        model=_as_str(capabilities["model"]),
        is_stub=bool(capabilities.get("is_stub")),
        error=_as_str(capabilities["error"]),
        modes=[mode.value for mode in PlannerMode],
        styles=[
            {
                "value": profile.style.value,
                "label": profile.label,
                "description": profile.description,
                "default_duration_ms": profile.default_duration_ms,
                "default_aspect": profile.default_aspect.value,
                "min_clip_ms": profile.min_clip_ms,
                "max_clip_ms": profile.max_clip_ms,
            }
            for profile in STYLE_PROFILES.values()
        ],
        aspect_ratios=[
            {"value": ratio.value, "width": size[0], "height": size[1]}
            for ratio, size in PRESET_DIMENSIONS.items()
        ],
        fps_presets=list(FPS_PRESETS),
        quality_presets=[preset.value for preset in QualityPreset],
        prompt_version=PROMPT_VERSION,
        max_request_chars=MAX_REQUEST_CHARS,
    )


@router.post(
    "/projects/{project_id}/edit-plan",
    response_model=EditPlanDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_edit_plan(
    body: PlanCreateRequest,
    project: Project = Depends(require_project),
    factory: EditServiceFactory = Depends(get_edit_service_factory),
) -> EditPlanDetail:
    """Generate and persist an edit plan from the project's analysed media.

    Still synchronous, and still 201, even though an AI plan takes seconds
    rather than milliseconds. The alternative -- making planning a job -- would
    add a queue round trip, a progress bar and a polling client to something the
    user is already waiting on, and would break a contract Phase 4 clients
    depend on. The service runs the planner on a worker thread instead, so a
    slow plan costs one thread rather than the event loop.

    Which planner runs is decided here from the mode, the style and whether the
    caller wrote anything -- and the decision is stored on the plan rather than
    left to be inferred later from which planner's name it carries.
    """
    profile = profile_for(body.style)

    # A style supplies defaults only where the caller stated nothing. An
    # explicit value always wins, including one that happens to equal the
    # style's own default -- which is why those fields default to None.
    aspect = body.aspect_ratio or profile.default_aspect
    width, height = PRESET_DIMENSIONS[aspect]
    order = body.order or (
        ClipOrder.SEQUENCE if profile.prefer_sequence_order else ClipOrder.SCORE_DESC
    )

    request = PlanRequest(
        project_id=ProjectId(project.id),
        target_duration_ms=body.target_duration_ms or profile.default_duration_ms,
        max_clips=body.max_clips,
        min_clips=body.min_clips,
        aspect_ratio=aspect,
        width=width,
        height=height,
        fps=body.fps,
        fit=body.fit,
        audio=body.audio or profile.default_audio,
        order=order,
        quality=body.quality,
        style=body.style,
        request_text=body.request_text,
    )

    service, _selection = factory.for_request(
        mode=body.mode, style=body.style, request_text=body.request_text
    )

    try:
        row, _outcome = await service.create_plan(project_id=ProjectId(project.id), request=request)
    except PlanInvalidError as exc:
        # A planner that emits an invalid plan is a bug, not user error. 422
        # with the full violation list, so it is diagnosable rather than a
        # generic failure.
        #
        # The LLM path does not normally arrive here: an invalid model plan is
        # caught inside the planner while the rules engine is still available.
        # Reaching this from an AI request means the fallback failed too.
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


@router.get("/projects/{project_id}/llm-runs")
async def list_llm_runs(
    limit: int = Query(default=20, ge=1, le=100),
    project: Project = Depends(require_project),
    repo: LlmRunRepository = Depends(get_llm_run_repo),
) -> dict[str, Any]:
    """Planning calls made for this project, successful or not.

    The failures are the point. A feature that falls back silently is a feature
    nobody can tell is broken, so every run is listed -- including the ones that
    produced no plan -- with its provider, latency, token usage and the reason
    it fell back.

    No prompt, no completion and no request text: the row holds a digest of what
    was asked, and that is all the table ever stored.
    """
    rows = await repo.list_for_project(project.id, limit=limit)
    return {
        "items": [
            {
                "id": str(row.id),
                "edit_plan_id": str(row.edit_plan_id) if row.edit_plan_id else None,
                "provider": row.provider,
                "model": row.model,
                "prompt_version": row.prompt_version,
                "status": row.status,
                "attempts": row.attempts,
                "latency_ms": row.latency_ms,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "fallback_reason": row.fallback_reason,
                "fallback_detail": row.fallback_detail,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
        "total": len(rows),
    }


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["PRESET_DIMENSIONS", "AudioMode", "ClipOrder", "FitMode", "router"]
