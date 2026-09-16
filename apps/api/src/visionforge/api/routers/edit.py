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
    get_media_repo,
    get_media_service,
    get_project_repo,
    get_session,
    require_project,
)
from visionforge.api.schemas.edit import (
    EditPlanDetail,
    EditPlanListResponse,
    EditPlanSummary,
    EffectBoundsResponse,
    ManualPlanCreateRequest,
    MeasurementResponse,
    MusicRequest,
    PlanCreateRequest,
    PlannerCapabilities,
    ReferenceProfileResponse,
    ReferenceRequest,
    ReferenceResponse,
    RenderCreateRequest,
    RenderListResponse,
    RenderResponse,
    SubtitlePresetResponse,
    SubtitleSuggestRequest,
    SubtitleSuggestResponse,
    SubtitleTrackRequest,
)
from visionforge.api.serializers import serialize_edit_plan, serialize_render
from visionforge.application.edit_service import EditService
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import DOWNLOAD_URL_TTL_S, MediaService
from visionforge.application.reference_service import PROFILE_ANALYZERS, ReferenceService
from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.beats import (
    MAX_BPM,
    MIN_BEAT_CONFIDENCE,
    MIN_BPM,
    BeatGrid,
    grid_from_payload,
)
from visionforge.domain.editbrief import MAX_REQUEST_CHARS
from visionforge.domain.editplan import (
    MAX_FADE_MS,
    MAX_GAIN,
    MAX_MUSIC_MS,
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_GAIN,
    MIN_MUSIC_MS,
    MIN_OUTPUT_MS,
    MIN_SEGMENT_MS,
    MIN_TRANSITION_MS,
    AudioMode,
    Cut,
    FitMode,
    MusicCue,
    OutputSpec,
    PlanInvalidError,
    TransitionKind,
    media_id_from,
)
from visionforge.domain.effects import EFFECT_BOUNDS, MAX_EFFECTS_PER_SEGMENT, EffectKind
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import ProjectId
from visionforge.domain.jobs import JobType
from visionforge.domain.llm import LlmProvider
from visionforge.domain.llm_planner import PlannerMode
from visionforge.domain.planner import ClipOrder, PlanRequest
from visionforge.domain.policy import StyleStrength, blend
from visionforge.domain.prompts import PROMPT_VERSION
from visionforge.domain.reference import Measurement, ReferenceProfile
from visionforge.domain.render import RenderStatus
from visionforge.domain.style import (
    FPS_PRESETS,
    PRESET_DIMENSIONS,
    STYLE_PROFILES,
    QualityPreset,
    profile_for,
)
from visionforge.domain.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUE_MS,
    MAX_CUES,
    MIN_CUE_MS,
    STYLE_PRESETS,
    SubtitlePosition,
)
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import (
    EditPlanRepository,
    LlmRunRepository,
    MediaRepository,
    ProjectRepository,
)
from visionforge.infra.llm import describe_capabilities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["edit"])


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
        segment_bounds={
            "min_clip_ms": MIN_SEGMENT_MS,
            "max_clip_ms": MAX_SEGMENT_MS,
            "max_clips": MAX_SEGMENTS,
            "min_total_ms": MIN_OUTPUT_MS,
            "max_total_ms": MAX_OUTPUT_MS,
        },
        audio_bounds={
            "min_gain": MIN_GAIN,
            "max_gain": MAX_GAIN,
            "min_music_ms": MIN_MUSIC_MS,
            "max_music_ms": MAX_MUSIC_MS,
            "max_fade_ms": MAX_FADE_MS,
        },
        beat_sync={
            "analyzer": AnalyzerName.BEATS.value,
            "min_bpm": MIN_BPM,
            "max_bpm": MAX_BPM,
            "min_confidence": MIN_BEAT_CONFIDENCE,
        },
        # Phase 9. Every one of these is read from the domain's own tables
        # rather than retyped, so a kind added there appears here without an
        # edit and a kind removed there stops being offered.
        transitions=[
            {
                "value": kind.value,
                "consumes_time": kind.consumes_time,
                "needs_previous": kind.needs_previous,
                "min_ms": 0 if kind is TransitionKind.CUT else MIN_TRANSITION_MS,
                "max_ms": 0 if kind is TransitionKind.CUT else MAX_TRANSITION_MS,
                "max_share": MAX_TRANSITION_SHARE,
            }
            for kind in TransitionKind
        ],
        effects=[
            EffectBoundsResponse(
                kind=kind.value,
                minimum=EFFECT_BOUNDS[kind][0],
                maximum=EFFECT_BOUNDS[kind][1],
                neutral=EFFECT_BOUNDS[kind][2],
                whole_segment_only=kind.spans_whole_segment,
            )
            for kind in EffectKind
        ],
        subtitle_styles=[
            SubtitlePresetResponse(
                id=preset.style.value,
                label=preset.label,
                font=preset.font,
                size=preset.size,
                bold=preset.bold,
            )
            for preset in STYLE_PRESETS.values()
        ],
        subtitle_positions=[position.value for position in SubtitlePosition],
        subtitle_bounds={
            "min_cue_ms": MIN_CUE_MS,
            "max_cue_ms": MAX_CUE_MS,
            "max_chars": MAX_CUE_CHARS,
            "max_cues": MAX_CUES,
            "max_effects_per_clip": MAX_EFFECTS_PER_SEGMENT,
        },
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
    repo: MediaRepository = Depends(get_media_repo),
    projects: ProjectRepository = Depends(get_project_repo),
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

    # The reference is read from the project, never from the request: which clip
    # is the reference is server state, so a caller cannot aim one plan at
    # another project's media and read its measurements out of the result.
    reference = ReferenceService(projects, repo)
    reference_profile = await reference.profile_for_project(project)
    policy = blend(profile, reference_profile, body.style_strength)

    # A style supplies defaults only where the caller stated nothing. An
    # explicit value always wins, including one that happens to equal the
    # style's own default -- which is why those fields default to None.
    aspect = body.aspect_ratio or profile.default_aspect
    width, height = PRESET_DIMENSIONS[aspect]
    order = body.order or (
        ClipOrder.SEQUENCE if profile.prefer_sequence_order else ClipOrder.SCORE_DESC
    )

    # The track's own facts -- its length and its analysed beat grid -- are read
    # here from the database rather than taken from the request. The client
    # names a track it owns; everything the planner then knows about that track
    # comes from the server.
    music_duration_ms, beats = (
        await _music_facts(repo, ProjectId(project.id), body.music.media_id)
        if body.music is not None
        else (None, None)
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
        style_policy=policy,
        reference_media_id=ReferenceService.reference_id(project),
        music_media_id=media_id_from(body.music.media_id) if body.music else None,
        music_duration_ms=music_duration_ms,
        beats=beats,
        beat_sync=body.beat_sync,
        music_gain=body.music.volume if body.music else 0.7,
        music_fade_in_ms=body.music.fade_in_ms if body.music else 0,
        music_fade_out_ms=body.music.fade_out_ms if body.music else 1_500,
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


@router.post(
    "/projects/{project_id}/edit-plan/manual",
    response_model=EditPlanDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_manual_edit_plan(
    body: ManualPlanCreateRequest,
    project: Project = Depends(require_project),
    service: EditService = Depends(get_edit_service),
) -> EditPlanDetail:
    """Store a timeline the user cut themselves.

    The editor's timeline is client state while it is being dragged; this is
    where it stops being a proposal. No planner runs, and nothing here is
    trusted: the cuts go through the same validator as an automatic plan, are
    checked against the same media rows, and produce the same kind of plan row
    that the same render route consumes.

    The route deliberately takes clips and trims rather than a rendering
    description. The server still chooses the geometry from the aspect ratio the
    caller asked for -- the client has no way to name a width, a height or an
    encoder setting, which is the property Phase 4 established and a timeline
    must not be the thing that erodes.
    """
    width, height = PRESET_DIMENSIONS[body.aspect_ratio]
    output = OutputSpec(
        aspect_ratio=body.aspect_ratio,
        width=width,
        height=height,
        fps=body.fps,
        fit=body.fit,
        audio=body.audio,
        quality=body.quality,
        source_gain=body.source_gain,
    )
    cuts = [
        Cut(
            media_id=media_id_from(segment.media_id),
            source_in_ms=segment.source_in_ms,
            source_out_ms=segment.source_out_ms,
            transition_in=segment.transition_in,
            transition_ms=segment.transition_ms,
            effects=tuple(effect.to_domain() for effect in segment.effects),
        )
        for segment in body.segments
    ]

    try:
        row = await service.create_manual_plan(
            project_id=ProjectId(project.id),
            cuts=cuts,
            output=output,
            music=_cue_from(body.music),
            subtitles=body.subtitles.to_domain() if body.subtitles else None,
            derived_from=body.derived_from_edit_plan_id,
        )
    except PlanInvalidError as exc:
        # A rejected hand-cut edit is a rejected *request*, not a planner bug:
        # the user trimmed past the end of a source, or left a clip shorter than
        # the renderer can produce. Every violation comes back, so the editor
        # can point at the clip rather than saying the render failed.
        raise ValidationError(
            "this timeline cannot be rendered",
            hint="; ".join(v.message for v in exc.violations),
        ) from exc

    return serialize_edit_plan(row, include_plan=True)


@router.post(
    "/projects/{project_id}/edit-plan/{edit_plan_id}/subtitles/suggest",
    response_model=SubtitleSuggestResponse,
)
async def suggest_plan_subtitles(
    edit_plan_id: UUID,
    body: SubtitleSuggestRequest,
    project: Project = Depends(require_project),
    service: EditService = Depends(get_edit_service),
    provider: LlmProvider | None = Depends(get_llm_provider),
) -> SubtitleSuggestResponse:
    """Ask a model to draft cues for a stored plan.

    Read-only. Nothing here writes a plan, a render or a cue: the suggestion
    comes back to the editor, and the lines reach the database only if the user
    submits a plan containing them. A model proposes and a person accepts --
    the same boundary the planner has had since Phase 5.

    The model is given the compiled timing of *this* plan and the user's own
    words, and nothing else: no media ids, no filenames, no storage keys, no
    project identity. What comes back is read as three keys per cue, each
    bounded, with the text stripped to display characters. It cannot name a
    font, a colour or a position, because this response has no field for one --
    the look comes from the plan's recorded style through the server's preset
    table.

    A failure is a state, not an exception: ``ok: false`` with a named reason.
    Inventing subtitles for footage nobody transcribed would be worse than an
    empty panel, so nothing is filled in.
    """
    suggestion = await service.suggest_subtitles(
        project_id=ProjectId(project.id),
        edit_plan_id=edit_plan_id,
        provider=provider,
        request_text=body.request_text,
        style=body.style,
        position=body.position,
    )

    track = suggestion.track
    return SubtitleSuggestResponse(
        ok=suggestion.ok,
        subtitles=(
            SubtitleTrackRequest(
                cues=[
                    {"start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text}
                    for cue in track.ordered
                ],
                style=track.style,
                position=track.position,
            )
            if track
            else None
        ),
        failure=suggestion.failure.value if suggestion.failure else None,
        detail=suggestion.detail,
        provider=suggestion.provider,
        model=suggestion.model,
        prompt_version=suggestion.prompt_version,
        latency_ms=round(suggestion.latency_ms, 1),
    )


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


def _cue_from(body: MusicRequest | None) -> MusicCue | None:
    """The request's music block as a domain cue.

    A field-by-field copy with no defaulting and no inference: the route's job
    is to translate, and anything it decided for the caller here would be a
    second place where an edit gets its character.
    """
    if body is None:
        return None
    return MusicCue(
        media_id=media_id_from(body.media_id),
        source_in_ms=body.source_in_ms,
        source_out_ms=body.source_out_ms,
        timeline_start_ms=body.timeline_start_ms,
        gain=body.volume,
        fade_in_ms=body.fade_in_ms,
        fade_out_ms=body.fade_out_ms,
    )


async def _music_facts(
    repo: MediaRepository, project_id: ProjectId, media_id: UUID
) -> tuple[int | None, BeatGrid | None]:
    """How long the chosen track is, and what its beats were measured to be.

    Read from the database, never from the request. A client that could state
    its own track length could state one longer than the file, and a client that
    could state its own beat grid could make the planner cut to a tempo nothing
    in the audio has.

    Missing analysis returns ``None``, which the planner treats exactly as it
    treats an untrusted grid: plan the way Phase 4 planned. A track that has not
    been analysed yet is a reason to skip beat-syncing, not to refuse the edit.
    """
    row = await repo.get_in_project(media_id, project_id)
    if row is None:
        # Not this project's, or not a media id at all. Left to the plan
        # validator, which reports it as a violation the editor can show rather
        # than a bare 404 from a field the user never filled in.
        return None, None

    payload = await repo.analysis_payload(media_id, AnalyzerName.BEATS)
    return row.duration_ms, grid_from_payload(payload)


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["PRESET_DIMENSIONS", "AudioMode", "ClipOrder", "FitMode", "router"]


# ----------------------------------------------------------- reference (P8)
def _profile_response(
    media_id: UUID, profile: ReferenceProfile, suggests_beat_sync: bool
) -> ReferenceProfileResponse:
    """Serialise a profile. Nulls stay null; nothing is defaulted on the way out."""

    def measured(value: Measurement | None) -> MeasurementResponse | None:
        return (
            MeasurementResponse(value=round(value.value, 4), confidence=round(value.confidence, 3))
            if value
            else None
        )

    return ReferenceProfileResponse(
        media_id=media_id,
        version=profile.version,
        duration_ms=profile.source_duration_ms,
        confidence=profile.confidence,
        usable=profile.is_usable,
        pacing=profile.pacing.value if profile.pacing else None,
        scene_count=profile.scene_count,
        shot_ms=measured(profile.shot_ms),
        shot_ms_p25=profile.shot_ms_p25,
        shot_ms_p75=profile.shot_ms_p75,
        cut_rate=measured(profile.cut_rate),
        luminance=measured(profile.luminance),
        contrast=measured(profile.contrast),
        saturation=measured(profile.saturation),
        motion=measured(profile.motion),
        bpm=profile.bpm,
        beat_confidence=profile.beat_confidence,
        beat_sync=measured(profile.beat_sync),
        suggests_beat_sync=suggests_beat_sync,
    )


async def _reference_response(
    service: ReferenceService, project: Project, repo: MediaRepository
) -> ReferenceResponse:
    media_id = ReferenceService.reference_id(project)
    if media_id is None:
        return ReferenceResponse()

    profile = await service.profile_for_project(project)
    if profile is None:
        return ReferenceResponse(
            media_id=media_id,
            pending_analyzers=[a.value for a in PROFILE_ANALYZERS],
        )

    # What is still missing, so a client can say "analyse it" instead of showing
    # an empty profile and leaving the user to guess why.
    present = await repo.analysis_payloads(media_id, PROFILE_ANALYZERS)
    pending = [a.value for a in PROFILE_ANALYZERS if a not in present]

    suggests = blend(profile_for(None), profile, StyleStrength.FULL).suggests_beat_sync
    return ReferenceResponse(
        media_id=media_id,
        pending_analyzers=pending,
        profile=_profile_response(media_id, profile, suggests),
    )


@router.get("/projects/{project_id}/reference", response_model=ReferenceResponse)
async def get_reference(
    project: Project = Depends(require_project),
    projects: ProjectRepository = Depends(get_project_repo),
    repo: MediaRepository = Depends(get_media_repo),
) -> ReferenceResponse:
    """The project's reference clip and what it measures as.

    The profile is recomputed here from the reference's analysis rows rather
    than stored, so it can never be stale against a re-analysis.
    """
    return await _reference_response(ReferenceService(projects, repo), project, repo)


@router.put("/projects/{project_id}/reference", response_model=ReferenceResponse)
async def set_reference(
    body: ReferenceRequest,
    project: Project = Depends(require_project),
    projects: ProjectRepository = Depends(get_project_repo),
    repo: MediaRepository = Depends(get_media_repo),
    session: AsyncSession = Depends(get_session),
) -> ReferenceResponse:
    """Nominate a clip in this project as the style reference.

    The id is resolved through the project before it is written, so a media id
    belonging to someone else is a 404 -- the same answer a nonexistent id gets,
    because a distinct "not yours" would confirm the id exists.
    """
    service = ReferenceService(projects, repo)
    await service.set_reference(project, media_id_from(body.media_id))
    await session.commit()
    return await _reference_response(service, project, repo)


@router.delete("/projects/{project_id}/reference", response_model=ReferenceResponse)
async def clear_reference(
    project: Project = Depends(require_project),
    projects: ProjectRepository = Depends(get_project_repo),
    repo: MediaRepository = Depends(get_media_repo),
    session: AsyncSession = Depends(get_session),
) -> ReferenceResponse:
    """Stop styling this project after anything. The clip itself is untouched.

    Returns the resulting state rather than 204, which is what GET and PUT on
    this path return: a client that has just cleared the reference wants to
    render the empty state, and making all three shapes identical means it can
    do that from one response handler.
    """
    await ReferenceService(projects, repo).clear_reference(project)
    await session.commit()
    return ReferenceResponse()
