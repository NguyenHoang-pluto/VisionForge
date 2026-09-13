"""Request and response models for planning and rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from visionforge.domain.editbrief import MAX_REQUEST_CHARS
from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENTS,
    MIN_OUTPUT_MS,
    AspectRatio,
    AudioMode,
    FitMode,
    QualityPreset,
    TransitionKind,
)
from visionforge.domain.llm_planner import PlannerMode
from visionforge.domain.planner import ClipOrder
from visionforge.domain.style import FPS_PRESETS, EditStyle


class PlanCreateRequest(BaseModel):
    """Preferences for an automatic edit.

    Note what is absent: no width, no height, no codec, no CRF, no path. The
    caller chooses a shape, a length and a quality *level*; the server chooses
    the geometry and the encoder settings. There is no field here through which
    a client -- or a model reading a field a client set -- could influence what
    FFmpeg is handed.

    Several fields are ``None`` by default rather than carrying a literal. That
    is how "the user did not say" stays distinguishable from "the user chose the
    default", which is what lets a style supply its own defaults without
    overriding an explicit choice.
    """

    # --- how to plan ---
    mode: PlannerMode = PlannerMode.AUTOMATIC
    style: EditStyle | None = None
    #: What the user wants, in their own words. Bounded, and the only free text
    #: in the API. It reaches a model as data inside a delimited block, and it
    #: is never stored -- only a digest of it is.
    request_text: str | None = Field(default=None, max_length=MAX_REQUEST_CHARS)

    # --- what to produce ---
    target_duration_ms: int | None = Field(
        default=None, ge=MIN_OUTPUT_MS, le=MAX_OUTPUT_MS, examples=[25_000]
    )
    max_clips: int = Field(default=8, ge=1, le=40)
    min_clips: int = Field(default=2, ge=1, le=40)
    aspect_ratio: AspectRatio | None = None
    fps: int = Field(default=30, ge=1, le=60)
    fit: FitMode = FitMode.COVER
    audio: AudioMode | None = None
    order: ClipOrder | None = None
    quality: QualityPreset = QualityPreset.BALANCED

    @field_validator("fps")
    @classmethod
    def _fps_must_be_a_preset(cls, value: int) -> int:
        """Only the offered frame rates.

        The plan validator permits 1-120, which is what the *renderer* can
        survive. This is the narrower question of what a client may ask for, and
        a closed set is the same argument as the aspect-ratio preset map: an
        arbitrary number here is a request nobody has reason to make.
        """
        if value not in FPS_PRESETS:
            raise ValueError(f"fps must be one of {', '.join(str(f) for f in FPS_PRESETS)}")
        return value


class ManualCutRequest(BaseModel):
    """One clip on a hand-cut timeline.

    Note what is absent, again: no ``order`` and no timeline position. The
    sequence of this list is the sequence of the edit, which is the only way to
    express it that cannot describe an overlap or a hole.
    """

    media_id: UUID
    source_in_ms: int = Field(ge=0)
    source_out_ms: int = Field(gt=0)
    transition_in: TransitionKind = TransitionKind.CUT


class ManualPlanCreateRequest(BaseModel):
    """A timeline the user assembled, submitted for validation and storage.

    Carries the same output *intent* as an automatic plan -- a shape, a frame
    rate, a quality level -- and the same nothing-else. The editor sends which
    clips and which trims; it has no field for geometry, an encoder setting or a
    path, so a timeline cannot smuggle one through the route an automatic plan
    is protected from.
    """

    segments: list[ManualCutRequest] = Field(min_length=1, max_length=MAX_SEGMENTS)

    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    fps: int = Field(default=30, ge=1, le=60)
    fit: FitMode = FitMode.COVER
    audio: AudioMode = AudioMode.NONE
    quality: QualityPreset = QualityPreset.BALANCED

    #: The automatic plan this timeline was cut from, when there was one.
    derived_from_edit_plan_id: UUID | None = None

    @field_validator("fps")
    @classmethod
    def _fps_must_be_a_preset(cls, value: int) -> int:
        if value not in FPS_PRESETS:
            raise ValueError(f"fps must be one of {', '.join(str(f) for f in FPS_PRESETS)}")
        return value


class EditPlanSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    planner: str
    planner_version: str
    segment_count: int
    total_duration_ms: int
    created_at: datetime


class EditPlanDetail(EditPlanSummary):
    """A plan with its full document and the selection that produced it."""

    plan: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)

    #: How the planner was chosen, and why. Present on every plan.
    mode: dict[str, Any] | None = None
    #: Provider, model, prompt version, latency, tokens, and the fallback reason
    #: if there was one. Present only when a model was involved. Never contains
    #: a key, a prompt or a completion.
    llm: dict[str, Any] | None = None


class PlannerCapabilities(BaseModel):
    """What this server can do, for the UI to render honestly.

    Deliberately thin. The frontend needs to know whether to offer the AI option
    and what to label a plan with; it does not need, and never receives, the
    credential that makes AI possible.
    """

    ai_available: bool
    provider: str | None = None
    model: str | None = None
    #: True when the configured "provider" is the deterministic local stub. The
    #: UI says so plainly rather than letting a stub edit pass as AI.
    is_stub: bool = False
    error: str | None = None

    modes: list[str]
    styles: list[dict[str, Any]]
    aspect_ratios: list[dict[str, Any]]
    fps_presets: list[int]
    quality_presets: list[str]
    prompt_version: str
    max_request_chars: int

    #: The bounds every plan is validated against, declared rather than left to
    #: be rediscovered. An editor has to clamp a trim handle *while the pointer
    #: is moving*, which means it needs these numbers client-side; getting them
    #: from here is what stops a copy in the browser from silently drifting out
    #: of agreement with the validator that actually enforces them.
    segment_bounds: dict[str, int]


class EditPlanListResponse(BaseModel):
    items: list[EditPlanSummary]
    total: int


class RenderCreateRequest(BaseModel):
    edit_plan_id: UUID


class RenderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    edit_plan_id: UUID
    job_id: UUID | None
    status: str
    bytes_size: int | None = None
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    spec: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime

    #: Presigned and short-lived; present only once the render is ready.
    playback_url: str | None = None
    playback_expires_in_s: int | None = None


class RenderListResponse(BaseModel):
    items: list[RenderResponse]
    total: int
