"""Request and response models for planning and rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MIN_OUTPUT_MS,
    AspectRatio,
    AudioMode,
    FitMode,
)
from visionforge.domain.planner import ClipOrder


class PlanCreateRequest(BaseModel):
    """Preferences for an automatic edit.

    Note what is absent: no width, no height, no codec, no path. The caller
    chooses a shape and a length; the server chooses the geometry. There is no
    field here through which a client could influence what FFmpeg is handed.
    """

    target_duration_ms: int = Field(
        default=25_000, ge=MIN_OUTPUT_MS, le=MAX_OUTPUT_MS, examples=[25_000]
    )
    max_clips: int = Field(default=8, ge=1, le=40)
    min_clips: int = Field(default=2, ge=1, le=40)
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    fps: int = Field(default=30, ge=1, le=60)
    fit: FitMode = FitMode.COVER
    audio: AudioMode = AudioMode.NONE
    order: ClipOrder = ClipOrder.SCORE_DESC


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
