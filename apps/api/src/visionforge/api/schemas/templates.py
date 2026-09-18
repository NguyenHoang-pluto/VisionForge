"""Wire shapes for templates (Phase 12)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from visionforge.domain.template import MAX_TEMPLATE_NAME, EditTemplate


class TemplateSlotResponse(BaseModel):
    duration_ms: int
    energy: float
    role: str
    transition_in: str
    transition_ms: int
    motion: str | None
    prefer: str
    beats: int | None


class TemplateResponse(BaseModel):
    #: A library template's id is a short name (``travel``); a user's is a UUID.
    id: str
    name: str
    source: str
    aspect: str
    bpm: float | None
    total_ms: int
    slot_count: int
    still_slots: int
    slots: list[TemplateSlotResponse]
    #: The video a user's template was measured from, while it exists.
    source_media_id: UUID | None = None
    #: How the measurement went. ``None`` for a library template.
    extraction: dict[str, Any] | None = None
    created_at: datetime | None = None

    @classmethod
    def of(
        cls,
        template: EditTemplate,
        *,
        source_media_id: UUID | None = None,
        extraction: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TemplateResponse:
        payload = template.as_payload()
        return cls(
            id=template.id,
            name=template.name,
            source=template.source.value,
            aspect=template.aspect.value,
            bpm=template.bpm,
            total_ms=template.total_ms,
            slot_count=len(template.slots),
            still_slots=template.still_slots,
            slots=[TemplateSlotResponse(**slot) for slot in payload["slots"]],
            source_media_id=source_media_id,
            extraction=extraction,
            created_at=created_at,
        )


class TemplateListResponse(BaseModel):
    library: list[TemplateResponse]
    mine: list[TemplateResponse]


class TemplateCreateRequest(BaseModel):
    """Measure a template from a video in this project.

    Carries a media id and a name, and nothing about the template itself: the
    slots are measured by the server from its own analysis of that video.
    """

    media_id: UUID
    name: str = Field(min_length=1, max_length=MAX_TEMPLATE_NAME)


class TemplateRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_TEMPLATE_NAME)
