"""Templates: listing, resolving, measuring and keeping them (Phase 12).

Thin, like ``ReferenceService``. The measuring is ``template_extract``'s job and
the meaning of a template is the domain's; what is left for this layer is the
part that touches the database and the part that enforces ownership.

Two rules live here and nowhere else:

**A user's template is theirs.** Every lookup of a stored template is scoped to
its owner, and an id that is not theirs gets the same 404 an id that does not
exist gets -- a distinct "not yours" would confirm the id exists.

**A template is measured from a video in a project the user owns.** The media
id is resolved through the project before anything is read, so nobody can
measure someone else's footage by guessing its id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID, uuid4

from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.template import (
    MAX_TEMPLATE_NAME,
    EditTemplate,
    TemplateInvalidError,
    template_from_payload,
)
from visionforge.domain.template_extract import extract_template
from visionforge.domain.template_library import LIBRARY, builtin_template
from visionforge.infra.db.models import EditTemplateRow, Project
from visionforge.infra.db.repositories import MediaRepository, TemplateRepository

logger = logging.getLogger(__name__)

#: What a template is measured from. Scenes are required -- without cuts there
#: is no structure -- and the other two sharpen it when present.
TEMPLATE_ANALYZERS: tuple[AnalyzerName, ...] = (
    AnalyzerName.SCENES,
    AnalyzerName.DYNAMICS,
    AnalyzerName.BEATS,
)


@dataclass(frozen=True, slots=True)
class StoredTemplate:
    """A user's template, with what the interface needs to show about it."""

    template: EditTemplate
    source_media_id: UUID | None
    extraction: dict[str, Any] | None
    created_at: Any


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned or len(cleaned) > MAX_TEMPLATE_NAME:
        raise ValidationError(
            f"a template name must be 1-{MAX_TEMPLATE_NAME} characters",
            hint="Give it a short name you will recognise later.",
        )
    return cleaned


def _stored(row: EditTemplateRow) -> StoredTemplate:
    # The row id is authoritative. The payload's own id is overwritten rather
    # than trusted, so a hand-edited row cannot claim another template's id.
    template = replace(template_from_payload(row.payload), id=str(row.id), name=row.name)
    return StoredTemplate(
        template=template,
        source_media_id=row.source_media_id,
        extraction=row.extraction,
        created_at=row.created_at,
    )


class TemplateService:
    def __init__(self, templates: TemplateRepository, media: MediaRepository) -> None:
        self._templates = templates
        self._media = media

    # ------------------------------------------------------------------ read
    @staticmethod
    def library() -> tuple[EditTemplate, ...]:
        return LIBRARY

    async def mine(self, user_id: UUID) -> list[StoredTemplate]:
        stored: list[StoredTemplate] = []
        for row in await self._templates.list_for_user(user_id):
            try:
                stored.append(_stored(row))
            except TemplateInvalidError:
                # A row this build cannot read is skipped and logged rather
                # than failing the whole list: one bad row should not hide the
                # rest of a user's templates.
                logger.warning("unreadable template row", extra={"template_id": str(row.id)})
        return stored

    async def resolve(self, template_id: str, user_id: UUID) -> EditTemplate:
        """A library template by name, or one of the user's by id. Else 404."""
        builtin = builtin_template(template_id)
        if builtin is not None:
            return builtin
        try:
            row_id = UUID(template_id)
        except ValueError:
            raise NotFoundError("template not found") from None
        row = await self._templates.get_owned(row_id, user_id)
        if row is None:
            raise NotFoundError("template not found")
        try:
            return _stored(row).template
        except TemplateInvalidError as exc:
            raise ValidationError(
                "this template can no longer be read",
                hint="Measure it again from its video, or delete it.",
            ) from exc

    # ----------------------------------------------------------------- write
    async def measure(
        self, *, project: Project, media_id: MediaId, name: str, user_id: UUID
    ) -> StoredTemplate:
        """Measure a template from a video in ``project`` and keep it."""
        cleaned = _clean_name(name)
        asset = await self._media.get_in_project(media_id, project.id)
        if asset is None:
            raise NotFoundError("media not found in this project")

        kind = MediaKind(asset.kind)
        if kind is not MediaKind.VIDEO:
            raise ValidationError(
                f"a template is measured from a video, not {kind.value}",
                hint="Pick an edited video whose cutting you want to copy.",
            )
        if MediaStatus(asset.status) is not MediaStatus.READY:
            raise ValidationError(
                "this video is still being processed",
                hint="Wait for ingest to finish, then try again.",
            )

        payloads = await self._media.analysis_payloads(media_id, TEMPLATE_ANALYZERS)
        if AnalyzerName.SCENES not in payloads:
            raise ValidationError(
                "this video has not been analysed for cuts yet",
                hint="Run analysis on it, then make the template.",
            )

        template_id = uuid4()
        try:
            extraction = extract_template(
                template_id=str(template_id),
                name=cleaned,
                width=asset.width,
                height=asset.height,
                scenes=payloads.get(AnalyzerName.SCENES),
                dynamics=payloads.get(AnalyzerName.DYNAMICS),
                beats=payloads.get(AnalyzerName.BEATS),
            )
        except TemplateInvalidError as exc:
            raise ValidationError(
                str(exc), hint="A template copies an edit; pick a video with several shots."
            ) from exc

        row = await self._templates.create(
            template_id=template_id,
            user_id=user_id,
            name=cleaned,
            source_media_id=media_id,
            payload=extraction.template.as_payload(),
            extraction=extraction.as_payload(),
        )
        return _stored(row)

    async def rename(self, template_id: UUID, name: str, user_id: UUID) -> StoredTemplate:
        row = await self._owned(template_id, user_id)
        cleaned = _clean_name(name)
        payload = {**row.payload, "name": cleaned}
        await self._templates.rename(row, cleaned, payload)
        return _stored(row)

    async def delete(self, template_id: UUID, user_id: UUID) -> None:
        await self._templates.delete(await self._owned(template_id, user_id))

    async def _owned(self, template_id: UUID, user_id: UUID) -> EditTemplateRow:
        row = await self._templates.get_owned(template_id, user_id)
        if row is None:
            raise NotFoundError("template not found")
        return row
