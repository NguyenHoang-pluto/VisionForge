"""Templates (Phase 12): the library, and the ones a user measured.

Four routes. Listing and measuring are the feature; renaming and deleting are
what a user needs once they have more than a couple. Filling a template is not
here -- it is the ordinary plan route with a ``template_id``, so a templated
edit is stored, versioned and rendered exactly like any other.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.api.dependencies import (
    current_user_id,
    get_session,
    get_template_service,
    require_project,
)
from visionforge.api.schemas.templates import (
    TemplateCreateRequest,
    TemplateListResponse,
    TemplateRenameRequest,
    TemplateResponse,
)
from visionforge.application.template_service import StoredTemplate, TemplateService
from visionforge.domain.editplan import media_id_from
from visionforge.domain.ids import UserId
from visionforge.infra.db.models import Project

router = APIRouter(prefix="/api", tags=["templates"])


def _mine(stored: StoredTemplate) -> TemplateResponse:
    return TemplateResponse.of(
        stored.template,
        source_media_id=stored.source_media_id,
        extraction=stored.extraction,
        created_at=stored.created_at,
    )


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    service: TemplateService = Depends(get_template_service),
    user_id: UserId = Depends(current_user_id),
) -> TemplateListResponse:
    """The library, and the caller's own templates, newest first."""
    return TemplateListResponse(
        library=[TemplateResponse.of(template) for template in service.library()],
        mine=[_mine(stored) for stored in await service.mine(user_id)],
    )


@router.post(
    "/projects/{project_id}/templates",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def measure_template(
    body: TemplateCreateRequest,
    project: Project = Depends(require_project),
    service: TemplateService = Depends(get_template_service),
    session: AsyncSession = Depends(get_session),
    user_id: UserId = Depends(current_user_id),
) -> TemplateResponse:
    """Measure a template from a video in this project and keep it.

    The video must be analysed: its cuts become the slots, its motion their
    energy, and its beats -- when trusted -- how the slots re-time to other
    music. A video with no detected cuts is refused with the reason.
    """
    stored = await service.measure(
        project=project, media_id=media_id_from(body.media_id), name=body.name, user_id=user_id
    )
    await session.commit()
    return _mine(stored)


@router.patch("/templates/{template_id}", response_model=TemplateResponse)
async def rename_template(
    template_id: UUID,
    body: TemplateRenameRequest,
    service: TemplateService = Depends(get_template_service),
    session: AsyncSession = Depends(get_session),
    user_id: UserId = Depends(current_user_id),
) -> TemplateResponse:
    stored = await service.rename(template_id, body.name, user_id)
    await session.commit()
    return _mine(stored)


@router.delete(
    "/templates/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_template(
    template_id: UUID,
    service: TemplateService = Depends(get_template_service),
    session: AsyncSession = Depends(get_session),
    user_id: UserId = Depends(current_user_id),
) -> Response:
    """Delete one of the caller's templates. Library templates cannot be."""
    await service.delete(template_id, user_id)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
