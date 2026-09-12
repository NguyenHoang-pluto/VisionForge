"""Project and media routes.

Note what is absent: no endpoint accepts file bytes. Uploads are presigned and go
browser -> object storage directly (Phase 0 rule #1).
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.api.dependencies import (
    DEV_USER_EMAIL,
    current_user_id,
    get_job_dispatcher,
    get_job_repo,
    get_media_repo,
    get_media_service,
    get_project_repo,
    get_session,
    get_user_repo,
    require_project,
)
from visionforge.api.schemas.media import (
    CompleteUploadResponse,
    JobResponse,
    MediaListResponse,
    MediaResponse,
    ProjectCreateRequest,
    ProjectResponse,
    SignedUrlResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from visionforge.api.serializers import serialize_job, serialize_media
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.application.media_service import DOWNLOAD_URL_TTL_S, MediaService
from visionforge.domain.errors import NotFoundError
from visionforge.domain.ids import MediaId, ProjectId, UserId
from visionforge.domain.jobs import JobType
from visionforge.domain.media import DerivativeKind, MediaStatus
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import (
    JobRepository,
    MediaRepository,
    ProjectRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["projects", "media"])


# ------------------------------------------------------------------- projects
@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateRequest,
    session: AsyncSession = Depends(get_session),
    projects: ProjectRepository = Depends(get_project_repo),
    users: UserRepository = Depends(get_user_repo),
    user_id: UserId = Depends(current_user_id),
) -> ProjectResponse:
    # Phase 2 runs as a single development user; seed the row on first use so
    # the foreign key is real rather than assumed.
    if await users.get(user_id) is None:
        user = await users.create(email=DEV_USER_EMAIL, display_name="Development User")
        user.id = user_id
        await session.flush()

    project = await projects.create(user_id=user_id, title=body.title, description=body.description)
    await session.commit()
    return ProjectResponse.model_validate(project)


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects(
    projects: ProjectRepository = Depends(get_project_repo),
    media: MediaRepository = Depends(get_media_repo),
    user_id: UserId = Depends(current_user_id),
) -> list[ProjectResponse]:
    rows = await projects.list_for_user(user_id)
    return [
        ProjectResponse(
            id=p.id,
            title=p.title,
            description=p.description,
            created_at=p.created_at,
            media_count=await media.count_in_project(p.id),
        )
        for p in rows
    ]


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project: Project = Depends(require_project),
    media: MediaRepository = Depends(get_media_repo),
) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        title=project.title,
        description=project.description,
        created_at=project.created_at,
        media_count=await media.count_in_project(project.id),
    )


# ---------------------------------------------------------------------- media
@router.post(
    "/projects/{project_id}/media/upload-url",
    response_model=UploadUrlResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_upload_url(
    body: UploadUrlRequest,
    project: Project = Depends(require_project),
    service: MediaService = Depends(get_media_service),
) -> UploadUrlResponse:
    """Reserve a media row and presign a PUT for its bytes.

    The object key is derived from server-side identifiers. The client's filename
    is kept for display only and never becomes a path component.
    """
    ticket = await service.create_upload_ticket(
        project_id=ProjectId(project.id),
        filename=body.filename,
        declared_size=body.size_bytes,
    )
    return UploadUrlResponse(
        media_id=ticket.media_id,
        object_key=ticket.object_key,
        upload_url=ticket.upload.url,
        expires_at=ticket.upload.expires_at,
        max_bytes=ticket.upload.max_bytes,
    )


@router.post(
    "/projects/{project_id}/media/{media_id}/complete",
    response_model=CompleteUploadResponse,
)
async def complete_upload(
    media_id: UUID,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    project: Project = Depends(require_project),
    service: MediaService = Depends(get_media_service),
    dispatcher: JobDispatcher = Depends(get_job_dispatcher),
) -> CompleteUploadResponse:
    """Confirm the bytes landed, then create and dispatch the ingest job."""
    media, should_process = await service.complete_upload(
        project_id=ProjectId(project.id), media_id=MediaId(media_id)
    )

    if not should_process:
        return CompleteUploadResponse(media_id=media.id, status=str(media.status))

    job, _created = await dispatcher.create_and_dispatch(
        project_id=ProjectId(project.id),
        job_type=JobType.MEDIA_INGEST,
        params={"media_id": str(media_id)},
        media_id=media_id,
        # Default to the media id: completing the same upload twice must not
        # start a second ingest of the same bytes.
        idempotency_key=idempotency_key or f"ingest:{media_id}",
    )
    return CompleteUploadResponse(media_id=media.id, status=str(media.status), job_id=job.id)


@router.get("/projects/{project_id}/media", response_model=MediaListResponse)
async def list_media(
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    project: Project = Depends(require_project),
    media: MediaRepository = Depends(get_media_repo),
) -> MediaListResponse:
    rows = await media.list_in_project(project.id, limit=limit, offset=offset)
    total = await media.count_in_project(project.id)
    return MediaListResponse(items=[serialize_media(m) for m in rows], total=total)


@router.get("/projects/{project_id}/media/{media_id}", response_model=MediaResponse)
async def get_media(
    media_id: UUID,
    project: Project = Depends(require_project),
    media: MediaRepository = Depends(get_media_repo),
) -> MediaResponse:
    row = await media.get_in_project(media_id, project.id)
    if row is None:
        raise NotFoundError("media not found in this project")
    return serialize_media(row)


@router.get("/projects/{project_id}/media/{media_id}/thumbnail", response_model=SignedUrlResponse)
async def get_thumbnail_url(
    media_id: UUID,
    project: Project = Depends(require_project),
    media: MediaRepository = Depends(get_media_repo),
    service: MediaService = Depends(get_media_service),
) -> SignedUrlResponse:
    return await _derivative_url(media_id, project, media, service, DerivativeKind.THUMBNAIL)


@router.get("/projects/{project_id}/media/{media_id}/proxy", response_model=SignedUrlResponse)
async def get_proxy_url(
    media_id: UUID,
    project: Project = Depends(require_project),
    media: MediaRepository = Depends(get_media_repo),
    service: MediaService = Depends(get_media_service),
) -> SignedUrlResponse:
    return await _derivative_url(media_id, project, media, service, DerivativeKind.PROXY)


@router.get("/projects/{project_id}/jobs", response_model=list[JobResponse])
async def list_project_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    project: Project = Depends(require_project),
    jobs: JobRepository = Depends(get_job_repo),
) -> list[JobResponse]:
    rows = await jobs.list_in_project(project.id, limit=limit)
    return [serialize_job(j) for j in rows]


# -------------------------------------------------------------------- helpers
async def _derivative_url(
    media_id: UUID,
    project: Project,
    media: MediaRepository,
    service: MediaService,
    kind: DerivativeKind,
) -> SignedUrlResponse:
    """Presign a GET for a derivative, having first proven project ownership.

    The API returns a URL; it never streams the bytes itself.
    """
    row = await media.get_in_project(media_id, project.id)
    if row is None:
        raise NotFoundError("media not found in this project")

    derivative = next((d for d in row.derivatives if d.kind == kind), None)
    if derivative is None:
        hint = (
            "Processing may still be running."
            if row.status is not MediaStatus.READY
            else f"This media has no {kind.value}."
        )
        raise NotFoundError(f"{kind.value} not available", hint=hint)

    return SignedUrlResponse(
        url=service.presign_download(derivative.storage_key),
        expires_in_s=DOWNLOAD_URL_TTL_S,
    )
