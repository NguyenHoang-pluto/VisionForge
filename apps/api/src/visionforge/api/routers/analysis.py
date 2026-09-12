"""Analysis routes.

Minimal on purpose: trigger analysis, read it back, and search by similarity.
There is no natural-language search UI in Phase 3 -- the similarity endpoint
exists to prove the pgvector storage is usable, not to be a product surface.

Every route resolves access through ``require_project``. There is no endpoint
here that reaches a media row or an analysis row by id alone.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from visionforge.api.dependencies import (
    get_analysis_repo,
    get_job_dispatcher,
    get_media_repo,
    get_session,
    require_project,
)
from visionforge.api.schemas.analysis import (
    AnalysisListResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    SimilarityHit,
    SimilarMediaResponse,
)
from visionforge.api.schemas.media import JobResponse
from visionforge.api.serializers import serialize_analysis, serialize_job
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import ProjectId
from visionforge.domain.jobs import JobType
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import AnalysisRepository, MediaRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

#: Which job type runs which analyzers. The API knows the queue split so it can
#: dispatch the right job; it does not know what an analyzer does.
LANE_JOB_TYPES: dict[str, JobType] = {
    "cpu": JobType.MEDIA_ANALYZE_CPU,
    "gpu": JobType.MEDIA_ANALYZE_GPU,
}


@router.post(
    "/projects/{project_id}/analysis",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_analysis(
    body: AnalyzeRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    project: Project = Depends(require_project),
    media_repo: MediaRepository = Depends(get_media_repo),
    dispatcher: JobDispatcher = Depends(get_job_dispatcher),
) -> AnalyzeResponse:
    """Queue analysis for one asset or for every ready asset in the project.

    Returns 202: the work is queued, not done. Progress is followed through the
    existing job endpoints and the SSE stream -- Phase 3 adds no second
    progress mechanism.
    """
    lanes = body.lanes or ["cpu", "gpu"]
    for lane in lanes:
        if lane not in LANE_JOB_TYPES:
            raise ValidationError(
                f"unknown analysis lane {lane!r}",
                hint=f"Supported lanes: {', '.join(sorted(LANE_JOB_TYPES))}",
            )

    if body.media_id is not None:
        media = await media_repo.get_in_project(body.media_id, project.id)
        if media is None:
            raise NotFoundError("media not found in this project")
        targets = [media]
    else:
        # Only ready assets: analysing something mid-ingest would read a proxy
        # that does not exist yet.
        targets = [
            m for m in await media_repo.list_in_project(project.id) if str(m.status) == "ready"
        ]

    if not targets:
        raise ValidationError(
            "no media ready for analysis",
            hint="Upload media and let ingest finish first.",
        )

    jobs: list[JobResponse] = []
    for media in targets:
        for lane in lanes:
            job_type = LANE_JOB_TYPES[lane]
            # Default key is (media, lane), so re-requesting analysis of the same
            # asset does not queue a second identical job.
            key = idempotency_key or f"analyze:{lane}:{media.id}"
            job, _created = await dispatcher.create_and_dispatch(
                project_id=ProjectId(project.id),
                job_type=job_type,
                params={"media_id": str(media.id), "lane": lane},
                media_id=media.id,
                idempotency_key=key,
            )
            jobs.append(serialize_job(job))

    return AnalyzeResponse(queued=len(jobs), media_count=len(targets), lanes=lanes, jobs=jobs)


@router.get("/projects/{project_id}/analysis", response_model=AnalysisListResponse)
async def list_project_analysis(
    analyzer: AnalyzerName | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
    project: Project = Depends(require_project),
    repo: AnalysisRepository = Depends(get_analysis_repo),
) -> AnalysisListResponse:
    rows = await repo.list_for_project(project.id, analyzer=analyzer, limit=limit)
    return AnalysisListResponse(items=[serialize_analysis(row) for row in rows], total=len(rows))


@router.get(
    "/projects/{project_id}/media/{media_id}/analysis",
    response_model=AnalysisListResponse,
)
async def get_media_analysis(
    media_id: UUID,
    project: Project = Depends(require_project),
    media_repo: MediaRepository = Depends(get_media_repo),
    repo: AnalysisRepository = Depends(get_analysis_repo),
) -> AnalysisListResponse:
    """Latest result per analyzer for one asset.

    Superseded versions are not returned: they stay in the table for history,
    but a caller asking "what do we know about this file" wants current answers.
    """
    if await media_repo.get_in_project(media_id, project.id) is None:
        raise NotFoundError("media not found in this project")

    rows = await repo.latest_for_media(media_id)
    return AnalysisListResponse(items=[serialize_analysis(row) for row in rows], total=len(rows))


@router.get(
    "/projects/{project_id}/media/{media_id}/similar",
    response_model=SimilarMediaResponse,
)
async def find_similar(
    media_id: UUID,
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
    project: Project = Depends(require_project),
    media_repo: MediaRepository = Depends(get_media_repo),
    repo: AnalysisRepository = Depends(get_analysis_repo),
) -> SimilarMediaResponse:
    """Visually similar media, by cosine distance over CLIP embeddings.

    The retrieval foundation Phase 3 was asked to establish, exposed as one
    endpoint so the HNSW index is demonstrably wired rather than merely created.
    The search is project-scoped in SQL, so similarity can never cross the
    ownership boundary.
    """
    if await media_repo.get_in_project(media_id, project.id) is None:
        raise NotFoundError("media not found in this project")

    row = await repo.get(media_id=media_id, analyzer=AnalyzerName.CLIP, version="1")
    if row is None or row.embedding is None:
        raise NotFoundError(
            "no CLIP embedding for this media",
            hint="Run GPU analysis on it first.",
        )

    hits = await repo.find_similar(project_id=project.id, embedding=row.embedding, limit=limit + 1)
    return SimilarMediaResponse(
        query_media_id=media_id,
        results=[
            SimilarityHit(
                media_id=hit.media_id,
                distance=round(distance, 6),
                similarity=round(1.0 - distance, 6),
            )
            for hit, distance in hits
            if hit.media_id != media_id
        ][:limit],
    )


__all__ = ["router"]
