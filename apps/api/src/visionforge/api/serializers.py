"""ORM row -> response model.

Kept out of the routers so that the derived fields (progress, has_thumbnail) are
computed in exactly one place and cannot drift between endpoints.
"""

from __future__ import annotations

from visionforge.api.schemas.analysis import AnalysisResponse
from visionforge.api.schemas.media import (
    DerivativeResponse,
    JobResponse,
    JobStepResponse,
    MediaResponse,
)
from visionforge.domain.jobs import StepStatus
from visionforge.domain.media import DerivativeKind
from visionforge.infra.db.models import Job, MediaAnalysis, MediaAsset


def serialize_media(media: MediaAsset) -> MediaResponse:
    derivatives = list(media.derivatives)
    return MediaResponse(
        id=media.id,
        project_id=media.project_id,
        original_filename=media.original_filename,
        kind=str(media.kind),
        status=str(media.status),
        bytes_size=media.bytes_size,
        mime_type=media.mime_type,
        sha256=media.sha256,
        container_format=media.container_format,
        duration_ms=media.duration_ms,
        width=media.width,
        height=media.height,
        fps=media.fps,
        codec=media.codec,
        pix_fmt=media.pix_fmt,
        bit_rate=media.bit_rate,
        sample_rate=media.sample_rate,
        channels=media.channels,
        error=media.error,
        created_at=media.created_at,
        derivatives=[
            DerivativeResponse(
                kind=str(d.kind),
                variant=d.variant,
                bytes_size=d.bytes_size,
                width=d.width,
                height=d.height,
            )
            for d in derivatives
        ],
        has_thumbnail=any(d.kind == DerivativeKind.THUMBNAIL for d in derivatives),
        has_proxy=any(d.kind == DerivativeKind.PROXY for d in derivatives),
    )


def job_progress(job: Job) -> float:
    """Fraction of steps completed. Derived from real step rows, never invented."""
    if not job.steps:
        return 0.0
    done = sum(1 for s in job.steps if s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED))
    return round(done / len(job.steps), 4)


def serialize_job(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        media_id=job.media_id,
        type=str(job.type),
        status=str(job.status),
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        cancel_requested=job.cancel_requested,
        progress=job_progress(job),
        result=job.result,
        error=job.error,
        retry_at=job.retry_at,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        steps=[
            JobStepResponse(
                seq=s.seq,
                name=s.name,
                status=str(s.status),
                attempt=s.attempt,
                error=s.error,
                metrics=s.metrics,
                started_at=s.started_at,
                finished_at=s.finished_at,
            )
            for s in sorted(job.steps, key=lambda s: s.seq)
        ],
    )


def serialize_analysis(row: MediaAnalysis) -> AnalysisResponse:
    """Serialise one analysis row.

    The embedding is reported as a boolean rather than 512 floats: the vector is
    large, and the similarity endpoint is a better answer to every question a
    client would use it for.
    """
    return AnalysisResponse(
        id=row.id,
        media_id=row.media_id,
        analyzer=str(row.analyzer),
        analyzer_version=row.analyzer_version,
        status=str(row.status),
        payload=row.payload,
        metrics=row.metrics,
        has_embedding=row.embedding is not None,
        created_at=row.created_at,
    )
