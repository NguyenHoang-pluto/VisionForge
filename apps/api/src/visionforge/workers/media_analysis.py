"""The media analysis pipelines, CPU and GPU.

Both follow Phase 2's shape exactly: plain step functions over a ``JobContext``,
with retry, cancellation, progress and state transitions owned by ``JobRunner``.
Nothing about the job machinery is re-invented here.

**Proxy-first (Phase 3 rule).** RESOLVE prefers the 720p proxy over the original
wherever one exists. Analysing a 4K master to compute a blur score would decode
eight times the pixels for a number that is then normalised to 512px anyway. The
original stays the source of truth for rendering and is never opened here. Which
representation was used is recorded on every result, because a score computed on
a proxy is not interchangeable with one computed on a master.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import select

from visionforge.domain.analysis import (
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    Analyzer,
    AnalyzerName,
)
from visionforge.domain.errors import PermanentError
from visionforge.domain.ids import MediaId
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.infra.db.models import MediaAnalysis, MediaAsset, MediaDerivative
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers.runtime import JobContext

logger = logging.getLogger(__name__)


def _media(ctx: JobContext) -> MediaAsset:
    media = ctx.session.get(MediaAsset, ctx.job.media_id)
    if media is None:
        raise PermanentError("media row disappeared")
    return media


def _store(ctx: JobContext) -> S3ObjectStore:
    store = ctx.data.get("store")
    if store is None:
        store = S3ObjectStore()
        ctx.data["store"] = store
    return store


# ---------------------------------------------------------------------- RESOLVE
def step_resolve(ctx: JobContext) -> None:
    """Download the best analysis representation and build the AnalysisSource.

    Idempotent and lazy in the same way as Phase 2's ``_local_copy``: a retry
    that resumes past RESOLVE still finds a file, because every step goes through
    ``_source`` rather than assuming an earlier step left one behind.
    """
    _source(ctx)


def _source(ctx: JobContext) -> AnalysisSource:
    cached: AnalysisSource | None = ctx.data.get("source")
    if cached is not None and os.path.exists(cached.local_path):
        return cached

    media = _media(ctx)
    if media.status is MediaStatus.FAILED:
        raise PermanentError(
            "cannot analyse media that failed ingest",
            hint="Re-upload the file or re-run ingest first.",
        )

    store = _store(ctx)

    # Proxy first. Falls back to the original for images, audio, and video that
    # never needed a proxy because it was already at or below 720p.
    proxy = ctx.session.execute(
        select(MediaDerivative).where(
            MediaDerivative.media_id == media.id,
            MediaDerivative.kind == DerivativeKind.PROXY,
        )
    ).scalar_one_or_none()

    key = proxy.storage_key if proxy is not None else media.storage_key
    used_proxy = proxy is not None

    workdir = ctx.data.get("workdir")
    if not workdir or not os.path.isdir(workdir):
        workdir = tempfile.mkdtemp(prefix=f"vfa-{ctx.job_id}-")
        ctx.data["workdir"] = workdir

    local = os.path.join(workdir, "input" + Path(key).suffix)
    store.download_to(key, local)

    source = AnalysisSource(
        media_id=MediaId(media.id),
        kind=MediaKind(media.kind),
        local_path=local,
        used_proxy=used_proxy,
        duration_ms=media.duration_ms,
        width=proxy.width if proxy is not None and proxy.width else media.width,
        height=proxy.height if proxy is not None and proxy.height else media.height,
    )
    ctx.data["source"] = source
    logger.info(
        "analysis source resolved",
        extra={
            "media_id": str(media.id),
            "used_proxy": used_proxy,
            "kind": str(media.kind),
        },
    )
    return source


# ------------------------------------------------------------------- persistence
def _record(ctx: JobContext, outcome: AnalysisOutcome) -> None:
    """Upsert one analyzer result on ``(media_id, analyzer, analyzer_version)``.

    Synchronous session: this runs inside a Celery worker, not the API.
    """
    media = _media(ctx)
    existing = ctx.session.execute(
        select(MediaAnalysis).where(
            MediaAnalysis.media_id == media.id,
            MediaAnalysis.analyzer == outcome.analyzer,
            MediaAnalysis.analyzer_version == outcome.version,
        )
    ).scalar_one_or_none()

    embedding = list(outcome.embedding) if outcome.embedding is not None else None

    if existing is not None:
        existing.status = outcome.status
        existing.payload = dict(outcome.payload)
        existing.metrics = dict(outcome.metrics) or None
        existing.embedding = embedding
        return

    ctx.session.add(
        MediaAnalysis(
            media_id=media.id,
            project_id=media.project_id,
            analyzer=outcome.analyzer,
            analyzer_version=outcome.version,
            status=outcome.status,
            payload=dict(outcome.payload),
            metrics=dict(outcome.metrics) or None,
            embedding=embedding,
        )
    )


def _run_analyzer(ctx: JobContext, analyzer: Analyzer) -> AnalysisOutcome:
    """Run one analyzer and persist whatever it reports.

    ``unsupported`` is persisted too. A stored "audio has no scenes" row is what
    stops the pipeline re-attempting it on every pass, and it is a genuine
    answer rather than a gap.
    """
    source = _source(ctx)
    outcome = analyzer.analyze(source)
    _record(ctx, outcome)

    summary = ctx.data.setdefault("summary", {})
    summary[str(analyzer.name)] = outcome.status.value
    if outcome.metrics:
        ctx.data.setdefault("metrics", {})[str(analyzer.name)] = outcome.metrics
    return outcome


# -------------------------------------------------------------------- CPU steps
def step_quality(ctx: JobContext) -> None:
    from visionforge.infra.analysis import QualityAnalyzer

    _run_analyzer(ctx, QualityAnalyzer())


def step_scenes(ctx: JobContext) -> None:
    from visionforge.infra.analysis import SceneAnalyzer

    _run_analyzer(ctx, SceneAnalyzer())


def step_phash(ctx: JobContext) -> None:
    from visionforge.infra.analysis import PerceptualHashAnalyzer

    _run_analyzer(ctx, PerceptualHashAnalyzer())


# -------------------------------------------------------------------- GPU steps
def step_embed(ctx: JobContext) -> None:
    from visionforge.infra.ml import ClipEmbeddingAnalyzer

    _run_analyzer(ctx, ClipEmbeddingAnalyzer())


def step_faces(ctx: JobContext) -> None:
    from visionforge.infra.ml import FaceDetectionAnalyzer

    _run_analyzer(ctx, FaceDetectionAnalyzer())


# --------------------------------------------------------------------- FINALIZE
def step_finalize(ctx: JobContext) -> None:
    source = ctx.data.get("source")
    ctx.job.result = {
        "media_id": str(ctx.job.media_id),
        "analyzers": ctx.data.get("summary", {}),
        "used_proxy": source.used_proxy if source else None,
        "kind": str(source.kind) if source else None,
        "gpu": ctx.data.get("metrics", {}),
    }
    cleanup(ctx)


def cleanup(ctx: JobContext) -> None:
    """Remove the scratch directory. Safe to call more than once."""
    ctx.data.pop("source", None)
    workdir = ctx.data.pop("workdir", None)
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


CPU_STEPS = {
    "RESOLVE": step_resolve,
    "QUALITY": step_quality,
    "SCENES": step_scenes,
    "PHASH": step_phash,
    "FINALIZE": step_finalize,
}

GPU_STEPS = {
    "RESOLVE": step_resolve,
    "EMBED": step_embed,
    "FACES": step_faces,
    "FINALIZE": step_finalize,
}

#: Which analyzers each queue is responsible for. Used by the API to report what
#: a project still has outstanding.
CPU_ANALYZERS = (AnalyzerName.QUALITY, AnalyzerName.SCENES, AnalyzerName.PHASH)
GPU_ANALYZERS = (AnalyzerName.CLIP, AnalyzerName.FACES)

__all__ = [
    "CPU_ANALYZERS",
    "CPU_STEPS",
    "GPU_ANALYZERS",
    "GPU_STEPS",
    "AnalysisStatus",
    "cleanup",
]
