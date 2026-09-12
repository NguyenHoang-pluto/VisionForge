"""Analysis pipelines against real Postgres, MinIO, FFmpeg and — where present — a real GPU.

The CPU lane runs unconditionally. GPU tests are marked ``gpu`` and skipped when
no CUDA device is available, so CI on a plain runner stays green.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from visionforge.domain.analysis import (
    CLIP_EMBEDDING_DIM,
    AnalysisStatus,
    AnalyzerName,
)
from visionforge.domain.jobs import JOB_STEP_PLANS, JobStatus, JobType, StepStatus
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.domain.storage import derivative_key, original_key
from visionforge.infra.db.models import (
    Job,
    JobStep,
    MediaAnalysis,
    MediaAsset,
    MediaDerivative,
    Project,
)
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers import media_analysis
from visionforge.workers.runtime import JobContext, JobRunner

pytestmark = pytest.mark.integration

try:  # pragma: no cover - import guard, not logic
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:  # pragma: no cover
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="no CUDA device available")


# -------------------------------------------------------------------- helpers
def _ingest(
    store: S3ObjectStore,
    db: Session,
    project: Project,
    path: Path,
    kind: MediaKind,
    *,
    duration_ms: int | None = None,
    width: int | None = None,
    height: int | None = None,
    with_proxy: Path | None = None,
) -> MediaAsset:
    """Create a READY media row with its bytes in storage, as ingest would."""
    media_id = uuid.uuid4()
    key = original_key(project.id, media_id, path.suffix)  # type: ignore[arg-type]
    store.upload_file(str(path), key)

    media = MediaAsset(
        id=media_id,
        project_id=project.id,
        original_filename=path.name,
        storage_key=key,
        kind=kind,
        status=MediaStatus.READY,
        sha256=uuid.uuid4().hex * 2,
        duration_ms=duration_ms,
        width=width,
        height=height,
    )
    db.add(media)
    db.flush()

    if with_proxy is not None:
        proxy_key = derivative_key(
            project.id,  # type: ignore[arg-type]
            media_id,  # type: ignore[arg-type]
            DerivativeKind.PROXY,
            "720p",
            ".mp4",
        )
        store.upload_file(str(with_proxy), proxy_key)
        db.add(
            MediaDerivative(
                media_id=media_id,
                kind=DerivativeKind.PROXY,
                variant="720p",
                storage_key=proxy_key,
                height=720,
                width=1280,
            )
        )
    db.commit()
    return media


def _job(db: Session, media: MediaAsset, job_type: JobType) -> Job:
    job = Job(
        id=uuid.uuid4(),
        project_id=media.project_id,
        media_id=media.id,
        type=job_type,
        status=JobStatus.QUEUED,
        params={"media_id": str(media.id)},
    )
    db.add(job)
    db.flush()
    for seq, name in enumerate(JOB_STEP_PLANS[job_type]):
        db.add(JobStep(job_id=job.id, seq=seq, name=name))
    db.commit()
    return job


def _run(db: Session, job: Job, steps: dict[str, object]) -> None:
    JobRunner(db, job).run(steps)  # type: ignore[arg-type]
    media_analysis.cleanup(JobContext(db, job))


def _vector_literal(embedding: object) -> str:
    """Render an embedding as a pgvector literal.

    pgvector hands back a numpy array, and ``str(list(arr))`` renders each
    element as ``np.float32(...)`` -- not valid vector input syntax.
    """
    return "[" + ", ".join(str(float(v)) for v in embedding) + "]"  # type: ignore[union-attr]


def _analyses(db: Session, media_id: uuid.UUID) -> dict[str, MediaAnalysis]:
    rows = (
        db.execute(select(MediaAnalysis).where(MediaAnalysis.media_id == media_id)).scalars().all()
    )
    return {str(r.analyzer): r for r in rows}


# ------------------------------------------------------------------- CPU lane
class TestCpuAnalysisJob:
    def test_image_produces_quality_and_phash(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        job = _job(db, media, JobType.MEDIA_ANALYZE_CPU)

        _run(db, job, media_analysis.CPU_STEPS)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED

        rows = _analyses(db, media.id)
        assert AnalysisStatus(rows["quality"].status) is AnalysisStatus.OK
        assert AnalysisStatus(rows["phash"].status) is AnalysisStatus.OK
        # An image has no scenes; that is recorded as a fact, not skipped.
        assert AnalysisStatus(rows["scenes"].status) is AnalysisStatus.UNSUPPORTED

        quality = rows["quality"].payload
        assert quality["blur_score"] > 0
        assert 0 <= quality["mean_luminance"] <= 255
        assert len(rows["phash"].payload["phash"]) == 16

    def test_video_produces_scenes(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(
            store,
            db,
            project,
            media_fixtures["video_1080.mp4"],
            MediaKind.VIDEO,
            duration_ms=1000,
            width=1920,
            height=1080,
        )
        _run(db, _job(db, media, JobType.MEDIA_ANALYZE_CPU), media_analysis.CPU_STEPS)

        rows = _analyses(db, media.id)
        assert AnalysisStatus(rows["scenes"].status) is AnalysisStatus.OK
        assert rows["scenes"].payload["scene_count"] >= 1
        assert rows["quality"].payload["frame_count"] > 1, "video must sample several frames"

    def test_audio_is_recorded_as_unsupported_and_the_job_still_succeeds(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """The Phase 3 requirement: audio must not fail analysis."""
        media = _ingest(
            store, db, project, media_fixtures["audio.mp3"], MediaKind.AUDIO, duration_ms=1000
        )
        job = _job(db, media, JobType.MEDIA_ANALYZE_CPU)

        _run(db, job, media_analysis.CPU_STEPS)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED
        assert all(
            AnalysisStatus(row.status) is AnalysisStatus.UNSUPPORTED
            for row in _analyses(db, media.id).values()
        )

    def test_every_step_is_recorded(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        job = _job(db, media, JobType.MEDIA_ANALYZE_CPU)

        _run(db, job, media_analysis.CPU_STEPS)

        db.refresh(job)
        steps = sorted(job.steps, key=lambda s: s.seq)
        assert [s.name for s in steps] == ["RESOLVE", "QUALITY", "SCENES", "PHASH", "FINALIZE"]
        assert all(StepStatus(s.status) is StepStatus.SUCCEEDED for s in steps)


# ---------------------------------------------------------------- proxy-first
class TestProxyFirst:
    def test_proxy_is_preferred_over_the_original(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """The Phase 3 architecture rule, asserted rather than assumed.

        Decoding a 1080p master to compute a score that is normalised to 512px
        anyway is wasted work; the proxy exists precisely for this.
        """
        media = _ingest(
            store,
            db,
            project,
            media_fixtures["video_1080.mp4"],
            MediaKind.VIDEO,
            duration_ms=1000,
            width=1920,
            height=1080,
            with_proxy=media_fixtures["video_720.mp4"],
        )
        job = _job(db, media, JobType.MEDIA_ANALYZE_CPU)

        _run(db, job, media_analysis.CPU_STEPS)

        db.refresh(job)
        assert job.result["used_proxy"] is True
        assert all(
            row.payload.get("used_proxy") is True
            for row in _analyses(db, media.id).values()
            if AnalysisStatus(row.status) is AnalysisStatus.OK
        )

    def test_original_is_used_when_no_proxy_exists(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        job = _job(db, media, JobType.MEDIA_ANALYZE_CPU)

        _run(db, job, media_analysis.CPU_STEPS)

        db.refresh(job)
        assert job.result["used_proxy"] is False

    def test_original_bytes_are_never_modified(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(
            store,
            db,
            project,
            media_fixtures["video_1080.mp4"],
            MediaKind.VIDEO,
            duration_ms=1000,
        )
        before = store.stat(media.storage_key)

        _run(db, _job(db, media, JobType.MEDIA_ANALYZE_CPU), media_analysis.CPU_STEPS)

        after = store.stat(media.storage_key)
        assert before is not None and after is not None
        assert before.size_bytes == after.size_bytes


# ------------------------------------------------------------------ versioning
class TestAnalyzerVersioning:
    def test_same_version_is_replaced_not_duplicated(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)

        _run(db, _job(db, media, JobType.MEDIA_ANALYZE_CPU), media_analysis.CPU_STEPS)
        _run(db, _job(db, media, JobType.MEDIA_ANALYZE_CPU), media_analysis.CPU_STEPS)

        count = (
            db.execute(
                select(MediaAnalysis).where(
                    MediaAnalysis.media_id == media.id,
                    MediaAnalysis.analyzer == AnalyzerName.QUALITY,
                )
            )
            .scalars()
            .all()
        )
        assert len(count) == 1

    def test_two_versions_of_one_analyzer_coexist(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """The whole point of versioning: a model upgrade must not erase history."""
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)

        for version in ("1", "2"):
            db.add(
                MediaAnalysis(
                    media_id=media.id,
                    project_id=project.id,
                    analyzer=AnalyzerName.QUALITY,
                    analyzer_version=version,
                    status=AnalysisStatus.OK,
                    payload={"blur_score": 100.0 if version == "1" else 200.0},
                )
            )
        db.commit()

        rows = (
            db.execute(
                select(MediaAnalysis).where(
                    MediaAnalysis.media_id == media.id,
                    MediaAnalysis.analyzer == AnalyzerName.QUALITY,
                )
            )
            .scalars()
            .all()
        )
        assert {r.analyzer_version for r in rows} == {"1", "2"}

    def test_unique_constraint_blocks_a_third_identical_row(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        import sqlalchemy.exc

        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        for _ in range(2):
            db.add(
                MediaAnalysis(
                    media_id=media.id,
                    project_id=project.id,
                    analyzer=AnalyzerName.PHASH,
                    analyzer_version="9",
                    status=AnalysisStatus.OK,
                    payload={},
                )
            )
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            db.commit()
        db.rollback()


# ------------------------------------------------------------------- GPU lane
@requires_gpu
@pytest.mark.gpu
class TestGpuAnalysisJob:
    def test_image_gets_a_clip_embedding_and_a_face_result(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        job = _job(db, media, JobType.MEDIA_ANALYZE_GPU)

        _run(db, job, media_analysis.GPU_STEPS)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED

        rows = _analyses(db, media.id)
        clip = rows["clip"]
        assert AnalysisStatus(clip.status) is AnalysisStatus.OK
        assert clip.embedding is not None
        assert len(clip.embedding) == CLIP_EMBEDDING_DIM

        # L2-normalised, so cosine distance and inner product agree.
        norm = sum(v * v for v in clip.embedding) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

        faces = rows["faces"]
        assert AnalysisStatus(faces.status) is AnalysisStatus.OK
        assert faces.payload["identity_stored"] is False
        assert isinstance(faces.payload["face_count"], int)

    def test_gpu_metrics_are_recorded(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        job = _job(db, media, JobType.MEDIA_ANALYZE_GPU)

        _run(db, job, media_analysis.GPU_STEPS)

        metrics = _analyses(db, media.id)["clip"].metrics
        assert metrics is not None
        assert metrics["device"] == "cuda"
        assert metrics["vram_peak_mb"] > 0
        assert metrics["duration_ms"] > 0
        assert "vit-b-32" in metrics["model"]

    def test_audio_is_unsupported_on_the_gpu_lane_too(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _ingest(
            store, db, project, media_fixtures["audio.mp3"], MediaKind.AUDIO, duration_ms=1000
        )
        job = _job(db, media, JobType.MEDIA_ANALYZE_GPU)

        _run(db, job, media_analysis.GPU_STEPS)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED
        rows = _analyses(db, media.id)
        assert AnalysisStatus(rows["clip"].status) is AnalysisStatus.UNSUPPORTED
        assert rows["clip"].embedding is None


# ------------------------------------------------------------------- pgvector
@requires_gpu
@pytest.mark.gpu
class TestVectorSearch:
    def test_similar_media_ranks_a_near_duplicate_first(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """End-to-end proof that the HNSW index and the embeddings work together."""
        same = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        copy = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        different = _ingest(store, db, project, media_fixtures["image2.jpg"], MediaKind.IMAGE)

        for media in (same, copy, different):
            _run(db, _job(db, media, JobType.MEDIA_ANALYZE_GPU), media_analysis.GPU_STEPS)

        query = _analyses(db, same.id)["clip"].embedding
        assert query is not None

        rows = db.execute(
            text(
                "SELECT media_id, embedding <=> CAST(:q AS vector) AS distance "
                "FROM media_analysis "
                "WHERE project_id = :p AND analyzer = 'clip' AND embedding IS NOT NULL "
                "ORDER BY distance LIMIT 5"
            ),
            {"q": _vector_literal(query), "p": str(project.id)},
        ).all()

        ranked = [r[0] for r in rows]
        assert ranked[0] == same.id, "an asset is its own nearest neighbour"
        assert ranked[1] == copy.id, "an identical image must rank above a different one"
        assert ranked.index(copy.id) < ranked.index(different.id)

    def test_similarity_search_is_project_scoped(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Similarity must never cross the authorization boundary."""
        other = Project(id=uuid.uuid4(), user_id=project.user_id, title="Other project")
        db.add(other)
        db.commit()

        mine = _ingest(store, db, project, media_fixtures["image.jpg"], MediaKind.IMAGE)
        theirs = _ingest(store, db, other, media_fixtures["image.jpg"], MediaKind.IMAGE)
        for media in (mine, theirs):
            _run(db, _job(db, media, JobType.MEDIA_ANALYZE_GPU), media_analysis.GPU_STEPS)

        query = _analyses(db, mine.id)["clip"].embedding
        rows = db.execute(
            text(
                "SELECT media_id FROM media_analysis "
                "WHERE project_id = :p AND analyzer = 'clip' AND embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 10"
            ),
            {"q": _vector_literal(query), "p": str(project.id)},  # type: ignore[arg-type]
        ).all()

        found = {r[0] for r in rows}
        assert mine.id in found
        assert theirs.id not in found, "another project's media must never be returned"


@requires_gpu
@pytest.mark.gpu
def test_gpu_smoke_pytorch_to_driver_to_card() -> None:
    """The chain Phase 3 asked to prove: PyTorch -> CUDA runtime -> driver -> RTX 3050."""
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1

    name = torch.cuda.get_device_name(0)
    assert "NVIDIA" in name

    x = torch.randn(64, 64, device="cuda")
    result = (x @ x.T).sum().item()
    torch.cuda.synchronize()

    assert result == result  # not NaN
    assert torch.cuda.get_device_capability(0) >= (7, 0), "fp16 requires SM 7.0+"
    del x
    torch.cuda.empty_cache()
