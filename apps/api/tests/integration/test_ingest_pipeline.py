"""The media ingest pipeline end to end, against real Postgres, MinIO and FFmpeg.

The worker is driven in-process (``JobRunner``) rather than through Celery, so
each scenario is deterministic. Celery delivery itself is covered separately by
the end-to-end script.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from visionforge.domain.errors import DuplicateMediaError, PermanentError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.jobs import (
    JOB_STEP_PLANS,
    JobStatus,
    JobType,
    StepStatus,
)
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.domain.storage import original_key
from visionforge.infra.db.models import Job, JobStep, MediaAsset, Project
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers import media_ingest
from visionforge.workers.runtime import JobContext, JobRunner

pytestmark = pytest.mark.integration


def _upload(store: S3ObjectStore, db: Session, project: Project, path: Path) -> MediaAsset:
    """Create a media row and put its bytes in object storage, as the API would."""
    media_id = MediaId(uuid.uuid4())
    key = original_key(ProjectId(project.id), media_id, path.suffix)
    store.upload_file(str(path), key)

    media = MediaAsset(
        id=media_id,
        project_id=project.id,
        original_filename=path.name,
        storage_key=key,
        kind=MediaKind.IMAGE,  # a guess from the extension; ffprobe overrides it
        status=MediaStatus.UPLOADED,
    )
    db.add(media)
    db.commit()
    return media


def _job_for(db: Session, media: MediaAsset) -> Job:
    job = Job(
        id=uuid.uuid4(),
        project_id=media.project_id,
        media_id=media.id,
        type=JobType.MEDIA_INGEST,
        status=JobStatus.QUEUED,
        params={"media_id": str(media.id)},
    )
    db.add(job)
    db.flush()
    for seq, name in enumerate(JOB_STEP_PLANS[JobType.MEDIA_INGEST]):
        db.add(JobStep(job_id=job.id, seq=seq, name=name))
    db.commit()
    return job


def _run(db: Session, job: Job) -> None:
    JobRunner(db, job).run(media_ingest.STEPS)
    media_ingest.cleanup(JobContext(db, job))


class TestSuccessfulIngest:
    def test_image_produces_metadata_hash_and_thumbnail(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        _run(db, job)

        db.refresh(job)
        db.refresh(media)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED
        assert MediaStatus(media.status) is MediaStatus.READY

        # ffprobe overrode the extension-derived guess with the truth.
        assert MediaKind(media.kind) is MediaKind.IMAGE
        assert (media.width, media.height) == (64, 48)
        assert media.sha256 is not None and len(media.sha256) == 64
        assert media.bytes_size and media.bytes_size > 0
        assert media.mime_type == "image/jpeg"
        assert media.probe is not None

        derivatives = {d.kind: d for d in media.derivatives}
        assert DerivativeKind.THUMBNAIL in derivatives
        assert DerivativeKind.PROXY not in derivatives, "images get no proxy"
        assert store.stat(derivatives[DerivativeKind.THUMBNAIL].storage_key) is not None

    def test_1080p_video_gets_thumbnail_and_720p_proxy(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _upload(store, db, project, media_fixtures["video_1080.mp4"])
        job = _job_for(db, media)

        _run(db, job)

        db.refresh(media)
        assert MediaKind(media.kind) is MediaKind.VIDEO
        assert (media.width, media.height) == (1920, 1080)
        assert media.duration_ms and media.duration_ms > 0
        assert media.fps and media.fps > 0
        assert media.codec == "h264"

        derivatives = {d.kind: d for d in media.derivatives}
        assert DerivativeKind.THUMBNAIL in derivatives
        proxy = derivatives[DerivativeKind.PROXY]
        assert proxy.height == 720
        assert store.stat(proxy.storage_key) is not None

    def test_720p_video_gets_no_proxy(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Proxying a file that is already proxy-sized would waste CPU and disk."""
        media = _upload(store, db, project, media_fixtures["video_720.mp4"])
        _run(db, _job_for(db, media))

        db.refresh(media)
        kinds = {d.kind for d in media.derivatives}
        assert DerivativeKind.THUMBNAIL in kinds
        assert DerivativeKind.PROXY not in kinds

    def test_audio_gets_neither_thumbnail_nor_proxy(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _upload(store, db, project, media_fixtures["audio.mp3"])
        job = _job_for(db, media)

        _run(db, job)

        db.refresh(job)
        db.refresh(media)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED
        assert MediaKind(media.kind) is MediaKind.AUDIO
        assert media.sample_rate == 44100
        assert media.derivatives == []

    def test_all_steps_are_recorded_in_order_with_timings(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Progress is derived from real step rows, never invented."""
        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        _run(db, job)

        db.refresh(job)
        steps = sorted(job.steps, key=lambda s: s.seq)
        assert [s.name for s in steps] == list(JOB_STEP_PLANS[JobType.MEDIA_INGEST])
        assert all(StepStatus(s.status) is StepStatus.SUCCEEDED for s in steps)
        assert all(s.started_at and s.finished_at for s in steps)
        assert all(s.metrics and "duration_ms" in s.metrics for s in steps)


class TestDeduplication:
    def test_identical_bytes_in_one_project_are_rejected_as_duplicate(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        first = _upload(store, db, project, media_fixtures["image.jpg"])
        _run(db, _job_for(db, first))
        db.refresh(first)
        original_sha = first.sha256

        second = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, second)

        with pytest.raises(DuplicateMediaError) as exc_info:
            JobRunner(db, job).run(media_ingest.STEPS)

        assert exc_info.value.duplicate_of == first.id
        assert original_sha is not None

    def test_different_files_are_not_duplicates(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        first = _upload(store, db, project, media_fixtures["image.jpg"])
        second = _upload(store, db, project, media_fixtures["image2.jpg"])

        _run(db, _job_for(db, first))
        _run(db, _job_for(db, second))

        db.refresh(first)
        db.refresh(second)
        assert first.sha256 != second.sha256
        assert MediaStatus(first.status) is MediaStatus.READY
        assert MediaStatus(second.status) is MediaStatus.READY

    def test_same_bytes_in_a_different_project_are_allowed(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Dedupe is scoped to the project. Cross-project is Phase 3+ territory."""
        other = Project(id=uuid.uuid4(), user_id=project.user_id, title="Other")
        db.add(other)
        db.commit()

        first = _upload(store, db, project, media_fixtures["image.jpg"])
        second = _upload(store, db, other, media_fixtures["image.jpg"])

        _run(db, _job_for(db, first))
        _run(db, _job_for(db, second))

        db.refresh(first)
        db.refresh(second)
        assert first.sha256 == second.sha256
        assert MediaStatus(second.status) is MediaStatus.READY


class TestFailures:
    def test_corrupt_media_fails_permanently_without_retrying(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Invalid media is never worth a second attempt."""
        media = _upload(store, db, project, media_fixtures["corrupt.jpg"])
        job = _job_for(db, media)

        _run(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.FAILED
        assert job.attempts == 1, "a permanent error must not consume retries"
        assert job.error is not None
        assert job.error["retryable"] is False

        failed = [s for s in job.steps if StepStatus(s.status) is StepStatus.FAILED]
        assert [s.name for s in failed] == ["VALIDATE"]

    def test_missing_object_fails_permanently(self, db: Session, project: Project) -> None:
        media = MediaAsset(
            id=uuid.uuid4(),
            project_id=project.id,
            original_filename="ghost.jpg",
            storage_key=f"projects/{project.id}/media/{uuid.uuid4()}/original.jpg",
            kind=MediaKind.IMAGE,
            status=MediaStatus.UPLOADED,
        )
        db.add(media)
        db.commit()

        job = _job_for(db, media)
        _run(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.FAILED
        assert job.error is not None and job.error["retryable"] is False

    def test_transient_failure_enters_retry_wait_with_a_future_time(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        media_fixtures: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A storage blip must schedule a retry, not fail the job."""
        from visionforge.domain.errors import TransientError

        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        def boom(ctx: JobContext) -> None:
            raise TransientError("storage timed out")

        steps = dict(media_ingest.STEPS) | {"METADATA": boom}
        JobRunner(db, job).run(steps)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.RETRY_WAIT
        assert job.attempts == 1
        assert job.retry_at is not None
        assert job.error is not None and job.error["retryable"] is True

    def test_retries_are_exhausted_then_the_job_fails(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        from visionforge.domain.errors import TransientError

        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        def boom(ctx: JobContext) -> None:
            raise TransientError("still broken")

        steps = dict(media_ingest.STEPS) | {"VALIDATE": boom}
        for _ in range(job.max_attempts):
            db.refresh(job)
            if JobStatus(job.status) is JobStatus.RETRY_WAIT:
                job.status = JobStatus.QUEUED
                db.commit()
            JobRunner(db, job).run(steps)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.FAILED
        assert job.attempts == job.max_attempts

    def test_completed_steps_are_not_redone_on_retry(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """A retry resumes; it does not re-run work that already succeeded."""
        from visionforge.domain.errors import TransientError

        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)
        calls: list[str] = []
        metadata_attempts = 0

        def counting_validate(ctx: JobContext) -> None:
            calls.append("VALIDATE")
            media_ingest.step_validate(ctx)

        def flaky_metadata(ctx: JobContext) -> None:
            nonlocal metadata_attempts
            metadata_attempts += 1
            if metadata_attempts == 1:
                raise TransientError("first attempt fails")
            media_ingest.step_metadata(ctx)

        steps = dict(media_ingest.STEPS) | {
            "VALIDATE": counting_validate,
            "METADATA": flaky_metadata,
        }

        JobRunner(db, job).run(steps)
        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.RETRY_WAIT

        job.status = JobStatus.QUEUED
        db.commit()
        JobRunner(db, job).run(steps)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED
        assert calls == ["VALIDATE"], "VALIDATE already succeeded and must not repeat"
        assert metadata_attempts == 2, "METADATA had failed, so it must be retried"


class TestCancellation:
    def test_cancellation_is_observed_at_the_next_step_boundary(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        def cancel_after_metadata(ctx: JobContext) -> None:
            media_ingest.step_metadata(ctx)
            ctx.job.cancel_requested = True
            ctx.session.commit()

        steps = dict(media_ingest.STEPS) | {"METADATA": cancel_after_metadata}
        JobRunner(db, job).run(steps)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.CANCELLED

        by_name = {s.name: StepStatus(s.status) for s in job.steps}
        assert by_name["VALIDATE"] is StepStatus.SUCCEEDED
        assert by_name["METADATA"] is StepStatus.SUCCEEDED
        assert by_name["HASH"] is StepStatus.SKIPPED
        assert by_name["FINALIZE"] is StepStatus.SKIPPED

    def test_completed_work_is_preserved_when_cancelled(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """Cancellation stops further work; it does not roll back finished work."""
        media = _upload(store, db, project, media_fixtures["image.jpg"])
        job = _job_for(db, media)

        def cancel_after_metadata(ctx: JobContext) -> None:
            media_ingest.step_metadata(ctx)
            ctx.job.cancel_requested = True
            ctx.session.commit()

        JobRunner(db, job).run(dict(media_ingest.STEPS) | {"METADATA": cancel_after_metadata})

        db.refresh(media)
        assert media.width == 64, "metadata written before cancellation survives"


class TestIdempotencyOfSteps:
    def test_rerunning_derivative_steps_does_not_duplicate_rows(
        self, db: Session, project: Project, store: S3ObjectStore, media_fixtures: dict[str, Path]
    ) -> None:
        """The (media, kind, variant) unique constraint makes retries safe."""
        media = _upload(store, db, project, media_fixtures["video_1080.mp4"])
        job = _job_for(db, media)
        _run(db, job)

        db.refresh(media)
        before = {(d.kind, d.variant) for d in media.derivatives}
        assert len(before) == 2  # thumbnail + 720p proxy

        # Re-ingest the same asset, as a redelivered message would.
        _run(db, _job_for(db, media))

        db.refresh(media)
        after = {(d.kind, d.variant) for d in media.derivatives}
        assert after == before
        assert len(media.derivatives) == len(before)


def test_permanent_error_type_is_distinct_from_transient() -> None:
    """The retry decision reads exactly one thing: which base class was raised."""
    from visionforge.domain.errors import TransientError

    assert not issubclass(PermanentError, TransientError)
    assert not issubclass(TransientError, PermanentError)
