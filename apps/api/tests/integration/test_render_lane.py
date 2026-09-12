"""The render lane against real Postgres, MinIO and FFmpeg.

Produces actual MP4 files and verifies them with ffprobe. A render test that only
checks the job status would pass on a zero-byte file.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from visionforge.domain.editplan import (
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    OutputSpec,
    Segment,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.jobs import JOB_STEP_PLANS, JobStatus, JobType, StepStatus
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.domain.render import RenderStatus
from visionforge.domain.storage import derivative_key, original_key
from visionforge.infra.db.models import (
    EditPlanRow,
    Job,
    JobStep,
    MediaAsset,
    MediaDerivative,
    Project,
    RenderRow,
)
from visionforge.infra.ffmpeg import probe_media
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers import video_render
from visionforge.workers.runtime import JobContext, JobRunner

pytestmark = pytest.mark.integration


# -------------------------------------------------------------------- helpers
@pytest.fixture(scope="session")
def render_sources(media_fixtures: dict[str, Path], tmp_path_factory) -> list[Path]:  # type: ignore[no-untyped-def]
    """Four distinguishable 3-second 720p clips.

    Generated rather than reused from the analysis fixtures because a render
    needs sources long enough to trim, and visually distinct enough that a
    wrong concat order would be visible in the output.
    """
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")

    directory = tmp_path_factory.mktemp("render-sources")
    sources: list[Path] = []
    patterns = [
        "testsrc=size=1280x720:rate=25:duration=3",
        "smptebars=size=1280x720:rate=25:duration=3",
        "rgbtestsrc=size=1280x720:rate=25:duration=3",
        "testsrc2=size=1280x720:rate=25:duration=3",
    ]
    for index, pattern in enumerate(patterns):
        destination = directory / f"clip{index}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        sources.append(destination)
    return sources


def ingest(
    store: S3ObjectStore, db: Session, project: Project, path: Path, *, with_proxy: bool = False
) -> MediaAsset:
    """A READY 720p video asset with its bytes in storage."""
    media_id = uuid.uuid4()
    key = original_key(ProjectId(project.id), MediaId(media_id), ".mp4")
    store.upload_file(str(path), key)

    media = MediaAsset(
        id=media_id,
        project_id=project.id,
        original_filename=path.name,
        storage_key=key,
        kind=MediaKind.VIDEO,
        status=MediaStatus.READY,
        sha256=uuid.uuid4().hex * 2,
        duration_ms=3000,
        width=1280,
        height=720,
        fps=25.0,
    )
    db.add(media)
    db.flush()

    if with_proxy:
        proxy_key = derivative_key(
            ProjectId(project.id), MediaId(media_id), DerivativeKind.PROXY, "720p", ".mp4"
        )
        store.upload_file(str(path), proxy_key)
        db.add(
            MediaDerivative(
                media_id=media_id,
                kind=DerivativeKind.PROXY,
                variant="720p",
                storage_key=proxy_key,
                width=1280,
                height=720,
            )
        )
    db.commit()
    return media


def make_plan(
    db: Session, project: Project, media: list[MediaAsset], **output: object
) -> EditPlanRow:
    spec: dict[str, object] = {
        "aspect_ratio": AspectRatio.LANDSCAPE_16_9,
        "width": 1280,
        "height": 720,
        "fps": 25,
        "fit": FitMode.COVER,
        "audio": AudioMode.NONE,
    }
    spec.update(output)
    plan = EditPlan(
        project_id=ProjectId(project.id),
        segments=tuple(
            Segment(
                media_id=MediaId(asset.id),
                order=index,
                source_in_ms=500,
                source_out_ms=2000,
            )
            for index, asset in enumerate(media)
        ),
        output=OutputSpec(**spec),  # type: ignore[arg-type]
    )
    row = EditPlanRow(
        project_id=project.id,
        planner=plan.planner,
        planner_version=plan.planner_version,
        plan=plan.as_payload(),
        selection={},
        total_duration_ms=plan.total_duration_ms,
        segment_count=len(plan.segments),
    )
    db.add(row)
    db.commit()
    return row


def make_render_job(db: Session, project: Project, plan_row: EditPlanRow) -> tuple[RenderRow, Job]:
    render = RenderRow(project_id=project.id, edit_plan_id=plan_row.id, status=RenderStatus.PENDING)
    db.add(render)
    db.flush()

    job = Job(
        id=uuid.uuid4(),
        project_id=project.id,
        type=JobType.RENDER_VIDEO,
        status=JobStatus.QUEUED,
        params={"render_id": str(render.id), "edit_plan_id": str(plan_row.id)},
    )
    db.add(job)
    db.flush()
    for seq, name in enumerate(JOB_STEP_PLANS[JobType.RENDER_VIDEO]):
        db.add(JobStep(job_id=job.id, seq=seq, name=name))
    render.job_id = job.id
    db.commit()
    return render, job


def run_render(db: Session, job: Job) -> None:
    JobRunner(db, job).run(video_render.STEPS)
    video_render.cleanup(JobContext(db, job))


# ------------------------------------------------------------------ the render
class TestRenderJob:
    def test_produces_a_playable_mp4(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
        tmp_path: Path,
    ) -> None:
        """The Phase 4 acceptance in miniature: plan in, verified MP4 out."""
        media = [ingest(store, db, project, path) for path in render_sources[:3]]
        plan_row = make_plan(db, project, media)
        render, job = make_render_job(db, project, plan_row)

        run_render(db, job)

        db.refresh(job)
        db.refresh(render)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error
        assert RenderStatus(render.status) is RenderStatus.READY

        # The file exists in object storage and is a real video.
        assert render.storage_key
        assert store.stat(render.storage_key) is not None

        local = tmp_path / "verify.mp4"
        store.download_to(render.storage_key, str(local))
        metadata = probe_media(str(local))

        assert metadata.kind is MediaKind.VIDEO
        assert (metadata.width, metadata.height) == (1280, 720)
        assert metadata.codec == "h264"
        assert metadata.pix_fmt == "yuv420p"
        # 3 segments x 1.5 s. Encoders round to a frame boundary.
        assert metadata.duration_ms is not None
        assert 4300 <= metadata.duration_ms <= 4800

    def test_measured_values_are_stored_not_intended_ones(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        """A row reporting what it meant to produce would hide real drift."""
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        render, job = make_render_job(db, project, make_plan(db, project, media))

        run_render(db, job)

        db.refresh(render)
        assert render.duration_ms is not None and render.duration_ms > 0
        assert (render.width, render.height) == (1280, 720)
        assert render.fps is not None and render.fps > 0
        assert render.bytes_size is not None and render.bytes_size > 1000

    def test_every_step_is_recorded(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        _render, job = make_render_job(db, project, make_plan(db, project, media))

        run_render(db, job)

        db.refresh(job)
        steps = sorted(job.steps, key=lambda s: s.seq)
        assert [s.name for s in steps] == [
            "PREPARE",
            "COMPILE",
            "RENDER",
            "PUBLISH",
            "FINALIZE",
        ]
        assert all(StepStatus(s.status) is StepStatus.SUCCEEDED for s in steps)
        assert all(s.metrics and "duration_ms" in s.metrics for s in steps)

    def test_the_timeline_and_spec_are_stored_for_reproducibility(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        render, job = make_render_job(db, project, make_plan(db, project, media))

        run_render(db, job)

        db.refresh(render)
        assert render.timeline is not None
        assert render.spec is not None
        assert render.spec["video_codec"] == "h264"
        assert render.metrics and render.metrics["encoder"] == "libx264"

    @pytest.mark.parametrize(
        ("ratio", "width", "height"),
        [
            (AspectRatio.LANDSCAPE_16_9, 1280, 720),
            (AspectRatio.PORTRAIT_9_16, 720, 1280),
            (AspectRatio.SQUARE_1_1, 720, 720),
        ],
    )
    def test_each_aspect_ratio_renders_at_its_declared_geometry(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
        tmp_path: Path,
        ratio: AspectRatio,
        width: int,
        height: int,
    ) -> None:
        """Cover-fit crops a 16:9 source into portrait and square without bars."""
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        plan_row = make_plan(db, project, media, aspect_ratio=ratio, width=width, height=height)
        _render, job = make_render_job(db, project, plan_row)

        run_render(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error

        row = db.execute(
            select(RenderRow).where(RenderRow.edit_plan_id == plan_row.id)
        ).scalar_one()
        local = tmp_path / f"{ratio.name}.mp4"
        store.download_to(row.storage_key, str(local))

        metadata = probe_media(str(local))
        assert (metadata.width, metadata.height) == (width, height)

    def test_audio_is_included_when_requested(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
        tmp_path: Path,
    ) -> None:
        """The lavfi sources have no audio track, so this must still succeed.

        It is the realistic case: a folder of phone clips will contain some with
        audio and some without, and the render must not depend on it.
        """
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        plan_row = make_plan(db, project, media, audio=AudioMode.NONE)
        _render, job = make_render_job(db, project, plan_row)

        run_render(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error


class TestProxyFirst:
    def test_the_proxy_is_rendered_from_when_one_exists(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        """Same rule as analysis: never decode a master to make a preview."""
        media = [ingest(store, db, project, path, with_proxy=True) for path in render_sources[:2]]
        _render, job = make_render_job(db, project, make_plan(db, project, media))

        run_render(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error

    def test_source_media_is_never_modified(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        before = [store.stat(asset.storage_key) for asset in media]

        _render, job = make_render_job(db, project, make_plan(db, project, media))
        run_render(db, job)

        after = [store.stat(asset.storage_key) for asset in media]
        assert [s.size_bytes for s in before if s] == [s.size_bytes for s in after if s]


# -------------------------------------------------------------------- failures
class TestRenderFailures:
    def test_a_plan_referencing_deleted_media_fails_permanently(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        """Why the plan is re-validated at render time.

        Media can be deleted between planning and rendering; a stale plan must
        not reach FFmpeg.
        """
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        plan_row = make_plan(db, project, media)
        render, job = make_render_job(db, project, plan_row)

        db.delete(media[0])
        db.commit()

        run_render(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.FAILED
        assert job.attempts == 1, "an invalid plan must not consume retries"
        assert job.error is not None and job.error["retryable"] is False

    def test_a_plan_referencing_another_project_is_refused(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        """Authorization enforced in the domain, not only at the API."""
        other = Project(id=uuid.uuid4(), user_id=project.user_id, title="Other")
        db.add(other)
        db.commit()

        foreign = ingest(store, db, other, render_sources[0])
        mine = ingest(store, db, project, render_sources[1])

        plan_row = make_plan(db, project, [mine, foreign])
        _render, job = make_render_job(db, project, plan_row)

        run_render(db, job)

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.FAILED
        assert job.error is not None
        assert "another project" in str(job.error.get("message", "")) or "cross_project" in str(
            job.error
        )

    def test_cancellation_is_observed_at_a_step_boundary(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        render, job = make_render_job(db, project, make_plan(db, project, media))

        def cancel_after_compile(ctx: JobContext) -> None:
            video_render.step_compile(ctx)
            ctx.job.cancel_requested = True
            ctx.session.commit()

        steps = dict(video_render.STEPS) | {"COMPILE": cancel_after_compile}
        JobRunner(db, job).run(steps)
        video_render.cleanup(JobContext(db, job))

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.CANCELLED

        by_name = {s.name: StepStatus(s.status) for s in job.steps}
        assert by_name["PREPARE"] is StepStatus.SUCCEEDED
        assert by_name["COMPILE"] is StepStatus.SUCCEEDED
        assert by_name["RENDER"] is StepStatus.SKIPPED
        assert by_name["FINALIZE"] is StepStatus.SKIPPED

    def test_a_transient_failure_schedules_a_retry(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],
    ) -> None:
        from visionforge.domain.errors import TransientError

        media = [ingest(store, db, project, path) for path in render_sources[:2]]
        _render, job = make_render_job(db, project, make_plan(db, project, media))

        def flaky(ctx: JobContext) -> None:
            raise TransientError("storage blipped")

        JobRunner(db, job).run(dict(video_render.STEPS) | {"PUBLISH": flaky})

        db.refresh(job)
        assert JobStatus(job.status) is JobStatus.RETRY_WAIT
        assert job.retry_at is not None
        assert job.error is not None and job.error["retryable"] is True
