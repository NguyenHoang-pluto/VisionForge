"""Photos and videos in one render, against real Postgres, MinIO and FFmpeg.

Phase 12's render claims, checked on actual output rather than on argv:

- a plan of photos and videos renders, with the photos held and moving;
- the clips' own audio survives a photo between them, which has none;
- an output larger than the 720p proxy is rendered from the original.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from tests.integration.test_render_lane import (  # noqa: F401 -- render_sources is a fixture
    ingest,
    make_render_job,
    render_sources,
    run_render,
)
from visionforge.domain.editplan import (
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    OutputSpec,
    Segment,
    TransitionKind,
)
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.jobs import JobStatus
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.domain.storage import derivative_key, original_key
from visionforge.infra.db.models import EditPlanRow, MediaAsset, MediaDerivative, Project
from visionforge.infra.ffmpeg import probe_media
from visionforge.infra.storage import S3ObjectStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def photo_file(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """A 3:2 photo, larger than the output, as a phone would take one."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    destination = tmp_path_factory.mktemp("photos") / "photo.jpg"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1800x1200",
         "-frames:v", "1", str(destination)],
        check=True,
        capture_output=True,
        timeout=120,
    )  # fmt: skip
    return destination


@pytest.fixture(scope="session")
def voiced_clip(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """A 3-second 720p clip *with* an audio track, which the lavfi sources lack."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    destination = tmp_path_factory.mktemp("voiced") / "voiced.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(destination)],
        check=True,
        capture_output=True,
        timeout=120,
    )  # fmt: skip
    return destination


def ingest_photo(store: S3ObjectStore, db: Session, project: Project, path: Path) -> MediaAsset:
    media_id = uuid.uuid4()
    key = original_key(ProjectId(project.id), MediaId(media_id), ".jpg")
    store.upload_file(str(path), key)
    media = MediaAsset(
        id=media_id,
        project_id=project.id,
        original_filename=path.name,
        storage_key=key,
        kind=MediaKind.IMAGE,
        status=MediaStatus.READY,
        sha256=uuid.uuid4().hex * 2,
        # What ingest records for a photo: a probed single frame, not a length.
        duration_ms=40,
        width=1800,
        height=1200,
    )
    db.add(media)
    db.commit()
    return media


def store_plan(db: Session, project: Project, plan: EditPlan) -> EditPlanRow:
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


def output(**overrides: object) -> OutputSpec:
    values: dict[str, object] = {
        "aspect_ratio": AspectRatio.LANDSCAPE_16_9,
        "width": 1280,
        "height": 720,
        "fps": 25,
        "fit": FitMode.COVER,
        "audio": AudioMode.NONE,
    }
    values.update(overrides)
    return OutputSpec(**values)  # type: ignore[arg-type]


class TestStillsRender:
    def test_photos_and_a_video_render_together_with_source_audio(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        photo_file: Path,
        voiced_clip: Path,
        tmp_path: Path,
    ) -> None:
        photo = ingest_photo(store, db, project, photo_file)
        clip = ingest(store, db, project, voiced_clip)
        plan = EditPlan(
            project_id=ProjectId(project.id),
            segments=(
                Segment(
                    MediaId(photo.id), 0, 0, 2_000, effects=(Effect(EffectKind.ZOOM_IN, 0.12),)
                ),
                Segment(MediaId(clip.id), 1, 500, 2_000, TransitionKind.CROSSFADE, 400),
                Segment(
                    MediaId(photo.id),
                    2,
                    0,
                    2_500,
                    TransitionKind.CROSSFADE,
                    400,
                    (Effect(EffectKind.PAN_LEFT, 0.12),),
                ),
            ),
            output=output(audio=AudioMode.SOURCE),
        )
        render, job = make_render_job(db, project, store_plan(db, project, plan))

        run_render(db, job)

        db.refresh(job)
        db.refresh(render)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error
        local = tmp_path / "mixed.mp4"
        store.download_to(render.storage_key, str(local))
        metadata = probe_media(str(local))
        assert (metadata.width, metadata.height) == (1280, 720)
        # 2.0 + 1.5 + 2.5 s, less two 0.4 s dissolves.
        assert metadata.duration_ms is not None
        assert 5_000 <= metadata.duration_ms <= 5_400
        assert metadata.channels, "the clip audio was lost across the photos"

    def test_a_1080p_output_is_rendered_from_the_original_not_the_proxy(
        self,
        db: Session,
        project: Project,
        store: S3ObjectStore,
        render_sources: list[Path],  # noqa: F811
        tmp_path: Path,
    ) -> None:
        """The proxy row points at nothing. Reading it would fail the render."""
        clip = ingest(store, db, project, render_sources[1])
        db.add(
            MediaDerivative(
                media_id=clip.id,
                kind=DerivativeKind.PROXY,
                variant="720p",
                storage_key=derivative_key(
                    ProjectId(project.id), MediaId(clip.id), DerivativeKind.PROXY, "720p", ".mp4"
                ),
                width=1280,
                height=720,
            )
        )
        db.commit()
        plan = EditPlan(
            project_id=ProjectId(project.id),
            segments=(Segment(MediaId(clip.id), 0, 500, 2_500),),
            output=output(width=1920, height=1080),
        )
        render, job = make_render_job(db, project, store_plan(db, project, plan))

        run_render(db, job)

        db.refresh(job)
        db.refresh(render)
        assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error
        local = tmp_path / "full.mp4"
        store.download_to(render.storage_key, str(local))
        assert (probe_media(str(local)).width, probe_media(str(local)).height) == (1920, 1080)
