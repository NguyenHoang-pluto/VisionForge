"""Reference style against real Postgres, MinIO and FFmpeg.

The unit tests prove the arithmetic on synthetic payloads. These prove the part
that cannot be faked: that the analyzers, run for real on a real file, produce
payloads the profile can actually read, and that the database enforces what the
migration claims.

The reference here is built the way a real one behaves -- a concatenation of
visually unrelated segments, so the scene detector has genuine hard cuts to
find. A crossfade or a single take would measure nothing, which is a fact about
the footage and not something the test should paper over.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from visionforge.domain.analysis import AnalysisStatus, AnalyzerName
from visionforge.domain.ids import MediaId
from visionforge.domain.jobs import JOB_STEP_PLANS, JobStatus, JobType
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.policy import StyleStrength, blend
from visionforge.domain.reference import profile_from
from visionforge.domain.storage import original_key
from visionforge.domain.style import profile_for
from visionforge.infra.db.models import Job, JobStep, MediaAnalysis, MediaAsset, Project
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers import media_analysis
from visionforge.workers.runtime import JobContext, JobRunner

pytestmark = pytest.mark.integration

#: The reference cuts every 600 ms. Small enough to build quickly, long enough
#: that the detector has something to find at 25 fps.
SHOT_MS = 600
SHOTS = 10


@pytest.fixture(scope="session")
def reference_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A video with known, hard cuts at a known rate."""
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:  # pragma: no cover - environment guard
        pytest.skip("ffmpeg not on PATH")

    directory = tmp_path_factory.mktemp("reference")
    sources = [
        "smptebars=size=320x240:rate=25",
        "testsrc=size=320x240:rate=25",
        "rgbtestsrc=size=320x240:rate=25",
        "color=c=navy:size=320x240:rate=25",
    ]

    parts: list[Path] = []
    for index in range(SHOTS):
        part = directory / f"part_{index:02d}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"{sources[index % len(sources)]}:duration={SHOT_MS / 1000}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(part),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        parts.append(part)

    listing = directory / "parts.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8")

    output = directory / "reference.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return output


def _ingest(store: S3ObjectStore, db: Session, project: Project, path: Path) -> MediaAsset:
    """A READY video row with its bytes in storage, as ingest would leave it."""
    media_id = uuid.uuid4()
    key = original_key(project.id, media_id, path.suffix)  # type: ignore[arg-type]
    store.upload_file(str(path), key)

    media = MediaAsset(
        id=media_id,
        project_id=project.id,
        original_filename=path.name,
        storage_key=key,
        kind=MediaKind.VIDEO,
        status=MediaStatus.READY,
        sha256=uuid.uuid4().hex * 2,
        duration_ms=SHOTS * SHOT_MS,
        width=320,
        height=240,
    )
    db.add(media)
    db.commit()
    return media


def _analyse(db: Session, media: MediaAsset) -> None:
    """Run the real CPU analysis lane over this asset."""
    job = Job(
        id=uuid.uuid4(),
        project_id=media.project_id,
        media_id=media.id,
        type=JobType.MEDIA_ANALYZE_CPU,
        status=JobStatus.QUEUED,
        params={"media_id": str(media.id)},
    )
    db.add(job)
    db.flush()
    for seq, name in enumerate(JOB_STEP_PLANS[JobType.MEDIA_ANALYZE_CPU]):
        db.add(JobStep(job_id=job.id, seq=seq, name=name))
    db.commit()

    JobRunner(db, job).run(media_analysis.CPU_STEPS)  # type: ignore[arg-type]
    media_analysis.cleanup(JobContext(db, job))
    db.refresh(job)
    assert JobStatus(job.status) is JobStatus.SUCCEEDED, job.error


def _payloads(db: Session, media: MediaAsset) -> dict[AnalyzerName, dict]:
    rows = (
        db.execute(
            select(MediaAnalysis).where(
                MediaAnalysis.media_id == media.id,
                MediaAnalysis.status == AnalysisStatus.OK,
            )
        )
        .scalars()
        .all()
    )
    return {AnalyzerName(str(row.analyzer)): row.payload for row in rows}


class TestTheAnalyzersProduceWhatTheProfileReads:
    """The contract between the two halves, which only a real run can check."""

    def test_the_cpu_lane_writes_a_dynamics_row(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)

        payloads = _payloads(db, media)
        assert AnalyzerName.DYNAMICS in payloads

        dynamics = payloads[AnalyzerName.DYNAMICS]
        assert isinstance(dynamics["motion"], int | float)
        assert isinstance(dynamics["saturation"], int | float)
        assert 0.0 <= dynamics["motion"] <= 1.0
        assert 0.0 <= dynamics["saturation"] <= 1.0
        assert 0.0 < dynamics["motion_confidence"] <= 1.0

    def test_a_video_with_no_audio_is_unsupported_not_a_failure(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        """The regression this file caught.

        Widening the beat analyzer to video made `-map a:0` run against files
        that have no audio stream. FFmpeg exits non-zero, and the analyzer was
        raising -- which failed the entire CPU job, taking quality, scenes,
        phash and dynamics down with it for every silent video in the product.

        The right answer is an `unsupported` row: nothing to measure, stored
        once, never retried, and the rest of the lane unaffected.
        """
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)

        row = db.execute(
            select(MediaAnalysis).where(
                MediaAnalysis.media_id == media.id,
                MediaAnalysis.analyzer == AnalyzerName.BEATS,
            )
        ).scalar_one()
        assert AnalysisStatus(row.status) is AnalysisStatus.UNSUPPORTED
        assert "no audio stream" in row.payload["reason"]

        # And the analyzers that *do* apply still produced their results.
        payloads = _payloads(db, media)
        assert AnalyzerName.DYNAMICS in payloads
        assert AnalyzerName.SCENES in payloads

    def test_a_profile_built_from_the_real_rows_measures_the_cutting(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        """The end of the measuring chain: real file, real analyzers, real
        profile, and a shot length that matches how the clip was built."""
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)
        payloads = _payloads(db, media)

        profile = profile_from(
            media_id=MediaId(media.id),
            duration_ms=media.duration_ms or 0,
            scenes=payloads.get(AnalyzerName.SCENES),
            quality=payloads.get(AnalyzerName.QUALITY),
            dynamics=payloads.get(AnalyzerName.DYNAMICS),
            beats=payloads.get(AnalyzerName.BEATS),
        )

        assert profile.is_usable
        assert profile.shot_ms is not None
        # Scene boundaries can land a frame either side of the true cut.
        assert abs(profile.shot_ms.value - SHOT_MS) <= 200
        assert profile.pacing is not None
        assert profile.motion is not None
        assert profile.saturation is not None
        assert profile.luminance is not None
        assert 0.0 < profile.confidence <= 1.0

    def test_the_profile_is_deterministic_over_the_same_rows(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)
        payloads = _payloads(db, media)

        def build() -> object:
            return profile_from(
                media_id=MediaId(media.id),
                duration_ms=media.duration_ms or 0,
                scenes=payloads.get(AnalyzerName.SCENES),
                quality=payloads.get(AnalyzerName.QUALITY),
                dynamics=payloads.get(AnalyzerName.DYNAMICS),
                beats=payloads.get(AnalyzerName.BEATS),
            ).as_payload()

        assert build() == build()

    def test_analysing_twice_measures_the_same_thing(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        """Determinism at the analyzer level, not just the arithmetic level.

        The sample points are a function of duration alone, so a second pass
        over the same bytes has to read the same frames and produce the same
        numbers -- which is what makes a profile safe to derive on read.
        """
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)
        first = dict(_payloads(db, media)[AnalyzerName.DYNAMICS])

        _analyse(db, media)
        second = dict(_payloads(db, media)[AnalyzerName.DYNAMICS])

        assert first["motion"] == second["motion"]
        assert first["saturation"] == second["saturation"]


class TestThePointerIsEnforcedByTheDatabase:
    def test_a_project_can_reference_its_own_media(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        media = _ingest(store, db, project, reference_clip)
        project.reference_media_id = media.id
        db.commit()
        db.refresh(project)
        assert project.reference_media_id == media.id

    def test_deleting_the_media_clears_the_pointer(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        """``ON DELETE SET NULL``, verified against the real constraint.

        A dangling reference id would be a row every reader downstream has to
        defend against, and the defence would eventually be forgotten.
        """
        media = _ingest(store, db, project, reference_clip)
        project.reference_media_id = media.id
        db.commit()

        db.execute(text("DELETE FROM media_assets WHERE id = :id"), {"id": media.id})
        db.commit()
        db.refresh(project)

        assert project.reference_media_id is None

    def test_the_project_relationship_still_means_its_own_media(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        """The ambiguity the second foreign key introduced, checked against a
        live query rather than against the mapper's own opinion."""
        media = _ingest(store, db, project, reference_clip)
        project.reference_media_id = media.id
        db.commit()
        db.refresh(project)

        assert [asset.id for asset in project.media] == [media.id]


class TestStyleStrengthOverRealMeasurements:
    def test_the_dial_moves_pacing_and_zero_does_not(
        self, db: Session, project: Project, store: S3ObjectStore, reference_clip: Path
    ) -> None:
        media = _ingest(store, db, project, reference_clip)
        _analyse(db, media)
        payloads = _payloads(db, media)

        profile = profile_from(
            media_id=MediaId(media.id),
            duration_ms=media.duration_ms or 0,
            scenes=payloads.get(AnalyzerName.SCENES),
            quality=payloads.get(AnalyzerName.QUALITY),
            dynamics=payloads.get(AnalyzerName.DYNAMICS),
            beats=payloads.get(AnalyzerName.BEATS),
        )
        preset = profile_for(None)

        off = blend(preset, profile, StyleStrength.ZERO)
        half = blend(preset, profile, StyleStrength.HALF)
        full = blend(preset, profile, StyleStrength.FULL)

        assert off.target_clip_ms == preset.target_clip_ms
        assert off.weights == preset.weights
        # The reference cuts far faster than any preset's default.
        assert full.target_clip_ms < half.target_clip_ms < off.target_clip_ms
        assert full.weights.affinity > 0
