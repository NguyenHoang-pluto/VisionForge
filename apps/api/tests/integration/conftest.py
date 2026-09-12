"""Fixtures for tests that need real infrastructure.

These tests talk to the Docker Compose stack. They are marked ``integration`` and
excluded from the default run so CI stays green on a machine with no Docker.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from visionforge.core.config import get_settings
from visionforge.domain.ids import ProjectId
from visionforge.infra.db.models import Project, User
from visionforge.infra.db.sync_session import get_sync_sessionmaker
from visionforge.infra.storage import S3ObjectStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        pytest.skip("ffmpeg not on PATH")
    return binary


@pytest.fixture(scope="session")
def media_fixtures() -> dict[str, Path]:
    """Generate small synthetic media with FFmpeg, once per session.

    Generated rather than committed: a handful of binary fixtures in git is a
    repository that grows forever, and FFmpeg can make exactly what each test
    needs deterministically.
    """
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    specs: dict[str, list[str]] = {
        # A 64x48 still image.
        "image.jpg": [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=1",
            "-frames:v",
            "1",
        ],
        # A different still, so dedupe tests have two distinct files.
        "image2.jpg": [
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=64x48:rate=1",
            "-frames:v",
            "1",
        ],
        # 1s of 1280x720 video: below the proxy threshold, so no proxy expected.
        "video_720.mp4": [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1280x720:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
        ],
        # 1s of 1920x1080: above the threshold, so a 720p proxy is expected.
        "video_1080.mp4": [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1920x1080:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
        ],
        # 1s of silence: audio gets neither a thumbnail nor a proxy.
        "audio.mp3": [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "1",
        ],
    }

    paths: dict[str, Path] = {}
    for name, args in specs.items():
        destination = FIXTURE_DIR / name
        if not destination.exists():
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", *args, str(destination)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        paths[name] = destination

    # Not media at all: a renamed executable header.
    corrupt = FIXTURE_DIR / "corrupt.jpg"
    corrupt.write_bytes(b"MZ\x90\x00" + b"\x00" * 512)
    paths["corrupt.jpg"] = corrupt

    return paths


#: Domain tables, ordered so that FK cascades never block a truncate.
#: ``users`` last, because everything hangs off it.
_DOMAIN_TABLES = (
    "media_analysis",
    "media_derivatives",
    "job_steps",
    "jobs",
    "media_assets",
    "events",
    "projects",
    "users",
)


@pytest.fixture(scope="session", autouse=True)
def _clean_database() -> None:
    """Truncate domain tables once, before any integration test runs.

    These tests share a database with the acceptance scripts and with manual
    dev use. A run against a database carrying data from several previous
    acceptance runs failed 11 of 74 tests once, and passed on a fresh schema --
    the kind of failure that is worse than a bug because it looks like
    flakiness. Starting from a known-empty state removes the variable.

    Session-scoped rather than per-test: the fixtures below already clean up
    after themselves, and truncating between every test would triple the
    suite's runtime for no additional isolation.
    """
    with get_sync_sessionmaker()() as session:
        session.execute(
            text(f"TRUNCATE TABLE {', '.join(_DOMAIN_TABLES)} RESTART IDENTITY CASCADE")
        )
        session.commit()


@pytest.fixture
def db() -> Iterator[Session]:
    with get_sync_sessionmaker()() as session:
        try:
            yield session
        finally:
            # A test that failed mid-transaction leaves the session in an
            # aborted state, and every later statement on it -- including the
            # fixture teardowns below -- fails with InFailedSqlTransaction.
            # Rolling back here contains the damage to the test that caused it.
            session.rollback()


@pytest.fixture
def project(db: Session) -> Iterator[Project]:
    """A throwaway project owned by a throwaway user, cleaned up afterwards."""
    user = User(
        id=uuid.uuid4(),
        email=f"test-{uuid.uuid4().hex[:8]}@visionforge.test",
        display_name="Integration Test",
    )
    db.add(user)
    db.flush()

    row = Project(id=uuid.uuid4(), user_id=user.id, title="Integration Test Project")
    db.add(row)
    db.commit()

    yield row

    # Roll back first: if the test failed mid-transaction, this DELETE would
    # otherwise fail too and leak the project into the next test.
    db.rollback()
    # Cascades remove media, derivatives, analyses, jobs, steps and events.
    db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
    db.commit()


@pytest.fixture
def project_id(project: Project) -> ProjectId:
    return ProjectId(project.id)


@pytest.fixture(scope="session")
def store() -> S3ObjectStore:
    return S3ObjectStore()


@pytest.fixture(autouse=True)
def _reset_sse_exit_event() -> None:
    """Work around sse-starlette caching its shutdown Event across event loops.

    ``AppStatus.should_exit_event`` is module-level and bound to the loop that
    created it. Each ``TestClient`` runs on a fresh loop, so the second SSE test
    in a session would otherwise fail with "bound to a different event loop".
    Purely a test-harness concern: in production there is one loop for the life
    of the process.
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest.fixture(autouse=True)
def _settings_are_local() -> None:
    """Fail loudly rather than run destructive tests against a real environment."""
    settings = get_settings()
    assert "localhost" in settings.database_url or "127.0.0.1" in settings.database_url
