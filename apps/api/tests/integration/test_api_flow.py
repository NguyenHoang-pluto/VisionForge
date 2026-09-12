"""HTTP API against real infrastructure: projects, presigned upload, jobs, SSE.

Uses ``TestClient``, so the real routers, dependencies, repositories and
authorization checks all run. Only Celery delivery is bypassed -- the publisher
is replaced so tests do not depend on a worker being up.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from visionforge.api.dependencies import get_job_dispatcher
from visionforge.api.main import create_app
from visionforge.application.job_dispatch import JobDispatcher
from visionforge.domain.jobs import JobStatus, JobType
from visionforge.infra.db import get_sessionmaker
from visionforge.infra.db.models import Job, MediaAsset
from visionforge.infra.db.repositories import JobRepository
from visionforge.infra.db.sync_session import get_sync_sessionmaker
from visionforge.workers import media_ingest
from visionforge.workers.runtime import JobContext, JobRunner

pytestmark = pytest.mark.integration


class NoopPublisher:
    """Records dispatches without needing a broker or a worker."""

    def __init__(self) -> None:
        self.published: list[UUID] = []

    def publish(self, job_type: JobType, job_id: UUID) -> str | None:
        self.published.append(job_id)
        return f"test-{job_id}"


@pytest.fixture
def publisher() -> NoopPublisher:
    return NoopPublisher()


@pytest.fixture
def client(publisher: NoopPublisher) -> Iterator[TestClient]:
    app = create_app()

    async def _dispatcher() -> Any:
        async with get_sessionmaker()() as session:
            yield JobDispatcher(session, JobRepository(session), publisher)

    app.dependency_overrides[get_job_dispatcher] = _dispatcher
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def api_project(client: TestClient) -> Iterator[dict[str, Any]]:
    response = client.post("/api/projects", json={"title": "API Flow Test"})
    assert response.status_code == 201, response.text
    project = response.json()

    yield project

    with get_sync_sessionmaker()() as session:
        session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": UUID(project["id"])})
        session.commit()


def _upload_bytes(url: str, path: Path) -> None:
    """PUT straight to object storage, exactly as a browser would.

    That this works without the API in the path is the point: media bytes never
    transit FastAPI.
    """
    response = httpx.put(url, content=path.read_bytes(), timeout=30)
    response.raise_for_status()


class TestProjects:
    def test_create_and_fetch(self, client: TestClient) -> None:
        created = client.post(
            "/api/projects", json={"title": "My Trip", "description": "Summer 2026"}
        )
        assert created.status_code == 201
        body = created.json()
        assert body["title"] == "My Trip"
        assert body["media_count"] == 0

        fetched = client.get(f"/api/projects/{body['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == body["id"]

        # Phase 2 has no delete endpoint; clean up directly.
        with get_sync_sessionmaker()() as session:
            session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": UUID(body["id"])})
            session.commit()

    def test_unknown_project_is_404(self, client: TestClient) -> None:
        response = client.get(f"/api/projects/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_title_is_required(self, client: TestClient) -> None:
        assert client.post("/api/projects", json={"title": ""}).status_code == 422


class TestUploadFlow:
    def test_presigned_upload_round_trip(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
        publisher: NoopPublisher,
    ) -> None:
        project_id = api_project["id"]

        ticket = client.post(
            f"/api/projects/{project_id}/media/upload-url",
            json={"filename": "photo.jpg", "size_bytes": 1234},
        )
        assert ticket.status_code == 201, ticket.text
        body = ticket.json()
        assert body["object_key"].startswith(f"projects/{project_id}/media/")
        assert body["object_key"].endswith("/original.jpg")

        _upload_bytes(body["upload_url"], media_fixtures["image.jpg"])

        completed = client.post(f"/api/projects/{project_id}/media/{body['media_id']}/complete")
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "uploaded"
        assert completed.json()["job_id"] is not None
        assert len(publisher.published) == 1

    def test_complete_without_uploading_is_rejected(
        self, client: TestClient, api_project: dict[str, Any], publisher: NoopPublisher
    ) -> None:
        ticket = client.post(
            f"/api/projects/{api_project['id']}/media/upload-url",
            json={"filename": "ghost.jpg"},
        ).json()

        response = client.post(
            f"/api/projects/{api_project['id']}/media/{ticket['media_id']}/complete"
        )
        assert response.status_code == 422
        assert publisher.published == [], "no job for an upload that never happened"

    def test_unsupported_extension_is_rejected(
        self, client: TestClient, api_project: dict[str, Any]
    ) -> None:
        response = client.post(
            f"/api/projects/{api_project['id']}/media/upload-url",
            json={"filename": "payload.exe"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unsupported_media"

    def test_oversized_declaration_is_rejected(
        self, client: TestClient, api_project: dict[str, Any]
    ) -> None:
        response = client.post(
            f"/api/projects/{api_project['id']}/media/upload-url",
            json={"filename": "huge.mp4", "size_bytes": 10 * 1024**3},
        )
        assert response.status_code == 413

    def test_upload_url_for_another_project_is_404(self, client: TestClient) -> None:
        response = client.post(
            f"/api/projects/{uuid.uuid4()}/media/upload-url",
            json={"filename": "a.jpg"},
        )
        assert response.status_code == 404

    def test_completing_twice_does_not_create_a_second_job(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
        publisher: NoopPublisher,
    ) -> None:
        project_id = api_project["id"]
        ticket = client.post(
            f"/api/projects/{project_id}/media/upload-url", json={"filename": "a.jpg"}
        ).json()
        _upload_bytes(ticket["upload_url"], media_fixtures["image.jpg"])

        first = client.post(f"/api/projects/{project_id}/media/{ticket['media_id']}/complete")
        second = client.post(f"/api/projects/{project_id}/media/{ticket['media_id']}/complete")

        assert first.status_code == second.status_code == 200
        assert len(publisher.published) == 1


class TestMediaListing:
    def test_ready_media_exposes_metadata_and_derivative_urls(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        project_id = api_project["id"]
        ticket = client.post(
            f"/api/projects/{project_id}/media/upload-url", json={"filename": "v.mp4"}
        ).json()
        _upload_bytes(ticket["upload_url"], media_fixtures["video_1080.mp4"])
        job_id = client.post(
            f"/api/projects/{project_id}/media/{ticket['media_id']}/complete"
        ).json()["job_id"]

        _run_job_inline(UUID(job_id))

        media = client.get(f"/api/projects/{project_id}/media/{ticket['media_id']}").json()
        assert media["status"] == "ready"
        assert media["kind"] == "video"
        assert (media["width"], media["height"]) == (1920, 1080)
        assert media["has_thumbnail"] and media["has_proxy"]

        thumb = client.get(f"/api/projects/{project_id}/media/{ticket['media_id']}/thumbnail")
        proxy = client.get(f"/api/projects/{project_id}/media/{ticket['media_id']}/proxy")
        assert thumb.status_code == proxy.status_code == 200
        assert thumb.json()["url"].startswith("http")

        # The signed URL really serves the bytes.
        assert httpx.get(thumb.json()["url"], timeout=30).status_code == 200

    def test_listing_reports_totals(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        project_id = api_project["id"]
        for name in ("image.jpg", "image2.jpg"):
            ticket = client.post(
                f"/api/projects/{project_id}/media/upload-url",
                json={"filename": name},
            ).json()
            _upload_bytes(ticket["upload_url"], media_fixtures[name])
            client.post(f"/api/projects/{project_id}/media/{ticket['media_id']}/complete")

        listing = client.get(f"/api/projects/{project_id}/media").json()
        assert listing["total"] == 2
        assert len(listing["items"]) == 2

    def test_thumbnail_before_processing_is_404_with_a_hint(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        project_id = api_project["id"]
        ticket = client.post(
            f"/api/projects/{project_id}/media/upload-url", json={"filename": "a.jpg"}
        ).json()
        _upload_bytes(ticket["upload_url"], media_fixtures["image.jpg"])
        client.post(f"/api/projects/{project_id}/media/{ticket['media_id']}/complete")

        response = client.get(f"/api/projects/{project_id}/media/{ticket['media_id']}/thumbnail")
        assert response.status_code == 404
        assert "Processing" in (response.json()["error"]["hint"] or "")


class TestJobs:
    def test_job_reports_steps_before_the_worker_starts(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        """Progress is honest from the first poll: the step rows already exist."""
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "queued"
        assert job["progress"] == 0.0
        assert [s["name"] for s in job["steps"]] == [
            "VALIDATE",
            "METADATA",
            "HASH",
            "THUMBNAIL",
            "PROXY",
            "FINALIZE",
        ]
        assert all(s["status"] == "pending" for s in job["steps"])

    def test_job_progress_after_completion(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])
        _run_job_inline(UUID(job_id))

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "succeeded"
        assert job["progress"] == 1.0
        assert job["result"]["thumbnail"] is True

    def test_cancel_a_queued_job(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        """A job that has not started yet is cancelled outright, not merely flagged."""
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])

        response = client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "cancelled"
        assert body["cancel_requested"] is True

    def test_cancelled_job_is_not_processed_by_a_worker(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])
        client.post(f"/api/jobs/{job_id}/cancel")

        with get_sync_sessionmaker()() as session:
            job = session.get(Job, UUID(job_id))
            assert job is not None
            assert JobStatus(job.status) is JobStatus.CANCELLED

        media_id = client.get(f"/api/jobs/{job_id}").json()["media_id"]
        media = client.get(f"/api/projects/{api_project['id']}/media/{media_id}").json()
        assert media["status"] != "ready"

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/jobs/{uuid.uuid4()}").status_code == 404

    def test_unknown_job_type_is_rejected(
        self, client: TestClient, api_project: dict[str, Any]
    ) -> None:
        response = client.post(
            "/api/jobs",
            json={"project_id": api_project["id"], "type": "mine_bitcoin"},
        )
        assert response.status_code == 422
        assert "media_ingest" in (response.json()["error"]["hint"] or "")

    def test_idempotency_key_prevents_a_duplicate_job(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
        publisher: NoopPublisher,
    ) -> None:
        project_id = api_project["id"]
        ticket = client.post(
            f"/api/projects/{project_id}/media/upload-url", json={"filename": "a.jpg"}
        ).json()
        _upload_bytes(ticket["upload_url"], media_fixtures["image.jpg"])
        client.post(f"/api/projects/{project_id}/media/{ticket['media_id']}/complete")
        publisher.published.clear()

        payload = {
            "project_id": project_id,
            "type": "media_ingest",
            "media_id": ticket["media_id"],
        }
        first = client.post("/api/jobs", json=payload, headers={"Idempotency-Key": "k1"})
        second = client.post("/api/jobs", json=payload, headers={"Idempotency-Key": "k1"})

        assert first.json()["id"] == second.json()["id"]
        assert len(publisher.published) == 1


class TestSSE:
    def test_stream_opens_with_current_state_from_the_database(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        """A client learns where the job stands immediately, not on the next event."""
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])
        client.post(f"/api/jobs/{job_id}/cancel")  # make it terminal so the stream ends

        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            payload = _first_event(response)

        assert payload["job_id"] == job_id
        assert payload["status"] == "cancelled"
        assert payload["steps_total"] == 6

    def test_reconnect_replays_the_last_known_state(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        """The defining SSE requirement: a refresh mid-job is cheap and correct."""
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])
        _run_job_inline(UUID(job_id))

        seen = []
        for _ in range(2):  # two independent "page loads"
            with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
                seen.append(_first_event(response))

        assert seen[0] == seen[1]
        assert seen[0]["status"] == "succeeded"
        assert seen[0]["progress"] == 1.0

    def test_terminal_job_stream_closes_immediately(
        self,
        client: TestClient,
        api_project: dict[str, Any],
        media_fixtures: dict[str, Path],
    ) -> None:
        """A finished job must not hold a connection open for 30 minutes."""
        job_id = _queue_ingest(client, api_project, media_fixtures["image.jpg"])
        _run_job_inline(UUID(job_id))

        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            events = [line for line in response.iter_lines() if line.startswith("data:")]

        assert len(events) == 1

    def test_sse_for_an_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/jobs/{uuid.uuid4()}/events").status_code == 404


# ------------------------------------------------------------------- helpers
def _queue_ingest(client: TestClient, project: dict[str, Any], fixture: Path) -> str:
    ticket = client.post(
        f"/api/projects/{project['id']}/media/upload-url",
        json={"filename": fixture.name},
    ).json()
    _upload_bytes(ticket["upload_url"], fixture)
    completed = client.post(
        f"/api/projects/{project['id']}/media/{ticket['media_id']}/complete"
    ).json()
    return str(completed["job_id"])


def _run_job_inline(job_id: UUID) -> None:
    """Execute a queued job synchronously, standing in for a Celery worker."""
    with get_sync_sessionmaker()() as session:
        job = session.get(Job, job_id)
        assert job is not None
        JobRunner(session, job).run(media_ingest.STEPS)
        media_ingest.cleanup(JobContext(session, job))


def _first_event(response: Any) -> dict[str, Any]:
    for line in response.iter_lines():
        if line.startswith("data:"):
            return dict(json.loads(line[5:].strip()))
    raise AssertionError("no SSE data frame received")


def _media_row(session: Session, media_id: UUID) -> MediaAsset | None:
    return session.get(MediaAsset, media_id)
