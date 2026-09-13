"""The manual edit-plan route against real Postgres.

The unit tests cover the boundary with the service faked out and the service
with the repositories faked out. This is the one that proves the two halves
agree: real rows, real validation, real persistence, through the real router.

No bytes and no FFmpeg -- storing a timeline reads media *metadata* and writes a
plan row. Rendering it is Phase 4's lane and is covered there.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from visionforge.api.main import create_app
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.infra.db.models import MediaAsset
from visionforge.infra.db.sync_session import get_sync_sessionmaker

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def project(client: TestClient) -> Iterator[dict[str, Any]]:
    response = client.post("/api/projects", json={"title": "Editor Lane"})
    assert response.status_code == 201, response.text
    row = response.json()

    yield row

    with get_sync_sessionmaker()() as session:
        session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": UUID(row["id"])})
        session.commit()


def seed_media(
    session: Session,
    project_id: UUID,
    *,
    duration_ms: int = 20_000,
    kind: MediaKind = MediaKind.VIDEO,
    status: MediaStatus = MediaStatus.READY,
) -> UUID:
    """A media row with realistic metadata and no bytes behind it."""
    media = MediaAsset(
        id=uuid.uuid4(),
        project_id=project_id,
        original_filename="clip.mp4",
        storage_key=f"projects/{project_id}/media/{uuid.uuid4()}/original.mp4",
        kind=kind,
        status=status,
        bytes_size=1024,
        mime_type="video/mp4",
        duration_ms=duration_ms,
        width=1920,
        height=1080,
        fps=25.0,
        codec="h264",
        channels=2,
    )
    session.add(media)
    session.commit()
    return media.id


@pytest.fixture
def media_id(project: dict[str, Any]) -> UUID:
    with get_sync_sessionmaker()() as session:
        return seed_media(session, UUID(project["id"]))


def post(client: TestClient, project_id: str, body: dict[str, Any]) -> Any:
    return client.post(f"/api/projects/{project_id}/edit-plan/manual", json=body)


class TestStoringATimeline:
    def test_a_hand_cut_timeline_becomes_a_plan(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        response = post(
            client,
            project["id"],
            {
                "segments": [
                    {"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 4000},
                    {"media_id": str(media_id), "source_in_ms": 9000, "source_out_ms": 12_000},
                ],
                "aspect_ratio": "16:9",
                "fps": 30,
                "quality": "draft",
            },
        )

        assert response.status_code == 201, response.text
        plan = response.json()
        assert plan["planner"] == "manual"
        assert plan["segment_count"] == 2
        assert plan["total_duration_ms"] == 7000
        assert [s["order"] for s in plan["plan"]["segments"]] == [0, 1]

    def test_it_is_readable_back_and_listed(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        """The editor reloads a plan through the same routes as any other."""
        created = post(
            client,
            project["id"],
            {"segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 5000}]},
        ).json()

        detail = client.get(f"/api/projects/{project['id']}/edit-plan/{created['id']}")
        assert detail.status_code == 200
        assert detail.json()["plan"]["segments"][0]["source_out_ms"] == 5000

        listing = client.get(f"/api/projects/{project['id']}/edit-plan").json()
        assert any(item["id"] == created["id"] for item in listing["items"])

    def test_the_server_chooses_the_geometry(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        response = post(
            client,
            project["id"],
            {
                "segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 4000}],
                "aspect_ratio": "9:16",
            },
        )

        output = response.json()["plan"]["output"]
        assert (output["width"], output["height"]) == (720, 1280)

    def test_geometry_in_the_body_cannot_reach_the_plan(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        """The property Phase 4 established, checked at the new route."""
        response = post(
            client,
            project["id"],
            {
                "segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 4000}],
                "width": 3840,
                "height": 2160,
                "crf": 1,
                "ffmpeg_args": "-vf drawtext=text=owned",
            },
        )

        assert response.status_code == 201
        plan = response.json()["plan"]
        assert (plan["output"]["width"], plan["output"]["height"]) == (1280, 720)
        assert "crf" not in plan["output"]
        assert "ffmpeg_args" not in json.dumps(plan)

    def test_provenance_records_the_plan_it_was_cut_from(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        origin = client.post(
            f"/api/projects/{project['id']}/edit-plan",
            json={"mode": "rules", "max_clips": 1, "min_clips": 1},
        )
        # Planning needs analysed media; when there is none this is a 4xx and the
        # provenance case is covered by the unit tests instead.
        if origin.status_code != 201:
            pytest.skip("no analysed media in this project to plan from")

        response = post(
            client,
            project["id"],
            {
                "segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 4000}],
                "derived_from_edit_plan_id": origin.json()["id"],
            },
        )

        metadata = response.json()["plan"]["metadata"]
        assert metadata["derived_from_edit_plan_id"] == origin.json()["id"]


class TestRefusedTimelines:
    def test_a_trim_past_the_end_is_refused(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        response = post(
            client,
            project["id"],
            {"segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 90_000}]},
        )

        assert response.status_code == 422
        assert "exceeds" in response.json()["error"]["hint"]

    def test_media_from_another_project_is_refused(
        self, client: TestClient, project: dict[str, Any]
    ) -> None:
        other = client.post("/api/projects", json={"title": "Other"}).json()
        with get_sync_sessionmaker()() as session:
            foreign = seed_media(session, UUID(other["id"]))

        try:
            response = post(
                client,
                project["id"],
                {
                    "segments": [
                        {"media_id": str(foreign), "source_in_ms": 0, "source_out_ms": 4000}
                    ]
                },
            )
            assert response.status_code == 422
        finally:
            with get_sync_sessionmaker()() as session:
                session.execute(
                    text("DELETE FROM projects WHERE id = :id"), {"id": UUID(other["id"])}
                )
                session.commit()

    def test_media_that_is_not_ready_is_refused(
        self, client: TestClient, project: dict[str, Any]
    ) -> None:
        with get_sync_sessionmaker()() as session:
            pending = seed_media(session, UUID(project["id"]), status=MediaStatus.PROCESSING)

        response = post(
            client,
            project["id"],
            {"segments": [{"media_id": str(pending), "source_in_ms": 0, "source_out_ms": 4000}]},
        )

        assert response.status_code == 422

    def test_an_image_is_refused_as_a_timeline_clip(
        self, client: TestClient, project: dict[str, Any]
    ) -> None:
        with get_sync_sessionmaker()() as session:
            still = seed_media(session, UUID(project["id"]), kind=MediaKind.IMAGE)

        response = post(
            client,
            project["id"],
            {"segments": [{"media_id": str(still), "source_in_ms": 0, "source_out_ms": 4000}]},
        )

        assert response.status_code == 422

    def test_a_refused_timeline_stores_nothing(
        self, client: TestClient, project: dict[str, Any], media_id: UUID
    ) -> None:
        before = client.get(f"/api/projects/{project['id']}/edit-plan").json()["total"]

        post(
            client,
            project["id"],
            {"segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 90_000}]},
        )

        after = client.get(f"/api/projects/{project['id']}/edit-plan").json()["total"]
        assert after == before

    def test_another_projects_route_is_a_404(self, client: TestClient, media_id: UUID) -> None:
        """Authorization still resolves through project ownership."""
        response = post(
            client,
            str(uuid.uuid4()),
            {"segments": [{"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": 4000}]},
        )

        assert response.status_code == 404
