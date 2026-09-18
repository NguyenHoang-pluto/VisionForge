"""Templates at the API boundary (Phase 12).

What is pinned here, by test rather than by reading:

- the library is listed for everyone, a user's own templates only for them;
- a template is measured only from a video in a project the caller owns, and
  only once that video has been analysed for cuts;
- another user's template is a 404, the same answer an unknown id gets.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_reference_route import PROJECT_ID, FakeProject
from visionforge.api.dependencies import (
    DEV_USER_ID,
    get_template_service,
    require_project,
)
from visionforge.api.main import create_app
from visionforge.application.template_service import TemplateService
from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.errors import NotFoundError
from visionforge.domain.template_library import LIBRARY

VIDEO_ID = uuid.uuid4()
IMAGE_ID = uuid.uuid4()
SINGLE_TAKE_ID = uuid.uuid4()
UNANALYSED_ID = uuid.uuid4()
SOMEONE_ELSE = uuid.uuid4()


def scenes(*lengths: int) -> dict[str, Any]:
    out, start = [], 0
    for index, length in enumerate(lengths):
        out.append({"scene_id": index, "start_ms": start, "end_ms": start + length})
        start += length
    return {"scenes": out}


class Asset:
    def __init__(self, media_id: uuid.UUID, kind: str = "video") -> None:
        self.id = media_id
        self.kind = kind
        self.status = "ready"
        self.width = 1920
        self.height = 1080


class MediaRepo:
    def __init__(self) -> None:
        self.assets = {
            VIDEO_ID: Asset(VIDEO_ID),
            IMAGE_ID: Asset(IMAGE_ID, kind="image"),
            SINGLE_TAKE_ID: Asset(SINGLE_TAKE_ID),
            UNANALYSED_ID: Asset(UNANALYSED_ID),
        }
        self.payloads: dict[uuid.UUID, dict[AnalyzerName, dict[str, Any]]] = {
            VIDEO_ID: {AnalyzerName.SCENES: scenes(1_000, 2_000, 1_500, 1_000)},
            SINGLE_TAKE_ID: {AnalyzerName.SCENES: scenes(20_000)},
        }

    async def get_in_project(self, media_id: Any, project_id: Any) -> Asset | None:
        return self.assets.get(uuid.UUID(str(media_id))) if project_id == PROJECT_ID else None

    async def analysis_payloads(self, media_id: Any, analyzers: Any) -> dict[Any, Any]:
        found = self.payloads.get(uuid.UUID(str(media_id)), {})
        return {analyzer: found[analyzer] for analyzer in analyzers if analyzer in found}


class Row:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        self.created_at = datetime.now(UTC)


class TemplateRepo:
    """In memory, and scoped by owner exactly as the real one is."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, Row] = {}

    async def create(self, *, template_id: uuid.UUID, user_id: Any, **values: Any) -> Row:
        row = Row(id=template_id, user_id=user_id, **values)
        self.rows[template_id] = row
        return row

    async def get_owned(self, template_id: uuid.UUID, user_id: Any) -> Row | None:
        row = self.rows.get(template_id)
        return row if row is not None and row.user_id == user_id else None

    async def list_for_user(self, user_id: Any, *, limit: int = 100) -> list[Row]:
        return [row for row in self.rows.values() if row.user_id == user_id]

    async def rename(self, row: Row, name: str, payload: dict[str, Any]) -> None:
        row.name, row.payload = name, payload

    async def delete(self, row: Row) -> None:
        self.rows.pop(row.id)


@pytest.fixture
def repo() -> TemplateRepo:
    return TemplateRepo()


@pytest.fixture
def client(repo: TemplateRepo) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_template_service] = lambda: TemplateService(
        repo,  # type: ignore[arg-type]
        MediaRepo(),  # type: ignore[arg-type]
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def measure(client: TestClient, media_id: uuid.UUID, name: str = "My cut") -> Any:
    return client.post(
        f"/api/projects/{PROJECT_ID}/templates", json={"media_id": str(media_id), "name": name}
    )


class TestListing:
    def test_the_library_is_listed(self, client: TestClient) -> None:
        body = client.get("/api/templates").json()
        assert [item["id"] for item in body["library"]] == [t.id for t in LIBRARY]
        assert body["mine"] == []

    def test_a_measured_template_appears_under_mine(self, client: TestClient) -> None:
        measure(client, VIDEO_ID)
        mine = client.get("/api/templates").json()["mine"]
        assert len(mine) == 1 and mine[0]["name"] == "My cut"


class TestMeasuring:
    def test_an_analysed_video_becomes_a_template_of_its_shots(self, client: TestClient) -> None:
        response = measure(client, VIDEO_ID)
        assert response.status_code == 201
        body = response.json()
        assert [slot["duration_ms"] for slot in body["slots"]] == [1_000, 2_000, 1_500, 1_000]
        assert body["source"] == "user"
        assert body["extraction"]["transitions"] == "cuts_only"

    def test_a_photo_cannot_be_measured(self, client: TestClient) -> None:
        assert measure(client, IMAGE_ID).status_code == 422

    def test_a_video_without_cuts_is_refused_with_the_reason(self, client: TestClient) -> None:
        response = measure(client, SINGLE_TAKE_ID)
        assert response.status_code == 422
        assert "no cuts" in response.text

    def test_an_unanalysed_video_is_refused(self, client: TestClient) -> None:
        assert measure(client, UNANALYSED_ID).status_code == 422

    def test_media_outside_the_project_is_not_found(self, client: TestClient) -> None:
        assert measure(client, uuid.uuid4()).status_code == 404

    def test_a_blank_name_is_refused(self, client: TestClient) -> None:
        assert measure(client, VIDEO_ID, name="   ").status_code == 422


class TestOwnership:
    def test_rename_and_delete_are_the_owners(self, client: TestClient) -> None:
        template_id = measure(client, VIDEO_ID).json()["id"]
        renamed = client.patch(f"/api/templates/{template_id}", json={"name": "Renamed"})
        assert renamed.status_code == 200 and renamed.json()["name"] == "Renamed"
        assert client.delete(f"/api/templates/{template_id}").status_code == 204
        assert client.get("/api/templates").json()["mine"] == []

    def test_another_users_template_is_not_found(
        self, client: TestClient, repo: TemplateRepo
    ) -> None:
        template_id = measure(client, VIDEO_ID).json()["id"]
        repo.rows[uuid.UUID(template_id)].user_id = SOMEONE_ELSE
        assert client.delete(f"/api/templates/{template_id}").status_code == 404
        assert client.get("/api/templates").json()["mine"] == []


class TestResolving:
    @pytest.mark.asyncio
    async def test_a_library_name_resolves(self) -> None:
        service = TemplateService(TemplateRepo(), MediaRepo())  # type: ignore[arg-type]
        assert (await service.resolve("travel", DEV_USER_ID)).id == "travel"

    @pytest.mark.asyncio
    async def test_an_unknown_name_is_not_found(self) -> None:
        service = TemplateService(TemplateRepo(), MediaRepo())  # type: ignore[arg-type]
        with pytest.raises(NotFoundError):
            await service.resolve("not-a-template", DEV_USER_ID)
