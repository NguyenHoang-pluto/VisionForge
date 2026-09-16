"""The co-editor routes, with the service faked out.

What is under test is the *boundary*: which fields a change request may carry,
what the route does with a refusal, and that the operation vocabulary the UI is
handed is the one the parser actually implements. The service's own behaviour is
covered in ``test_coeditor`` and ``test_plan_patch``; its persistence in
``tests/integration/test_coedit_lane``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from visionforge.api.dependencies import (
    get_coedit_service,
    get_llm_provider,
    get_version_repo,
    require_project,
)
from visionforge.api.main import create_app
from visionforge.application.coedit_service import CoEditPreview, CoEditRejected
from visionforge.domain.coeditor import CoEditFailure, CoEditOutcome, CoEditSource
from visionforge.domain.editdelta import (
    MAX_OPERATIONS,
    ChangeMusicVolume,
    DeltaViolation,
    EditDelta,
    OperationKind,
)
from visionforge.domain.patch import DiffEntry, PlanDiff
from visionforge.domain.versions import VersionOrigin, VersionSummary

PROJECT_ID = uuid.uuid4()
PLAN_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


class FakeProject:
    id = PROJECT_ID
    title = "Co-edit"


def summary(*, version: int = 2, is_current: bool = True) -> VersionSummary:
    return VersionSummary(
        id=VERSION_ID,
        version=version,
        parent_id=None,
        edit_plan_id=PLAN_ID,
        origin=VersionOrigin.CO_EDIT,
        is_current=is_current,
        applied=("Music volume to 40%",),
        operation_count=1,
        source="rules",
        provider=None,
        model=None,
        latency_ms=None,
        request_digest="abc123",
        total_duration_ms=12_000,
        segment_count=3,
        created_at=datetime.now(UTC).isoformat(),
    )


class PlanRow:
    id = PLAN_ID
    project_id = PROJECT_ID
    planner = "co-editor"
    planner_version = "1"
    segment_count = 3
    total_duration_ms = 12_000
    created_at = datetime.now(UTC)
    plan: ClassVar[dict[str, Any]] = {"metadata": {}}
    selection: ClassVar[dict[str, Any]] = {}


class VersionRow:
    id = VERSION_ID
    version = 2
    parent_version_id = None
    edit_plan_id = PLAN_ID
    is_current = True


class FakeService:
    """Stands in for ``CoEditService`` and records what the route handed it."""

    def __init__(self, *, rejects: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._rejects = rejects

    async def preview(self, **kwargs: Any) -> CoEditPreview:
        self.calls.append(("preview", kwargs))
        return CoEditPreview(
            ok=True,
            outcome=CoEditOutcome(
                ok=True,
                delta=EditDelta(operations=(ChangeMusicVolume(value=0.4),), source="rules"),
                source=CoEditSource.RULES,
            ),
            patch=_patch_outcome(),
            base_version_id=VERSION_ID,
            base_version=1,
        )

    async def apply(self, **kwargs: Any) -> Any:
        self.calls.append(("apply", kwargs))
        if self._rejects is not None:
            raise self._rejects

        class Applied:
            version = VersionRow()
            plan_row = PlanRow()
            diff = PlanDiff(
                entries=(DiffEntry("music_gain", "Music", "70%", "40%"),),
                applied=("Music volume to 40%",),
            )
            outcome = CoEditOutcome(
                ok=True,
                delta=EditDelta(
                    operations=(ChangeMusicVolume(value=0.4),),
                    rationale="Quieter bed.",
                    source="rules",
                ),
                source=CoEditSource.RULES,
            )

        return Applied()

    async def history(self, **kwargs: Any) -> list[VersionSummary]:
        self.calls.append(("history", kwargs))
        return [summary()]

    async def undo(self, project_id: Any) -> Any:
        self.calls.append(("undo", {"project_id": project_id}))
        return VersionRow()

    async def redo(self, project_id: Any) -> Any:
        self.calls.append(("redo", {"project_id": project_id}))
        return VersionRow()

    async def restore(self, project_id: Any, version_id: Any) -> Any:
        self.calls.append(("restore", {"project_id": project_id, "version_id": version_id}))
        return VersionRow()


def _patch_outcome() -> Any:
    class Patch:
        ok = True
        diff = PlanDiff(
            entries=(DiffEntry("music_gain", "Music", "70%", "40%"),),
            applied=("Music volume to 40%",),
        )
        violations: ClassVar[tuple[Any, ...]] = ()

    return Patch()


class FakeVersionRepo:
    async def list_for_project(self, project_id: Any, *, limit: int = 50) -> list[VersionRow]:
        return [VersionRow()]


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(service: FakeService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_coedit_service] = lambda: service
    app.dependency_overrides[get_version_repo] = lambda: FakeVersionRepo()
    app.dependency_overrides[get_llm_provider] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def url(suffix: str = "") -> str:
    return f"/api/projects/{PROJECT_ID}/edit-plan{suffix}"


# -------------------------------------------------------------------- preview
def test_preview_returns_a_diff_and_writes_nothing(
    client: TestClient, service: FakeService
) -> None:
    response = client.post(url("/co-edit/preview"), json={"request_text": "lower the music to 40%"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["diff"]["entries"][0] == {
        "field": "music_gain",
        "label": "Music",
        "before": "70%",
        "after": "40%",
        "clip": None,
    }
    assert body["operations"] == [{"kind": "CHANGE_MUSIC_VOLUME", "value": 0.4}]
    # The rules resolved it, so the editor may apply it without a confirmation.
    assert body["needs_confirmation"] is False
    assert [name for name, _ in service.calls] == ["preview"]


def test_preview_passes_the_request_through_unchanged(
    client: TestClient, service: FakeService
) -> None:
    client.post(url("/co-edit/preview"), json={"request_text": "make it snappier"})
    _, kwargs = service.calls[0]
    assert kwargs["request_text"] == "make it snappier"
    assert kwargs["operations"] is None


# ---------------------------------------------------------------------- apply
def test_apply_creates_a_version(client: TestClient, service: FakeService) -> None:
    response = client.post(url("/co-edit"), json={"request_text": "lower the music to 40%"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"]["version"] == 2
    assert body["version"]["is_current"] is True
    assert body["plan"]["id"] == str(PLAN_ID)
    assert body["diff"]["applied"] == ["Music volume to 40%"]
    assert body["rationale"] == "Quieter bed."
    assert body["source"] == "rules"


def test_apply_forwards_the_base_version(client: TestClient, service: FakeService) -> None:
    """A change previewed against one version must not land on another."""
    base = uuid.uuid4()
    client.post(
        url("/co-edit"),
        json={
            "operations": [{"kind": "CHANGE_MUSIC_VOLUME", "value": 0.4}],
            "base_version_id": str(base),
        },
    )
    _, kwargs = next(call for call in service.calls if call[0] == "apply")
    assert kwargs["base_version_id"] == base


def test_a_rejected_change_is_a_422_naming_every_reason() -> None:
    rejection = CoEditRejected(
        CoEditOutcome(ok=False, failure=CoEditFailure.UNREADABLE, detail="unreadable"),
        (
            DeltaViolation("no_such_segment", "this edit has no clip 9", 0),
            DeltaViolation("no_music", "this edit has no music to adjust", 1),
        ),
    )
    failing = FakeService(rejects=rejection)
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_coedit_service] = lambda: failing
    app.dependency_overrides[get_llm_provider] = lambda: None

    with TestClient(app) as client:
        response = client.post(url("/co-edit"), json={"request_text": "remove clip 9"})

    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert "unchanged" in error["message"]
    assert "no clip 9" in error["hint"]
    assert "no music" in error["hint"]
    app.dependency_overrides.clear()


def test_a_provider_failure_is_reported_without_touching_the_plan() -> None:
    rejection = CoEditRejected(
        CoEditOutcome(
            ok=False,
            failure=CoEditFailure.PROVIDER_UNAVAILABLE,
            detail="no route to host",
            source=CoEditSource.LLM,
        ),
        (),
    )
    failing = FakeService(rejects=rejection)
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_coedit_service] = lambda: failing
    app.dependency_overrides[get_llm_provider] = lambda: None

    with TestClient(app) as client:
        response = client.post(url("/co-edit"), json={"request_text": "make it cinematic"})

    assert response.status_code == 422
    assert "no route to host" in response.json()["error"]["hint"]
    app.dependency_overrides.clear()


# ------------------------------------------------------------ what may be sent
def test_a_request_must_ask_for_something(client: TestClient) -> None:
    assert client.post(url("/co-edit"), json={}).status_code == 422


def test_request_text_is_length_capped(client: TestClient) -> None:
    response = client.post(url("/co-edit"), json={"request_text": "x" * 5_000})
    assert response.status_code == 422


def test_too_many_operations_are_refused_at_the_boundary(client: TestClient) -> None:
    operations = [{"kind": "REMOVE_SEGMENT", "segment": 0}] * (MAX_OPERATIONS + 1)
    response = client.post(url("/co-edit"), json={"operations": operations})
    assert response.status_code == 422


def test_there_is_no_field_for_a_plan_or_a_path(client: TestClient, service: FakeService) -> None:
    """Extra keys are ignored; the route reads three fields and no others."""
    client.post(
        url("/co-edit"),
        json={
            "request_text": "lower the music to 40%",
            "plan": {"segments": []},
            "output_path": "/tmp/out.mp4",
            "ffmpeg_args": ["-i", "x"],
        },
    )
    _, kwargs = next(call for call in service.calls if call[0] == "apply")
    assert set(kwargs) == {
        "project_id",
        "request_text",
        "operations",
        "base_version_id",
        "provider",
    }


# -------------------------------------------------------------------- history
def test_history_lists_versions_with_undo_state(client: TestClient) -> None:
    response = client.get(url("/versions"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["current_version_id"] == str(VERSION_ID)
    # One version with no parent: nothing to undo to, nothing to redo.
    assert body["can_undo"] is False
    assert body["can_redo"] is False
    assert body["items"][0]["applied"] == ["Music volume to 40%"]


def test_history_never_carries_the_request_text(client: TestClient) -> None:
    body = client.get(url("/versions")).json()
    assert body["items"][0]["request_digest"] == "abc123"
    assert "request_text" not in body["items"][0]


def test_undo_and_redo_move_the_head(client: TestClient, service: FakeService) -> None:
    assert client.post(url("/versions/undo")).status_code == 200
    assert client.post(url("/versions/redo")).status_code == 200
    names = [name for name, _ in service.calls]
    assert "undo" in names and "redo" in names


def test_restore_jumps_to_a_named_version(client: TestClient, service: FakeService) -> None:
    target = uuid.uuid4()
    response = client.post(url(f"/versions/{target}/restore"))

    assert response.status_code == 200
    _, kwargs = next(call for call in service.calls if call[0] == "restore")
    assert kwargs["version_id"] == target


# --------------------------------------------------------------- capabilities
def test_capabilities_declare_the_operation_vocabulary() -> None:
    """The UI offers what the parser accepts, generated from one enum."""
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: None
    with TestClient(app) as client:
        body = client.get("/api/planner/capabilities").json()

    assert [item["kind"] for item in body["operations"]] == [kind.value for kind in OperationKind]
    assert body["coedit_prompt_version"]
    app.dependency_overrides.clear()
