"""The manual edit-plan route, with the service faked out.

What is under test here is the *boundary*, not the planning: which fields a
timeline may send, what the server derives rather than accepts, and how a
rejected timeline is reported. The service's own behaviour is covered in
``test_manual_plan``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from visionforge.api.dependencies import get_edit_service, require_project
from visionforge.api.main import create_app
from visionforge.domain.editplan import PlanInvalidError, PlanViolation

PROJECT_ID = uuid.uuid4()
MEDIA_ID = uuid.uuid4()


class CapturingService:
    """Stands in for ``EditService`` and records what the route handed it."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises

    async def create_manual_plan(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises

        class Row:
            id = uuid.uuid4()
            project_id = PROJECT_ID
            planner = "manual"
            planner_version = "1"
            segment_count = len(kwargs["cuts"])
            total_duration_ms = sum(c.source_out_ms - c.source_in_ms for c in kwargs["cuts"])
            created_at = datetime.now(UTC)
            plan: ClassVar[dict[str, Any]] = {"metadata": {"source": "manual"}}
            selection: ClassVar[dict[str, Any]] = {}

        return Row()


class FakeProject:
    id = PROJECT_ID
    title = "Editor"


def client_for(service: CapturingService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_edit_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def service() -> CapturingService:
    return CapturingService()


@pytest.fixture
def client(service: CapturingService) -> Iterator[TestClient]:
    yield from client_for(service)


def body(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "segments": [
            {"media_id": str(MEDIA_ID), "source_in_ms": 0, "source_out_ms": 4000},
            {"media_id": str(MEDIA_ID), "source_in_ms": 6000, "source_out_ms": 9000},
        ],
        "aspect_ratio": "16:9",
        "fps": 30,
        "quality": "balanced",
    }
    payload.update(overrides)
    return payload


def post(client: TestClient, payload: dict[str, Any]) -> Any:
    return client.post(f"/api/projects/{PROJECT_ID}/edit-plan/manual", json=payload)


class TestManualPlanRoute:
    def test_accepts_a_timeline_and_returns_the_stored_plan(
        self, client: TestClient, service: CapturingService
    ) -> None:
        response = post(client, body())

        assert response.status_code == 201, response.text
        assert response.json()["planner"] == "manual"
        assert len(service.calls[0]["cuts"]) == 2

    def test_list_position_becomes_segment_order(
        self, client: TestClient, service: CapturingService
    ) -> None:
        post(client, body())
        cuts = service.calls[0]["cuts"]

        assert [c.source_in_ms for c in cuts] == [0, 6000]

    @pytest.mark.parametrize(
        ("aspect", "geometry"),
        [("16:9", (1280, 720)), ("9:16", (720, 1280)), ("1:1", (720, 720))],
    )
    def test_the_server_chooses_the_geometry(
        self,
        client: TestClient,
        service: CapturingService,
        aspect: str,
        geometry: tuple[int, int],
    ) -> None:
        """The caller names a shape. It never names a width.

        This is the Phase 4 property the timeline must not erode: there is no
        field on this route through which a client could set the output size.
        """
        post(client, body(aspect_ratio=aspect))
        output = service.calls[0]["output"]

        assert (output.width, output.height) == geometry

    def test_geometry_fields_in_the_body_are_ignored(self, client: TestClient) -> None:
        """Extra keys are not a way in: pydantic drops what the model lacks."""
        response = post(client, body(width=3840, height=2160, crf=1, ffmpeg_args="-vf evil"))

        assert response.status_code == 201, response.text

    def test_an_empty_timeline_is_rejected_by_the_schema(self, client: TestClient) -> None:
        assert post(client, body(segments=[])).status_code == 422

    def test_only_offered_frame_rates_are_accepted(self, client: TestClient) -> None:
        assert post(client, body(fps=29)).status_code == 422
        assert post(client, body(fps=60)).status_code == 201

    def test_a_negative_trim_is_rejected_by_the_schema(self, client: TestClient) -> None:
        response = post(
            client,
            body(segments=[{"media_id": str(MEDIA_ID), "source_in_ms": -1, "source_out_ms": 4000}]),
        )

        assert response.status_code == 422

    def test_too_many_segments_are_rejected(self, client: TestClient) -> None:
        segment = {"media_id": str(MEDIA_ID), "source_in_ms": 0, "source_out_ms": 1000}
        response = post(client, body(segments=[segment] * 41))

        assert response.status_code == 422

    def test_the_plan_it_was_cut_from_is_passed_through(
        self, client: TestClient, service: CapturingService
    ) -> None:
        origin = uuid.uuid4()
        post(client, body(derived_from_edit_plan_id=str(origin)))

        assert service.calls[0]["derived_from"] == origin

    def test_defaults_are_applied_when_the_editor_states_nothing(
        self, client: TestClient, service: CapturingService
    ) -> None:
        post(client, {"segments": body()["segments"]})
        output = service.calls[0]["output"]

        assert (output.width, output.height, output.fps) == (1280, 720, 30)


class TestRejectedTimeline:
    def test_violations_are_reported_so_the_editor_can_point_at_the_clip(self) -> None:
        """A rejected hand-cut edit explains itself.

        The failure a user will actually hit -- dragging a trim handle past the
        end of a source -- must come back as a readable reason, not as a render
        that fails minutes later.
        """
        service = CapturingService(
            raises=PlanInvalidError(
                [
                    PlanViolation(
                        "trim_past_end",
                        "source_out_ms 9000 exceeds the source's 5000 ms",
                        1,
                    )
                ]
            )
        )
        for test_client in client_for(service):
            response = post(test_client, body())

            assert response.status_code == 422
            payload = response.json()["error"]
            assert payload["code"] == "validation_error"
            assert "trim_past_end" not in payload["message"]
            assert "exceeds the source's 5000 ms" in payload["hint"]
