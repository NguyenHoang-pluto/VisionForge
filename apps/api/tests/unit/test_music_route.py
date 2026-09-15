"""The music block at the API boundary.

What is under test is the *boundary*: which fields a caller may send, what the
server derives rather than accepts, and how a bad cue is reported. The domain's
own rules are covered in ``test_music_validation``; the planner's in
``test_beat_planning``.

The security half of this file is the point. A music bed reaches an FFmpeg
audio filter graph, which runs commands as readily as a video one, so the
question "is there any field here a path or a flag could travel in" has to be
answered by tests rather than by reading.
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
from visionforge.api.schemas.edit import MusicRequest
from visionforge.domain.editplan import (
    MAX_FADE_MS,
    MAX_GAIN,
    MIN_MUSIC_MS,
    PlanInvalidError,
    PlanViolation,
)

PROJECT_ID = uuid.uuid4()
MEDIA_ID = uuid.uuid4()
TRACK_ID = uuid.uuid4()


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


@pytest.fixture
def service() -> CapturingService:
    return CapturingService()


@pytest.fixture
def client(service: CapturingService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[require_project] = lambda: FakeProject()
    app.dependency_overrides[get_edit_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def music(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "media_id": str(TRACK_ID),
        "source_in_ms": 0,
        "source_out_ms": 20_000,
        "timeline_start_ms": 0,
        "volume": 0.6,
        "fade_in_ms": 500,
        "fade_out_ms": 1_500,
    }
    payload.update(overrides)
    return payload


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


# ------------------------------------------------------------- accepted cues
class TestAcceptedMusic:
    def test_a_timeline_with_no_music_is_unchanged(
        self, client: TestClient, service: CapturingService
    ) -> None:
        """Every plan written before Phase 7 must still post exactly as it did."""
        assert post(client, body()).status_code == 201
        assert service.calls[0]["music"] is None

    def test_a_cue_reaches_the_service_field_for_field(
        self, client: TestClient, service: CapturingService
    ) -> None:
        assert post(client, body(music=music())).status_code == 201

        cue = service.calls[0]["music"]
        assert str(cue.media_id) == str(TRACK_ID)
        assert cue.source_in_ms == 0
        assert cue.source_out_ms == 20_000
        assert cue.timeline_start_ms == 0
        assert cue.gain == pytest.approx(0.6)
        assert cue.fade_in_ms == 500
        assert cue.fade_out_ms == 1_500

    def test_volume_becomes_gain_without_reinterpretation(
        self, client: TestClient, service: CapturingService
    ) -> None:
        """The wire calls it volume and the domain calls it gain; the route
        renames and does nothing else. A scale conversion hidden here would be
        a second place an edit gets its character."""
        post(client, body(music=music(volume=0.25)))
        assert service.calls[0]["music"].gain == pytest.approx(0.25)

    def test_source_gain_reaches_the_output_spec(
        self, client: TestClient, service: CapturingService
    ) -> None:
        post(client, body(source_gain=0.3))
        assert service.calls[0]["output"].source_gain == pytest.approx(0.3)

    def test_source_gain_defaults_to_unity(
        self, client: TestClient, service: CapturingService
    ) -> None:
        post(client, body())
        assert service.calls[0]["output"].source_gain == 1.0

    def test_fades_are_optional(self, client: TestClient, service: CapturingService) -> None:
        payload = music()
        del payload["fade_in_ms"]
        del payload["fade_out_ms"]
        assert post(client, body(music=payload)).status_code == 201


# --------------------------------------------------------------- rejections
class TestRejectedMusic:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("volume", -0.1),
            ("volume", MAX_GAIN + 0.1),
            ("source_in_ms", -1),
            ("timeline_start_ms", -1),
            ("fade_in_ms", -1),
            ("fade_out_ms", MAX_FADE_MS + 1),
            ("source_out_ms", 0),
        ],
    )
    def test_out_of_range_values_are_refused_with_a_field_error(
        self, client: TestClient, field: str, value: Any
    ) -> None:
        response = post(client, body(music=music(**{field: value})))
        assert response.status_code == 422
        assert field in response.text

    def test_an_inverted_range_is_refused(self, client: TestClient) -> None:
        response = post(client, body(music=music(source_in_ms=9_000, source_out_ms=1_000)))
        assert response.status_code == 422
        assert "source_out_ms" in response.text

    def test_a_cue_below_the_floor_is_refused(self, client: TestClient) -> None:
        response = post(client, body(music=music(source_in_ms=0, source_out_ms=MIN_MUSIC_MS - 1)))
        assert response.status_code == 422

    def test_fades_longer_than_the_cue_are_refused(self, client: TestClient) -> None:
        """Overlapping ramps make afade's behaviour a coin toss, so they are
        stopped at the boundary rather than resolved arbitrarily downstream."""
        response = post(
            client,
            body(music=music(source_out_ms=4_000, fade_in_ms=3_000, fade_out_ms=3_000)),
        )
        assert response.status_code == 422

    def test_a_media_id_that_is_not_a_uuid_is_refused(self, client: TestClient) -> None:
        response = post(client, body(music=music(media_id="../../etc/passwd")))
        assert response.status_code == 422

    def test_a_domain_violation_comes_back_as_a_readable_rejection(self) -> None:
        """A cue naming another project's track is refused by the validator, and
        the editor is told which rule it broke rather than that a render failed."""
        failing = CapturingService(
            raises=PlanInvalidError(
                [
                    PlanViolation(
                        "music_cross_project_media",
                        "music asset belongs to another project",
                        is_music=True,
                    )
                ]
            )
        )
        app = create_app()
        app.dependency_overrides[require_project] = lambda: FakeProject()
        app.dependency_overrides[get_edit_service] = lambda: failing
        with TestClient(app) as test_client:
            response = post(test_client, body(music=music()))
        app.dependency_overrides.clear()

        assert response.status_code == 422
        assert "another project" in response.text


# ----------------------------------------------------------------- security
class TestNoInjectionSurface:
    """The cue schema is a media id and six numbers, and nothing else.

    Pinned as a field set so that adding a field is a deliberate act that fails
    this test until somebody confirms the new field cannot carry a path, a
    filter or a flag -- the same guard the video segment schema has.
    """

    def test_the_field_set_is_pinned(self) -> None:
        assert set(MusicRequest.model_fields) == {
            "media_id",
            "source_in_ms",
            "source_out_ms",
            "timeline_start_ms",
            "volume",
            "fade_in_ms",
            "fade_out_ms",
        }

    def test_every_field_but_the_id_is_a_number(self) -> None:
        for name, field in MusicRequest.model_fields.items():
            if name == "media_id":
                assert field.annotation is uuid.UUID
            else:
                assert field.annotation in (int, float), f"{name} is not numeric"

    @pytest.mark.parametrize(
        "extra",
        [
            {"filter": "volume=1;rm -rf /"},
            {"path": "/etc/passwd"},
            {"storage_key": "projects/other/secret.mp3"},
            {"codec": "libmp3lame"},
            {"ffmpeg_args": ["-i", "/dev/zero"]},
            {"af": "aresample=48000"},
        ],
    )
    def test_unknown_fields_never_reach_the_domain(
        self, client: TestClient, service: CapturingService, extra: dict[str, Any]
    ) -> None:
        """Pydantic ignores unknown keys by default, so the assertion that
        matters is not the status code -- it is that nothing the caller invented
        survives into the cue the service receives."""
        response = post(client, body(music=music(**extra)))
        assert response.status_code == 201

        cue = service.calls[0]["music"]
        for key in extra:
            assert not hasattr(cue, key)

    def test_the_cue_handed_to_the_service_carries_no_strings_but_the_id(
        self, client: TestClient, service: CapturingService
    ) -> None:
        post(client, body(music=music()))
        cue = service.calls[0]["music"]
        for name in type(cue).__dataclass_fields__:
            value = getattr(cue, name)
            if name == "media_id":
                assert uuid.UUID(str(value))
            else:
                assert isinstance(value, int | float), f"{name} is not numeric"


# ------------------------------------------------------------- capabilities
class TestCapabilitiesDeclareAudioBounds:
    def test_audio_bounds_are_declared(self, client: TestClient) -> None:
        """The editor clamps a volume slider and two fade handles while the
        pointer is moving. It gets the numbers from here for the same reason it
        gets the trim bounds from here: a copy in the browser would drift."""
        bounds = client.get("/api/planner/capabilities").json()["audio_bounds"]
        assert bounds["min_gain"] == 0.0
        assert bounds["max_gain"] == MAX_GAIN
        assert bounds["max_fade_ms"] == MAX_FADE_MS
        assert bounds["min_music_ms"] == MIN_MUSIC_MS

    def test_beat_sync_capability_is_declared(self, client: TestClient) -> None:
        """So the UI can explain a grid the server has decided not to use,
        rather than offering a toggle that silently does nothing."""
        beat_sync = client.get("/api/planner/capabilities").json()["beat_sync"]
        assert beat_sync["analyzer"] == "beats"
        assert beat_sync["min_bpm"] < beat_sync["max_bpm"]
        assert 0.0 < beat_sync["min_confidence"] <= 1.0
