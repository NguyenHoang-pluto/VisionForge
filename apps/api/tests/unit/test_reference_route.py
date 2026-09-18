"""The reference at the API boundary, and the boundaries it must not cross.

Two questions are answered here by test rather than by reading:

**Whose footage can become a reference?** Only media already in the project.
A media id belonging to someone else gets the answer a nonexistent id gets,
because a distinct "not yours" would confirm the id exists.

**Can a reference end up in the render?** No. The clip being imitated is
excluded from the candidate list before any planner sees it, so neither the
rules engine nor a model can select what it was never handed -- and the
acceptance requirement that the output contain only the user's other media
rests on exactly that.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.api.dependencies import get_media_repo, get_project_repo, require_project
from visionforge.api.main import create_app
from visionforge.api.schemas.edit import PlanCreateRequest
from visionforge.application.reference_service import PROFILE_ANALYZERS, ReferenceService
from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.domain.policy import StyleStrength
from visionforge.domain.reference import without_reference
from visionforge.domain.selection import Candidate

PROJECT_ID = uuid.uuid4()
REFERENCE_ID = uuid.uuid4()
OTHER_PROJECT_MEDIA = uuid.uuid4()
AUDIO_ID = uuid.uuid4()

SCENES = {
    "scenes": [
        {"scene_id": i, "start_ms": i * 800, "end_ms": (i + 1) * 800, "duration_ms": 800}
        for i in range(12)
    ],
    "scene_count": 12,
}
QUALITY = {"frame_count": 5, "mean_luminance": 120.0, "contrast": 55.0}
DYNAMICS = {"motion": 0.5, "saturation": 0.6, "motion_confidence": 1.0, "colour_confidence": 1.0}


class FakeAsset:
    def __init__(self, media_id: uuid.UUID, kind: str = "video", status: str = "ready") -> None:
        self.id = media_id
        self.kind = kind
        self.status = status
        self.duration_ms = 30_000


class FakeProject:
    def __init__(self, reference_media_id: uuid.UUID | None = None) -> None:
        self.id = PROJECT_ID
        self.title = "Styled"
        self.reference_media_id = reference_media_id


class FakeProjectRepo:
    def __init__(self) -> None:
        self.set_calls: list[uuid.UUID | None] = []

    async def set_reference(self, project: Any, media_id: uuid.UUID | None) -> None:
        self.set_calls.append(media_id)
        project.reference_media_id = media_id


class FakeMediaRepo:
    """Only media in this project resolves. Anything else is not found."""

    def __init__(self, assets: dict[uuid.UUID, FakeAsset] | None = None) -> None:
        self.assets = assets if assets is not None else {REFERENCE_ID: FakeAsset(REFERENCE_ID)}
        self.payloads: dict[AnalyzerName, dict[str, Any]] = {
            AnalyzerName.SCENES: SCENES,
            AnalyzerName.QUALITY: QUALITY,
            AnalyzerName.DYNAMICS: DYNAMICS,
        }

    async def get_in_project(self, media_id: Any, project_id: Any) -> FakeAsset | None:
        if project_id != PROJECT_ID:
            return None
        return self.assets.get(uuid.UUID(str(media_id)))

    async def analysis_payload(self, media_id: Any, analyzer: AnalyzerName) -> Any:
        return self.payloads.get(analyzer)

    async def analysis_payloads(self, media_id: Any, analyzers: Any) -> dict[Any, Any]:
        return {a: self.payloads[a] for a in analyzers if a in self.payloads}


@pytest.fixture
def project() -> FakeProject:
    return FakeProject()


@pytest.fixture
def media_repo() -> FakeMediaRepo:
    return FakeMediaRepo()


@pytest.fixture
def client(project: FakeProject, media_repo: FakeMediaRepo) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[require_project] = lambda: project
    app.dependency_overrides[get_project_repo] = lambda: FakeProjectRepo()
    app.dependency_overrides[get_media_repo] = lambda: media_repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestNominating:
    def test_a_clip_in_the_project_is_accepted(
        self, client: TestClient, project: FakeProject
    ) -> None:
        response = client.put(
            f"/api/projects/{PROJECT_ID}/reference", json={"media_id": str(REFERENCE_ID)}
        )
        assert response.status_code == 200
        assert response.json()["media_id"] == str(REFERENCE_ID)
        assert project.reference_media_id == REFERENCE_ID

    def test_media_from_another_project_is_not_found(self, client: TestClient) -> None:
        """Not 403. A distinct answer would confirm the id exists somewhere."""
        response = client.put(
            f"/api/projects/{PROJECT_ID}/reference", json={"media_id": str(OTHER_PROJECT_MEDIA)}
        )
        assert response.status_code == 404

    def test_a_nonexistent_id_gets_the_same_answer(self, client: TestClient) -> None:
        response = client.put(
            f"/api/projects/{PROJECT_ID}/reference", json={"media_id": str(uuid.uuid4())}
        )
        assert response.status_code == 404

    def test_the_body_accepts_nothing_but_an_id(self, client: TestClient) -> None:
        """No path, no key, no description, no measurements. What the reference
        measures as is read from the analysers, never asserted by the caller."""
        assert set(
            __import__(
                "visionforge.api.schemas.edit", fromlist=["ReferenceRequest"]
            ).ReferenceRequest.model_fields
        ) == {"media_id"}

    def test_extra_fields_are_ignored_rather_than_honoured(self, client: TestClient) -> None:
        response = client.put(
            f"/api/projects/{PROJECT_ID}/reference",
            json={
                "media_id": str(REFERENCE_ID),
                "local_path": "/etc/passwd",
                "storage_key": "bucket/secret",
                "shot_ms": 1,
            },
        )
        assert response.status_code == 200
        assert "local_path" not in response.text
        assert "passwd" not in response.text

    def test_clearing_returns_the_empty_state(
        self, client: TestClient, project: FakeProject
    ) -> None:
        project.reference_media_id = REFERENCE_ID
        response = client.delete(f"/api/projects/{PROJECT_ID}/reference")
        assert response.status_code == 200
        assert response.json()["media_id"] is None
        assert project.reference_media_id is None


class TestKindAndReadiness:
    @pytest.mark.anyio
    async def test_audio_cannot_be_a_reference(self, media_repo: FakeMediaRepo) -> None:
        """An audio file has no shot length and no cuts. Nominating it would
        produce an empty profile and no explanation."""
        media_repo.assets[AUDIO_ID] = FakeAsset(AUDIO_ID, kind="audio")
        service = ReferenceService(FakeProjectRepo(), media_repo)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            await service.set_reference(FakeProject(), MediaId(AUDIO_ID))

    @pytest.mark.anyio
    async def test_media_still_processing_cannot_be_a_reference(
        self, media_repo: FakeMediaRepo
    ) -> None:
        pending = uuid.uuid4()
        media_repo.assets[pending] = FakeAsset(pending, status="processing")
        service = ReferenceService(FakeProjectRepo(), media_repo)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            await service.set_reference(FakeProject(), MediaId(pending))

    @pytest.mark.anyio
    async def test_a_foreign_id_raises_not_found(self, media_repo: FakeMediaRepo) -> None:
        service = ReferenceService(FakeProjectRepo(), media_repo)  # type: ignore[arg-type]
        with pytest.raises(NotFoundError):
            await service.set_reference(FakeProject(), MediaId(OTHER_PROJECT_MEDIA))


class TestReadingTheProfile:
    def test_no_reference_is_an_empty_response_not_an_error(self, client: TestClient) -> None:
        response = client.get(f"/api/projects/{PROJECT_ID}/reference")
        assert response.status_code == 200
        assert response.json() == {"media_id": None, "pending_analyzers": [], "profile": None}

    def test_the_profile_is_measured_from_the_analysis_rows(
        self, client: TestClient, project: FakeProject
    ) -> None:
        project.reference_media_id = REFERENCE_ID
        payload = client.get(f"/api/projects/{PROJECT_ID}/reference").json()

        profile = payload["profile"]
        assert profile is not None
        assert profile["usable"] is True
        assert profile["shot_ms"]["value"] == 800
        assert profile["pacing"] == "rapid"
        assert 0.0 < profile["confidence"] <= 1.0

    def test_missing_analysis_is_reported_rather_than_defaulted(
        self, client: TestClient, project: FakeProject, media_repo: FakeMediaRepo
    ) -> None:
        project.reference_media_id = REFERENCE_ID
        del media_repo.payloads[AnalyzerName.DYNAMICS]

        payload = client.get(f"/api/projects/{PROJECT_ID}/reference").json()
        assert payload["profile"]["motion"] is None
        assert payload["profile"]["saturation"] is None
        assert AnalyzerName.DYNAMICS.value in payload["pending_analyzers"]

    def test_an_unanalysed_reference_lists_everything_as_pending(
        self, client: TestClient, project: FakeProject, media_repo: FakeMediaRepo
    ) -> None:
        project.reference_media_id = REFERENCE_ID
        media_repo.payloads.clear()

        payload = client.get(f"/api/projects/{PROJECT_ID}/reference").json()
        assert payload["profile"] is not None
        assert payload["profile"]["usable"] is False
        assert set(payload["pending_analyzers"]) == {a.value for a in PROFILE_ANALYZERS}

    def test_the_response_carries_no_path_or_key(
        self, client: TestClient, project: FakeProject
    ) -> None:
        project.reference_media_id = REFERENCE_ID
        body = client.get(f"/api/projects/{PROJECT_ID}/reference").text
        for forbidden in ("local_path", "storage_key", "bucket", "original_filename", ".mp4"):
            assert forbidden not in body


class TestTheStyleDial:
    def test_strength_is_a_closed_set(self) -> None:
        field = PlanCreateRequest.model_fields["style_strength"]
        assert field.default is StyleStrength.ZERO

    def test_an_unknown_strength_is_refused(self, client: TestClient) -> None:
        response = client.post(
            f"/api/projects/{PROJECT_ID}/edit-plan",
            json={"style_strength": "63"},
        )
        assert response.status_code == 422

    def test_the_request_cannot_name_a_reference(self) -> None:
        """Which clip is the reference is project state, resolved server-side.

        A per-request reference id would let a caller aim one plan at another
        project's media and read its measurements back out of the result.
        """
        assert "reference_media_id" not in PlanCreateRequest.model_fields
        assert "reference" not in PlanCreateRequest.model_fields


class TestTheReferenceIsNeverFootage:
    """The rule the acceptance depends on."""

    def candidate(self, media_id: uuid.UUID) -> Candidate:
        return Candidate(
            media_id=MediaId(media_id),
            kind=MediaKind.VIDEO,
            is_ready=True,
            duration_ms=10_000,
            width=1920,
            height=1080,
        )

    def test_the_reference_is_removed_from_the_candidate_list(self) -> None:
        others = [self.candidate(uuid.uuid4()) for _ in range(3)]
        candidates = [*others, self.candidate(REFERENCE_ID)]

        kept = without_reference(candidates, MediaId(REFERENCE_ID))
        assert MediaId(REFERENCE_ID) not in [c.media_id for c in kept]
        assert len(kept) == 3

    def test_everything_else_survives_untouched_and_in_order(self) -> None:
        others = [self.candidate(uuid.uuid4()) for _ in range(4)]
        kept = without_reference(list(others), MediaId(REFERENCE_ID))
        assert kept == others

    def test_no_reference_removes_nothing(self) -> None:
        others = [self.candidate(uuid.uuid4()) for _ in range(4)]
        assert without_reference(list(others), None) == others

    def test_it_is_safe_when_the_reference_was_never_a_candidate(self) -> None:
        """An unanalysed or non-ready reference never reaches the list at all."""
        others = [self.candidate(uuid.uuid4()) for _ in range(2)]
        assert without_reference(list(others), MediaId(REFERENCE_ID)) == others


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
