"""The LLM planning lane against real Postgres, MinIO and the real routers.

Three configurations, because the feature has to be correct in all three:

- **provider disabled** -- the state of every developer without a key. Every
  mode still produces a plan, and the AI option reports itself unavailable
  rather than failing when used.
- **stub provider enabled** -- the whole LLM path runs: prompt, provider, parse,
  handle resolution, clamping, compilation, validation, persistence.
- **a hostile provider** -- a provider that answers with paths, commands,
  invented handles and other projects' media ids, to prove that none of it
  reaches a plan.

No test here makes a network call. The provider is overridden through the same
dependency the real one is injected by, so everything between the route and the
database is the production code path.
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

from visionforge.api.dependencies import get_llm_provider
from visionforge.api.main import create_app
from visionforge.domain.analysis import AnalysisStatus, AnalyzerName
from visionforge.domain.editplan import QualityPreset, plan_from_payload
from visionforge.domain.llm import LlmRequest, LlmResponse, LlmUsage
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.style import QUALITY_SETTINGS, EditStyle
from visionforge.infra.db.models import LlmRunRow, MediaAnalysis, MediaAsset
from visionforge.infra.db.sync_session import get_sync_sessionmaker
from visionforge.infra.llm.stub_provider import StubProvider

pytestmark = pytest.mark.integration


# -------------------------------------------------------------------- doubles
class RecordingStub(StubProvider):
    """The stub, with the prompts it received kept for inspection."""

    def __init__(self) -> None:
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return super().complete(request)


class HostileProvider:
    """Answers with everything a compromised or jailbroken model might say."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return "hostile"

    @property
    def model(self) -> str:
        return "hostile-1"

    def complete(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text=json.dumps(self.payload),
            provider=self.name,
            model=self.model,
            latency_ms=1.0,
            usage=LlmUsage(input_tokens=10, output_tokens=5),
        )


# ------------------------------------------------------------------- fixtures
def _make_client(provider: Any) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app)


@pytest.fixture
def rules_client() -> Iterator[TestClient]:
    """No provider configured. The default deployment."""
    client = _make_client(None)
    with client:
        yield client
    client.app.dependency_overrides.clear()  # type: ignore[attr-defined]


@pytest.fixture
def stub() -> RecordingStub:
    return RecordingStub()


@pytest.fixture
def ai_client(stub: RecordingStub) -> Iterator[TestClient]:
    client = _make_client(stub)
    with client:
        yield client
    client.app.dependency_overrides.clear()  # type: ignore[attr-defined]


def _analysed_project(client: TestClient, db: Session, count: int = 5) -> UUID:
    """A project with analysed, plannable video. No files, no workers.

    The project is created *through the API* so it belongs to the same
    development user ``require_project`` resolves against -- a project inserted
    directly with its own owner is correctly invisible to every route, which is
    the authorization anchor working.

    The media and analysis rows go in directly, because planning reads
    ``media_assets`` and ``media_analysis`` and nothing else. Rendering is
    Phase 4's lane and has its own tests against real files.
    """
    response = client.post("/api/projects", json={"title": "LLM Lane Test"})
    assert response.status_code == 201, response.text
    project_id = UUID(response.json()["id"])

    for index in range(count):
        media = MediaAsset(
            id=uuid.uuid4(),
            project_id=project_id,
            original_filename=f"clip_{index}.mp4",
            storage_key=f"projects/{project_id}/media/{uuid.uuid4()}/original.mp4",
            kind=MediaKind.VIDEO,
            status=MediaStatus.READY,
            duration_ms=12_000,
            width=1920,
            height=1080,
            fps=30.0,
            channels=2,
            sha256=uuid.uuid4().hex * 2,
            bytes_size=1024,
        )
        db.add(media)
        db.flush()
        db.add(
            MediaAnalysis(
                id=uuid.uuid4(),
                project_id=project_id,
                media_id=media.id,
                analyzer=AnalyzerName.QUALITY,
                analyzer_version="1",
                status=AnalysisStatus.OK,
                payload={
                    "blur_score": 400.0 - index * 15,
                    "contrast": 60.0,
                    "mean_luminance": 128.0,
                    "frames": [{"underexposed_ratio": 0.01, "overexposed_ratio": 0.01}],
                },
            )
        )
        db.add(
            MediaAnalysis(
                id=uuid.uuid4(),
                project_id=project_id,
                media_id=media.id,
                analyzer=AnalyzerName.PHASH,
                analyzer_version="1",
                status=AnalysisStatus.OK,
                # Far apart, so the near-duplicate collapser leaves them alone.
                payload={"phash": f"{(index * 0x1111111111111111) & ((1 << 64) - 1):016x}"},
            )
        )
    db.commit()
    return project_id


def _drop_project(db: Session, project_id: UUID) -> None:
    """Remove a project and everything that cascades from it.

    The development user is shared and seeded at startup, so it is left alone --
    deleting it would take every other test's fixtures with it.
    """
    db.rollback()
    with get_sync_sessionmaker()() as session:
        session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": project_id})
        session.commit()


@pytest.fixture
def analysed_project(rules_client: TestClient, db: Session) -> Iterator[UUID]:
    project_id = _analysed_project(rules_client, db)
    yield project_id
    _drop_project(db, project_id)


def _plan(client: TestClient, project_id: UUID, **body: Any) -> dict[str, Any]:
    response = client.post(f"/api/projects/{project_id}/edit-plan", json=body)
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------- provider disabled
class TestProviderDisabled:
    """The default deployment: no key, no AI, everything still works."""

    def test_capabilities_report_ai_unavailable(self, rules_client: TestClient) -> None:
        body = rules_client.get("/api/planner/capabilities").json()

        assert body["ai_available"] is False
        assert body["provider"] is None
        assert body["styles"], "styles are available with or without a model"

    def test_automatic_mode_produces_a_rules_plan(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        plan = _plan(rules_client, analysed_project, mode="automatic")

        assert plan["planner"] == "rules-engine"
        assert plan["mode"]["mode"] == "rules"
        assert plan["segment_count"] >= 2

    def test_ai_mode_degrades_to_rules_rather_than_failing(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        """Asking for AI on a server without it is answered, not refused."""
        plan = _plan(rules_client, analysed_project, mode="ai", request_text="make it cinematic")

        assert plan["planner"] == "rules-engine"
        assert plan["mode"]["mode"] == "rules"
        assert "no provider" in plan["mode"]["reason"]

    def test_a_written_request_still_matches_a_style_by_keyword(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        plan = _plan(
            rules_client,
            analysed_project,
            mode="automatic",
            request_text="make an energetic gaming montage",
        )

        assert plan["mode"]["inferred_style"] == EditStyle.GAMING.value

    def test_no_llm_run_is_recorded_when_no_model_ran(
        self, rules_client: TestClient, analysed_project: UUID, db: Session
    ) -> None:
        _plan(rules_client, analysed_project, mode="automatic")
        rows = db.query(LlmRunRow).filter(LlmRunRow.project_id == analysed_project).all()

        assert rows == []

    def test_a_styled_rules_plan_honours_the_style_pacing(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        plan = _plan(rules_client, analysed_project, style="cinematic", max_clips=4)
        segments = plan["plan"]["segments"]

        assert all(2_500 <= s["duration_ms"] <= 8_000 for s in segments)


# ------------------------------------------------------------ provider on
class TestStubProviderEnabled:
    """The full LLM path, with a deterministic provider instead of a model."""

    def test_capabilities_report_the_provider_and_flag_the_stub(
        self, ai_client: TestClient
    ) -> None:
        body = ai_client.get("/api/planner/capabilities").json()

        assert body["ai_available"] is True
        assert body["provider"] == "stub"
        assert body["is_stub"] is True, "a stub must never be presented as a model"

    def test_a_written_request_routes_to_the_llm_planner(
        self, ai_client: TestClient, analysed_project: UUID
    ) -> None:
        plan = _plan(
            ai_client,
            analysed_project,
            mode="automatic",
            request_text="make a fast energetic montage",
        )

        assert plan["mode"]["mode"] == "ai"
        assert plan["planner"] == "llm"
        assert plan["llm"]["provider"] == "stub"
        assert plan["llm"]["fallback_reason"] is None

    def test_the_plan_is_valid_and_renderable(
        self, ai_client: TestClient, analysed_project: UUID
    ) -> None:
        """An LLM plan is an ordinary plan: it rebuilds into the typed domain."""
        plan = _plan(ai_client, analysed_project, mode="ai", request_text="fast montage")
        rebuilt = plan_from_payload(plan["plan"])

        assert rebuilt.segments
        assert rebuilt.total_duration_ms == plan["total_duration_ms"]
        assert [s.order for s in rebuilt.ordered_segments] == list(range(len(rebuilt.segments)))

    def test_the_prompt_contains_no_media_id_or_filename(
        self, ai_client: TestClient, analysed_project: UUID, stub: RecordingStub, db: Session
    ) -> None:
        """The central security claim, checked against the real prompt sent."""
        _plan(ai_client, analysed_project, mode="ai", request_text="fast montage")
        assert stub.calls, "the provider should have been called"
        prompt = stub.calls[0].system + stub.calls[0].user

        media_ids = [
            str(row.id)
            for row in db.query(MediaAsset).filter(MediaAsset.project_id == analysed_project)
        ]
        assert media_ids
        for media_id in media_ids:
            assert media_id not in prompt
        for name in ("clip_0.mp4", "clip_1.mp4", "original.mp4", "projects/"):
            assert name not in prompt

    def test_an_llm_run_is_recorded_with_its_provenance(
        self, ai_client: TestClient, analysed_project: UUID, db: Session
    ) -> None:
        plan = _plan(ai_client, analysed_project, mode="ai", request_text="fast montage")
        row = db.query(LlmRunRow).filter(LlmRunRow.edit_plan_id == UUID(plan["id"])).one()

        assert row.provider == "stub"
        assert row.model
        assert row.prompt_version
        assert row.status == "ok"
        assert row.attempts >= 1

    def test_the_run_stores_a_digest_not_the_request(
        self, ai_client: TestClient, analysed_project: UUID, db: Session
    ) -> None:
        secret = "my private trip to Sa Pa with Linh"
        plan = _plan(ai_client, analysed_project, mode="ai", request_text=secret)
        row = db.query(LlmRunRow).filter(LlmRunRow.edit_plan_id == UUID(plan["id"])).one()

        assert row.request_digest and len(row.request_digest) == 16
        assert row.request_chars == len(secret)
        for column in (row.request_digest, row.fallback_detail or "", row.model, row.provider):
            assert "Sa Pa" not in column
            assert "Linh" not in column

    def test_a_bare_style_does_not_spend_a_model_call(
        self, ai_client: TestClient, analysed_project: UUID, stub: RecordingStub
    ) -> None:
        """A style is a complete instruction; paying a model to restate it is waste."""
        plan = _plan(ai_client, analysed_project, mode="automatic", style="cinematic")

        assert plan["planner"] == "rules-engine"
        assert stub.calls == []

    def test_rules_mode_never_calls_the_provider(
        self, ai_client: TestClient, analysed_project: UUID, stub: RecordingStub
    ) -> None:
        _plan(ai_client, analysed_project, mode="rules", request_text="cinematic please")
        assert stub.calls == []

    def test_the_llm_runs_endpoint_lists_the_run(
        self, ai_client: TestClient, analysed_project: UUID
    ) -> None:
        _plan(ai_client, analysed_project, mode="ai", request_text="fast montage")
        body = ai_client.get(f"/api/projects/{analysed_project}/llm-runs").json()

        assert body["total"] == 1
        assert body["items"][0]["provider"] == "stub"


# ----------------------------------------------------------- hostile provider
class TestHostileProvider:
    """A model that answers with paths, commands and other people's media."""

    def _client(self, payload: dict[str, Any]) -> TestClient:
        return _make_client(HostileProvider(payload))

    def test_a_response_naming_filesystem_paths_falls_back(self, analysed_project: UUID) -> None:
        client = self._client(
            {
                "style": "custom",
                "clips": ["/etc/passwd", "C:\\Windows\\System32\\cmd.exe"],
                "output_path": "/tmp/pwned.mp4",
            }
        )
        with client:
            plan = _plan(client, analysed_project, mode="ai", request_text="do something")

        assert plan["planner"] == "rules-engine"
        assert plan["llm"]["fallback_reason"] == "invalid_output"
        assert "pwned" not in json.dumps(plan)

    def test_a_response_carrying_ffmpeg_arguments_falls_back(self, analysed_project: UUID) -> None:
        client = self._client(
            {
                "style": "custom",
                "clips": ["c1"],
                "ffmpeg_args": ["-i", "/dev/zero", "-f", "lavfi"],
                "command": "rm -rf /",
            }
        )
        with client:
            plan = _plan(client, analysed_project, mode="ai", request_text="do something")

        payload = json.dumps(plan)
        assert "/dev/zero" not in payload
        assert "rm -rf" not in payload
        assert "ffmpeg_args" not in payload

    def test_a_response_naming_another_projects_media_falls_back(
        self, analysed_project: UUID, db: Session, rules_client: TestClient
    ) -> None:
        """Even a real, valid media id from elsewhere is not a usable reference."""
        other_project = _analysed_project(rules_client, db, count=2)
        stranger = db.query(MediaAsset).filter(MediaAsset.project_id == other_project).first()
        assert stranger is not None

        client = self._client({"style": "custom", "clips": [str(stranger.id)]})
        try:
            with client:
                plan = _plan(client, analysed_project, mode="ai", request_text="use that clip")

            assert plan["planner"] == "rules-engine"
            assert str(stranger.id) not in json.dumps(plan["plan"]["segments"])
        finally:
            _drop_project(db, other_project)

    def test_an_absurd_duration_is_clamped_not_rejected(self, analysed_project: UUID) -> None:
        """A greedy but well-formed answer is corrected, not thrown away."""
        client = self._client(
            {
                "style": "custom",
                "pacing": "slow",
                "clips": [{"ref": f"c{i}", "duration_ms": 99_999_999} for i in (1, 2, 3)],
            }
        )
        with client:
            plan = _plan(client, analysed_project, mode="ai", request_text="make it long")

        assert plan["planner"] == "llm", "a clampable answer should still be used"
        assert plan["total_duration_ms"] <= 10 * 60 * 1_000
        assert all(s["duration_ms"] <= 30_000 for s in plan["plan"]["segments"])

    def test_the_fallback_is_recorded_rather_than_hidden(
        self, analysed_project: UUID, db: Session
    ) -> None:
        client = self._client({"nonsense": True})
        with client:
            plan = _plan(client, analysed_project, mode="ai", request_text="go")

        row = db.query(LlmRunRow).filter(LlmRunRow.edit_plan_id == UUID(plan["id"])).one()
        assert row.status == "fallback"
        assert row.fallback_reason
        assert row.violations is not None


# -------------------------------------------------------------------- limits
class TestServerSideLimits:
    def test_the_request_text_is_length_limited(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        response = rules_client.post(
            f"/api/projects/{analysed_project}/edit-plan",
            json={"request_text": "x" * 5_000},
        )
        assert response.status_code == 422

    def test_an_arbitrary_frame_rate_is_refused(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        response = rules_client.post(
            f"/api/projects/{analysed_project}/edit-plan", json={"fps": 47}
        )
        assert response.status_code == 422

    def test_geometry_cannot_be_requested(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        """Unknown fields are ignored, so the output stays on the preset map."""
        plan = _plan(
            rules_client,
            analysed_project,
            width=3840,
            height=2160,
            crf=1,
            output_path="/tmp/x.mp4",
        )
        assert (plan["plan"]["output"]["width"], plan["plan"]["output"]["height"]) == (1280, 720)

    def test_quality_presets_map_to_encoder_settings(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        for preset in QualityPreset:
            plan = _plan(rules_client, analysed_project, quality=preset.value)
            assert plan["plan"]["output"]["quality"] == preset.value
            assert QUALITY_SETTINGS[preset][0] in range(0, 52)

    def test_planning_requires_project_ownership(self, rules_client: TestClient) -> None:
        response = rules_client.post(
            f"/api/projects/{uuid.uuid4()}/edit-plan", json={"mode": "rules"}
        )
        assert response.status_code == 404

    def test_llm_runs_require_project_ownership(self, rules_client: TestClient) -> None:
        response = rules_client.get(f"/api/projects/{uuid.uuid4()}/llm-runs")
        assert response.status_code == 404

    def test_capabilities_never_expose_a_key(self, ai_client: TestClient) -> None:
        body = ai_client.get("/api/planner/capabilities").text.lower()
        for token in ("api_key", "secret", "sk-", "authorization", "bearer"):
            assert token not in body


# ------------------------------------------------------------ subtitles (P9)
class CueProvider:
    """Answers the subtitle prompt with whatever the test wants to send."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "cue-stub"

    @property
    def model(self) -> str:
        return "cue-stub-1"

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return LlmResponse(
            text=self._text,
            provider=self.name,
            model=self.model,
            latency_ms=2.0,
            usage=LlmUsage(input_tokens=20, output_tokens=10),
        )


def _suggest(client: TestClient, project_id: UUID, plan_id: str, **body: Any) -> dict[str, Any]:
    response = client.post(
        f"/api/projects/{project_id}/edit-plan/{plan_id}/subtitles/suggest", json=body
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


class TestSubtitleSuggestion:
    """The AI subtitle route, against a real plan in a real database.

    The route is read-only by design: it drafts cues for a stored plan and the
    user decides whether they become an edit. These tests hold that line and
    the two that matter more -- that the model is told nothing it could name a
    file with, and that a failure produces no subtitles rather than plausible
    ones.
    """

    def test_no_provider_is_a_named_failure_not_a_500(
        self, rules_client: TestClient, analysed_project: UUID
    ) -> None:
        plan = _plan(rules_client, analysed_project, mode="rules")

        body = _suggest(rules_client, analysed_project, plan["id"])

        assert body["ok"] is False
        assert body["failure"] == "provider_disabled"
        assert body["subtitles"] is None

    def test_a_plan_from_another_project_is_not_found(
        self, rules_client: TestClient, analysed_project: UUID, db: Session
    ) -> None:
        """The same ownership anchor every other route has."""
        other = _analysed_project(rules_client, db)
        try:
            plan = _plan(rules_client, other, mode="rules")
            response = rules_client.post(
                f"/api/projects/{analysed_project}/edit-plan/{plan['id']}/subtitles/suggest",
                json={},
            )
            assert response.status_code == 404
        finally:
            _drop_project(db, other)

    def test_usable_cues_come_back_as_a_track(self, analysed_project: UUID) -> None:
        provider = CueProvider(
            json.dumps(
                {
                    "cues": [
                        {"start_ms": 0, "end_ms": 2_000, "text": "First line"},
                        {"start_ms": 2_500, "end_ms": 4_500, "text": "Second line"},
                    ]
                }
            )
        )
        client = _make_client(provider)
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            body = _suggest(client, analysed_project, plan["id"])

        assert body["ok"] is True
        assert [cue["text"] for cue in body["subtitles"]["cues"]] == [
            "First line",
            "Second line",
        ]

    def test_the_cue_prompt_names_no_file_and_no_id(self, analysed_project: UUID) -> None:
        """The claim the whole design rests on, checked against real rows.

        The project has real media with real filenames and real storage keys in
        the database. None of it is in the prompt, because the prompt is built
        from a compiled timeline's durations.
        """
        provider = CueProvider(json.dumps({"cues": []}))
        client = _make_client(provider)
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            _suggest(client, analysed_project, plan["id"])

        assert provider.calls, "the provider was never called"
        prompt = provider.calls[0].system + provider.calls[0].user

        assert str(analysed_project) not in prompt
        assert "clip_0.mp4" not in prompt
        assert "storage_key" not in prompt
        assert "projects/" not in prompt

    def test_the_prompt_carries_the_timing_that_will_be_rendered(
        self, analysed_project: UUID
    ) -> None:
        provider = CueProvider(json.dumps({"cues": []}))
        client = _make_client(provider)
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            _suggest(client, analysed_project, plan["id"])

        assert str(plan["total_duration_ms"]) in provider.calls[0].user

    def test_a_model_that_answers_with_prose_produces_no_subtitles(
        self, analysed_project: UUID
    ) -> None:
        """Nothing is invented to fill the gap. The point of the feature."""
        client = _make_client(CueProvider("I am sorry, I cannot help with that."))
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            body = _suggest(client, analysed_project, plan["id"])

        assert body["ok"] is False
        assert body["failure"] == "no_usable_cues"
        assert body["subtitles"] is None

    def test_a_hostile_answer_reaches_the_editor_as_display_text(
        self, analysed_project: UUID
    ) -> None:
        hostile = {
            "cues": [
                {
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "text": "x; drawtext=text=y:fontfile=/etc/passwd",
                    "font": "C:\\Windows\\Fonts\\evil.ttf",
                    "position": {"x": -9999, "y": -9999},
                }
            ],
            "style": "made_up",
            "filter_complex": "scale=1:1",
        }
        client = _make_client(CueProvider(json.dumps(hostile)))
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            body = _suggest(client, analysed_project, plan["id"])

        track = body["subtitles"]
        assert track is not None
        # The style is the server's, the cue is three fields, and the text has
        # lost every character that means something to a document format.
        assert track["style"] in {"clean", "bold", "minimal", "cinematic", "social"}
        assert set(track["cues"][0]) == {"start_ms", "end_ms", "text"}
        assert "\\" not in track["cues"][0]["text"]
        # Kept as *words*: it is display text, not an instruction that was
        # neutralised. Nothing downstream parses it.
        assert "fontfile" in track["cues"][0]["text"]

    def test_the_route_writes_nothing(self, analysed_project: UUID) -> None:
        """A model proposes; a person accepts. The stored plan is untouched."""
        client = _make_client(
            CueProvider(json.dumps({"cues": [{"start_ms": 0, "end_ms": 2_000, "text": "Hi"}]}))
        )
        with client:
            plan = _plan(client, analysed_project, mode="rules")
            _suggest(client, analysed_project, plan["id"])
            stored = client.get(f"/api/projects/{analysed_project}/edit-plan/{plan['id']}")

        assert stored.status_code == 200
        assert stored.json()["plan"].get("subtitles") is None

    def test_the_preset_follows_the_plans_style_unless_the_caller_chooses(
        self, analysed_project: UUID
    ) -> None:
        """Phase 8's reference influence, reaching subtitles as a preference."""
        client = _make_client(
            CueProvider(json.dumps({"cues": [{"start_ms": 0, "end_ms": 2_000, "text": "Hi"}]}))
        )
        with client:
            plan = _plan(client, analysed_project, mode="rules", style=EditStyle.CINEMATIC.value)
            inherited = _suggest(client, analysed_project, plan["id"])
            chosen = _suggest(client, analysed_project, plan["id"], style="social")

        assert inherited["subtitles"]["style"] == "cinematic"
        assert chosen["subtitles"]["style"] == "social"
