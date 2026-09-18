"""The editorial engine against real Postgres, through the real router.

The unit tests prove the engine decides well over constructed candidates. This
proves the other half: that the analysis rows a real project actually contains
reach the engine, that its decisions survive being written to JSONB and read
back, and that the variants route returns four different edits of one library
without storing any of them.

No bytes and no FFmpeg. Analysis payloads are seeded directly, which is exactly
what the analyzers write -- the pipeline that produces them is Phase 3's lane and
is covered there. What is under test here is the wiring between a row and a
decision.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from visionforge.api.main import create_app
from visionforge.domain.analysis import AnalysisStatus, AnalyzerName
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.infra.db.models import MediaAnalysis, MediaAsset
from visionforge.infra.db.sync_session import get_sync_sessionmaker

pytestmark = pytest.mark.integration

#: Width of the seeded CLIP vectors. The column is 512-wide, so the fixtures
#: have to be too -- unlike the unit tests, which are free to use sixteen.
EMBEDDING_DIM = 512


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def project(client: TestClient) -> Iterator[dict[str, Any]]:
    response = client.post("/api/projects", json={"title": "Editorial Lane"})
    assert response.status_code == 201, response.text
    row = response.json()

    yield row

    with get_sync_sessionmaker()() as session:
        session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": UUID(row["id"])})
        session.commit()


def embedding(seed: int) -> list[float]:
    """A deterministic unit vector, distinct per seed.

    A linear congruential sequence rather than a trigonometric one, for the
    reason the unit fixtures record: ``sin`` over a shared phase makes
    consecutive seeds far more alike than two real clips are, and fixtures that
    accidentally encode "everything is a duplicate" make the diversity
    assertions pass for the wrong reason.
    """
    state = (seed + 1) * 2_654_435_761 % 2**32
    raw: list[float] = []
    for _ in range(EMBEDDING_DIM):
        state = (state * 1_664_525 + 1_013_904_223) % 2**32
        raw.append(state / 2**31 - 1.0)
    norm = sum(value * value for value in raw) ** 0.5 or 1.0
    return [value / norm for value in raw]


def seed_clip(
    session: Session,
    project_id: UUID,
    *,
    index: int,
    subject: int,
    motion: float,
    duration_ms: int = 8_000,
    blur: float = 320.0,
    faces: int = 0,
) -> UUID:
    """One analysed video: a media row plus the four analyzer rows the engine reads.

    Written exactly as the analyzers write them -- ``quality`` with its frame
    list, ``dynamics`` with its samples, ``faces`` with normalised boxes, and
    ``clip`` with its vector in the typed column rather than in the payload.
    Seeding a simplified shape here would test a pipeline this product does not
    have.
    """
    media = MediaAsset(
        id=uuid.uuid4(),
        project_id=project_id,
        original_filename=f"clip{index:02d}.mp4",
        storage_key=f"projects/{project_id}/media/{uuid.uuid4()}/original.mp4",
        kind=MediaKind.VIDEO,
        status=MediaStatus.READY,
        bytes_size=2048,
        mime_type="video/mp4",
        duration_ms=duration_ms,
        width=1280,
        height=720,
        fps=25.0,
        codec="h264",
        channels=2,
    )
    session.add(media)
    session.flush()

    stamps = [int(duration_ms * f) for f in (0.05, 0.25, 0.50, 0.75, 0.95)]
    rows = [
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media.id,
            project_id=project_id,
            analyzer=AnalyzerName.QUALITY,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            payload={
                "frames": [{"underexposed_ratio": 0.01, "overexposed_ratio": 0.01} for _ in stamps],
                "frame_count": len(stamps),
                "blur_score": blur,
                "contrast": 48.0,
                "mean_luminance": 128.0,
            },
        ),
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media.id,
            project_id=project_id,
            analyzer=AnalyzerName.PHASH,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            # Spread far apart so the pixel-level dedupe never fires: what this
            # lane is testing is the *semantic* gate, and a pHash collision
            # would remove the clips before the engine saw them.
            payload={"phash": f"{((index + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF:016x}"},
        ),
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media.id,
            project_id=project_id,
            analyzer=AnalyzerName.SCENES,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            payload={
                "scenes": [
                    {
                        "scene_id": 0,
                        "start_ms": 0,
                        "end_ms": duration_ms,
                        "duration_ms": duration_ms,
                    }
                ],
                "scene_count": 1,
                "mean_scene_ms": duration_ms,
            },
        ),
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media.id,
            project_id=project_id,
            analyzer=AnalyzerName.DYNAMICS,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            payload={
                "samples": [
                    {"timestamp_ms": stamp, "motion": motion, "saturation": 0.45}
                    for stamp in stamps
                ],
                "sample_count": len(stamps),
                "motion": motion,
                "motion_spread": 0.02,
                "saturation": 0.45,
                "motion_confidence": 1.0,
                "colour_confidence": 1.0,
            },
        ),
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media.id,
            project_id=project_id,
            analyzer=AnalyzerName.CLIP,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            payload={"dim": EMBEDDING_DIM, "normalized": True},
            embedding=embedding(subject),
        ),
    ]
    if faces:
        rows.append(
            MediaAnalysis(
                id=uuid.uuid4(),
                media_id=media.id,
                project_id=project_id,
                analyzer=AnalyzerName.FACES,
                analyzer_version="1",
                status=AnalysisStatus.OK,
                payload={
                    "face_count": faces,
                    "frame_count": len(stamps),
                    "frames": [
                        {
                            "timestamp_ms": stamp,
                            "face_count": faces,
                            "faces": [
                                {
                                    "x": 0.4,
                                    "y": 0.3,
                                    "width": 0.34,
                                    "height": 0.42,
                                    "confidence": 0.95,
                                }
                            ],
                        }
                        for stamp in stamps
                    ],
                    "identity_stored": False,
                },
            )
        )

    session.add_all(rows)
    session.commit()
    return media.id


@pytest.fixture
def library(project: dict[str, Any]) -> dict[str, Any]:
    """Sixteen analysed clips: a rising action curve, one repeated moment, faces.

    The same shape as the unit fixture, seeded through the database so the
    repository, the candidate builder and the engine are all exercised. Twelve
    distinct subjects rather than eight, because the claim that a fast policy
    cuts more shots than a slow one is untestable on a library that does not
    contain more shots than the slow one wants.
    """
    project_id = UUID(project["id"])
    with get_sync_sessionmaker()() as session:
        ids: list[UUID] = []
        # Twelve distinct clips, motion rising.
        for index in range(12):
            ids.append(
                seed_clip(
                    session,
                    project_id,
                    index=index,
                    subject=index,
                    motion=0.03 + index * 0.027,
                    duration_ms=7_000 + index * 300,
                )
            )
        # Three angles on one moment, technically excellent.
        repeated: list[UUID] = []
        for offset in range(3):
            repeated.append(
                seed_clip(
                    session,
                    project_id,
                    index=12 + offset,
                    subject=99,
                    motion=0.18,
                    blur=560.0,
                )
            )
        # A close-up with faces, calm.
        faces_id = seed_clip(session, project_id, index=15, subject=50, motion=0.04, faces=2)
    return {"all": ids + repeated + [faces_id], "repeated": repeated, "faces": faces_id}


def plan(client: TestClient, project_id: str, **body: Any) -> Any:
    payload = {"mode": "rules", "target_duration_ms": 30_000, "max_clips": 10, "min_clips": 2}
    payload.update(body)
    return client.post(f"/api/projects/{project_id}/edit-plan", json=payload)


# ------------------------------------------------------------------ the plan
class TestEditorialPlanning:
    def test_a_plan_carries_its_editorial_account(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        response = plan(client, project["id"], editorial_policy="football")
        assert response.status_code == 201, response.text
        body = response.json()

        assert body["planner"] == "editorial"
        editorial = body["editorial"]
        assert editorial is not None
        assert editorial["policy"] == "football"
        assert len(editorial["segments"]) == body["segment_count"]

    def test_every_segment_has_a_role_and_reasons(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        editorial = plan(client, project["id"], editorial_policy="football").json()["editorial"]
        roles = {"hook", "setup", "build", "peak", "reaction", "ending"}
        for segment in editorial["segments"]:
            assert segment["role"] in roles
            assert segment["reasons"]

    def test_the_repeated_moment_yields_at_most_two_clips(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """The failure this phase exists to remove, over real rows.

        Three technically excellent angles on one moment must not become three
        segments of the edit -- and the vector that proves they are one moment
        comes out of a pgvector column, which no unit test can exercise.
        """
        editorial = plan(client, project["id"], editorial_policy="football").json()["editorial"]
        repeated = {str(media_id) for media_id in library["repeated"]}
        used = [s for s in editorial["segments"] if s["media_id"] in repeated]
        assert len(used) <= 2

    def test_the_rejected_clips_say_what_they_repeat(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        editorial = plan(client, project["id"], editorial_policy="football").json()["editorial"]
        similar = [item for item in editorial["rejected"] if item.get("similar_to")]
        assert similar

    def test_clip_lengths_vary(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """The most visible difference from the even split this replaced."""
        body = plan(client, project["id"], editorial_policy="football").json()
        lengths = [segment["duration_ms"] for segment in body["plan"]["segments"]]
        assert len(set(lengths)) > 1

    def test_metrics_survive_the_round_trip_through_jsonb(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        created = plan(client, project["id"], editorial_policy="nature").json()
        detail = client.get(f"/api/projects/{project['id']}/edit-plan/{created['id']}")
        assert detail.status_code == 200
        metrics = detail.json()["editorial"]["metrics"]
        assert set(metrics) >= {"content_diversity", "repetition", "story_completeness"}
        assert metrics["beat_alignment"]["value"] is None

    def test_the_listing_omits_the_editorial_document(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """A few kilobytes per plan, times twenty plans, for a listing that
        shows none of it."""
        plan(client, project["id"], editorial_policy="football")
        listing = client.get(f"/api/projects/{project['id']}/edit-plan").json()
        assert listing["items"]
        assert all(item.get("editorial") is None for item in listing["items"])

    def test_two_policies_produce_materially_different_edits(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """Section 9 of the brief, over the same real library."""
        cinematic = plan(
            client, project["id"], editorial_policy="cinematic_travel", max_clips=16
        ).json()
        social = plan(client, project["id"], editorial_policy="social", max_clips=16).json()

        assert cinematic["segment_count"] != social["segment_count"]
        cinematic_mean = cinematic["total_duration_ms"] / cinematic["segment_count"]
        social_mean = social["total_duration_ms"] / social["segment_count"]
        assert cinematic_mean > social_mean * 1.5

        chosen_c = {s["media_id"] for s in cinematic["plan"]["segments"]}
        chosen_s = {s["media_id"] for s in social["plan"]["segments"]}
        assert chosen_c != chosen_s

    def test_a_plan_is_still_renderable(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """The validator ran server-side; a render row can be created from it.

        The render itself needs a worker and is Phase 4's lane -- what matters
        here is that nothing about the editorial plan makes it unacceptable to
        the route the renderer is dispatched from.
        """
        created = plan(client, project["id"], editorial_policy="football").json()
        response = client.post(
            f"/api/projects/{project['id']}/render",
            json={"edit_plan_id": created["id"]},
        )
        assert response.status_code in (201, 202), response.text


# -------------------------------------------------------------- engine choice
class TestEngineSelection:
    """Both generations are reachable, and they are not the same engine.

    The editorial engine is the default, so every other test in this file gets
    it without asking. What is under test here is the escape hatch: that a
    caller can still ask for the Phase 4 rules engine by name, that doing so
    really does bypass the editorial stages rather than relabelling them, and
    that the default has not quietly become the old behaviour again.
    """

    def test_the_default_engine_is_editorial(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        body = plan(client, project["id"], editorial_policy="football").json()

        assert body["planner"] == "editorial"
        assert body["editorial"] is not None

    def test_the_rules_engine_can_still_be_asked_for(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        body = plan(client, project["id"], engine="rules").json()

        assert body["planner"] == "rules-engine"
        # No roles, no pacing, no reasons: the Phase 4 engine has none of that
        # to report, and reporting an empty account would be worse than None.
        assert body["editorial"] is None

    def test_the_two_engines_cut_the_same_footage_differently(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """The claim Phase 11 exists to make, at the level of one request.

        An even split gives every shot the same length. The editorial engine
        derives lengths from an energy curve, so a plan whose durations are all
        equal would mean the curve never reached the plan.
        """
        editorial = plan(client, project["id"], editorial_policy="football").json()
        rules = plan(client, project["id"], engine="rules").json()

        rules_durations = {s["duration_ms"] for s in rules["plan"]["segments"]}
        editorial_durations = {s["duration_ms"] for s in editorial["plan"]["segments"]}

        assert len(rules_durations) == 1, "the Phase 4 engine splits the budget evenly"
        assert len(editorial_durations) > 1, "the editorial engine paces the edit"

    def test_an_invented_engine_is_refused(
        self, client: TestClient, project: dict[str, Any]
    ) -> None:
        response = plan(client, project["id"], engine="whatever")
        assert response.status_code == 422


# ------------------------------------------------------------------ variants
class TestVariantPreview:
    def test_four_alternatives_come_back(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/edit-plan/variants",
            json={
                "mode": "rules",
                "target_duration_ms": 30_000,
                "max_clips": 10,
                "editorial_policy": "football",
            },
        )
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert next(item["variant"] for item in items) is None
        assert {item["variant"] for item in items} >= {
            None,
            "high_energy",
            "cinematic",
            "social_fast_cut",
        }

    def test_the_variants_differ_from_each_other(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        items = client.post(
            f"/api/projects/{project['id']}/edit-plan/variants",
            json={
                "mode": "rules",
                "target_duration_ms": 30_000,
                "max_clips": 10,
                "editorial_policy": "football",
            },
        ).json()["items"]

        signatures = {
            tuple((s["media_id"], s["duration_ms"]) for s in item["editorial"]["segments"])
            for item in items
        }
        assert len(signatures) == len(items)

    def test_nothing_is_stored(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """Three plan rows and three versions per click, two of which nobody
        looks at again, is the cost this route exists to avoid."""
        before = client.get(f"/api/projects/{project['id']}/edit-plan").json()["total"]
        client.post(
            f"/api/projects/{project['id']}/edit-plan/variants",
            json={"mode": "rules", "target_duration_ms": 30_000, "max_clips": 10},
        )
        after = client.get(f"/api/projects/{project['id']}/edit-plan").json()["total"]
        assert after == before

    def test_choosing_a_variant_reproduces_what_was_previewed(
        self, client: TestClient, project: dict[str, Any], library: dict[str, Any]
    ) -> None:
        """What makes previewing without storing safe."""
        body = {
            "mode": "rules",
            "target_duration_ms": 30_000,
            "max_clips": 10,
            "editorial_policy": "football",
        }
        preview = next(
            item
            for item in client.post(
                f"/api/projects/{project['id']}/edit-plan/variants", json=body
            ).json()["items"]
            if item["variant"] == "cinematic"
        )

        created = plan(client, project["id"], **{**body, "variant": "cinematic"}).json()
        assert [(s["media_id"], s["duration_ms"]) for s in created["editorial"]["segments"]] == [
            (s["media_id"], s["duration_ms"]) for s in preview["editorial"]["segments"]
        ]

    def test_a_project_with_no_analysis_is_refused_clearly(
        self, client: TestClient, project: dict[str, Any]
    ) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/edit-plan/variants",
            json={"mode": "rules", "target_duration_ms": 20_000},
        )
        assert response.status_code == 422
        assert "analysed media" in response.text


# -------------------------------------------------------------- capabilities
class TestCapabilities:
    def test_the_editorial_vocabulary_is_declared(self, client: TestClient) -> None:
        """The editor offers what the server has, and nothing else."""
        body = client.get("/api/planner/capabilities").json()
        assert {p["id"] for p in body["editorial_policies"]} >= {
            "football",
            "gaming",
            "nature",
            "cinematic_travel",
            "social",
        }
        assert {v["id"] for v in body["variants"]} == {
            "high_energy",
            "cinematic",
            "social_fast_cut",
        }
        assert body["story_roles"] == ["hook", "setup", "build", "peak", "reaction", "ending"]
        assert body["editorial_version"]

    def test_metric_directions_are_declared(self, client: TestClient) -> None:
        body = client.get("/api/planner/capabilities").json()
        directions = {m["name"]: m["direction"] for m in body["editorial_metrics"]}
        assert directions["repetition"] == "lower_is_better"
        assert directions["content_diversity"] == "higher_is_better"
