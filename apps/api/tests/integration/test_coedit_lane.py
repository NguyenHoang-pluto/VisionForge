"""The co-editor against real Postgres: patch, version, undo, redo, reject.

The unit tests cover the delta with no database and the routes with the service
faked out. This is the one that proves the halves agree -- real plan rows, real
version rows, the real partial unique index that keeps one head per project, and
the real validator running against real media metadata.

No bytes and no FFmpeg. Patching a plan reads media *metadata* and writes a plan
row; rendering the result is the existing lane and is covered there and in the
acceptance script.
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

CLIP_MS = 4_000
SOURCE_MS = 20_000


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def project(client: TestClient) -> Iterator[dict[str, Any]]:
    response = client.post("/api/projects", json={"title": "Co-edit Lane"})
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
    kind: MediaKind = MediaKind.VIDEO,
    duration_ms: int = SOURCE_MS,
    name: str = "clip.mp4",
) -> UUID:
    media = MediaAsset(
        id=uuid.uuid4(),
        project_id=project_id,
        original_filename=name,
        storage_key=f"projects/{project_id}/media/{uuid.uuid4()}/original.mp4",
        kind=kind,
        status=MediaStatus.READY,
        bytes_size=1024,
        mime_type="video/mp4" if kind is MediaKind.VIDEO else "audio/mpeg",
        duration_ms=duration_ms,
        width=1920 if kind is MediaKind.VIDEO else None,
        height=1080 if kind is MediaKind.VIDEO else None,
        fps=25.0 if kind is MediaKind.VIDEO else None,
        codec="h264" if kind is MediaKind.VIDEO else "mp3",
        channels=2,
    )
    session.add(media)
    session.commit()
    return media.id


def seed_quality(session: Session, media_id: UUID, project_id: UUID) -> None:
    """The analysis the selector needs to consider a clip usable at all.

    Only the planner path reads this: a hand-cut timeline names its own clips
    and never goes through selection. Values are comfortably above every
    usability floor, because what is under test here is versioning, not the
    Phase 3 gates.
    """
    session.add(
        MediaAnalysis(
            id=uuid.uuid4(),
            media_id=media_id,
            project_id=project_id,
            analyzer=AnalyzerName.QUALITY,
            analyzer_version="1",
            status=AnalysisStatus.OK,
            payload={
                "blur_score": 240.0,
                "contrast": 58.0,
                "mean_luminance": 122.0,
                "frames": [{"underexposed_ratio": 0.01, "overexposed_ratio": 0.01}],
            },
        )
    )
    session.commit()


@pytest.fixture
def media(project: dict[str, Any]) -> dict[str, Any]:
    """Three clips and a music track, with no bytes behind them."""
    with get_sync_sessionmaker()() as session:
        project_id = UUID(project["id"])
        return {
            "clips": [
                seed_media(session, project_id, name=f"clip{index}.mp4") for index in range(3)
            ],
            "music": seed_media(session, project_id, kind=MediaKind.AUDIO, name="bed.mp3"),
        }


def base_timeline(media: dict[str, Any], *, with_music: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "segments": [
            {"media_id": str(media_id), "source_in_ms": 0, "source_out_ms": CLIP_MS}
            for media_id in media["clips"]
        ],
        "aspect_ratio": "16:9",
        "fps": 30,
        "quality": "draft",
        "audio": "none",
        "subtitles": {
            "style": "clean",
            "position": "bottom",
            "cues": [
                {"start_ms": 200, "end_ms": 2_200, "text": "First line"},
                {"start_ms": 4_500, "end_ms": 6_500, "text": "Second line"},
            ],
        },
    }
    if with_music:
        body["music"] = {
            "media_id": str(media["music"]),
            "source_in_ms": 0,
            "source_out_ms": 12_000,
            "timeline_start_ms": 0,
            "volume": 0.7,
            "fade_in_ms": 0,
            "fade_out_ms": 1_500,
        }
    return body


@pytest.fixture
def edit(client: TestClient, project: dict[str, Any], media: dict[str, Any]) -> dict[str, Any]:
    """A stored edit, which is version 1 of the project's history."""
    response = client.post(
        f"/api/projects/{project['id']}/edit-plan/manual", json=base_timeline(media)
    )
    assert response.status_code == 201, response.text
    return response.json()


def url(project: dict[str, Any], suffix: str = "") -> str:
    return f"/api/projects/{project['id']}/edit-plan{suffix}"


def versions(client: TestClient, project: dict[str, Any]) -> dict[str, Any]:
    response = client.get(url(project, "/versions"))
    assert response.status_code == 200, response.text
    return response.json()


def co_edit(client: TestClient, project: dict[str, Any], **body: Any) -> Any:
    return client.post(url(project, "/co-edit"), json=body)


# ----------------------------------------------------------- the first version
def test_a_stored_edit_becomes_version_one(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    history = versions(client, project)

    assert history["total"] == 1
    first = history["items"][0]
    assert first["version"] == 1
    assert first["origin"] == "manual"
    assert first["is_current"] is True
    assert first["edit_plan_id"] == edit["id"]
    assert history["can_undo"] is False
    assert history["can_redo"] is False


# ------------------------------------------------------------------ patching
def test_a_deterministic_change_creates_a_new_version(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = co_edit(client, project, request_text="lower the music to 40%")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"]["version"] == 2
    assert body["source"] == "rules"
    assert body["diff"]["applied"] == ["Music volume to 40%"]

    # The new plan is a *different* row: plans stay append-only.
    assert body["plan"]["id"] != edit["id"]
    assert body["plan"]["plan"]["music"]["gain"] == 0.4
    assert body["plan"]["planner"] == "co-editor"

    # And the original is untouched.
    original = client.get(url(project, f"/{edit['id']}")).json()
    assert original["plan"]["music"]["gain"] == 0.7


def test_the_original_plan_is_byte_identical_after_a_patch(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    before = client.get(url(project, f"/{edit['id']}")).json()["plan"]
    co_edit(client, project, request_text="remove the third clip")
    after = client.get(url(project, f"/{edit['id']}")).json()["plan"]
    assert before == after


def test_two_changes_in_one_request(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = co_edit(
        client, project, request_text="Remove the third clip and use bold subtitles."
    )

    assert response.status_code == 201, response.text
    document = response.json()["plan"]["plan"]
    assert len(document["segments"]) == 2
    assert document["subtitles"]["style"] == "bold"


def test_changes_compose_across_versions(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """Each change patches whatever is current, not the plan it started from."""
    co_edit(client, project, request_text="lower the music to 40%")
    second = co_edit(client, project, request_text="use bold subtitles")

    assert second.status_code == 201, second.text
    document = second.json()["plan"]["plan"]
    assert document["music"]["gain"] == 0.4
    assert document["subtitles"]["style"] == "bold"
    assert second.json()["version"]["version"] == 3


# ------------------------------------------------------------------- preview
def test_preview_writes_nothing(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = client.post(
        url(project, "/co-edit/preview"), json={"request_text": "lower the music to 30%"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    entry = next(item for item in body["diff"]["entries"] if item["field"] == "music_gain")
    assert (entry["before"], entry["after"]) == ("70%", "30%")

    # Still one version, and the plan still has its original volume.
    assert versions(client, project)["total"] == 1
    assert client.get(url(project, f"/{edit['id']}")).json()["plan"]["music"]["gain"] == 0.7


def test_previewed_operations_can_be_applied_verbatim(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """Apply takes the operations back and re-validates them from scratch."""
    preview = client.post(
        url(project, "/co-edit/preview"), json={"request_text": "remove the third clip"}
    ).json()

    response = co_edit(
        client,
        project,
        operations=preview["operations"],
        base_version_id=preview["base_version_id"],
    )
    assert response.status_code == 201, response.text
    assert response.json()["source"] == "client"
    assert len(response.json()["plan"]["plan"]["segments"]) == 2


def test_applying_against_a_stale_version_is_refused(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """A change approved against version 1 must not land on version 2."""
    preview = client.post(
        url(project, "/co-edit/preview"), json={"request_text": "remove the third clip"}
    ).json()

    # Somebody else changes the edit in between.
    co_edit(client, project, request_text="lower the music to 40%")

    response = co_edit(
        client,
        project,
        operations=preview["operations"],
        base_version_id=preview["base_version_id"],
    )
    assert response.status_code == 409, response.text
    assert versions(client, project)["total"] == 2


# ------------------------------------------------------------------ rejection
def test_an_impossible_change_leaves_the_edit_untouched(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = co_edit(client, project, operations=[{"kind": "REMOVE_SEGMENT", "segment": 9}])

    assert response.status_code == 422, response.text
    # Addresses are zero-based; the message the user reads counts from one.
    assert "no clip 10" in response.json()["error"]["hint"]
    # No version, no plan, no change.
    history = versions(client, project)
    assert history["total"] == 1
    assert history["items"][0]["edit_plan_id"] == edit["id"]


def test_a_partially_valid_change_applies_none_of_it(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = co_edit(
        client,
        project,
        operations=[
            {"kind": "CHANGE_MUSIC_VOLUME", "value": 0.2},
            {"kind": "REMOVE_SEGMENT", "segment": 7},
        ],
    )

    assert response.status_code == 422
    assert versions(client, project)["total"] == 1
    assert client.get(url(project, f"/{edit['id']}")).json()["plan"]["music"]["gain"] == 0.7


def test_an_unknown_operation_never_reaches_the_database(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = co_edit(
        client, project, operations=[{"kind": "RUN_FFMPEG", "args": "-i /etc/passwd"}]
    )
    assert response.status_code == 422
    assert versions(client, project)["total"] == 1


def test_a_change_that_would_break_the_plan_is_refused(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """Every operation is legal; the plan they would produce is not."""
    response = co_edit(
        client,
        project,
        operations=[
            {"kind": "REMOVE_SEGMENT", "segment": 1},
            {"kind": "REMOVE_SEGMENT", "segment": 2},
            {"kind": "TRIM_SEGMENT", "segment": 0, "source_out_ms": 400},
        ],
    )
    assert response.status_code == 422, response.text
    assert versions(client, project)["total"] == 1


def test_a_trim_past_the_real_source_is_refused(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """Source length comes from the database, never from the request."""
    response = co_edit(
        client,
        project,
        operations=[
            {"kind": "TRIM_SEGMENT", "segment": 0, "source_in_ms": 0, "source_out_ms": 25_000}
        ],
    )
    assert response.status_code == 422, response.text
    assert versions(client, project)["total"] == 1


def test_a_request_the_rules_cannot_read_fails_without_a_provider(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """No provider configured in tests, so a creative request is a clean refusal."""
    response = co_edit(client, project, request_text="make it feel like a summer blockbuster")

    assert response.status_code == 422, response.text
    assert versions(client, project)["total"] == 1
    assert client.get(url(project, f"/{edit['id']}")).json()["plan"]["music"]["gain"] == 0.7


# -------------------------------------------------------------- cross-project
def test_a_change_cannot_reach_another_projects_edit(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any], media: dict[str, Any]
) -> None:
    """The co-edit routes resolve through the project, like every other route."""
    other = client.post("/api/projects", json={"title": "Somebody else"}).json()
    try:
        # The other project has no edit at all, so there is nothing to patch --
        # and critically, this project's version is not what it finds.
        response = client.post(
            f"/api/projects/{other['id']}/edit-plan/co-edit",
            json={"request_text": "lower the music to 40%"},
        )
        assert response.status_code == 404, response.text

        history = client.get(f"/api/projects/{other['id']}/edit-plan/versions").json()
        assert history["total"] == 0
    finally:
        with get_sync_sessionmaker()() as session:
            session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": UUID(other["id"])})
            session.commit()


def test_an_unknown_project_is_a_404(client: TestClient) -> None:
    response = client.post(
        f"/api/projects/{uuid.uuid4()}/edit-plan/co-edit",
        json={"request_text": "lower the music to 40%"},
    )
    assert response.status_code == 404


# ------------------------------------------------------------------ undo/redo
def test_undo_restores_the_previous_plan_exactly(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    before = client.get(url(project, f"/{edit['id']}")).json()["plan"]
    co_edit(client, project, request_text="lower the music to 40%")

    response = client.post(url(project, "/versions/undo"))
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["version"] == 1
    assert restored["edit_plan_id"] == edit["id"]

    # The exact bytes, not a recomputation of them.
    assert client.get(url(project, f"/{restored['edit_plan_id']}")).json()["plan"] == before


def test_redo_moves_forward_again(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    applied = co_edit(client, project, request_text="lower the music to 40%").json()
    client.post(url(project, "/versions/undo"))

    response = client.post(url(project, "/versions/redo"))
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 2
    assert response.json()["edit_plan_id"] == applied["plan"]["id"]


def test_undo_at_the_first_version_is_refused(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    response = client.post(url(project, "/versions/undo"))
    assert response.status_code == 409, response.text
    assert versions(client, project)["items"][0]["version"] == 1


def test_redo_at_the_tip_is_refused(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    co_edit(client, project, request_text="lower the music to 40%")
    response = client.post(url(project, "/versions/redo"))
    assert response.status_code == 409


def test_undo_then_edit_branches_and_redo_follows_the_new_branch(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    co_edit(client, project, request_text="lower the music to 40%")  # v2
    client.post(url(project, "/versions/undo"))  # back to v1
    branched = co_edit(client, project, request_text="use bold subtitles")  # v3, child of v1

    assert branched.status_code == 201, branched.text
    assert branched.json()["version"]["version"] == 3

    history = versions(client, project)
    assert history["total"] == 3
    assert history["current_version_id"] == branched.json()["version"]["id"]
    # The abandoned branch is still in the history; it is simply not current.
    assert {item["version"] for item in history["items"]} == {1, 2, 3}

    # And the branch that was abandoned is still reachable by undo, which is
    # what makes branching non-destructive.
    client.post(url(project, "/versions/undo"))
    assert versions(client, project)["items"][-1]["version"] == 1


def test_the_undo_flags_track_the_real_history(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    assert versions(client, project)["can_undo"] is False

    co_edit(client, project, request_text="lower the music to 40%")
    assert versions(client, project)["can_undo"] is True
    assert versions(client, project)["can_redo"] is False

    client.post(url(project, "/versions/undo"))
    state = versions(client, project)
    assert state["can_undo"] is False
    assert state["can_redo"] is True


def test_restore_jumps_to_any_version(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    co_edit(client, project, request_text="lower the music to 40%")
    co_edit(client, project, request_text="use bold subtitles")

    first = next(item for item in versions(client, project)["items"] if item["version"] == 1)
    response = client.post(url(project, f"/versions/{first['id']}/restore"))

    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1
    assert versions(client, project)["current_version_id"] == first["id"]


def test_only_one_version_is_ever_current(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    """The partial unique index, exercised through the routes that move the head."""
    co_edit(client, project, request_text="lower the music to 40%")
    co_edit(client, project, request_text="use bold subtitles")
    client.post(url(project, "/versions/undo"))
    client.post(url(project, "/versions/redo"))

    with get_sync_sessionmaker()() as session:
        count = session.execute(
            text(
                "SELECT count(*) FROM edit_plan_versions " "WHERE project_id = :id AND is_current"
            ),
            {"id": UUID(project["id"])},
        ).scalar_one()
    assert count == 1


# ---------------------------------------------------- manual and AI together
def test_a_hand_cut_edit_after_an_ai_change_continues_the_same_history(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any], media: dict[str, Any]
) -> None:
    """The hybrid workflow: AI, then manual, then AI again, one history."""
    co_edit(client, project, request_text="lower the music to 40%")

    manual = base_timeline(media)
    manual["segments"] = manual["segments"][:2]
    stored = client.post(url(project, "/manual"), json=manual)
    assert stored.status_code == 201, stored.text

    history = versions(client, project)
    assert [item["origin"] for item in history["items"]] == ["manual", "co_edit", "manual"]
    assert history["items"][0]["is_current"] is True

    # And the AI now patches the hand-cut edit, not the one before it.
    again = co_edit(client, project, request_text="use bold subtitles")
    assert again.status_code == 201, again.text
    assert len(again.json()["plan"]["plan"]["segments"]) == 2


def test_the_history_records_the_digest_and_never_the_request(
    client: TestClient, project: dict[str, Any], edit: dict[str, Any]
) -> None:
    secret = "lower the music to 40%"
    co_edit(client, project, request_text=secret)

    history = versions(client, project)
    newest = history["items"][0]
    assert newest["request_digest"]
    assert secret not in str(history)

    with get_sync_sessionmaker()() as session:
        stored = session.execute(
            text("SELECT operations::text, summary::text FROM edit_plan_versions WHERE version = 2")
        ).first()
    assert stored is not None
    assert secret not in (stored[0] or "")
    assert secret not in (stored[1] or "")


# --------------------------------------------------------------- generated
def test_a_generated_plan_also_starts_a_history(
    client: TestClient, project: dict[str, Any], media: dict[str, Any]
) -> None:
    """A planned edit joins the same history a hand-cut one does."""
    with get_sync_sessionmaker()() as session:
        for media_id in media["clips"]:
            seed_quality(session, media_id, UUID(project["id"]))

    response = client.post(
        url(project),
        json={"mode": "rules", "max_clips": 3, "min_clips": 1, "target_duration_ms": 9_000},
    )
    assert response.status_code == 201, response.text

    history = versions(client, project)
    assert history["total"] == 1
    assert history["items"][0]["origin"] == "generated"
    assert history["items"][0]["edit_plan_id"] == response.json()["id"]
