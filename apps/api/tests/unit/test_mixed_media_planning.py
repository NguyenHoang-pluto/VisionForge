"""Planning from a folder of photos and videos together (Phase 12).

Every planner must be able to place a still: held from zero, never slowed,
given a drift, and passing the same validator every plan passes.
"""

from __future__ import annotations

from dataclasses import replace

from tests.unit.editorial_fixtures import PROJECT, clip
from visionforge.domain.editorial_planner import EditorialPlanner
from visionforge.domain.editplan import MAX_STILL_MS, EditPlan, MediaFact, validate_plan
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.planner import PlanRequest, RulesEnginePlanner
from visionforge.domain.selection import Candidate
from visionforge.domain.stills import drifted, still_motion

_MOTIONS = {
    EffectKind.ZOOM_IN,
    EffectKind.ZOOM_OUT,
    EffectKind.PAN_LEFT,
    EffectKind.PAN_RIGHT,
    EffectKind.PAN_UP,
    EffectKind.PAN_DOWN,
}


def photo(index: int, *, portrait: bool = False) -> Candidate:
    """A still as ingest records one: no duration, no motion, one frame."""
    base = clip(index, motion=None, motion_spread=None, has_audio=False)
    return replace(
        base,
        kind=MediaKind.IMAGE,
        duration_ms=None,
        width=3000 if not portrait else 2000,
        height=2000 if not portrait else 3000,
        scene_count=None,
        mean_scene_ms=None,
        motion_peak_ms=None,
    )


def folder() -> list[Candidate]:
    """Photos and videos interleaved, the way a trip folder looks."""
    return [
        clip(0, motion=0.5),
        photo(1),
        photo(2, portrait=True),
        clip(3, motion=0.3),
        photo(4),
        clip(5, motion=0.6),
    ]


def facts(candidates: list[Candidate]) -> dict:
    return {
        candidate.media_id: MediaFact.from_media(
            media_id=candidate.media_id,
            project_id=PROJECT,
            kind=candidate.kind,
            status=MediaStatus.READY,
            duration_ms=candidate.duration_ms,
            width=candidate.width,
            height=candidate.height,
        )
        for candidate in candidates
    }


def request(**overrides: object) -> PlanRequest:
    values: dict[str, object] = {
        "project_id": PROJECT,
        "target_duration_ms": 20_000,
        "max_clips": 6,
        "min_clips": 2,
    }
    values.update(overrides)
    return PlanRequest(**values)  # type: ignore[arg-type]


def stills_in(plan: EditPlan, candidates: list[Candidate]) -> list:
    photos = {c.media_id for c in candidates if c.kind is MediaKind.IMAGE}
    return [segment for segment in plan.segments if segment.media_id in photos]


def assert_stills_are_held(plan: EditPlan, candidates: list[Candidate]) -> None:
    held = stills_in(plan, candidates)
    assert held, "no photo reached the edit"
    for segment in held:
        assert segment.source_in_ms == 0
        assert segment.duration_ms <= MAX_STILL_MS
        assert segment.speed == 1.0
        assert any(effect.kind in _MOTIONS for effect in segment.effects)


class TestEditorialEngine:
    def test_photos_and_videos_are_cut_together_and_validate(self) -> None:
        candidates = folder()
        outcome = EditorialPlanner().plan(request(), candidates)

        assert validate_plan(outcome.plan, facts(candidates)) == []
        assert_stills_are_held(outcome.plan, candidates)
        videos = {c.media_id for c in candidates if c.kind is MediaKind.VIDEO}
        assert any(segment.media_id in videos for segment in outcome.plan.segments)

    def test_a_folder_of_only_photos_is_a_slideshow(self) -> None:
        candidates = [photo(index) for index in range(5)]
        outcome = EditorialPlanner().plan(request(target_duration_ms=15_000), candidates)

        assert validate_plan(outcome.plan, facts(candidates)) == []
        assert_stills_are_held(outcome.plan, candidates)


class TestRulesEngine:
    def test_photos_and_videos_are_cut_together_and_validate(self) -> None:
        candidates = folder()
        outcome = RulesEnginePlanner().plan(request(), candidates)

        assert validate_plan(outcome.plan, facts(candidates)) == []
        assert_stills_are_held(outcome.plan, candidates)


class TestDrift:
    def test_neighbouring_stills_move_differently(self) -> None:
        kinds = [still_motion(position).kind for position in range(4)]
        assert len(set(kinds)) == 4

    def test_a_portrait_still_drifts_vertically(self) -> None:
        kinds = {still_motion(position, portrait=True).kind for position in range(4)}
        assert EffectKind.PAN_LEFT not in kinds and EffectKind.PAN_RIGHT not in kinds

    def test_a_still_that_already_moves_is_left_alone(self) -> None:
        own = (Effect(EffectKind.PAN_UP, 0.2),)
        assert drifted(own, 0) == own

    def test_a_speed_change_is_removed_from_a_still(self) -> None:
        result = drifted((Effect(EffectKind.SLOW_MOTION, 0.5),), 0)
        assert all(not effect.kind.changes_duration for effect in result)
