"""The editorial planner end to end, and the metrics that measure its output.

The tests here are the ones that matter for the phase's headline claim, and they
are deliberately phrased as things a sceptic would ask:

- *Is this just "pick N and concatenate"?* No: clip lengths vary, the order is
  not the score order, and clips are dropped for editorial rather than technical
  reasons.
- *Does the style actually change the edit?* Cinematic and social fast cut are
  compared directly on identical footage, and must differ in selection, length
  and treatment -- not merely in order.
- *Does it still produce a valid plan?* Every plan here goes through the same
  Phase 4 validator the renderer depends on.
"""

from __future__ import annotations

import pytest

from tests.unit.editorial_fixtures import (
    PROJECT,
    clip,
    football_project,
    gaming_project,
    nature_project,
)
from visionforge.domain.beats import BeatGrid
from visionforge.domain.editorial_planner import EditorialPlanner, editorial_of, roles_of
from visionforge.domain.editplan import (
    MAX_SEGMENTS,
    MIN_OUTPUT_MS,
    MediaFact,
    TransitionKind,
    validate_plan,
)
from visionforge.domain.metrics import METRIC_DIRECTIONS
from visionforge.domain.planner import NoUsableMediaError, PlanRequest
from visionforge.domain.story import PolicyId, StoryRole
from visionforge.domain.style import EditStyle
from visionforge.domain.variants import VariantId

PLANNER = EditorialPlanner()


def request(**overrides: object) -> PlanRequest:
    defaults: dict[str, object] = {
        "project_id": PROJECT,
        "target_duration_ms": 30_000,
        "max_clips": 10,
        "min_clips": 2,
    }
    defaults.update(overrides)
    return PlanRequest(**defaults)  # type: ignore[arg-type]


def facts(candidates: list) -> dict:
    return {
        candidate.media_id: MediaFact(
            media_id=candidate.media_id,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=candidate.duration_ms,
            width=candidate.width,
            height=candidate.height,
        )
        for candidate in candidates
    }


# ------------------------------------------------------------------- the plan
class TestPlanShape:
    def test_the_plan_passes_the_phase_four_validator(self) -> None:
        """Everything downstream is unchanged, so this is the only gate that
        decides whether the new engine can ship at all."""
        clips = football_project()
        outcome = PLANNER.plan(request(), clips)
        assert validate_plan(outcome.plan, facts(clips)) == []

    def test_clip_lengths_vary(self) -> None:
        """The single most visible difference from an even split."""
        outcome = PLANNER.plan(request(editorial_policy=PolicyId.FOOTBALL), football_project())
        lengths = [segment.duration_ms for segment in outcome.plan.segments]
        assert len(set(lengths)) > 1
        assert max(lengths) > min(lengths) * 1.3

    def test_the_target_duration_is_respected(self) -> None:
        outcome = PLANNER.plan(request(target_duration_ms=24_000), football_project())
        assert abs(outcome.plan.total_duration_ms - 24_000) <= 24_000 * 0.15

    def test_the_edit_never_exceeds_the_segment_ceiling(self) -> None:
        outcome = PLANNER.plan(request(max_clips=40), nature_project())
        assert len(outcome.plan.segments) <= MAX_SEGMENTS

    def test_orders_are_contiguous_from_zero(self) -> None:
        outcome = PLANNER.plan(request(), football_project())
        assert [s.order for s in outcome.plan.segments] == list(range(len(outcome.plan.segments)))

    def test_planning_is_deterministic(self) -> None:
        clips = football_project()
        first = PLANNER.plan(request(), clips).plan
        second = PLANNER.plan(request(), clips).plan
        assert first.as_payload()["segments"] == second.as_payload()["segments"]

    def test_too_little_usable_media_raises_rather_than_inventing(self) -> None:
        with pytest.raises(NoUsableMediaError):
            PLANNER.plan(request(min_clips=4), [clip(0, blur_score=10.0, contrast=5.0)])

    def test_a_project_of_one_clip_still_plans(self) -> None:
        outcome = PLANNER.plan(request(min_clips=1, target_duration_ms=5_000), [clip(0)])
        assert len(outcome.plan.segments) == 1
        assert outcome.plan.total_duration_ms >= MIN_OUTPUT_MS


# ----------------------------------------------------------------- editorial
class TestEditorialPayload:
    def test_the_plan_records_a_role_for_every_segment(self) -> None:
        outcome = PLANNER.plan(request(), football_project())
        roles = roles_of(outcome.plan)
        assert len(roles) == len(outcome.plan.segments)
        assert all(role in {member.value for member in StoryRole} for role in roles)

    def test_the_plan_records_every_metric_by_name(self) -> None:
        outcome = PLANNER.plan(request(), football_project())
        editorial = editorial_of(outcome.plan)
        assert editorial is not None
        assert set(editorial["metrics"]) == set(METRIC_DIRECTIONS)

    def test_no_metric_is_collapsed_into_a_single_score(self) -> None:
        """Eight numbers that can each be argued with, and no ninth that
        cannot."""
        editorial = editorial_of(PLANNER.plan(request(), football_project()).plan)
        assert editorial is not None
        assert "score" not in editorial["metrics"]
        assert "overall" not in editorial["metrics"]

    def test_an_unmeasurable_metric_says_so_rather_than_reporting_zero(self) -> None:
        editorial = editorial_of(PLANNER.plan(request(), football_project()).plan)
        assert editorial is not None
        beat = editorial["metrics"]["beat_alignment"]
        assert beat["value"] is None
        assert beat["unmeasurable"] == "not_requested"

    def test_the_versions_that_produced_the_edit_are_recorded(self) -> None:
        editorial = editorial_of(PLANNER.plan(request(), football_project()).plan)
        assert editorial is not None
        for key in ("signal_version", "events_version", "policy_version", "engine_version"):
            assert editorial[key]

    def test_the_fit_reports_what_the_footage_supported(self) -> None:
        editorial = editorial_of(PLANNER.plan(request(), football_project()).plan)
        assert editorial is not None
        fit = editorial["fit"]
        assert fit["achieved_clips"] <= fit["requested_clips"]
        assert fit["target_ms"] == 30_000

    def test_rejections_come_back_with_editorial_reasons(self) -> None:
        clips = [clip(index, subject=1, blur_score=560.0) for index in range(6)]
        clips += [clip(20 + index, subject=20 + index) for index in range(4)]
        outcome = PLANNER.plan(request(), clips)
        reasons = {rejection.reason.value for rejection in outcome.selection.rejected}
        assert "semantic_duplicate" in reasons

    def test_the_selection_reports_only_what_the_edit_used(self) -> None:
        """The ranker's "selected" meant "usable", which is true of it and not
        of the edit."""
        outcome = PLANNER.plan(request(), football_project())
        assert len(outcome.selection.selected) == len(outcome.plan.segments)


# ------------------------------------------------------- materially different
class TestStylesDiffer:
    """Section 9 of the phase brief, tested directly.

    The same footage under two policies must differ in *what was chosen and how
    long it is held*, not merely in the order of the same clips. Each assertion
    below fails if the system degenerates back into a shuffle.
    """

    #: One project, reused by every assertion in this class. Rebuilding it per
    #: call would give each plan different media ids and make "the same clips"
    #: unaskable.
    CLIPS = gaming_project()

    @classmethod
    def _plan(cls, policy: PolicyId):  # type: ignore[no-untyped-def]
        return PLANNER.plan(request(editorial_policy=policy, max_clips=12), list(cls.CLIPS)).plan

    def test_clip_lengths_differ_between_policies(self) -> None:
        cinematic = self._plan(PolicyId.CINEMATIC_TRAVEL)
        social = self._plan(PolicyId.SOCIAL)
        cinematic_mean = cinematic.total_duration_ms / len(cinematic.segments)
        social_mean = social.total_duration_ms / len(social.segments)
        assert cinematic_mean > social_mean * 1.5

    def test_clip_counts_differ_between_policies(self) -> None:
        assert len(self._plan(PolicyId.SOCIAL).segments) > len(
            self._plan(PolicyId.CINEMATIC_TRAVEL).segments
        )

    def test_treatment_differs_between_policies(self) -> None:
        cinematic = self._plan(PolicyId.CINEMATIC_TRAVEL)
        social = self._plan(PolicyId.SOCIAL)
        assert any(s.transition_in is TransitionKind.CROSSFADE for s in cinematic.segments)
        assert all(s.transition_in is TransitionKind.CUT for s in social.segments)

    def test_the_two_edits_are_not_the_same_clips_reordered(self) -> None:
        """The assertion that rules out a shuffle."""
        cinematic = {s.media_id for s in self._plan(PolicyId.CINEMATIC_TRAVEL).segments}
        social = {s.media_id for s in self._plan(PolicyId.SOCIAL).segments}
        assert cinematic != social

    def test_a_nature_policy_holds_longer_than_a_gaming_one(self) -> None:
        nature = PLANNER.plan(
            request(editorial_policy=PolicyId.NATURE, max_clips=12), nature_project()
        ).plan
        gaming = PLANNER.plan(
            request(editorial_policy=PolicyId.GAMING, max_clips=12), nature_project()
        ).plan
        nature_mean = nature.total_duration_ms / len(nature.segments)
        gaming_mean = gaming.total_duration_ms / len(gaming.segments)
        assert nature_mean > gaming_mean

    def test_a_style_still_selects_a_policy_for_callers_that_do_not_name_one(self) -> None:
        outcome = PLANNER.plan(request(style=EditStyle.NATURE), nature_project())
        editorial = editorial_of(outcome.plan)
        assert editorial is not None
        assert editorial["policy"] == PolicyId.NATURE.value


# -------------------------------------------------------------------- variants
class TestVariants:
    CLIPS = gaming_project()

    @classmethod
    def _plan(cls, variant: VariantId | None):  # type: ignore[no-untyped-def]
        return PLANNER.plan(
            request(editorial_policy=PolicyId.FOOTBALL, max_clips=12, variant=variant),
            list(cls.CLIPS),
        ).plan

    def test_each_variant_produces_a_different_decision_set(self) -> None:
        signatures = {
            variant: tuple(
                (str(s.media_id), s.duration_ms, s.transition_in.value)
                for s in self._plan(variant).segments
            )
            for variant in (None, *VariantId)
        }
        assert len(set(signatures.values())) == len(signatures)

    def test_high_energy_cuts_shorter_than_cinematic(self) -> None:
        fast = self._plan(VariantId.HIGH_ENERGY)
        slow = self._plan(VariantId.CINEMATIC)
        assert fast.total_duration_ms / len(fast.segments) < slow.total_duration_ms / len(
            slow.segments
        )

    def test_a_variant_does_not_change_how_long_the_result_runs(self) -> None:
        """The user asked for a length. Returning a shorter video because they
        clicked "social" would answer a question they did not ask."""
        for variant in (None, *VariantId):
            plan = self._plan(variant)
            assert abs(plan.total_duration_ms - 30_000) <= 30_000 * 0.2

    def test_a_variant_is_reproducible(self) -> None:
        """What makes previewing variants without storing them safe."""
        first = self._plan(VariantId.CINEMATIC)
        second = self._plan(VariantId.CINEMATIC)
        assert first.as_payload()["segments"] == second.as_payload()["segments"]

    def test_variants_still_validate(self) -> None:
        clips = gaming_project()
        for variant in VariantId:
            plan = PLANNER.plan(
                request(editorial_policy=PolicyId.FOOTBALL, max_clips=12, variant=variant),
                clips,
            ).plan
            assert validate_plan(plan, facts(clips)) == []


# ---------------------------------------------------------------------- music
def grid(bpm: float = 120.0, confidence: float = 0.9) -> BeatGrid:
    period = 60_000.0 / bpm
    return BeatGrid(
        bpm=bpm,
        confidence=confidence,
        beats_ms=tuple(int(round(index * period)) for index in range(400)),
        source_duration_ms=180_000,
    )


class TestMusic:
    def test_beat_sync_is_off_unless_asked_for(self) -> None:
        outcome = PLANNER.plan(request(beats=grid()), football_project())
        editorial = editorial_of(outcome.plan)
        assert editorial is not None
        assert editorial["pacing"]["beat_sync"] is None

    def test_beat_sync_changes_the_lengths_when_asked_for(self) -> None:
        clips = gaming_project()
        plain = PLANNER.plan(request(max_clips=12), clips).plan
        synced = PLANNER.plan(request(max_clips=12, beats=grid(), beat_sync=True), clips).plan
        assert [s.duration_ms for s in plain.segments] != [s.duration_ms for s in synced.segments]

    def test_an_untrusted_grid_degrades_rather_than_re_timing(self) -> None:
        clips = gaming_project()
        plain = PLANNER.plan(request(max_clips=12), clips).plan
        weak = PLANNER.plan(
            request(max_clips=12, beats=grid(confidence=0.05), beat_sync=True), clips
        ).plan
        assert [s.duration_ms for s in plain.segments] == [s.duration_ms for s in weak.segments]

    def test_beat_alignment_becomes_measurable_once_synced(self) -> None:
        outcome = PLANNER.plan(
            request(max_clips=12, beats=grid(), beat_sync=True), gaming_project()
        )
        editorial = editorial_of(outcome.plan)
        assert editorial is not None
        assert editorial["metrics"]["beat_alignment"]["value"] is not None

    def test_a_music_cue_is_written_when_a_track_is_named(self) -> None:
        clips = football_project()
        music_id = clip(99).media_id
        outcome = PLANNER.plan(request(music_media_id=music_id, music_duration_ms=180_000), clips)
        assert outcome.plan.music is not None
        assert outcome.plan.music.media_id == music_id
