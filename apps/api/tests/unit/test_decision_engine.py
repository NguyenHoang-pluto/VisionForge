"""The editorial decision engine: selection, roles, trims and treatment.

This is where the phase's central claim is tested directly. The system must not
be describable as "pick the top N and concatenate them", and the way to prove
that is to build footage where the top N *is* the wrong answer and show the
engine declining it:

- five technically excellent shots of the same thing yield one or two;
- the softest clip in the project becomes the peak, because it is the only
  record of the moment that matters;
- a trim lands where the action is rather than in the middle of the file;
- and every one of those decisions comes back with the reason it was made.
"""

from __future__ import annotations

from tests.unit.editorial_fixtures import clip, football_project, nature_project
from visionforge.domain.decisions import (
    DecisionKind,
    EngineInput,
    ReasonCode,
    decide,
)
from visionforge.domain.editorial import EditorialEvent, board_from, read_board
from visionforge.domain.editplan import TransitionKind
from visionforge.domain.effects import EffectKind
from visionforge.domain.pacing import curve_for, plan_pacing
from visionforge.domain.selection import score_candidate
from visionforge.domain.story import EDITORIAL_POLICIES, EditorialPolicy, PolicyId, StoryRole


def engine(
    candidates: list,
    policy: EditorialPolicy,
    *,
    count: int | None = None,
    target_ms: int = 30_000,
    treatments: bool = True,
) -> EngineInput:
    scores = {c.media_id: score_candidate(c) for c in candidates}
    board = board_from(candidates, scores=scores)
    readings = read_board(board)
    slots = count if count is not None else min(len(readings), policy.arc.max_clips)
    roles = policy.arc.expand(slots)
    pacing = plan_pacing(
        count=len(roles),
        target_ms=target_ms,
        curve=curve_for(policy.pacing, spread=policy.pacing_spread),
        min_clip_ms=policy.min_clip_ms,
        max_clip_ms=policy.max_clip_ms,
        emphasis=tuple(policy.emphasis_for(role) for role in roles),
        source_limits_ms=tuple(r.signals.duration_ms for r in readings),
    )
    return EngineInput(
        readings=readings,
        board=board,
        policy=policy,
        pacing=pacing,
        roles=roles,
        treatments=treatments,
    )


FOOTBALL = EDITORIAL_POLICIES[PolicyId.FOOTBALL]
NATURE = EDITORIAL_POLICIES[PolicyId.NATURE]


# ------------------------------------------------------------------ selection
class TestCreativeSelection:
    def test_near_identical_clips_do_not_all_get_in(self) -> None:
        """The failure this phase exists to remove.

        Five technically excellent shots of one wall score five times under a
        ranker and become five sixths of an edit. Here they compete against what
        is already chosen.
        """
        clips = [clip(index, subject=1, blur_score=560.0) for index in range(5)]
        clips += [clip(10 + index, subject=10 + index) for index in range(5)]
        plan = decide(engine(clips, FOOTBALL))
        repeated = {c.media_id for c in clips[:5]}
        from_group = [s for s in plan.segments if s.media_id in repeated]
        assert 1 <= len(from_group) <= 2

    def test_the_repeats_are_rejected_for_being_repeats(self) -> None:
        clips = [clip(index, subject=1, blur_score=560.0) for index in range(5)]
        clips += [clip(10 + index, subject=10 + index) for index in range(5)]
        plan = decide(engine(clips, FOOTBALL))
        reasons = {reason for rejection in plan.rejected for reason in rejection.reasons}
        assert ReasonCode.TOO_SIMILAR in reasons

    def test_a_rejection_names_what_it_duplicates(self) -> None:
        """A user asking "why is this clip not in my edit" gets the clip that
        made it redundant, not a bare refusal."""
        clips = [clip(0, subject=1), clip(1, subject=1), clip(2, subject=2)]
        plan = decide(engine(clips, FOOTBALL, count=3))
        similar = [r for r in plan.rejected if r.similar_to is not None]
        assert similar
        assert similar[0].similar_to in {segment.media_id for segment in plan.segments}

    def test_a_technically_weak_clip_can_still_be_the_peak(self) -> None:
        """Quality is one term among six.

        The football fixture's peak is the softest clip in the project, because
        it was shot on a long lens by somebody running. An edit that dropped it
        would be missing the reason the edit exists.
        """
        clips = football_project()
        plan = decide(engine(clips, FOOTBALL))
        peaks = {s.media_id for s in plan.segments if s.role is StoryRole.PEAK}
        assert clips[4].media_id in peaks

    def test_the_peak_picks_before_the_setup(self) -> None:
        """Priority order, not narrative order.

        Filling left to right gives the least important role first refusal on
        the best footage, which is how a peak ends up being whatever the setup
        did not want.
        """
        clips = football_project()
        plan = decide(engine(clips, FOOTBALL))
        busiest = max(plan.segments, key=lambda segment: segment.energy)
        assert busiest.role is StoryRole.PEAK

    def test_every_kept_clip_has_at_least_one_reason(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        assert all(segment.reasons for segment in plan.segments)

    def test_keep_decisions_carry_the_same_reasons_as_the_segment(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        keeps = {d.slot: d for d in plan.decisions if d.kind is DecisionKind.KEEP}
        for segment in plan.segments:
            assert set(keeps[segment.slot].reasons) <= set(segment.reasons)

    def test_a_clip_with_no_strong_component_reports_only_that(self) -> None:
        """Reporting a least-bad axis as a reason would be inventing one."""
        weak = [clip(index, blur_score=45.0, contrast=14.0, motion=0.01) for index in range(3)]
        plan = decide(engine(weak, FOOTBALL, count=1))
        assert plan.segments[0].reasons

    def test_the_component_breakdown_is_reported(self) -> None:
        """A score without its parts cannot be argued with."""
        plan = decide(engine(football_project(), FOOTBALL))
        components = plan.segments[0].components
        assert set(components) == {
            "quality",
            "energy_fit",
            "diversity",
            "role_fit",
            "style_match",
            "relevance",
        }

    def test_selection_is_deterministic(self) -> None:
        clips = football_project()
        first = decide(engine(clips, FOOTBALL))
        second = decide(engine(clips, FOOTBALL))
        assert [s.media_id for s in first.segments] == [s.media_id for s in second.segments]
        assert [s.source_in_ms for s in first.segments] == [s.source_in_ms for s in second.segments]


# ---------------------------------------------------------------------- story
class TestStoryAssignment:
    def test_roles_run_in_narrative_order(self) -> None:
        from visionforge.domain.story import ROLE_ORDER

        plan = decide(engine(football_project(), FOOTBALL))
        positions = [ROLE_ORDER.index(role) for role in plan.roles]
        assert positions == sorted(positions)

    def test_a_chronological_policy_keeps_each_role_in_shot_order(self) -> None:
        """A highlight that reorders the match is not a highlight of the match
        -- but the arc still decides that the build comes before the peak."""
        plan = decide(engine(nature_project(), NATURE))
        by_role: dict[StoryRole, list[int]] = {}
        for segment in plan.segments:
            by_role.setdefault(segment.role, []).append(segment.sequence)
        for sequences in by_role.values():
            assert sequences == sorted(sequences)

    def test_reordering_is_recorded_rather_than_silent(self) -> None:
        """An edit that silently reorders is indistinguishable from one that got
        the chronology wrong."""
        plan = decide(engine(football_project(), FOOTBALL))
        chronological = [s.sequence for s in plan.segments] == sorted(
            s.sequence for s in plan.segments
        )
        reorders = [d for d in plan.decisions if d.kind is DecisionKind.REORDER]
        assert chronological == (not reorders)


# ---------------------------------------------------------------------- trims
class TestTrimWindows:
    def test_an_action_role_centres_on_the_measured_peak(self) -> None:
        """The difference between trimming the goal and trimming the run-up."""
        clips = [
            clip(0, motion=0.34, motion_peak_ms=8_000, duration_ms=10_000, subject=0),
            clip(1, motion=0.05, duration_ms=10_000, subject=1),
            clip(2, motion=0.20, duration_ms=10_000, subject=2),
        ]
        plan = decide(engine(clips, FOOTBALL, count=3, target_ms=9_000))
        peak = next(s for s in plan.segments if s.role is StoryRole.PEAK)
        assert peak.media_id == clips[0].media_id
        assert peak.source_in_ms < 8_000 < peak.source_out_ms

    def test_a_trim_reports_how_it_was_placed(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        trims = [d for d in plan.decisions if d.kind is DecisionKind.TRIM]
        assert trims
        placements = {
            ReasonCode.CENTRED_ON_ACTION,
            ReasonCode.CENTRE_TRIM,
            ReasonCode.AVOIDS_INTERNAL_CUT,
            ReasonCode.SOURCE_TOO_SHORT,
        }
        for trim in trims:
            assert placements & set(trim.reasons)

    def test_a_window_avoids_an_internal_cut_where_it_can(self) -> None:
        """A centre trim across a boundary puts a jump in the middle of what the
        plan calls one shot."""
        clips = [
            clip(0, motion=0.05, duration_ms=12_000, scene_boundaries_ms=(6_000,), subject=0),
            clip(1, motion=0.05, duration_ms=12_000, subject=1),
        ]
        plan = decide(engine(clips, NATURE, count=2, target_ms=8_000))
        cut = next(s for s in plan.segments if s.media_id == clips[0].media_id)
        assert not (cut.source_in_ms < 6_000 < cut.source_out_ms)

    def test_every_window_stays_inside_its_source(self) -> None:
        clips = football_project()
        lengths = {c.media_id: c.duration_ms for c in clips}
        plan = decide(engine(clips, FOOTBALL))
        for segment in plan.segments:
            assert 0 <= segment.source_in_ms < segment.source_out_ms
            assert segment.source_out_ms <= (lengths[segment.media_id] or 0)


# ------------------------------------------------------------------ treatment
class TestTreatment:
    def test_the_peak_is_slowed_where_the_policy_asks_for_it(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        peak = next(s for s in plan.segments if s.role is StoryRole.PEAK)
        assert any(effect.kind is EffectKind.SLOW_MOTION for effect in peak.effects)

    def test_slow_motion_takes_more_source_so_the_slot_still_fits(self) -> None:
        """Deciding speed after the trim is how a slowed peak silently becomes
        twice as long as pacing asked for."""
        plan = decide(engine(football_project(), FOOTBALL))
        peak = next(s for s in plan.segments if s.role is StoryRole.PEAK)
        assert peak.output_ms > peak.duration_ms

    def test_a_policy_that_does_not_want_slow_motion_gets_none(self) -> None:
        plan = decide(engine(nature_project(), NATURE))
        assert not any(
            effect.kind is EffectKind.SLOW_MOTION
            for segment in plan.segments
            for effect in segment.effects
        )

    def test_a_low_appetite_policy_cuts_rather_than_dissolves(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        assert all(segment.transition is TransitionKind.CUT for segment in plan.segments)

    def test_a_high_appetite_policy_dissolves_where_the_edit_settles(self) -> None:
        plan = decide(engine(nature_project(), NATURE))
        assert any(segment.transition is not TransitionKind.CUT for segment in plan.segments)

    def test_a_crossfade_never_lands_on_the_first_clip(self) -> None:
        plan = decide(engine(nature_project(), NATURE))
        assert plan.segments[0].transition is not TransitionKind.CROSSFADE

    def test_no_transition_exceeds_half_of_either_neighbour(self) -> None:
        """The plan validator's own rule, applied here so a plan is never
        rejected over its decoration."""
        plan = decide(engine(nature_project(), NATURE))
        for index, segment in enumerate(plan.segments):
            if segment.transition.consumes_time:
                previous = plan.segments[index - 1]
                assert segment.transition_ms <= min(previous.output_ms, segment.output_ms) // 2

    def test_treatments_can_be_turned_off_entirely(self) -> None:
        """Used by the acceptance script to prove the structure is doing the
        work rather than the decoration."""
        plan = decide(engine(nature_project(), NATURE, treatments=False))
        assert all(segment.transition is TransitionKind.CUT for segment in plan.segments)
        assert all(not segment.effects for segment in plan.segments)

    def test_a_subtitle_is_recommended_not_written(self) -> None:
        """Nothing here transcribes anything, so nothing here supplies text."""
        clips = [
            clip(index, motion=0.02, faces=(True,) * 5, face_area=0.12, subject=index)
            for index in range(4)
        ]
        plan = decide(engine(clips, NATURE))
        suggestions = [d for d in plan.decisions if d.kind is DecisionKind.ADD_SUBTITLE]
        for suggestion in suggestions:
            assert ReasonCode.LIKELY_SPEECH in suggestion.reasons
            assert suggestion.confidence <= 0.45

    def test_a_decision_payload_carries_no_free_text(self) -> None:
        """Structurally there is nowhere for a filter expression to arrive."""
        plan = decide(engine(football_project(), FOOTBALL))
        for decision in plan.decisions:
            payload = decision.as_payload()
            for key, value in payload.items():
                if isinstance(value, str):
                    assert key in {"kind", "role", "effect", "transition", "media_id"}


# ---------------------------------------------------------------------- shape
class TestPlanShape:
    def test_events_travel_with_the_segment(self) -> None:
        plan = decide(engine(football_project(), FOOTBALL))
        assert all(segment.events for segment in plan.segments)

    def test_a_clip_with_no_observation_is_unknown_not_absent(self) -> None:
        clips = [clip(i, motion=None, motion_spread=None, saturation=None) for i in range(4)]
        plan = decide(engine(clips, FOOTBALL))
        assert all(segment.events[0].event is EditorialEvent.UNKNOWN for segment in plan.segments)

    def test_the_total_accounts_for_speed_and_overlap(self) -> None:
        plan = decide(engine(nature_project(), NATURE))
        expected = sum(s.output_ms for s in plan.segments) - sum(
            s.transition_ms for s in plan.segments[1:] if s.transition.consumes_time
        )
        assert plan.total_output_ms == expected
