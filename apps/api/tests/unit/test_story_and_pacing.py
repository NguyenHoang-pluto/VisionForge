"""Story arcs, editorial policies, and the pacing model.

Two modules, one file, because the claim they jointly make is one claim: **an
edit's structure and its rhythm are configuration, not code.** An arc is a table
of slots; a policy is a table of numbers; a pacing curve is a table of control
points. If any of those turned out to need a branch per genre, the "one pipeline,
many policies" argument would be wrong and this file is where it would show.
"""

from __future__ import annotations

import pytest

from visionforge.domain.beats import BeatGrid
from visionforge.domain.editplan import MAX_SEGMENT_MS, MIN_SEGMENT_MS
from visionforge.domain.pacing import (
    MAX_EMPHASIS,
    PACING_CURVES,
    PacingCurve,
    PacingShape,
    curve_for,
    plan_pacing,
)
from visionforge.domain.story import (
    EDITORIAL_POLICIES,
    STYLE_POLICIES,
    ArcSlot,
    CreativeWeights,
    PolicyId,
    StoryArc,
    StoryRole,
    policy_for,
    with_bounds,
)
from visionforge.domain.style import EditStyle


# --------------------------------------------------------------------- arcs
class TestStoryArc:
    def test_slots_must_be_in_narrative_order(self) -> None:
        """An arc that peaks before it sets up is a different story, not this
        one shuffled -- so it is refused at construction rather than producing a
        surprising edit."""
        with pytest.raises(ValueError, match="narrative order"):
            StoryArc(
                name="backwards",
                slots=(
                    ArcSlot(role=StoryRole.PEAK, weight=1.0),
                    ArcSlot(role=StoryRole.SETUP, weight=1.0),
                ),
            )

    def test_a_role_cannot_appear_twice(self) -> None:
        with pytest.raises(ValueError, match="same role twice"):
            StoryArc(
                name="doubled",
                slots=(
                    ArcSlot(role=StoryRole.BUILD, weight=1.0),
                    ArcSlot(role=StoryRole.BUILD, weight=1.0),
                ),
            )

    def test_expand_returns_exactly_the_requested_count(self) -> None:
        arc = EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc
        for count in range(1, arc.max_clips + 1):
            assert len(arc.expand(count)) == count

    def test_expand_stops_at_the_arcs_own_ceiling(self) -> None:
        """An arc that allows one hook and two peaks cannot carry twenty shots.

        The planner clamps its budget to this, so the ceiling is a fact callers
        can plan against rather than a silent truncation.
        """
        arc = EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc
        assert len(arc.expand(arc.max_clips + 5)) == arc.max_clips

    def test_expand_keeps_roles_in_narrative_order(self) -> None:
        from visionforge.domain.story import ROLE_ORDER

        roles = EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc.expand(10)
        positions = [ROLE_ORDER.index(role) for role in roles]
        assert positions == sorted(positions)

    def test_a_one_clip_edit_is_its_peak(self) -> None:
        """Degradation by priority: the most important role survives."""
        assert EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc.expand(1) == (StoryRole.PEAK,)

    def test_a_three_clip_edit_keeps_the_ends_not_the_middle(self) -> None:
        roles = set(EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc.expand(3))
        assert StoryRole.PEAK in roles
        assert StoryRole.ENDING in roles
        assert StoryRole.SETUP not in roles

    def test_surplus_goes_to_build_never_to_peak(self) -> None:
        """An edit with four peaks has none."""
        roles = EDITORIAL_POLICIES[PolicyId.FOOTBALL].arc.expand(16)
        assert roles.count(StoryRole.PEAK) <= 2
        assert roles.count(StoryRole.BUILD) >= 4

    def test_expansion_is_deterministic(self) -> None:
        arc = EDITORIAL_POLICIES[PolicyId.NATURE].arc
        assert arc.expand(9) == arc.expand(9)

    def test_role_fit_takes_the_best_argument_not_the_sum(self) -> None:
        """ "A bit like four things" is not a reason to make something a peak."""
        from visionforge.domain.editorial import EditorialEvent as E

        slot = ArcSlot(
            role=StoryRole.PEAK,
            weight=1.0,
            affinity={E.PEAK_MOTION: 1.0, E.ACTION: 0.6, E.IMPACT: 0.6, E.CELEBRATION: 0.6},
        )
        focused = slot.fit({E.PEAK_MOTION: 0.8})
        scattered = slot.fit({E.ACTION: 0.5, E.IMPACT: 0.5, E.CELEBRATION: 0.5})
        assert focused > scattered
        assert scattered <= 1.0


# ------------------------------------------------------------------ policies
class TestPolicies:
    def test_every_policy_has_ordered_clip_bounds(self) -> None:
        for policy in EDITORIAL_POLICIES.values():
            assert policy.min_clip_ms <= policy.target_clip_ms <= policy.max_clip_ms

    def test_every_creative_weight_set_is_convex(self) -> None:
        """Enforced at construction, so a typo fails at import rather than
        producing a quietly skewed edit."""
        for policy in EDITORIAL_POLICIES.values():
            total = sum(
                (
                    policy.weights.quality,
                    policy.weights.energy_fit,
                    policy.weights.diversity,
                    policy.weights.role_fit,
                    policy.weights.style_match,
                    policy.weights.relevance,
                )
            )
            assert abs(total - 1.0) < 1e-6

    def test_quality_is_never_a_majority_of_the_score(self) -> None:
        """The whole point of the phase.

        If quality could outvote everything else combined, five excellent shots
        of one wall would still become the edit.
        """
        for policy in EDITORIAL_POLICIES.values():
            assert policy.weights.quality < 0.5

    def test_weights_that_do_not_sum_to_one_are_refused(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            CreativeWeights(quality=0.9, energy_fit=0.9)

    def test_every_style_maps_to_a_policy_that_exists(self) -> None:
        for style in EditStyle:
            assert STYLE_POLICIES[style] in EDITORIAL_POLICIES

    def test_an_explicit_policy_beats_the_style(self) -> None:
        """The user who chose a policy chose it; a style is a default they may
        not have thought about."""
        chosen = policy_for(PolicyId.NATURE, EditStyle.GAMING)
        assert chosen.id is PolicyId.NATURE

    def test_no_policy_and_no_style_is_neutral(self) -> None:
        assert policy_for(None, None).id is PolicyId.NEUTRAL

    def test_genres_actually_differ_in_pacing(self) -> None:
        nature = EDITORIAL_POLICIES[PolicyId.NATURE]
        social = EDITORIAL_POLICIES[PolicyId.SOCIAL]
        assert nature.target_clip_ms > social.target_clip_ms * 3

    def test_with_bounds_cannot_invert_a_policy(self) -> None:
        """A reference video's measurements arrive through here, and a blend
        artefact must not produce a policy that raises."""
        policy = EDITORIAL_POLICIES[PolicyId.NATURE]
        widened = with_bounds(policy, min_clip_ms=9_000, target_clip_ms=500, max_clip_ms=1_000)
        assert widened.min_clip_ms <= widened.target_clip_ms <= widened.max_clip_ms


# -------------------------------------------------------------------- curves
class TestPacingCurve:
    def test_every_shape_has_a_table(self) -> None:
        for shape in PacingShape:
            assert len(PACING_CURVES[shape]) >= 2

    def test_energy_at_interpolates_between_control_points(self) -> None:
        curve = PacingCurve(shape=PacingShape.STEADY, points=(0.0, 1.0))
        assert curve.energy_at(0.0) == 0.0
        assert curve.energy_at(0.5) == 0.5
        assert curve.energy_at(1.0) == 1.0

    def test_energy_at_is_clamped_outside_the_range(self) -> None:
        curve = curve_for(PacingShape.RAMP)
        assert curve.energy_at(-5.0) == curve.energy_at(0.0)
        assert curve.energy_at(9.0) == curve.energy_at(1.0)

    def test_a_single_slot_sits_at_the_peak(self) -> None:
        """A one-shot edit has no arc to walk, and giving it the opening energy
        of a ramp would make it the slowest thing the system produces."""
        curve = curve_for(PacingShape.RAMP)
        assert curve.energies(1) == (max(curve.points),)

    def test_ramp_rises_then_releases(self) -> None:
        energies = curve_for(PacingShape.RAMP).energies(6)
        assert energies[0] < energies[3]
        assert energies[-1] < max(energies)

    def test_decay_starts_at_its_busiest(self) -> None:
        energies = curve_for(PacingShape.DECAY).energies(5)
        assert energies[0] == max(energies)

    def test_a_curve_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            PacingCurve(shape=PacingShape.RAMP, points=(0.5, 1.4))


# ---------------------------------------------------------------- the plan
def _plan(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "count": 6,
        "target_ms": 30_000,
        "curve": curve_for(PacingShape.RAMP),
        "min_clip_ms": 1_000,
        "max_clip_ms": 9_000,
    }
    defaults.update(overrides)
    return plan_pacing(**defaults)  # type: ignore[arg-type]


class TestPlanPacing:
    def test_the_total_matches_the_target(self) -> None:
        """The user stated a length. Missing it is a last resort, not a default."""
        plan = plan_pacing(
            count=6,
            target_ms=30_000,
            curve=curve_for(PacingShape.RAMP),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
        )
        assert abs(plan.total_ms - 30_000) <= 6

    def test_high_energy_slots_are_shorter_than_low_energy_ones(self) -> None:
        """The whole mechanism: pacing is expressed through clip length."""
        plan = plan_pacing(
            count=6,
            target_ms=30_000,
            curve=curve_for(PacingShape.RAMP),
            min_clip_ms=500,
            max_clip_ms=20_000,
        )
        busiest = max(plan.slots, key=lambda slot: slot.energy)
        calmest = min(plan.slots, key=lambda slot: slot.energy)
        assert busiest.duration_ms < calmest.duration_ms

    def test_a_steady_curve_produces_even_lengths(self) -> None:
        """Not "no pacing" -- a deliberately even rhythm is a real choice."""
        plan = plan_pacing(
            count=5,
            target_ms=25_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
        )
        lengths = {slot.duration_ms for slot in plan.slots}
        assert max(lengths) - min(lengths) <= 2

    def test_every_length_respects_the_bounds(self) -> None:
        plan = plan_pacing(
            count=8,
            target_ms=60_000,
            curve=curve_for(PacingShape.WAVE),
            min_clip_ms=2_000,
            max_clip_ms=8_000,
        )
        for slot in plan.slots:
            assert 2_000 <= slot.duration_ms <= 8_000

    def test_the_plan_bounds_win_over_a_policy_that_asks_for_more(self) -> None:
        plan = plan_pacing(
            count=2,
            target_ms=120_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=100,
            max_clip_ms=90_000,
        )
        for slot in plan.slots:
            assert MIN_SEGMENT_MS <= slot.duration_ms <= MAX_SEGMENT_MS

    def test_a_short_source_caps_its_own_slot_and_the_rest_absorb_it(self) -> None:
        """Clamping one slot changes what the others should be. Without the
        redistribution the curve survives only until the first bound bites."""
        plan = plan_pacing(
            count=4,
            target_ms=24_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=12_000,
            source_limits_ms=(2_000, 30_000, 30_000, 30_000),
        )
        assert plan.slots[0].duration_ms <= 2_000
        assert plan.total_ms >= 23_000

    def test_emphasis_moves_time_between_shots_rather_than_adding_it(self) -> None:
        plain = plan_pacing(
            count=4,
            target_ms=20_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=500,
            max_clip_ms=15_000,
        )
        stressed = plan_pacing(
            count=4,
            target_ms=20_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=500,
            max_clip_ms=15_000,
            emphasis=(1.0, 1.0, 1.8, 1.0),
        )
        assert stressed.slots[2].duration_ms > plain.slots[2].duration_ms
        assert abs(stressed.total_ms - plain.total_ms) <= 6

    def test_emphasis_is_capped(self) -> None:
        stressed = plan_pacing(
            count=3,
            target_ms=15_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=500,
            max_clip_ms=25_000,
            emphasis=(1.0, 99.0, 1.0),
        )
        plain = plan_pacing(
            count=3,
            target_ms=15_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=500,
            max_clip_ms=25_000,
            emphasis=(1.0, MAX_EMPHASIS, 1.0),
        )
        assert stressed.slots[1].duration_ms == plain.slots[1].duration_ms

    def test_shot_density_is_cuts_per_second(self) -> None:
        plan = plan_pacing(
            count=10,
            target_ms=20_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=500,
            max_clip_ms=9_000,
        )
        assert plan.shot_density == pytest.approx(0.5, abs=0.02)

    def test_pacing_is_deterministic(self) -> None:
        first = plan_pacing(
            count=7,
            target_ms=33_000,
            curve=curve_for(PacingShape.WAVE),
            min_clip_ms=900,
            max_clip_ms=9_000,
        )
        second = plan_pacing(
            count=7,
            target_ms=33_000,
            curve=curve_for(PacingShape.WAVE),
            min_clip_ms=900,
            max_clip_ms=9_000,
        )
        assert [s.duration_ms for s in first.slots] == [s.duration_ms for s in second.slots]


# ------------------------------------------------------------------- beats
def grid(bpm: float = 120.0, confidence: float = 0.9, count: int = 400) -> BeatGrid:
    period = 60_000.0 / bpm
    return BeatGrid(
        bpm=bpm,
        confidence=confidence,
        beats_ms=tuple(int(round(index * period)) for index in range(count)),
    )


class TestBeatQuantisation:
    def test_nothing_is_quantised_without_the_flag(self) -> None:
        """Adding music and re-timing an edit stayed separate decisions."""
        plan = plan_pacing(
            count=5,
            target_ms=20_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
            grid=grid(),
        )
        assert plan.beat_sync is None
        assert not plan.is_beat_synced

    def test_an_untrusted_grid_says_why_rather_than_applying(self) -> None:
        plan = plan_pacing(
            count=5,
            target_ms=20_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
            grid=grid(confidence=0.05),
            beat_sync=True,
        )
        assert plan.beat_sync is not None
        assert plan.beat_sync["applied"] is False
        assert plan.beat_sync["reason"] == "low_confidence"

    def test_quantised_lengths_are_whole_beats(self) -> None:
        period = 60_000.0 / 120.0
        plan = plan_pacing(
            count=6,
            target_ms=24_000,
            curve=curve_for(PacingShape.RAMP),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
            grid=grid(),
            beat_sync=True,
        )
        assert plan.is_beat_synced
        for slot in plan.slots:
            if slot.beats is not None:
                assert abs(slot.duration_ms - slot.beats * period) < 2

    def test_cuts_do_not_drift_across_a_long_edit(self) -> None:
        """The reason lengths are differences of cumulative positions.

        128 BPM is 468.75 ms a beat. Rounding each length separately and adding
        them accumulates error; rounding the running total cannot.
        """
        beat_grid = grid(bpm=128.0, count=400)
        period = beat_grid.period_ms
        plan = plan_pacing(
            count=20,
            target_ms=60_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=9_000,
            grid=beat_grid,
            beat_sync=True,
        )
        cursor = 0
        for slot in plan.slots:
            cursor += slot.duration_ms
            nearest = round(cursor / period) * period
            assert abs(cursor - nearest) < 2

    def test_a_slot_that_cannot_fit_a_beat_keeps_its_length_and_says_so(self) -> None:
        """Partial quantisation is honest: the plan reports which cuts landed."""
        plan = plan_pacing(
            count=4,
            target_ms=8_000,
            curve=curve_for(PacingShape.STEADY),
            min_clip_ms=1_000,
            max_clip_ms=2_200,
            # 20 BPM would be three seconds a beat, outside the bounds above.
            grid=grid(bpm=61.0),
            beat_sync=True,
        )
        assert plan.beat_sync is not None
        assert plan.beat_sync["slots_on_beat"] <= len(plan.slots)
        for slot in plan.slots:
            assert 1_000 <= slot.duration_ms <= 2_200
