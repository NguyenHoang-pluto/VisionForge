"""Transition timing: the arithmetic, and the rules that keep it renderable.

The property under test throughout is that **one number describes the edit**.
A plan's ``total_duration_ms``, the last clip's ``timeline_end_ms`` and the
render spec's ``duration_ms`` must agree, for every combination of transitions
and speeds -- because an editor that reports one length and renders another is
worse than one that refuses to render at all.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.editplan import (
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_TRANSITION_MS,
    EditPlan,
    MediaFact,
    Segment,
    TransitionKind,
    plan_from_payload,
    validate_plan,
)
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.timeline import build_render_spec, compile_timeline

PROJECT = ProjectId(uuid.uuid4())
MEDIA = MediaId(uuid.uuid4())


def segment(order: int, duration: int = 2_000, **kwargs: object) -> Segment:
    return Segment(
        media_id=MEDIA,
        order=order,
        source_in_ms=0,
        source_out_ms=duration,
        **kwargs,  # type: ignore[arg-type]
    )


def plan(*segments: Segment, **kwargs: object) -> EditPlan:
    return EditPlan(project_id=PROJECT, segments=segments, **kwargs)  # type: ignore[arg-type]


def facts(duration_ms: int = 60_000) -> dict[MediaId, MediaFact]:
    return {
        MEDIA: MediaFact(
            media_id=MEDIA,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=duration_ms,
            width=1920,
            height=1080,
        )
    }


def codes(violations: list) -> set[str]:  # type: ignore[type-arg]
    return {violation.code for violation in violations}


class TestTheVocabulary:
    def test_the_kinds_are_the_four_that_ship(self) -> None:
        assert {kind.value for kind in TransitionKind} == {
            "cut",
            "crossfade",
            "fade_in",
            "fade_to_black",
        }

    @pytest.mark.parametrize(
        ("kind", "consumes"),
        [
            (TransitionKind.CUT, False),
            (TransitionKind.CROSSFADE, True),
            (TransitionKind.FADE_IN, False),
            (TransitionKind.FADE_TO_BLACK, False),
        ],
    )
    def test_only_a_crossfade_consumes_time(self, kind: TransitionKind, consumes: bool) -> None:
        """The distinction the whole timing model rests on.

        Pinned per member so that a fifth kind cannot be added without someone
        answering this question for it.
        """
        assert kind.consumes_time is consumes

    def test_only_a_crossfade_needs_something_to_come_from(self) -> None:
        assert TransitionKind.CROSSFADE.needs_previous
        assert not TransitionKind.FADE_IN.needs_previous
        assert not TransitionKind.CUT.needs_previous


class TestDurationArithmetic:
    def test_cuts_are_the_sum(self) -> None:
        assert plan(segment(0), segment(1), segment(2)).total_duration_ms == 6_000

    def test_a_crossfade_shortens_the_programme_by_the_overlap(self) -> None:
        edit = plan(
            segment(0),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
        )
        assert edit.total_duration_ms == 3_500

    def test_several_crossfades_each_take_their_own_overlap(self) -> None:
        edit = plan(
            segment(0),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=400),
            segment(2, transition_in=TransitionKind.CROSSFADE, transition_ms=600),
        )
        assert edit.total_duration_ms == 6_000 - 400 - 600

    def test_fades_consume_no_time(self) -> None:
        edit = plan(
            segment(0, transition_in=TransitionKind.FADE_IN, transition_ms=800),
            segment(1, transition_in=TransitionKind.FADE_TO_BLACK, transition_ms=800),
        )
        assert edit.total_duration_ms == 4_000

    def test_a_first_segment_cannot_overlap_anything(self) -> None:
        """Even if the plan claims it does. The arithmetic is total rather than
        trusting the validator to have run."""
        edit = plan(segment(0, transition_in=TransitionKind.CROSSFADE, transition_ms=500))
        assert edit.total_duration_ms == 2_000

    def test_slow_motion_lengthens_and_speed_up_shortens(self) -> None:
        slow = plan(segment(0, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),)))
        fast = plan(segment(0, effects=(Effect(kind=EffectKind.SPEED_UP, amount=2.0),)))
        assert slow.total_duration_ms == 4_000
        assert fast.total_duration_ms == 1_000

    def test_speed_and_crossfade_compose(self) -> None:
        """The overlap is taken from the *played* length, not the trim."""
        edit = plan(
            segment(0, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),)),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
        )
        # 4000 (slowed) + 2000 - 500
        assert edit.total_duration_ms == 5_500


class TestTheTimelineAgrees:
    """The plan and the timeline must never disagree about length."""

    @pytest.mark.parametrize(
        "segments",
        [
            (segment(0), segment(1)),
            (segment(0), segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500)),
            (
                segment(0),
                segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=300),
                segment(2, transition_in=TransitionKind.CROSSFADE, transition_ms=700),
            ),
            (
                segment(0, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),)),
                segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=400),
            ),
            (
                segment(0, transition_in=TransitionKind.FADE_IN, transition_ms=500),
                segment(1),
                segment(2, transition_in=TransitionKind.FADE_TO_BLACK, transition_ms=500),
            ),
        ],
    )
    def test_plan_timeline_and_spec_report_the_same_length(
        self, segments: tuple[Segment, ...]
    ) -> None:
        edit = plan(*segments)
        timeline = compile_timeline(edit)
        spec = build_render_spec(
            timeline, local_paths={MEDIA: "/tmp/x.mp4"}, output_path="/tmp/out.mp4"
        )

        assert timeline.duration_ms == edit.total_duration_ms
        assert spec.duration_ms == edit.total_duration_ms

    def test_a_crossfaded_clip_starts_before_the_previous_one_ends(self) -> None:
        timeline = compile_timeline(
            plan(
                segment(0),
                segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
            )
        )
        first, second = timeline.video_track.clips
        assert first.timeline_end_ms == 2_000
        assert second.timeline_start_ms == 1_500
        assert second.timeline_start_ms < first.timeline_end_ms

    def test_clips_stay_in_order_and_never_start_before_zero(self) -> None:
        timeline = compile_timeline(
            plan(
                segment(0, duration=600),
                segment(1, duration=600, transition_in=TransitionKind.CROSSFADE, transition_ms=300),
            )
        )
        starts = [clip.timeline_start_ms for clip in timeline.video_track.clips]
        assert starts == sorted(starts)
        assert starts[0] == 0


class TestValidation:
    def test_a_cut_cannot_have_a_duration(self) -> None:
        """A slider that does nothing is worse than no slider."""
        edit = plan(segment(0), segment(1, transition_ms=400))
        assert "transition_duration_on_cut" in codes(validate_plan(edit, facts()))

    @pytest.mark.parametrize("duration", [1, MIN_TRANSITION_MS - 1, MAX_TRANSITION_MS + 1, 30_000])
    def test_a_transition_outside_the_bounds_is_refused(self, duration: int) -> None:
        edit = plan(
            segment(0, duration=20_000),
            segment(
                1,
                duration=20_000,
                transition_in=TransitionKind.CROSSFADE,
                transition_ms=duration,
            ),
        )
        assert "transition_duration" in codes(validate_plan(edit, facts()))

    def test_a_crossfade_on_the_first_segment_is_refused(self) -> None:
        edit = plan(segment(0, transition_in=TransitionKind.CROSSFADE, transition_ms=400))
        assert "transition_without_previous" in codes(validate_plan(edit, facts()))

    def test_a_fade_on_the_first_segment_is_fine(self) -> None:
        """Fading up from black is exactly what a first clip should be able to do."""
        edit = plan(segment(0, transition_in=TransitionKind.FADE_IN, transition_ms=400))
        assert "transition_without_previous" not in codes(validate_plan(edit, facts()))

    def test_a_crossfade_longer_than_half_its_neighbour_is_refused(self) -> None:
        """Beyond half, the shorter clip is more dissolve than clip -- and
        ``xfade`` has nowhere to put the offset."""
        edit = plan(
            segment(0, duration=1_000),
            segment(
                1,
                duration=8_000,
                transition_in=TransitionKind.CROSSFADE,
                transition_ms=800,
            ),
        )
        assert "transition_too_long_for_neighbours" in codes(validate_plan(edit, facts()))

    def test_exactly_half_the_shorter_neighbour_is_allowed(self) -> None:
        edit = plan(
            segment(0, duration=2_000),
            segment(
                1,
                duration=8_000,
                transition_in=TransitionKind.CROSSFADE,
                transition_ms=int(2_000 * MAX_TRANSITION_SHARE),
            ),
        )
        assert "transition_too_long_for_neighbours" not in codes(validate_plan(edit, facts()))

    def test_the_share_is_measured_against_played_length_not_trim(self) -> None:
        """A slowed clip is longer, and can lend more to a dissolve."""
        edit = plan(
            segment(0, duration=1_000, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.25),)),
            segment(
                1,
                duration=8_000,
                transition_in=TransitionKind.CROSSFADE,
                transition_ms=1_500,
            ),
        )
        # The first clip plays for 4000 ms, so 1500 is inside half of it.
        assert "transition_too_long_for_neighbours" not in codes(validate_plan(edit, facts()))

    def test_a_valid_transitioned_plan_has_no_violations(self) -> None:
        edit = plan(
            segment(0, transition_in=TransitionKind.FADE_IN, transition_ms=400),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
            segment(2, transition_in=TransitionKind.FADE_TO_BLACK, transition_ms=600),
        )
        assert validate_plan(edit, facts()) == []


class TestRoundTrip:
    def test_transitions_survive_storage(self) -> None:
        edit = plan(
            segment(0, transition_in=TransitionKind.FADE_IN, transition_ms=400),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
        )
        restored = plan_from_payload(edit.as_payload())

        assert [s.transition_in for s in restored.ordered_segments] == [
            TransitionKind.FADE_IN,
            TransitionKind.CROSSFADE,
        ]
        assert [s.transition_ms for s in restored.ordered_segments] == [400, 500]
        assert restored.total_duration_ms == edit.total_duration_ms

    def test_a_plan_written_before_phase_9_still_loads(self) -> None:
        """The compatibility that matters: no migration, no defaults invented."""
        payload = plan(segment(0), segment(1)).as_payload()
        for entry in payload["segments"]:
            entry.pop("transition_ms")
            entry.pop("effects")

        restored = plan_from_payload(payload)
        assert all(s.transition_ms == 0 for s in restored.segments)
        assert all(s.effects == () for s in restored.segments)
        assert restored.total_duration_ms == 4_000
