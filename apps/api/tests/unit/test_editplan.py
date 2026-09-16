"""EditPlan validation, including plans a hostile or broken planner might emit.

The gate exists so that nothing invalid reaches FFmpeg. These tests are the
evidence that it holds, and they matter more the moment an LLM takes the
planner's seat in Phase 5.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    MediaFact,
    MusicCue,
    OutputSpec,
    PlanInvalidError,
    QualityPreset,
    Segment,
    TransitionKind,
    assert_valid,
    plan_from_payload,
    validate_plan,
)
from visionforge.domain.ids import MediaId, ProjectId

PROJECT = ProjectId(uuid.uuid4())
OTHER_PROJECT = ProjectId(uuid.uuid4())


def fact(media_id: MediaId, **overrides: object) -> MediaFact:
    defaults: dict[str, object] = {
        "media_id": media_id,
        "project_id": PROJECT,
        "is_renderable": True,
        "duration_ms": 20_000,
        "width": 1920,
        "height": 1080,
    }
    defaults.update(overrides)
    return MediaFact(**defaults)  # type: ignore[arg-type]


def plan_with(segments: list[Segment], **output: object) -> EditPlan:
    spec: dict[str, object] = {
        "aspect_ratio": AspectRatio.LANDSCAPE_16_9,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fit": FitMode.COVER,
        "audio": AudioMode.NONE,
    }
    spec.update(output)
    return EditPlan(
        project_id=PROJECT,
        segments=tuple(segments),
        output=OutputSpec(**spec),  # type: ignore[arg-type]
    )


def simple_plan(count: int = 3) -> tuple[EditPlan, dict[MediaId, MediaFact]]:
    ids = [MediaId(uuid.uuid4()) for _ in range(count)]
    segments = [
        Segment(media_id=m, order=i, source_in_ms=1000, source_out_ms=6000)
        for i, m in enumerate(ids)
    ]
    return plan_with(segments), {m: fact(m) for m in ids}


def codes(plan: EditPlan, facts: dict[MediaId, MediaFact]) -> set[str]:
    return {v.code for v in validate_plan(plan, facts)}


# ------------------------------------------------------------------- happy path
class TestValidPlan:
    def test_a_well_formed_plan_passes(self) -> None:
        plan, facts = simple_plan()
        assert validate_plan(plan, facts) == []
        assert_valid(plan, facts)

    def test_total_duration_is_the_sum_of_segments(self) -> None:
        plan, _ = simple_plan(3)
        assert plan.total_duration_ms == 15_000

    def test_segments_are_exposed_in_order(self) -> None:
        ids = [MediaId(uuid.uuid4()) for _ in range(3)]
        shuffled = [
            Segment(media_id=ids[0], order=2, source_in_ms=0, source_out_ms=2000),
            Segment(media_id=ids[1], order=0, source_in_ms=0, source_out_ms=2000),
            Segment(media_id=ids[2], order=1, source_in_ms=0, source_out_ms=2000),
        ]
        assert [s.order for s in plan_with(shuffled).ordered_segments] == [0, 1, 2]


# ------------------------------------------------------------- structural rules
class TestStructure:
    def test_a_plan_with_no_segments_is_rejected(self) -> None:
        assert "no_segments" in codes(plan_with([]), {})

    def test_duplicate_orders_are_rejected(self) -> None:
        ids = [MediaId(uuid.uuid4()) for _ in range(2)]
        segments = [
            Segment(media_id=ids[0], order=0, source_in_ms=0, source_out_ms=3000),
            Segment(media_id=ids[1], order=0, source_in_ms=0, source_out_ms=3000),
        ]
        facts = {m: fact(m) for m in ids}
        assert "duplicate_order" in codes(plan_with(segments), facts)

    def test_too_many_segments_are_rejected(self) -> None:
        """A cap on how much one render can cost this machine."""
        ids = [MediaId(uuid.uuid4()) for _ in range(MAX_SEGMENTS + 1)]
        segments = [
            Segment(media_id=m, order=i, source_in_ms=0, source_out_ms=1000)
            for i, m in enumerate(ids)
        ]
        assert "too_many_segments" in codes(plan_with(segments), {m: fact(m) for m in ids})

    def test_every_violation_is_reported_not_just_the_first(self) -> None:
        """A planner fixing one error at a time would need many round trips."""
        media_id = MediaId(uuid.uuid4())
        bad = plan_with(
            [Segment(media_id=media_id, order=0, source_in_ms=-5, source_out_ms=-1)],
            width=1281,
            height=721,
            fps=999,
        )
        found = codes(bad, {media_id: fact(media_id)})
        assert len(found) >= 3


# ------------------------------------------------------------------ time ranges
class TestTimeRanges:
    def test_negative_start_is_rejected(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with(
            [Segment(media_id=media_id, order=0, source_in_ms=-1000, source_out_ms=4000)]
        )
        assert "negative_in" in codes(plan, {media_id: fact(media_id)})

    @pytest.mark.parametrize(("start", "end"), [(5000, 5000), (5000, 1000)])
    def test_end_must_exceed_start(self, start: int, end: int) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with(
            [Segment(media_id=media_id, order=0, source_in_ms=start, source_out_ms=end)]
        )
        assert "non_positive_duration" in codes(plan, {media_id: fact(media_id)})

    def test_a_segment_shorter_than_the_floor_is_rejected(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with(
            [
                Segment(
                    media_id=media_id,
                    order=0,
                    source_in_ms=0,
                    source_out_ms=MIN_SEGMENT_MS - 1,
                )
            ]
        )
        assert "segment_too_short" in codes(plan, {media_id: fact(media_id)})

    def test_a_segment_longer_than_the_ceiling_is_rejected(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with(
            [
                Segment(
                    media_id=media_id,
                    order=0,
                    source_in_ms=0,
                    source_out_ms=MAX_SEGMENT_MS + 1000,
                )
            ]
        )
        facts = {media_id: fact(media_id, duration_ms=MAX_SEGMENT_MS + 5000)}
        assert "segment_too_long" in codes(plan, facts)

    def test_trimming_past_the_end_of_the_source_is_rejected(self) -> None:
        """FFmpeg would silently produce a shorter clip; the plan would lie."""
        media_id = MediaId(uuid.uuid4())
        plan = plan_with([Segment(media_id=media_id, order=0, source_in_ms=0, source_out_ms=9000)])
        assert "trim_past_end" in codes(plan, {media_id: fact(media_id, duration_ms=5000)})


# ----------------------------------------------------------------- media checks
class TestMediaReferences:
    def test_a_plan_referencing_unknown_media_is_rejected(self) -> None:
        plan = plan_with(
            [
                Segment(
                    media_id=MediaId(uuid.uuid4()),
                    order=0,
                    source_in_ms=0,
                    source_out_ms=3000,
                )
            ]
        )
        assert "unknown_media" in codes(plan, {})

    def test_cross_project_media_is_rejected_in_the_domain(self) -> None:
        """Enforced here, not only at the API.

        A plan that reaches into another project is invalid even if something
        upstream failed to notice -- defence that does not depend on the caller.
        """
        media_id = MediaId(uuid.uuid4())
        plan = plan_with([Segment(media_id=media_id, order=0, source_in_ms=0, source_out_ms=3000)])
        facts = {media_id: fact(media_id, project_id=OTHER_PROJECT)}
        assert "cross_project_media" in codes(plan, facts)

    def test_unrenderable_media_is_rejected(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with([Segment(media_id=media_id, order=0, source_in_ms=0, source_out_ms=3000)])
        facts = {media_id: fact(media_id, is_renderable=False)}
        assert "media_not_renderable" in codes(plan, facts)


# ----------------------------------------------------------------------- output
class TestOutputSpec:
    @pytest.mark.parametrize(("width", "height"), [(0, 720), (8000, 720), (1280, 0)])
    def test_dimensions_outside_the_bounds_are_rejected(self, width: int, height: int) -> None:
        plan, facts = simple_plan(1)
        bad = plan_with(list(plan.segments), width=width, height=height)
        found = codes(bad, facts)
        assert {"output_width", "output_height"} & found

    def test_odd_dimensions_are_rejected(self) -> None:
        """H.264 4:2:0 cannot encode them.

        Catching it here turns a cryptic encoder failure into a plan-level
        rejection with a readable reason.
        """
        plan, facts = simple_plan(1)
        assert "output_dimensions_odd" in codes(
            plan_with(list(plan.segments), width=1281, height=721), facts
        )

    @pytest.mark.parametrize("fps", [0, -30, 500])
    def test_impossible_frame_rates_are_rejected(self, fps: int) -> None:
        plan, facts = simple_plan(1)
        assert "output_fps" in codes(plan_with(list(plan.segments), fps=fps), facts)

    def test_dimensions_must_match_the_declared_aspect_ratio(self) -> None:
        plan, facts = simple_plan(1)
        mismatched = plan_with(
            list(plan.segments),
            aspect_ratio=AspectRatio.PORTRAIT_9_16,
            width=1280,
            height=720,
        )
        assert "aspect_mismatch" in codes(mismatched, facts)

    def test_each_supported_aspect_ratio_has_a_consistent_geometry(self) -> None:
        for ratio, (width, height) in {
            AspectRatio.LANDSCAPE_16_9: (1280, 720),
            AspectRatio.PORTRAIT_9_16: (720, 1280),
            AspectRatio.SQUARE_1_1: (720, 720),
        }.items():
            plan, facts = simple_plan(1)
            candidate = plan_with(
                list(plan.segments), aspect_ratio=ratio, width=width, height=height
            )
            assert "aspect_mismatch" not in codes(candidate, facts)

    def test_an_output_shorter_than_the_floor_is_rejected(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_with([Segment(media_id=media_id, order=0, source_in_ms=0, source_out_ms=400)])
        assert "output_duration" in codes(plan, {media_id: fact(media_id)})


# -------------------------------------------------------------- hostile inputs
class TestHostilePlans:
    """A plan is structurally incapable of carrying an instruction.

    These assert that property directly: there is nowhere in the schema to put a
    command or a path, so the usual injection surface does not exist.
    """

    def test_a_media_id_must_be_a_uuid(self) -> None:
        with pytest.raises((ValueError, AttributeError, TypeError)):
            plan_from_payload(
                {
                    "project_id": str(PROJECT),
                    "output": {
                        "aspect_ratio": "16:9",
                        "width": 1280,
                        "height": 720,
                        "fps": 30,
                        "fit": "cover",
                        "audio": "none",
                    },
                    "segments": [
                        {
                            "media_id": "../../etc/passwd",
                            "order": 0,
                            "source_in_ms": 0,
                            "source_out_ms": 3000,
                        }
                    ],
                }
            )

    def test_an_unknown_transition_is_rejected(self) -> None:
        """The vocabulary is closed; an unrecognised operation cannot pass."""
        with pytest.raises(ValueError):
            plan_from_payload(
                {
                    "project_id": str(PROJECT),
                    "output": {
                        "aspect_ratio": "16:9",
                        "width": 1280,
                        "height": 720,
                        "fps": 30,
                        "fit": "cover",
                        "audio": "none",
                    },
                    "segments": [
                        {
                            "media_id": str(uuid.uuid4()),
                            "order": 0,
                            "source_in_ms": 0,
                            "source_out_ms": 3000,
                            "transition_in": "exec_shell",
                        }
                    ],
                }
            )

    def test_an_unknown_aspect_ratio_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            plan_from_payload(
                {
                    "project_id": str(PROJECT),
                    "output": {
                        "aspect_ratio": "$(whoami)",
                        "width": 1280,
                        "height": 720,
                        "fps": 30,
                        "fit": "cover",
                        "audio": "none",
                    },
                    "segments": [],
                }
            )

    def test_the_plan_schema_has_no_field_for_a_path_or_a_command(self) -> None:
        """The structural guarantee, asserted rather than assumed.

        The field sets are pinned exactly, so *adding* a field to the plan is a
        deliberate act that fails this test until someone confirms the new field
        cannot carry a path or a command. Phase 5 added ``quality``, which is an
        enum of three values. Phase 7 added ``source_gain`` and the music cue,
        which are numbers and one media id. Phase 9 added ``transition_ms``, an
        integer, and ``effects``, a list whose every entry is a closed enum plus
        one number and two optional integers -- deliberately not a parameter
        dictionary, which would be a hole in exactly the shape of an arbitrary
        filter argument.
        """
        plan, _ = simple_plan(1)
        payload = plan.as_payload()

        assert set(payload["segments"][0]) == {
            "media_id",
            "order",
            "source_in_ms",
            "source_out_ms",
            "duration_ms",
            "transition_in",
            "transition_ms",
            "effects",
        }
        assert set(payload["output"]) == {
            "aspect_ratio",
            "width",
            "height",
            "fps",
            "fit",
            "audio",
            "quality",
            "source_gain",
        }

    def test_an_effect_is_an_enum_and_a_number(self) -> None:
        """No parameter dictionary, in either direction.

        An effect that carried ``{"filter": "..."}`` would defeat every other
        guarantee in this file, so the payload's field set is pinned the same
        way the segment's is.
        """
        from visionforge.domain.effects import Effect, EffectKind

        payload = Effect(kind=EffectKind.BRIGHTNESS, amount=0.2).as_payload()
        assert set(payload) == {"kind", "amount", "start_ms", "end_ms"}
        assert isinstance(payload["kind"], str)
        assert isinstance(payload["amount"], float)

    def test_subtitle_text_is_the_only_string_and_it_is_not_a_filter(self) -> None:
        """The first user-supplied string in the plan, and what happens to it.

        Text reaches the renderer inside an ASS document, never inside a filter
        graph -- so the characters that would carry meaning to a filter or to
        the document format are removed before storage rather than escaped at
        render time. ``clean_text`` is where that happens; this pins that it
        happens at all.
        """
        from visionforge.domain.subtitles import SubtitleCue, SubtitleTrack, clean_text

        hostile = "hi':drawbox=c=red@1:t=fill,drawtext=text='pwned"
        cleaned = clean_text(hostile)
        assert "\\" not in cleaned
        assert "{" not in cleaned and "}" not in cleaned

        track = SubtitleTrack(cues=(SubtitleCue(0, 1_000, cleaned),))
        payload = track.as_payload()
        assert set(payload) == {"style", "position", "cues"}
        assert set(payload["cues"][0]) == {"start_ms", "end_ms", "text"}
        # Style and position are ids, never parameters: no font path, no size,
        # no colour, nothing the renderer would take as an instruction.
        assert isinstance(payload["style"], str)
        assert isinstance(payload["position"], str)

    def test_the_music_cue_has_no_field_for_a_path_or_a_command(self) -> None:
        """The same guard, for the audio track.

        An audio filter graph runs commands exactly as readily as a video one,
        so the cue gets the identical treatment: a media id and six numbers,
        pinned, with no free string anywhere in it.
        """
        plan, _ = simple_plan(1)
        cue = MusicCue(
            media_id=MediaId(uuid.uuid4()),
            source_in_ms=0,
            source_out_ms=8_000,
            timeline_start_ms=0,
            gain=0.8,
            fade_in_ms=250,
            fade_out_ms=750,
        )
        body = replace(plan, music=cue).as_payload()["music"]

        assert set(body) == {
            "media_id",
            "source_in_ms",
            "source_out_ms",
            "timeline_start_ms",
            "duration_ms",
            "gain",
            "fade_in_ms",
            "fade_out_ms",
        }
        # One string, and it parses as a UUID or it is not a media id.
        assert uuid.UUID(str(body["media_id"]))
        for key, value in body.items():
            if key != "media_id":
                assert isinstance(value, int | float), f"{key} is not a number"

    def test_a_music_media_id_must_be_a_uuid(self) -> None:
        """A cue cannot name a path, for the same reason a segment cannot."""
        plan, _ = simple_plan(1)
        payload = plan.as_payload()
        payload["music"] = {
            "media_id": "../../etc/passwd",
            "source_in_ms": 0,
            "source_out_ms": 8_000,
        }
        with pytest.raises((ValueError, AttributeError, TypeError)):
            plan_from_payload(payload)

    def test_every_plan_field_is_a_number_or_a_closed_enum(self) -> None:
        """The property behind the field list, checked rather than assumed.

        Pinning names catches a new field; this catches a new field of the wrong
        *kind*. Every string in a plan's output must be a member of its enum, so
        there is nowhere for ``/etc/passwd`` or ``-i /dev/zero`` to live even if
        a future field slips past review.
        """
        plan, _ = simple_plan(2)
        output = plan.as_payload()["output"]

        allowed = {
            "aspect_ratio": {r.value for r in AspectRatio},
            "fit": {f.value for f in FitMode},
            "audio": {a.value for a in AudioMode},
            "quality": {q.value for q in QualityPreset},
        }
        for key, value in output.items():
            if isinstance(value, str):
                assert value in allowed[key], f"{key} is a free string, not a closed enum"
            else:
                # Gains are fractional; everything else is a whole number. Both
                # are numbers, which is the property that matters here.
                assert isinstance(value, int | float), f"{key} is neither a number nor an enum"


# ------------------------------------------------------------------ round trip
class TestSerialisation:
    def test_a_plan_survives_a_round_trip(self) -> None:
        plan, _ = simple_plan(3)
        restored = plan_from_payload(plan.as_payload())

        assert restored.project_id == plan.project_id
        assert restored.total_duration_ms == plan.total_duration_ms
        assert [s.media_id for s in restored.ordered_segments] == [
            s.media_id for s in plan.ordered_segments
        ]
        assert restored.output == plan.output

    def test_transition_defaults_to_a_cut_when_absent(self) -> None:
        plan, _ = simple_plan(1)
        payload = plan.as_payload()
        del payload["segments"][0]["transition_in"]

        assert plan_from_payload(payload).segments[0].transition_in is TransitionKind.CUT


class TestAssertValid:
    def test_assert_valid_raises_with_every_violation(self) -> None:
        media_id = MediaId(uuid.uuid4())
        bad = plan_with(
            [Segment(media_id=media_id, order=0, source_in_ms=100, source_out_ms=50)],
            width=1281,
        )
        with pytest.raises(PlanInvalidError) as exc_info:
            assert_valid(bad, {media_id: fact(media_id)})

        assert len(exc_info.value.violations) >= 2

    def test_violations_are_machine_readable(self) -> None:
        media_id = MediaId(uuid.uuid4())
        bad = plan_with([Segment(media_id=media_id, order=0, source_in_ms=100, source_out_ms=50)])
        with pytest.raises(PlanInvalidError) as exc_info:
            assert_valid(bad, {media_id: fact(media_id)})

        payload = exc_info.value.violations[0].as_payload()
        assert set(payload) == {"code", "message", "segment_order", "is_music"}
