"""Effect parameters, subtitle cues, and the presets that keep both closed.

The security half of this file is the point. Effects and subtitles are the two
places Phase 9 added where a client's input reaches the renderer -- one as a
number, one as a string -- and the tests that matter are the ones that show
neither can become a filter argument.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.editplan import (
    EditPlan,
    MediaFact,
    Segment,
    plan_from_payload,
    validate_plan,
)
from visionforge.domain.effects import (
    EFFECT_BOUNDS,
    MAX_EFFECTS_PER_SEGMENT,
    MIN_EFFECT_MS,
    Effect,
    EffectKind,
    output_duration_ms,
    speed_of,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.subtitles import (
    ALIGNMENT,
    MAX_CUE_CHARS,
    MAX_CUE_MS,
    MIN_CUE_MS,
    STYLE_PRESETS,
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
    clean_text,
    preset_for,
)
from visionforge.infra.ffmpeg.subtitles import build_ass, escape_filter_path

PROJECT = ProjectId(uuid.uuid4())
MEDIA = MediaId(uuid.uuid4())


def segment(order: int = 0, duration: int = 4_000, **kwargs: object) -> Segment:
    return Segment(
        media_id=MEDIA,
        order=order,
        source_in_ms=0,
        source_out_ms=duration,
        **kwargs,  # type: ignore[arg-type]
    )


def plan(*segments: Segment, **kwargs: object) -> EditPlan:
    return EditPlan(project_id=PROJECT, segments=segments or (segment(),), **kwargs)  # type: ignore[arg-type]


def facts() -> dict[MediaId, MediaFact]:
    return {
        MEDIA: MediaFact(
            media_id=MEDIA,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=60_000,
            width=1920,
            height=1080,
        )
    }


def codes(violations: list) -> set[str]:  # type: ignore[type-arg]
    return {violation.code for violation in violations}


# ============================================================ effects
class TestEffectVocabulary:
    def test_the_kinds_are_the_eleven_that_ship(self) -> None:
        """Seven from Phase 9, and the four pans Phase 12 added for stills."""
        assert {kind.value for kind in EffectKind} == {
            "zoom_in",
            "zoom_out",
            "pan_left",
            "pan_right",
            "pan_up",
            "pan_down",
            "slow_motion",
            "speed_up",
            "brightness",
            "contrast",
            "saturation",
        }

    def test_every_kind_has_bounds(self) -> None:
        """A kind without a range is a kind whose parameter is unvalidated."""
        assert set(EFFECT_BOUNDS) == set(EffectKind)

    def test_every_neutral_value_is_inside_its_own_range(self) -> None:
        for kind, (low, high, neutral) in EFFECT_BOUNDS.items():
            assert low <= neutral <= high, kind

    @pytest.mark.parametrize(
        ("kind", "changes"),
        [
            (EffectKind.SLOW_MOTION, True),
            (EffectKind.SPEED_UP, True),
            (EffectKind.ZOOM_IN, False),
            (EffectKind.BRIGHTNESS, False),
        ],
    )
    def test_only_speed_changes_duration(self, kind: EffectKind, changes: bool) -> None:
        assert kind.changes_duration is changes

    def test_only_colour_may_be_ranged(self) -> None:
        assert not EffectKind.BRIGHTNESS.spans_whole_segment
        assert not EffectKind.CONTRAST.spans_whole_segment
        assert not EffectKind.SATURATION.spans_whole_segment
        assert EffectKind.ZOOM_IN.spans_whole_segment
        assert EffectKind.SLOW_MOTION.spans_whole_segment


class TestSpeedArithmetic:
    @pytest.mark.parametrize(
        ("rate", "expected"),
        [(0.25, 8_000), (0.5, 4_000), (1.0, 2_000), (2.0, 1_000), (4.0, 500)],
    )
    def test_output_duration_is_the_trim_divided_by_the_rate(
        self, rate: float, expected: int
    ) -> None:
        kind = EffectKind.SLOW_MOTION if rate <= 1.0 else EffectKind.SPEED_UP
        assert output_duration_ms(2_000, (Effect(kind=kind, amount=rate),)) == expected

    def test_no_effects_is_the_identity(self) -> None:
        assert output_duration_ms(2_000, ()) == 2_000
        assert speed_of(()) == 1.0

    def test_colour_effects_do_not_change_the_rate(self) -> None:
        effects = (Effect(kind=EffectKind.BRIGHTNESS, amount=0.5),)
        assert speed_of(effects) == 1.0
        assert output_duration_ms(2_000, effects) == 2_000

    def test_a_zero_rate_raises_rather_than_dividing(self) -> None:
        with pytest.raises(ValueError):
            output_duration_ms(2_000, (Effect(kind=EffectKind.SLOW_MOTION, amount=0.0),))


class TestEffectValidation:
    @pytest.mark.parametrize("kind", list(EffectKind))
    def test_a_value_below_the_floor_is_refused(self, kind: EffectKind) -> None:
        low, _, _ = EFFECT_BOUNDS[kind]
        edit = plan(segment(effects=(Effect(kind=kind, amount=low - 0.01),)))
        assert "effect_amount" in codes(validate_plan(edit, facts()))

    @pytest.mark.parametrize("kind", list(EffectKind))
    def test_a_value_above_the_ceiling_is_refused(self, kind: EffectKind) -> None:
        _, high, _ = EFFECT_BOUNDS[kind]
        edit = plan(segment(effects=(Effect(kind=kind, amount=high + 0.01),)))
        assert "effect_amount" in codes(validate_plan(edit, facts()))

    @pytest.mark.parametrize("kind", list(EffectKind))
    def test_the_bounds_themselves_are_accepted(self, kind: EffectKind) -> None:
        low, high, _ = EFFECT_BOUNDS[kind]
        for amount in (low, high):
            edit = plan(segment(effects=(Effect(kind=kind, amount=amount),)))
            assert "effect_amount" not in codes(validate_plan(edit, facts()))

    def test_two_speed_effects_are_refused(self) -> None:
        """Two rates multiply into a third nobody asked for."""
        edit = plan(
            segment(
                effects=(
                    Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),
                    Effect(kind=EffectKind.SPEED_UP, amount=2.0),
                )
            )
        )
        assert "conflicting_speed" in codes(validate_plan(edit, facts()))

    def test_the_same_effect_twice_is_refused(self) -> None:
        edit = plan(
            segment(
                effects=(
                    Effect(kind=EffectKind.BRIGHTNESS, amount=0.1),
                    Effect(kind=EffectKind.BRIGHTNESS, amount=0.2),
                )
            )
        )
        assert "duplicate_effect" in codes(validate_plan(edit, facts()))

    def test_too_many_effects_are_refused(self) -> None:
        kinds = list(EffectKind)[: MAX_EFFECTS_PER_SEGMENT + 1]
        edit = plan(
            segment(effects=tuple(Effect(kind=k, amount=EFFECT_BOUNDS[k][2]) for k in kinds))
        )
        assert "too_many_effects" in codes(validate_plan(edit, facts()))

    def test_a_ranged_zoom_is_refused(self) -> None:
        """``crop``'s ramp is written against the whole segment; a window would
        be silently ignored, which is worse than a rejection."""
        edit = plan(
            segment(
                effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.2, start_ms=100, end_ms=900),)
            )
        )
        assert "effect_cannot_be_ranged" in codes(validate_plan(edit, facts()))

    def test_a_ranged_speed_is_refused(self) -> None:
        edit = plan(
            segment(
                effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5, start_ms=0, end_ms=900),)
            )
        )
        assert "effect_cannot_be_ranged" in codes(validate_plan(edit, facts()))

    def test_a_ranged_colour_effect_is_allowed(self) -> None:
        edit = plan(
            segment(
                effects=(
                    Effect(kind=EffectKind.SATURATION, amount=1.5, start_ms=500, end_ms=2_000),
                )
            )
        )
        assert validate_plan(edit, facts()) == []

    def test_an_inverted_window_is_refused(self) -> None:
        edit = plan(
            segment(
                effects=(
                    Effect(kind=EffectKind.BRIGHTNESS, amount=0.2, start_ms=2_000, end_ms=1_000),
                )
            )
        )
        assert "effect_window_inverted" in codes(validate_plan(edit, facts()))

    def test_a_window_shorter_than_the_floor_is_refused(self) -> None:
        edit = plan(
            segment(
                effects=(
                    Effect(
                        kind=EffectKind.BRIGHTNESS,
                        amount=0.2,
                        start_ms=0,
                        end_ms=MIN_EFFECT_MS - 1,
                    ),
                )
            )
        )
        assert "effect_window_too_short" in codes(validate_plan(edit, facts()))

    def test_a_window_past_the_clip_is_refused(self) -> None:
        edit = plan(
            segment(
                duration=2_000,
                effects=(Effect(kind=EffectKind.CONTRAST, amount=1.2, start_ms=0, end_ms=5_000),),
            )
        )
        assert "effect_window_past_end" in codes(validate_plan(edit, facts()))

    def test_effects_survive_storage(self) -> None:
        edit = plan(
            segment(
                effects=(
                    Effect(kind=EffectKind.ZOOM_IN, amount=0.2),
                    Effect(kind=EffectKind.BRIGHTNESS, amount=0.1, start_ms=0, end_ms=1_000),
                )
            )
        )
        restored = plan_from_payload(edit.as_payload())
        assert restored.ordered_segments[0].effects == edit.ordered_segments[0].effects


# ========================================================== subtitles
class TestSubtitlePresets:
    def test_every_style_has_a_preset(self) -> None:
        assert set(STYLE_PRESETS) == set(SubtitleStyle)

    def test_every_position_has_an_alignment(self) -> None:
        assert set(ALIGNMENT) == set(SubtitlePosition)

    def test_no_preset_names_a_path(self) -> None:
        """A font *family*, resolved by libass. Never a file on disk."""
        for preset in STYLE_PRESETS.values():
            assert "/" not in preset.font
            assert "\\" not in preset.font
            assert ".ttf" not in preset.font.lower()

    def test_presets_keep_text_inside_a_safe_area(self) -> None:
        for preset in STYLE_PRESETS.values():
            assert preset.margin_v > 0
            assert preset.margin_h > 0
            assert preset.size > 0


class TestCleanText:
    @pytest.mark.parametrize(
        "hostile",
        [
            "hi':drawbox=c=red@1:t=fill,drawtext=text='pwned",
            "line one\nline two",
            "{\\an8}override tag",
            "back\\slash",
            "null\x00byte",
            "tab\there",
        ],
    )
    def test_the_characters_that_carry_meaning_are_removed(self, hostile: str) -> None:
        cleaned = clean_text(hostile)
        for forbidden in ("\n", "\r", "\\", "{", "}", "\x00", "\t"):
            assert forbidden not in cleaned

    def test_ordinary_punctuation_survives(self) -> None:
        """Cleaning must not mangle real subtitles."""
        assert clean_text("It's 12:30 - are you ready?") == "It's 12:30 - are you ready?"

    def test_vietnamese_survives_intact(self) -> None:
        text = "Xin chào thế giới — phụ đề tiếng Việt"
        assert clean_text(text) == text

    def test_whitespace_collapses(self) -> None:
        assert clean_text("  too    many   spaces  ") == "too many spaces"


class TestSubtitleValidation:
    def track(self, *cues: SubtitleCue, **kwargs: object) -> SubtitleTrack:
        return SubtitleTrack(cues=cues, **kwargs)  # type: ignore[arg-type]

    def test_a_valid_track_passes(self) -> None:
        edit = plan(subtitles=self.track(SubtitleCue(0, 2_000, "Hello")))
        assert validate_plan(edit, facts()) == []

    def test_a_cue_past_the_end_is_refused(self) -> None:
        edit = plan(segment(duration=2_000), subtitles=self.track(SubtitleCue(0, 2_000, "hi")))
        # The edit is 2000 ms; a cue ending at 3000 would never be drawn.
        edit = plan(segment(duration=2_000), subtitles=self.track(SubtitleCue(1_000, 3_000, "hi")))
        assert "cue_past_end" in codes(validate_plan(edit, facts()))

    def test_an_inverted_cue_is_refused(self) -> None:
        edit = plan(subtitles=self.track(SubtitleCue(2_000, 1_000, "hi")))
        assert "cue_inverted" in codes(validate_plan(edit, facts()))

    def test_a_negative_start_is_refused(self) -> None:
        edit = plan(subtitles=self.track(SubtitleCue(-100, 1_000, "hi")))
        assert "cue_negative_start" in codes(validate_plan(edit, facts()))

    @pytest.mark.parametrize("duration", [MIN_CUE_MS - 1, MAX_CUE_MS + 1])
    def test_a_cue_outside_the_duration_bounds_is_refused(self, duration: int) -> None:
        edit = plan(segment(duration=30_000), subtitles=self.track(SubtitleCue(0, duration, "hi")))
        assert "cue_duration" in codes(validate_plan(edit, facts()))

    def test_empty_text_is_refused(self) -> None:
        edit = plan(subtitles=self.track(SubtitleCue(0, 1_000, "   ")))
        assert "cue_empty" in codes(validate_plan(edit, facts()))

    def test_text_over_the_cap_is_refused(self) -> None:
        edit = plan(subtitles=self.track(SubtitleCue(0, 2_000, "x" * (MAX_CUE_CHARS + 1))))
        assert "cue_too_long" in codes(validate_plan(edit, facts()))

    def test_overlapping_cues_are_refused(self) -> None:
        """Two cues at once needs a stacking rule this phase does not have, and
        drawing them on top of each other is not that feature."""
        edit = plan(
            subtitles=self.track(
                SubtitleCue(0, 2_000, "first"),
                SubtitleCue(1_500, 3_000, "second"),
            )
        )
        assert "cue_overlap" in codes(validate_plan(edit, facts()))

    def test_touching_cues_are_refused(self) -> None:
        edit = plan(
            subtitles=self.track(
                SubtitleCue(0, 2_000, "first"),
                SubtitleCue(2_000, 3_000, "second"),
            )
        )
        assert "cue_overlap" in codes(validate_plan(edit, facts()))

    def test_consecutive_cues_with_a_gap_are_fine(self) -> None:
        edit = plan(
            subtitles=self.track(
                SubtitleCue(0, 1_500, "first"),
                SubtitleCue(1_600, 3_000, "second"),
            )
        )
        assert validate_plan(edit, facts()) == []

    def test_cues_are_ordered_regardless_of_input_order(self) -> None:
        track = self.track(SubtitleCue(2_000, 3_000, "b"), SubtitleCue(0, 1_000, "a"))
        assert [cue.text for cue in track.ordered] == ["a", "b"]

    def test_the_overlap_check_uses_the_ordered_cues(self) -> None:
        """Sent out of order, still caught."""
        edit = plan(
            subtitles=self.track(
                SubtitleCue(1_500, 3_000, "second"),
                SubtitleCue(0, 2_000, "first"),
            )
        )
        assert "cue_overlap" in codes(validate_plan(edit, facts()))

    def test_subtitles_survive_storage(self) -> None:
        edit = plan(
            subtitles=self.track(
                SubtitleCue(0, 1_500, "Xin chào"),
                style=SubtitleStyle.SOCIAL,
                position=SubtitlePosition.CENTER,
            )
        )
        restored = plan_from_payload(edit.as_payload())
        assert restored.subtitles is not None
        assert restored.subtitles.style is SubtitleStyle.SOCIAL
        assert restored.subtitles.position is SubtitlePosition.CENTER
        assert restored.subtitles.cues[0].text == "Xin chào"

    def test_a_plan_without_subtitles_round_trips_as_none(self) -> None:
        assert plan_from_payload(plan().as_payload()).subtitles is None


class TestSubtitlesReachTheRenderer:
    """The chain from plan to spec, which had a hole in it.

    Subtitles were stored on the plan, carried onto the timeline, and then
    dropped: ``build_render_spec`` never passed them to the ``RenderSpec``. The
    compiler then correctly mapped ``[vout]`` instead of ``[vsub]`` -- it was
    told there were no subtitles -- and the render came out clean, valid, the
    right length, and with no text on it.

    Nothing caught it. The unit tests built specs directly, the integration
    tests built specs directly, and only a real render through the worker showed
    two identical frames. These assertions walk the whole chain instead.
    """

    def track(self) -> SubtitleTrack:
        return SubtitleTrack(cues=(SubtitleCue(0, 1_500, "hello"),))

    def test_the_plan_carries_them_to_the_timeline(self) -> None:
        from visionforge.domain.timeline import compile_timeline

        timeline = compile_timeline(plan(subtitles=self.track()))
        assert timeline.subtitles is not None
        assert timeline.subtitles.cues[0].text == "hello"

    def test_the_timeline_carries_them_to_the_spec(self) -> None:
        from visionforge.domain.timeline import build_render_spec, compile_timeline

        spec = build_render_spec(
            compile_timeline(plan(subtitles=self.track())),
            local_paths={MEDIA: "/tmp/x.mp4"},
            output_path="/tmp/out.mp4",
        )
        assert spec.subtitles is not None
        assert spec.subtitles.cues[0].text == "hello"

    def test_the_compiler_maps_the_burned_output_when_given_a_document(self) -> None:
        from visionforge.domain.timeline import build_render_spec, compile_timeline
        from visionforge.infra.ffmpeg.compiler import compile_render_argv

        spec = build_render_spec(
            compile_timeline(plan(subtitles=self.track())),
            local_paths={MEDIA: "/tmp/x.mp4"},
            output_path="/tmp/out.mp4",
        )
        argv = compile_render_argv(spec, ass_path="/tmp/subs.ass")
        assert "[vsub]" in argv
        assert "[vout]" not in argv
        assert "subtitles=" in argv[argv.index("-filter_complex") + 1]

    def test_a_plan_without_subtitles_maps_the_plain_output(self) -> None:
        from visionforge.domain.timeline import build_render_spec, compile_timeline
        from visionforge.infra.ffmpeg.compiler import compile_render_argv

        spec = build_render_spec(
            compile_timeline(plan()),
            local_paths={MEDIA: "/tmp/x.mp4"},
            output_path="/tmp/out.mp4",
        )
        assert spec.subtitles is None
        argv = compile_render_argv(spec)
        assert "[vout]" in argv
        assert "[vsub]" not in argv


class TestAssDocument:
    def test_the_document_has_the_sections_libass_needs(self) -> None:
        document = build_ass(SubtitleTrack(cues=(SubtitleCue(0, 1_000, "hi"),)))
        assert "[Script Info]" in document
        assert "[V4+ Styles]" in document
        assert "[Events]" in document
        assert "PlayResX: 1920" in document

    def test_a_cue_becomes_one_dialogue_line(self) -> None:
        document = build_ass(
            SubtitleTrack(cues=(SubtitleCue(200, 1_600, "hello"), SubtitleCue(2_000, 3_000, "bye")))
        )
        dialogues = [line for line in document.splitlines() if line.startswith("Dialogue:")]
        assert len(dialogues) == 2
        assert dialogues[0].endswith("hello")

    def test_timestamps_are_centiseconds_and_rounded(self) -> None:
        document = build_ass(SubtitleTrack(cues=(SubtitleCue(1_499, 2_501, "x"),)))
        line = next(row for row in document.splitlines() if row.startswith("Dialogue:"))
        assert "0:00:01.50" in line
        assert "0:00:02.50" in line

    def test_hostile_text_cannot_add_a_line_or_an_override(self) -> None:
        """The document is line-oriented, so a newline in the text would create a
        second event -- and a brace would be an override tag."""
        document = build_ass(
            SubtitleTrack(cues=(SubtitleCue(0, 1_000, "a\nDialogue: 0,0:00:00.00,x"),))
        )
        dialogues = [line for line in document.splitlines() if line.startswith("Dialogue:")]
        assert len(dialogues) == 1

    def test_vietnamese_reaches_the_document_unchanged(self) -> None:
        text = "Xin chào thế giới"
        document = build_ass(SubtitleTrack(cues=(SubtitleCue(0, 1_000, text),)))
        assert text in document

    def test_every_preset_produces_a_style_line(self) -> None:
        for style in SubtitleStyle:
            document = build_ass(SubtitleTrack(cues=(SubtitleCue(0, 1_000, "x"),), style=style))
            assert "Style: Default," in document
            assert preset_for(style).font in document

    def test_position_selects_the_alignment(self) -> None:
        for position in SubtitlePosition:
            document = build_ass(
                SubtitleTrack(cues=(SubtitleCue(0, 1_000, "x"),), position=position)
            )
            style_line = next(
                row for row in document.splitlines() if row.startswith("Style: Default")
            )
            assert f",{ALIGNMENT[position]}," in style_line

    def test_a_cue_that_cleans_to_nothing_is_dropped(self) -> None:
        document = build_ass(SubtitleTrack(cues=(SubtitleCue(0, 1_000, "{}"),)))
        assert not [line for line in document.splitlines() if line.startswith("Dialogue:")]


class TestPathEscaping:
    def test_a_windows_path_is_escaped_once(self) -> None:
        escaped = escape_filter_path(r"C:\Users\me\render\subtitles.ass")
        assert escaped == "C\\:/Users/me/render/subtitles.ass"

    def test_a_posix_path_needs_no_escaping(self) -> None:
        assert escape_filter_path("/tmp/render/subtitles.ass") == "/tmp/render/subtitles.ass"
