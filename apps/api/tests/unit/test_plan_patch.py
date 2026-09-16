"""Applying a delta to a plan.

The properties under test, in order of how much they matter:

**Atomic.** A delta whose second operation fails changes nothing -- not the plan
object, not half of it, not a copy with the first operation applied.

**Addressed against the plan the user saw.** Two removals in one delta remove
the two clips that were named, not the named one and its neighbour.

**Gated.** A patch that would produce an unrenderable plan is refused by the
same ``validate_plan`` every plan passes, with the violation reported against
the clip the user can see.

**Honest.** What the diff says changed is what changed, and an adjustment the
patcher made on its own is reported rather than applied quietly.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from visionforge.domain.editdelta import (
    AddEffect,
    AddSubtitle,
    ChangeBeatSync,
    ChangeDuration,
    ChangeMusicFade,
    ChangeMusicVolume,
    ChangeOutputPreset,
    ChangeStyleStrength,
    ChangeTransition,
    EditDelta,
    EditOperation,
    ModifyEffect,
    ModifySubtitle,
    RemoveEffect,
    RemoveSegment,
    RemoveSubtitle,
    ReorderSegment,
    TrimSegment,
)
from visionforge.domain.editplan import (
    AspectRatio,
    AudioMode,
    EditPlan,
    MediaFact,
    MusicCue,
    OutputSpec,
    QualityPreset,
    Segment,
    TransitionKind,
)
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.patch import PATCH_PLANNER, PatchContext, PatchOutcome, apply_delta
from visionforge.domain.policy import StyleStrength
from visionforge.domain.style import PRESET_DIMENSIONS
from visionforge.domain.subtitles import (
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
)

PROJECT = ProjectId(uuid.uuid4())
OTHER_PROJECT = ProjectId(uuid.uuid4())
MEDIA = [MediaId(uuid.uuid4()) for _ in range(4)]
MUSIC = MediaId(uuid.uuid4())

#: Every source is this long, so a trim has room to move in either direction.
SOURCE_MS = 30_000
CLIP_MS = 4_000


def segment(index: int, **kwargs: Any) -> Segment:
    return Segment(
        media_id=MEDIA[index],
        order=index,
        source_in_ms=0,
        source_out_ms=CLIP_MS,
        **kwargs,
    )


def plan(
    *,
    segments: tuple[Segment, ...] | None = None,
    music: MusicCue | None = None,
    subtitles: SubtitleTrack | None = None,
    output: OutputSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> EditPlan:
    return EditPlan(
        project_id=PROJECT,
        segments=segments if segments is not None else tuple(segment(i) for i in range(3)),
        output=output or OutputSpec(),
        music=music,
        subtitles=subtitles,
        planner="rules-engine",
        planner_version="1",
        metadata=dict(metadata or {}),
    )


def context(*, project: ProjectId = PROJECT) -> PatchContext:
    facts = {
        media_id: MediaFact(
            media_id=media_id,
            project_id=project,
            is_renderable=True,
            duration_ms=SOURCE_MS,
            width=1920,
            height=1080,
        )
        for media_id in MEDIA
    }
    facts[MUSIC] = MediaFact(
        media_id=MUSIC,
        project_id=project,
        is_renderable=False,
        duration_ms=SOURCE_MS,
        is_audio_asset=True,
    )
    return PatchContext(
        source_durations={media_id: SOURCE_MS for media_id in [*MEDIA, MUSIC]},
        dimensions=PRESET_DIMENSIONS,
        fps_presets=(24, 30, 60),
        media_facts=facts,
    )


def patch(
    base: EditPlan, *operations: EditOperation, ctx: PatchContext | None = None
) -> PatchOutcome:
    return apply_delta(base, EditDelta(operations=operations), ctx or context())


def music_cue(**kwargs: Any) -> MusicCue:
    defaults: dict[str, Any] = {
        "media_id": MUSIC,
        "source_in_ms": 0,
        "source_out_ms": 12_000,
        "timeline_start_ms": 0,
        "gain": 0.7,
        "fade_in_ms": 0,
        "fade_out_ms": 1_500,
    }
    defaults.update(kwargs)
    return MusicCue(**defaults)


def codes(outcome: PatchOutcome) -> list[str]:
    return [violation.code for violation in outcome.violations]


# ------------------------------------------------------------------- atomicity
def test_a_failing_operation_changes_nothing() -> None:
    """The first operation is legal and the second is not, so neither lands."""
    base = plan(music=music_cue())
    before = base.as_payload()

    outcome = patch(base, ChangeMusicVolume(value=0.4), RemoveSegment(segment=9))

    assert not outcome.ok
    assert codes(outcome) == ["no_such_segment"]
    # The plan handed back is the one passed in, and the original is untouched
    # -- including the volume the first operation would have changed.
    assert outcome.plan is base
    assert base.as_payload() == before


def test_a_patch_rejected_by_the_plan_gate_changes_nothing() -> None:
    """Every operation is legal on its own; the plan they produce is not."""
    base = plan(segments=(segment(0),))
    before = base.as_payload()

    # 300 ms is a legal segment but the programme floor is 1000 ms.
    outcome = patch(base, TrimSegment(segment=0, source_out_ms=300))

    assert not outcome.ok
    assert "output_duration" in codes(outcome)
    assert outcome.plan is base
    assert base.as_payload() == before


def test_a_successful_patch_leaves_the_original_object_alone() -> None:
    base = plan()
    before = base.as_payload()

    outcome = patch(base, RemoveSegment(segment=1))

    assert outcome.ok
    assert outcome.plan is not base
    assert base.as_payload() == before
    assert len(outcome.plan.segments) == 2


# ---------------------------------------------------------------- addressing
def test_two_removals_address_the_plan_the_user_saw() -> None:
    """``[remove 0, remove 1]`` removes the first two clips, not the first and
    then whatever slid into position 1."""
    base = plan(segments=tuple(segment(i) for i in range(4)))

    outcome = patch(base, RemoveSegment(segment=0), RemoveSegment(segment=1))

    assert outcome.ok
    assert [s.media_id for s in outcome.plan.ordered_segments] == [MEDIA[2], MEDIA[3]]


def test_removing_a_clip_twice_says_so() -> None:
    outcome = patch(plan(), RemoveSegment(segment=1), RemoveSegment(segment=1))
    assert not outcome.ok
    assert codes(outcome) == ["segment_removed"]


def test_an_operation_after_a_reorder_still_means_the_same_clip() -> None:
    base = plan()
    outcome = patch(
        base,
        ReorderSegment(segment=2, to_index=0),
        TrimSegment(segment=2, source_out_ms=2_000),
    )
    assert outcome.ok
    moved = outcome.plan.ordered_segments[0]
    assert moved.media_id == MEDIA[2]
    assert moved.duration_ms == 2_000


def test_orders_are_renumbered_contiguously() -> None:
    outcome = patch(plan(), RemoveSegment(segment=1))
    assert outcome.ok
    assert [s.order for s in outcome.plan.ordered_segments] == [0, 1]


# ------------------------------------------------------------------ structure
def test_the_last_clip_cannot_be_removed() -> None:
    outcome = patch(plan(segments=(segment(0),)), RemoveSegment(segment=0))
    assert not outcome.ok
    assert codes(outcome) == ["last_segment"]


def test_reorder_moves_a_clip_to_the_front() -> None:
    outcome = patch(plan(), ReorderSegment(segment=2, to_index=0))
    assert outcome.ok
    assert [s.media_id for s in outcome.plan.ordered_segments] == [MEDIA[2], MEDIA[0], MEDIA[1]]


def test_reorder_past_the_end_is_refused() -> None:
    outcome = patch(plan(), ReorderSegment(segment=0, to_index=7))
    assert not outcome.ok
    assert codes(outcome) == ["no_such_position"]


def test_trim_past_the_end_of_the_source_is_refused() -> None:
    # Inside the per-segment length bound, so it is the *source* that refuses it.
    outcome = patch(
        plan(), TrimSegment(segment=0, source_in_ms=10_000, source_out_ms=SOURCE_MS + 5_000)
    )
    assert not outcome.ok
    assert codes(outcome) == ["trim_past_end"]


def test_change_duration_of_one_clip_moves_its_out_point() -> None:
    outcome = patch(plan(), ChangeDuration(duration_ms=2_500, segment=1))
    assert outcome.ok
    changed = outcome.plan.ordered_segments[1]
    assert (changed.source_in_ms, changed.source_out_ms) == (0, 2_500)


def test_change_duration_pulls_the_in_point_back_when_the_tail_is_short() -> None:
    """A clip cut from the end of its source can still be lengthened."""
    tight = Segment(
        media_id=MEDIA[0], order=0, source_in_ms=SOURCE_MS - 2_000, source_out_ms=SOURCE_MS
    )
    base = plan(segments=(tight, segment(1)))

    outcome = patch(base, ChangeDuration(duration_ms=5_000, segment=0))

    assert outcome.ok
    first = outcome.plan.ordered_segments[0]
    assert (first.source_in_ms, first.source_out_ms) == (SOURCE_MS - 5_000, SOURCE_MS)


def test_change_duration_refuses_what_the_source_cannot_give() -> None:
    short = PatchContext(
        source_durations={media_id: 3_000 for media_id in MEDIA},
        dimensions=PRESET_DIMENSIONS,
        fps_presets=(24, 30, 60),
        media_facts={
            media_id: MediaFact(
                media_id=media_id, project_id=PROJECT, is_renderable=True, duration_ms=3_000
            )
            for media_id in MEDIA
        },
    )
    base = plan(
        segments=(
            Segment(media_id=MEDIA[0], order=0, source_in_ms=0, source_out_ms=3_000),
            Segment(media_id=MEDIA[1], order=1, source_in_ms=0, source_out_ms=3_000),
        )
    )
    outcome = patch(base, ChangeDuration(duration_ms=10_000, segment=0), ctx=short)
    assert not outcome.ok
    assert codes(outcome) == ["source_too_short"]


def test_change_duration_of_the_whole_edit_scales_every_clip() -> None:
    base = plan()  # three clips of 4 s = 12 s
    outcome = patch(base, ChangeDuration(duration_ms=6_000))

    assert outcome.ok
    assert [s.duration_ms for s in outcome.plan.ordered_segments] == [2_000, 2_000, 2_000]
    assert outcome.plan.total_duration_ms == 6_000


# ----------------------------------------------------------------- transitions
def test_change_transition_fits_a_crossfade_it_was_not_given_a_length_for() -> None:
    outcome = patch(plan(), ChangeTransition(segment=1, transition=TransitionKind.CROSSFADE))
    assert outcome.ok
    second = outcome.plan.ordered_segments[1]
    assert second.transition_in is TransitionKind.CROSSFADE
    assert 0 < second.transition_ms <= CLIP_MS // 2


def test_a_crossfade_on_the_first_clip_is_refused() -> None:
    outcome = patch(plan(), ChangeTransition(segment=0, transition=TransitionKind.CROSSFADE))
    assert not outcome.ok
    assert codes(outcome) == ["no_previous_clip"]


def test_a_crossfade_longer_than_the_clips_can_spare_is_refused() -> None:
    outcome = patch(
        plan(),
        ChangeTransition(segment=1, transition=TransitionKind.CROSSFADE, duration_ms=3_500),
    )
    assert not outcome.ok
    assert codes(outcome) == ["transition_duration"]


def test_removing_the_first_clip_refits_a_dissolve_that_lost_its_source() -> None:
    """The second clip becomes the first, so its crossfade becomes a fade in."""
    base = plan(
        segments=(
            segment(0),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=500),
            segment(2),
        )
    )
    outcome = patch(base, RemoveSegment(segment=0))

    assert outcome.ok
    assert outcome.plan.ordered_segments[0].transition_in is TransitionKind.FADE_IN
    assert any("fade in" in note for note in outcome.diff.adjustments)


def test_shortening_a_clip_shortens_a_dissolve_it_can_no_longer_carry() -> None:
    base = plan(
        segments=(
            segment(0),
            segment(1, transition_in=TransitionKind.CROSSFADE, transition_ms=1_800),
            segment(2),
        )
    )
    outcome = patch(base, ChangeDuration(duration_ms=1_000, segment=1))

    assert outcome.ok
    second = outcome.plan.ordered_segments[1]
    assert second.transition_ms <= 500
    assert any("shortened" in note for note in outcome.diff.adjustments)


# --------------------------------------------------------------------- effects
def test_add_effect_to_one_clip() -> None:
    outcome = patch(plan(), AddEffect(effect=EffectKind.SLOW_MOTION, amount=0.5, segment=0))
    assert outcome.ok
    assert outcome.plan.ordered_segments[0].effects == (
        Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),
    )
    # Half speed doubles the clip, and the plan's own arithmetic says so.
    assert outcome.plan.ordered_segments[0].output_duration_ms == CLIP_MS * 2


def test_add_effect_to_every_clip() -> None:
    outcome = patch(plan(), AddEffect(effect=EffectKind.CONTRAST, amount=1.15))
    assert outcome.ok
    assert all(
        any(effect.kind is EffectKind.CONTRAST for effect in s.effects)
        for s in outcome.plan.ordered_segments
    )


def test_adding_a_speed_replaces_the_other_direction() -> None:
    base = plan(
        segments=(
            segment(0, effects=(Effect(kind=EffectKind.SPEED_UP, amount=2.0),)),
            segment(1),
        )
    )
    outcome = patch(base, AddEffect(effect=EffectKind.SLOW_MOTION, amount=0.5, segment=0))

    assert outcome.ok
    kinds = [effect.kind for effect in outcome.plan.ordered_segments[0].effects]
    assert kinds == [EffectKind.SLOW_MOTION]


def test_modify_effect_needs_the_effect_to_be_there() -> None:
    outcome = patch(plan(), ModifyEffect(effect=EffectKind.ZOOM_IN, amount=0.2, segment=0))
    assert not outcome.ok
    assert codes(outcome) == ["effect_not_present"]


def test_modify_effect_changes_the_amount_in_place() -> None:
    base = plan(
        segments=(
            segment(0, effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.1),)),
            segment(1),
        )
    )
    outcome = patch(base, ModifyEffect(effect=EffectKind.ZOOM_IN, amount=0.25, segment=0))

    assert outcome.ok
    assert outcome.plan.ordered_segments[0].effects == (
        Effect(kind=EffectKind.ZOOM_IN, amount=0.25),
    )


def test_remove_effect_reports_when_there_was_none() -> None:
    outcome = patch(plan(), RemoveEffect(effect=EffectKind.ZOOM_IN))
    assert not outcome.ok
    assert codes(outcome) == ["effect_not_present"]


def test_remove_effect_across_the_edit_removes_what_is_there() -> None:
    base = plan(
        segments=(
            segment(0, effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.1),)),
            segment(1),
            segment(2, effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.2),)),
        )
    )
    outcome = patch(base, RemoveEffect(effect=EffectKind.ZOOM_IN))

    assert outcome.ok
    assert all(not s.effects for s in outcome.plan.ordered_segments)


# ------------------------------------------------------------------- subtitles
def track() -> SubtitleTrack:
    return SubtitleTrack(
        cues=(
            SubtitleCue(start_ms=0, end_ms=2_000, text="one"),
            SubtitleCue(start_ms=3_000, end_ms=5_000, text="two"),
        ),
        style=SubtitleStyle.CLEAN,
        position=SubtitlePosition.BOTTOM,
    )


def test_change_the_subtitle_style_for_the_whole_track() -> None:
    outcome = patch(plan(subtitles=track()), ModifySubtitle(cue=None, style=SubtitleStyle.BOLD))
    assert outcome.ok
    assert outcome.plan.subtitles is not None
    assert outcome.plan.subtitles.style is SubtitleStyle.BOLD
    assert any(entry.field == "subtitle_style" for entry in outcome.diff.entries)


def test_restyling_an_edit_with_no_subtitles_is_refused() -> None:
    outcome = patch(plan(), ModifySubtitle(cue=None, style=SubtitleStyle.BOLD))
    assert not outcome.ok
    assert codes(outcome) == ["no_subtitles"]


def test_add_a_cue() -> None:
    outcome = patch(
        plan(subtitles=track()), AddSubtitle(start_ms=6_000, end_ms=8_000, text="three")
    )
    assert outcome.ok
    assert outcome.plan.subtitles is not None
    assert len(outcome.plan.subtitles.cues) == 3


def test_an_overlapping_cue_is_refused() -> None:
    outcome = patch(plan(subtitles=track()), AddSubtitle(start_ms=1_000, end_ms=4_000, text="x"))
    assert not outcome.ok
    assert codes(outcome) == ["cue_overlap"]


def test_edit_one_cue_by_index() -> None:
    outcome = patch(plan(subtitles=track()), ModifySubtitle(cue=1, text="rewritten"))
    assert outcome.ok
    assert outcome.plan.subtitles is not None
    assert outcome.plan.subtitles.ordered[1].text == "rewritten"


def test_remove_one_cue_and_then_the_track() -> None:
    outcome = patch(plan(subtitles=track()), RemoveSubtitle(cue=0))
    assert outcome.ok
    assert outcome.plan.subtitles is not None
    assert len(outcome.plan.subtitles.cues) == 1

    outcome = patch(plan(subtitles=track()), RemoveSubtitle(cue=None))
    assert outcome.ok
    assert outcome.plan.subtitles is None


def test_shortening_the_edit_trims_the_cues_and_says_so() -> None:
    """Rather than refusing to shorten a subtitled edit, the tail is clipped."""
    base = plan(
        subtitles=SubtitleTrack(
            cues=(
                SubtitleCue(start_ms=0, end_ms=2_000, text="kept"),
                SubtitleCue(start_ms=9_000, end_ms=11_000, text="past the end"),
            )
        )
    )
    outcome = patch(base, ChangeDuration(duration_ms=6_000))

    assert outcome.ok
    assert outcome.plan.subtitles is not None
    assert [cue.text for cue in outcome.plan.subtitles.ordered] == ["kept"]
    assert any("removed" in note for note in outcome.diff.adjustments)


# ----------------------------------------------------------------------- music
def test_change_music_volume() -> None:
    outcome = patch(plan(music=music_cue()), ChangeMusicVolume(value=0.4))
    assert outcome.ok
    assert outcome.plan.music is not None
    assert outcome.plan.music.gain == 0.4
    entry = next(entry for entry in outcome.diff.entries if entry.field == "music_gain")
    assert (entry.before, entry.after) == ("70%", "40%")


def test_music_operations_on_a_silent_edit_are_refused() -> None:
    assert codes(patch(plan(), ChangeMusicVolume(value=0.4))) == ["no_music"]
    assert codes(patch(plan(), ChangeMusicFade(fade_in_ms=500))) == ["no_music"]


def test_change_music_fade() -> None:
    outcome = patch(plan(music=music_cue()), ChangeMusicFade(fade_in_ms=800, fade_out_ms=2_000))
    assert outcome.ok
    assert outcome.plan.music is not None
    assert (outcome.plan.music.fade_in_ms, outcome.plan.music.fade_out_ms) == (800, 2_000)


def test_fades_longer_than_the_cue_are_refused() -> None:
    outcome = patch(
        plan(music=music_cue(source_out_ms=3_000)),
        ChangeMusicFade(fade_in_ms=2_000, fade_out_ms=2_000),
    )
    assert not outcome.ok
    assert codes(outcome) == ["fades_overlap"]


# ---------------------------------------------------------------------- output
def test_change_output_preset_derives_geometry_server_side() -> None:
    outcome = patch(plan(), ChangeOutputPreset(aspect_ratio=AspectRatio.PORTRAIT_9_16))
    assert outcome.ok
    assert (outcome.plan.output.width, outcome.plan.output.height) == PRESET_DIMENSIONS[
        AspectRatio.PORTRAIT_9_16
    ]


def test_an_fps_this_server_does_not_offer_is_refused() -> None:
    outcome = patch(plan(), ChangeOutputPreset(fps=120))
    assert not outcome.ok
    assert codes(outcome) == ["fps_not_offered"]


def test_change_output_preset_leaves_unmentioned_fields_alone() -> None:
    base = plan(output=OutputSpec(quality=QualityPreset.HIGH, audio=AudioMode.SOURCE, fps=60))
    outcome = patch(base, ChangeOutputPreset(source_gain=0.3))

    assert outcome.ok
    assert outcome.plan.output.quality is QualityPreset.HIGH
    assert outcome.plan.output.audio is AudioMode.SOURCE
    assert outcome.plan.output.fps == 60
    assert outcome.plan.output.source_gain == 0.3


# ------------------------------------------------------- planning-time settings
def test_style_strength_and_beat_sync_are_recorded_for_the_next_plan() -> None:
    """Both are inputs to planning, so a patch records them rather than
    pretending the existing cuts were restyled or re-timed."""
    base = plan(metadata={"style_strength": "100", "beat_sync": True})
    outcome = patch(
        base, ChangeStyleStrength(value=StyleStrength.HALF), ChangeBeatSync(enabled=False)
    )

    assert outcome.ok
    assert outcome.plan.metadata["style_strength"] == "50"
    assert outcome.plan.metadata["beat_sync"] is False
    # The cuts themselves are identical -- which is exactly what the diff says.
    assert outcome.plan.ordered_segments == base.ordered_segments
    fields = {entry.field for entry in outcome.diff.entries}
    assert fields == {"style_strength", "beat_sync"}
    labels = " ".join(entry.label for entry in outcome.diff.entries)
    assert "next plan" in labels


# -------------------------------------------------------------------- ownership
def test_a_patch_cannot_reach_another_projects_media() -> None:
    """The gate runs on the patched plan with this project's facts.

    A plan whose media belongs elsewhere fails here exactly as it would at
    creation -- the co-editor gets no weaker a check than the planner.
    """
    base = plan()
    outcome = patch(
        base, ChangeOutputPreset(quality=QualityPreset.DRAFT), ctx=context(project=OTHER_PROJECT)
    )
    assert not outcome.ok
    assert "cross_project_media" in codes(outcome)
    assert outcome.plan is base


# -------------------------------------------------------------------- the diff
def test_the_diff_reports_what_actually_changed() -> None:
    base = plan(music=music_cue(), subtitles=track())
    outcome = patch(
        base,
        ChangeMusicVolume(value=0.4),
        ChangeDuration(duration_ms=3_200, segment=0),
        ModifySubtitle(cue=None, style=SubtitleStyle.BOLD),
    )

    assert outcome.ok
    fields = {entry.field for entry in outcome.diff.entries}
    assert {"music_gain", "segment_duration", "subtitle_style", "total_duration"} <= fields

    clip = next(entry for entry in outcome.diff.entries if entry.field == "segment_duration")
    assert (clip.before, clip.after, clip.clip) == ("4.0s", "3.2s", 1)

    subtitles = next(entry for entry in outcome.diff.entries if entry.field == "subtitle_style")
    assert (subtitles.before, subtitles.after) == ("Clean", "Bold")


def test_the_diff_lists_what_the_operations_said_they_would_do() -> None:
    outcome = patch(plan(music=music_cue()), ChangeMusicVolume(value=0.4))
    assert outcome.diff.applied == ("Music volume to 40%",)


def test_an_operation_that_changes_nothing_shows_an_empty_diff() -> None:
    """Setting a value to what it already is is not an error, and not a change."""
    outcome = patch(plan(music=music_cue(gain=0.4)), ChangeMusicVolume(value=0.4))
    assert outcome.ok
    assert outcome.diff.is_empty


# ------------------------------------------------------------------ provenance
def test_a_patched_plan_says_it_was_patched() -> None:
    outcome = patch(plan(), RemoveSegment(segment=1))
    assert outcome.ok
    assert outcome.plan.planner == PATCH_PLANNER
    assert outcome.plan.metadata["patched_from_planner"] == "rules-engine"

    recorded = outcome.plan.metadata["co_edit"]
    assert recorded["source"] == "rules"
    assert recorded["operations"] == [{"kind": "REMOVE_SEGMENT", "segment": 1}]


def test_the_patched_plan_round_trips_through_its_payload() -> None:
    """A patched plan is an ordinary plan: it stores and reloads like any other."""
    from visionforge.domain.editplan import plan_from_payload

    outcome = patch(plan(music=music_cue(), subtitles=track()), ChangeMusicVolume(value=0.55))
    assert outcome.ok
    assert plan_from_payload(outcome.plan.as_payload()).as_payload() == outcome.plan.as_payload()


@pytest.mark.parametrize(
    "operation",
    [
        RemoveSegment(segment=1),
        ReorderSegment(segment=0, to_index=2),
        TrimSegment(segment=0, source_in_ms=500, source_out_ms=3_000),
        ChangeDuration(duration_ms=2_000, segment=0),
        ChangeTransition(segment=1, transition=TransitionKind.FADE_IN, duration_ms=300),
        AddEffect(effect=EffectKind.BRIGHTNESS, amount=0.1, segment=0),
        ChangeOutputPreset(quality=QualityPreset.DRAFT),
    ],
)
def test_every_structural_operation_produces_a_renderable_plan(operation: EditOperation) -> None:
    """Whatever it does, what comes out passes the same gate a plan must pass."""
    outcome = patch(plan(music=music_cue()), operation)
    assert outcome.ok, codes(outcome)
    assert outcome.plan.total_duration_ms > 0
