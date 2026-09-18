"""Stills in the edit (Phase 12): how a photo is validated, looped and moved.

A photo has no length. A segment of one is a *hold*: it starts at zero, runs for
as long as the edit wants up to ``MAX_STILL_MS``, cannot change speed, and is
decoded as a looped picture with silence where its audio would be.
"""

from __future__ import annotations

import uuid

from visionforge.domain.editplan import (
    MAX_STILL_MS,
    AudioMode,
    EditPlan,
    MediaFact,
    OutputSpec,
    Segment,
    TransitionKind,
    validate_plan,
)
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.timeline import build_render_spec, compile_timeline
from visionforge.infra.ffmpeg.compiler import build_filter_graph, compile_render_argv

PROJECT = ProjectId(uuid.uuid4())
PHOTO = MediaId(uuid.uuid4())
CLIP = MediaId(uuid.uuid4())


def facts() -> dict[MediaId, MediaFact]:
    return {
        PHOTO: MediaFact.from_media(
            media_id=PHOTO,
            project_id=PROJECT,
            kind=MediaKind.IMAGE,
            status=MediaStatus.READY,
            # What ffprobe reports for a JPEG: one frame. It must be ignored.
            duration_ms=40,
        ),
        CLIP: MediaFact.from_media(
            media_id=CLIP,
            project_id=PROJECT,
            kind=MediaKind.VIDEO,
            status=MediaStatus.READY,
            duration_ms=10_000,
        ),
    }


def plan(*segments: Segment, audio: AudioMode = AudioMode.NONE) -> EditPlan:
    return EditPlan(project_id=PROJECT, segments=segments, output=OutputSpec(audio=audio))


def hold(order: int, out_ms: int, *, in_ms: int = 0, effects: tuple[Effect, ...] = ()) -> Segment:
    return Segment(
        media_id=PHOTO, order=order, source_in_ms=in_ms, source_out_ms=out_ms, effects=effects
    )


def cut(order: int) -> Segment:
    return Segment(media_id=CLIP, order=order, source_in_ms=1_000, source_out_ms=3_000)


def codes(edit: EditPlan) -> set[str]:
    return {violation.code for violation in validate_plan(edit, facts())}


# -------------------------------------------------------------- validation
def test_a_photo_held_from_zero_is_valid_beside_a_video() -> None:
    assert codes(plan(hold(0, 3_000), cut(1))) == set()


def test_a_photo_may_be_held_longer_than_its_probed_single_frame() -> None:
    """The 40 ms ffprobe reports for a JPEG is not a length to trim against."""
    assert "trim_past_end" not in codes(plan(hold(0, 5_000)))


def test_a_photo_must_be_held_from_zero() -> None:
    assert "still_not_from_zero" in codes(plan(hold(0, 3_000, in_ms=500)))


def test_a_photo_cannot_be_held_past_the_still_limit() -> None:
    assert "still_too_long" in codes(plan(hold(0, MAX_STILL_MS + 1)))


def test_a_photo_cannot_change_speed() -> None:
    slowed = hold(0, 3_000, effects=(Effect(EffectKind.SLOW_MOTION, 0.5),))
    assert "still_speed" in codes(plan(slowed))


# ------------------------------------------------------------------ render
def spec_for(edit: EditPlan):  # type: ignore[no-untyped-def]
    return build_render_spec(
        compile_timeline(edit),
        local_paths={PHOTO: "/w/photo.jpg", CLIP: "/w/clip.mp4"},
        output_path="/w/out.mp4",
        still_ids=frozenset({PHOTO}),
    )


def test_each_hold_of_a_photo_is_its_own_looped_input() -> None:
    """A photo used twice is opened twice, each looped for its own hold.

    Sharing one decoded stream made FFmpeg queue the later hold's frames in
    memory until the edit reached it -- over a gigabyte at 4K.
    """
    spec = spec_for(plan(hold(0, 2_000), cut(1), hold(2, 3_500)))
    argv = compile_render_argv(spec)

    holds = [i for i, arg in enumerate(argv) if arg == "/w/photo.jpg"]
    assert len(holds) == 2
    assert argv[holds[0] - 7 : holds[0]] == ["-loop", "1", "-framerate", "30", "-t", "2.000", "-i"]
    assert argv[holds[1] - 7 : holds[1]] == ["-loop", "1", "-framerate", "30", "-t", "3.500", "-i"]
    # The video is an ordinary input.
    clip_at = argv.index("/w/clip.mp4")
    assert argv[clip_at - 1] == "-i" and "-loop" not in argv[clip_at - 3 : clip_at]


def test_a_photo_gets_silence_when_source_audio_is_kept() -> None:
    graph = build_filter_graph(spec_for(plan(hold(0, 2_000), cut(1), audio=AudioMode.SOURCE)))
    assert "anullsrc=r=48000:cl=stereo,atrim=end=2.000" in graph
    # The photo's input is never asked for an audio stream; the video's is.
    photo_index = 0
    assert f"[{photo_index}:a]" not in graph
    assert "[1:a]" in graph


def test_a_pan_is_a_sliding_crop() -> None:
    panned = hold(0, 4_000, effects=(Effect(EffectKind.PAN_RIGHT, 0.2),))
    graph = build_filter_graph(spec_for(plan(panned)))
    assert "crop=w='iw*0.8':h='ih*0.8':x='(iw-ow)*min(t/4\\,1)':y='(ih-oh)/2'" in graph


def test_a_pan_left_starts_at_the_right_edge() -> None:
    panned = hold(0, 4_000, effects=(Effect(EffectKind.PAN_LEFT, 0.2),))
    graph = build_filter_graph(spec_for(plan(panned)))
    assert "x='(iw-ow)*(1-min(t/4\\,1))'" in graph


def test_a_single_faded_clip_still_produces_the_output_label() -> None:
    """One segment with a fade takes the transition path, which joins pairs.

    With nothing to join, the path used to emit no ``[vout]`` at all, and FFmpeg
    refused the output ("Output with label 'vout' does not exist").
    """
    faded = Segment(
        media_id=CLIP,
        order=0,
        source_in_ms=1_000,
        source_out_ms=3_000,
        transition_in=TransitionKind.FADE_IN,
        transition_ms=400,
    )
    graph = build_filter_graph(spec_for(plan(faded)))
    assert "[vout]" in graph
