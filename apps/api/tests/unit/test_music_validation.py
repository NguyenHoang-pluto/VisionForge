"""The music cue, and the gate it has to pass.

The video track's validator is the model here: every check it makes about a
segment -- a range that is not a range, a trim past the end, an asset from
another project -- has an audio counterpart, because a music cue is exactly as
capable of naming somebody else's file as a video segment is.

The audio-only checks are gain, fades and whether the cue plays at all.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.editplan import (
    MAX_FADE_MS,
    MAX_GAIN,
    MIN_MUSIC_MS,
    AudioMode,
    EditPlan,
    MediaFact,
    MusicCue,
    OutputSpec,
    PlanInvalidError,
    Segment,
    assert_valid,
    media_id_from,
    plan_from_payload,
    validate_plan,
)
from visionforge.domain.ids import ProjectId

PROJECT = ProjectId(uuid.uuid4())
OTHER_PROJECT = ProjectId(uuid.uuid4())
VIDEO = media_id_from(uuid.uuid4())
TRACK = media_id_from(uuid.uuid4())


def facts(
    *,
    track_duration: int | None = 60_000,
    track_project: ProjectId = PROJECT,
    track_is_audio: bool = True,
) -> dict:
    return {
        VIDEO: MediaFact(
            media_id=VIDEO,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=30_000,
        ),
        TRACK: MediaFact(
            media_id=TRACK,
            project_id=track_project,
            is_renderable=False,
            duration_ms=track_duration,
            is_audio_asset=track_is_audio,
        ),
    }


def plan(music: MusicCue | None = None, **output: object) -> EditPlan:
    return EditPlan(
        project_id=PROJECT,
        segments=(
            Segment(media_id=VIDEO, order=0, source_in_ms=0, source_out_ms=5_000),
            Segment(media_id=VIDEO, order=1, source_in_ms=5_000, source_out_ms=10_000),
        ),
        output=OutputSpec(**output),  # type: ignore[arg-type]
        music=music,
    )


def cue(**overrides: object) -> MusicCue:
    values: dict = {
        "media_id": TRACK,
        "source_in_ms": 0,
        "source_out_ms": 10_000,
        "timeline_start_ms": 0,
        "gain": 0.7,
        "fade_in_ms": 500,
        "fade_out_ms": 1_000,
    }
    values.update(overrides)
    return MusicCue(**values)  # type: ignore[arg-type]


def codes(p: EditPlan, media_facts: dict | None = None) -> set[str]:
    return {v.code for v in validate_plan(p, media_facts if media_facts is not None else facts())}


# ------------------------------------------------------------------ the happy path
class TestValidCue:
    def test_a_well_formed_cue_passes(self) -> None:
        assert codes(plan(cue())) == set()

    def test_a_plan_with_no_music_is_unaffected(self) -> None:
        """Phase 4-6 plans must validate exactly as they did."""
        assert codes(plan(None)) == set()

    def test_duration_is_derived_not_stated(self) -> None:
        assert cue(source_in_ms=2_000, source_out_ms=9_000).duration_ms == 7_000

    def test_timeline_end_follows_the_start(self) -> None:
        c = cue(source_in_ms=0, source_out_ms=8_000, timeline_start_ms=1_500)
        assert c.timeline_end_ms == 9_500


# ----------------------------------------------------------------------- ranges
class TestRanges:
    def test_rejects_a_negative_in_point(self) -> None:
        assert "music_negative_in" in codes(plan(cue(source_in_ms=-1)))

    def test_rejects_an_out_point_that_does_not_exceed_the_in_point(self) -> None:
        assert "music_non_positive_duration" in codes(
            plan(cue(source_in_ms=5_000, source_out_ms=5_000))
        )

    def test_an_inverted_range_suppresses_the_dependent_checks(self) -> None:
        """Everything downstream needs a sane range; reporting noise from one
        broken field helps nobody find the broken field."""
        found = codes(plan(cue(source_in_ms=9_000, source_out_ms=1_000, gain=99.0)))
        assert found == {"music_non_positive_duration"}

    def test_rejects_a_cue_shorter_than_the_floor(self) -> None:
        assert "music_too_short" in codes(plan(cue(source_out_ms=MIN_MUSIC_MS - 1)))

    def test_rejects_a_trim_past_the_end_of_the_track(self) -> None:
        found = codes(plan(cue(source_out_ms=40_000)), facts(track_duration=30_000))
        assert "music_trim_past_end" in found

    def test_accepts_a_trim_to_exactly_the_end(self) -> None:
        found = codes(plan(cue(source_out_ms=30_000)), facts(track_duration=30_000))
        assert "music_trim_past_end" not in found

    def test_a_track_of_unknown_duration_is_not_second_guessed(self) -> None:
        """ffprobe could not measure it; refusing every trim would be worse."""
        found = codes(plan(cue(source_out_ms=999_000)), facts(track_duration=None))
        assert "music_trim_past_end" not in found


# ------------------------------------------------------------------- placement
class TestPlacement:
    def test_rejects_a_negative_start(self) -> None:
        assert "music_negative_start" in codes(plan(cue(timeline_start_ms=-1)))

    def test_rejects_music_that_starts_after_the_video_ends(self) -> None:
        """Silence the user asked for by mistake. Saying so beats rendering it."""
        assert "music_starts_after_end" in codes(plan(cue(timeline_start_ms=10_000)))

    def test_allows_music_that_starts_inside_the_timeline(self) -> None:
        assert "music_starts_after_end" not in codes(plan(cue(timeline_start_ms=9_999)))

    def test_allows_music_longer_than_the_video(self) -> None:
        """Overhang is trimmed by the compiler, not rejected here: a bed that
        outlasts the picture is normal, and the render is what cuts it."""
        assert codes(plan(cue(source_out_ms=60_000))) == set()


# ----------------------------------------------------------------------- levels
class TestLevels:
    @pytest.mark.parametrize("gain", [-0.1, MAX_GAIN + 0.1, 99.0])
    def test_rejects_gain_outside_the_range(self, gain: float) -> None:
        assert "music_gain_range" in codes(plan(cue(gain=gain)))

    @pytest.mark.parametrize("gain", [0.0, 0.5, 1.0, MAX_GAIN])
    def test_accepts_gain_inside_the_range(self, gain: float) -> None:
        assert "music_gain_range" not in codes(plan(cue(gain=gain)))

    def test_rejects_source_gain_outside_the_range(self) -> None:
        assert "source_gain_range" in codes(plan(cue(), source_gain=5.0))

    def test_source_gain_defaults_to_unity(self) -> None:
        assert OutputSpec().source_gain == 1.0


# ------------------------------------------------------------------------ fades
class TestFades:
    def test_rejects_a_negative_fade(self) -> None:
        assert "music_negative_fade" in codes(plan(cue(fade_in_ms=-1)))

    def test_rejects_a_fade_longer_than_the_ceiling(self) -> None:
        assert "music_fade_too_long" in codes(plan(cue(fade_out_ms=MAX_FADE_MS + 1)))

    def test_rejects_fades_that_together_exceed_the_cue(self) -> None:
        """Overlapping ramps make afade's behaviour a coin toss."""
        found = codes(plan(cue(source_out_ms=4_000, fade_in_ms=3_000, fade_out_ms=3_000)))
        assert "music_fades_overlap" in found

    def test_accepts_fades_that_exactly_fill_the_cue(self) -> None:
        found = codes(plan(cue(source_out_ms=4_000, fade_in_ms=2_000, fade_out_ms=2_000)))
        assert "music_fades_overlap" not in found

    def test_accepts_no_fades(self) -> None:
        assert codes(plan(cue(fade_in_ms=0, fade_out_ms=0))) == set()


# --------------------------------------------------------------------- ownership
class TestOwnership:
    def test_rejects_an_unknown_asset(self) -> None:
        stripped = facts()
        del stripped[TRACK]
        assert "music_unknown_media" in codes(plan(cue()), stripped)

    def test_rejects_an_asset_from_another_project(self) -> None:
        """The authorisation check, enforced in the domain and not only at the
        route -- the same rule the video track already has."""
        found = codes(plan(cue()), facts(track_project=OTHER_PROJECT))
        assert "music_cross_project_media" in found

    def test_rejects_a_video_used_as_a_music_bed(self) -> None:
        found = codes(plan(cue()), facts(track_is_audio=False))
        assert "music_not_audio" in found


# ------------------------------------------------------------------- the gate
class TestGate:
    def test_assert_valid_raises_with_every_violation(self) -> None:
        bad = plan(cue(gain=9.0, fade_in_ms=-5, timeline_start_ms=-1))
        with pytest.raises(PlanInvalidError) as caught:
            assert_valid(bad, facts())
        found = {v.code for v in caught.value.violations}
        assert {"music_gain_range", "music_negative_fade", "music_negative_start"} <= found

    def test_music_violations_are_flagged_as_such(self) -> None:
        """So an editor can point at the audio lane instead of guessing."""
        violations = validate_plan(plan(cue(gain=9.0)), facts())
        music = [v for v in violations if v.code == "music_gain_range"]
        assert music and music[0].is_music is True

    def test_segment_violations_are_not(self) -> None:
        bad = EditPlan(
            project_id=PROJECT,
            segments=(Segment(media_id=VIDEO, order=0, source_in_ms=0, source_out_ms=10),),
        )
        violations = [v for v in validate_plan(bad, facts()) if v.code == "segment_too_short"]
        assert violations and violations[0].is_music is False


# ----------------------------------------------------------------- persistence
class TestPersistence:
    def test_a_plan_with_music_round_trips(self) -> None:
        original = plan(cue(), audio=AudioMode.SOURCE, source_gain=0.4)
        restored = plan_from_payload(original.as_payload())
        assert restored.music == original.music
        assert restored.output.source_gain == pytest.approx(0.4)
        assert restored.output.audio is AudioMode.SOURCE

    def test_a_phase_six_payload_deserialises_with_no_music(self) -> None:
        """The compatibility guarantee: an older plan renders to the same bytes."""
        payload = plan(None).as_payload()
        del payload["music"]
        del payload["output"]["source_gain"]
        restored = plan_from_payload(payload)
        assert restored.music is None
        assert restored.output.source_gain == 1.0

    def test_the_cue_payload_states_its_derived_duration(self) -> None:
        body = cue(source_in_ms=1_000, source_out_ms=6_000).as_payload()
        assert body["duration_ms"] == 5_000
