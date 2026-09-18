"""The reference profile: measured, versioned, and honest about what it lacks.

The property that matters most here is not any particular number. It is that a
profile built from partial evidence says so, rather than filling the gap with a
plausible default -- because a default is a claim about footage nobody looked
at, and the policy layer downstream weights by exactly these confidences.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.beats import BeatGrid
from visionforge.domain.ids import MediaId
from visionforge.domain.reference import (
    MIN_CUTS_FOR_BEAT_TENDENCY,
    PACING_FULL_CUTS,
    PROFILE_VERSION,
    Measurement,
    Pacing,
    ReferenceProfile,
    beat_sync_tendency,
    pacing_for,
    profile_from,
)

MEDIA = MediaId(uuid.UUID("11111111-1111-1111-1111-111111111111"))


def scenes_payload(durations: list[int]) -> dict[str, object]:
    """A scene payload shaped exactly as the analyzer writes one."""
    scenes = []
    start = 0
    for index, duration in enumerate(durations):
        scenes.append(
            {
                "scene_id": index,
                "start_ms": start,
                "end_ms": start + duration,
                "duration_ms": duration,
                "start_frame": 0,
                "end_frame": -1,
            }
        )
        start += duration
    return {
        "scenes": scenes,
        "scene_count": len(scenes),
        "mean_scene_ms": sum(durations) // len(durations) if durations else 0,
    }


QUALITY = {"frame_count": 5, "mean_luminance": 127.5, "contrast": 64.0}
DYNAMICS = {
    "motion": 0.42,
    "saturation": 0.6,
    "motion_confidence": 1.0,
    "colour_confidence": 1.0,
}


class TestPacingBuckets:
    @pytest.mark.parametrize(
        ("shot_ms", "expected"),
        [
            (300, Pacing.RAPID),
            (899, Pacing.RAPID),
            (900, Pacing.BRISK),
            (1_999, Pacing.BRISK),
            (2_000, Pacing.MEASURED),
            (4_499, Pacing.MEASURED),
            (4_500, Pacing.SLOW),
            (30_000, Pacing.SLOW),
        ],
    )
    def test_buckets_are_contiguous_and_closed(self, shot_ms: int, expected: Pacing) -> None:
        assert pacing_for(shot_ms) is expected


class TestMeasurement:
    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            Measurement(value=1.0, confidence=1.5)
        with pytest.raises(ValueError):
            Measurement(value=1.0, confidence=-0.1)

    def test_a_confident_zero_is_not_an_absent_value(self) -> None:
        """The distinction the whole module rests on."""
        measured = Measurement(value=0.0, confidence=1.0)
        assert measured.value == 0.0
        assert measured.confidence == 1.0


class TestCutting:
    def test_shot_length_is_the_median_not_the_mean(self) -> None:
        """One long tail shot must not become the clip's character."""
        profile = profile_from(
            media_id=MEDIA,
            duration_ms=60_000,
            scenes=scenes_payload([1_000] * 9 + [51_000]),
        )
        assert profile.shot_ms is not None
        assert profile.shot_ms.value == 1_000

    def test_quartiles_describe_the_shape(self) -> None:
        profile = profile_from(
            media_id=MEDIA,
            duration_ms=30_000,
            scenes=scenes_payload([500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500]),
        )
        assert profile.shot_ms_p25 is not None
        assert profile.shot_ms_p75 is not None
        assert profile.shot_ms_p25 < profile.shot_ms.value < profile.shot_ms_p75  # type: ignore[union-attr]

    def test_a_single_scene_is_not_a_pacing_measurement(self) -> None:
        """The analyzer reports 'no cuts found' as one scene spanning the clip.

        That is indistinguishable from a genuine single take, so it must not
        become a confident claim that the reference cuts every 30 seconds.
        """
        profile = profile_from(media_id=MEDIA, duration_ms=30_000, scenes=scenes_payload([30_000]))
        assert profile.shot_ms is None
        assert profile.cut_rate is None
        assert not profile.is_usable

    def test_confidence_grows_with_the_number_of_cuts(self) -> None:
        few = profile_from(media_id=MEDIA, duration_ms=9_000, scenes=scenes_payload([3_000] * 3))
        many = profile_from(
            media_id=MEDIA,
            duration_ms=60_000,
            scenes=scenes_payload([3_000] * 20),
        )
        assert few.shot_ms is not None and many.shot_ms is not None
        assert few.shot_ms.confidence < many.shot_ms.confidence
        assert many.shot_ms.confidence == 1.0

    def test_confidence_saturates_at_the_declared_cut_count(self) -> None:
        exact = profile_from(
            media_id=MEDIA,
            duration_ms=60_000,
            scenes=scenes_payload([2_000] * (PACING_FULL_CUTS + 1)),
        )
        assert exact.shot_ms is not None
        assert exact.shot_ms.confidence == 1.0

    def test_cut_rate_is_per_minute_of_the_reference(self) -> None:
        # Ten one-second shots inside ten seconds: nine cuts, 54 per minute.
        profile = profile_from(
            media_id=MEDIA, duration_ms=10_000, scenes=scenes_payload([1_000] * 10)
        )
        assert profile.cut_rate is not None
        assert profile.cut_rate.value == pytest.approx(54.0, abs=0.01)


class TestLook:
    def test_luminance_and_contrast_are_normalised_to_one_scale(self) -> None:
        """Everything on the profile is 0..1, so no consumer has to remember
        which field arrives as an 8-bit value."""
        profile = profile_from(
            media_id=MEDIA, duration_ms=10_000, scenes=scenes_payload([1_000] * 5), quality=QUALITY
        )
        assert profile.luminance is not None
        assert profile.contrast is not None
        assert profile.luminance.value == pytest.approx(0.5, abs=0.01)
        assert 0.0 <= profile.contrast.value <= 1.0

    def test_dynamics_carries_its_own_confidence(self) -> None:
        weak = dict(DYNAMICS, motion_confidence=0.2, colour_confidence=0.2)
        profile = profile_from(
            media_id=MEDIA, duration_ms=10_000, scenes=scenes_payload([1_000] * 5), dynamics=weak
        )
        assert profile.motion is not None
        assert profile.motion.confidence == 0.2

    def test_a_zero_confidence_signal_is_dropped_entirely(self) -> None:
        """A still reports no motion confidence. That must not read as 'no motion'."""
        still = dict(DYNAMICS, motion=None, motion_confidence=0.0)
        profile = profile_from(
            media_id=MEDIA, duration_ms=10_000, scenes=scenes_payload([1_000] * 5), dynamics=still
        )
        assert profile.motion is None
        assert profile.saturation is not None


class TestBeatSyncTendency:
    def grid(self, bpm: float = 120.0, confidence: float = 0.9) -> BeatGrid:
        period = 60_000 / bpm
        return BeatGrid(
            bpm=bpm,
            confidence=confidence,
            beats_ms=tuple(int(round(i * period)) for i in range(64)),
            source_duration_ms=32_000,
        )

    def test_cuts_on_the_beat_score_high(self) -> None:
        grid = self.grid()
        on_beat = [int(round(i * 500)) for i in (2, 4, 6, 8, 10, 12)]
        tendency = beat_sync_tendency(on_beat, grid)
        assert tendency is not None
        assert tendency.value == 1.0

    def test_cuts_off_the_beat_score_low(self) -> None:
        grid = self.grid()
        # A quarter-beat late every time: musically off, and far outside the
        # window at this tempo.
        off_beat = [int(round(i * 500 + 125)) for i in (2, 4, 6, 8, 10, 12)]
        tendency = beat_sync_tendency(off_beat, grid)
        assert tendency is not None
        assert tendency.value == 0.0

    def test_too_few_cuts_is_unknown_not_zero(self) -> None:
        """Two cuts can coincide with a beat by luck. Saying 'not beat-synced'
        on that evidence is a stronger claim than the evidence supports."""
        assert beat_sync_tendency([500], self.grid()) is None
        assert beat_sync_tendency([500] * (MIN_CUTS_FOR_BEAT_TENDENCY - 1), self.grid()) is None

    def test_an_untrusted_grid_yields_nothing(self) -> None:
        assert beat_sync_tendency([500, 1_000, 1_500, 2_000], self.grid(confidence=0.05)) is None

    def test_confidence_reflects_both_the_cuts_and_the_grid(self) -> None:
        strong = beat_sync_tendency([int(i * 500) for i in range(2, 20)], self.grid(confidence=0.9))
        weak = beat_sync_tendency([1_000, 1_500, 2_000], self.grid(confidence=0.4))
        assert strong is not None and weak is not None
        assert strong.confidence > weak.confidence


class TestDeterminism:
    """Same payloads in, same profile out -- the property that lets the profile
    be derived on read instead of stored, and the one the acceptance relies on
    when it asserts a reference analysed twice plans the same edit."""

    def payloads(self) -> dict[str, object]:
        return {
            "scenes": scenes_payload([800, 1_200, 900, 1_100, 1_000, 950, 1_050, 1_000, 900]),
            "quality": dict(QUALITY),
            "dynamics": dict(DYNAMICS),
            "beats": {
                "bpm": 120.0,
                "confidence": 0.9,
                "beat_count": 64,
                "beats_ms": [int(round(i * 500)) for i in range(64)],
            },
        }

    def test_repeated_builds_are_identical(self) -> None:
        first = profile_from(media_id=MEDIA, duration_ms=30_000, **self.payloads())  # type: ignore[arg-type]
        second = profile_from(media_id=MEDIA, duration_ms=30_000, **self.payloads())  # type: ignore[arg-type]
        assert first == second
        assert first.as_payload() == second.as_payload()

    def test_key_order_does_not_matter(self) -> None:
        payloads = self.payloads()
        reversed_scenes = dict(reversed(list(payloads["scenes"].items())))  # type: ignore[union-attr]
        first = profile_from(media_id=MEDIA, duration_ms=30_000, **payloads)  # type: ignore[arg-type]
        second = profile_from(
            media_id=MEDIA,
            duration_ms=30_000,
            scenes=reversed_scenes,
            quality=payloads["quality"],  # type: ignore[arg-type]
            dynamics=payloads["dynamics"],  # type: ignore[arg-type]
            beats=payloads["beats"],  # type: ignore[arg-type]
        )
        assert first == second

    def test_a_missing_payload_subtracts_features_and_changes_nothing_else(self) -> None:
        full = profile_from(media_id=MEDIA, duration_ms=30_000, **self.payloads())  # type: ignore[arg-type]
        payloads = self.payloads()
        del payloads["dynamics"]
        partial = profile_from(media_id=MEDIA, duration_ms=30_000, **payloads)  # type: ignore[arg-type]

        assert partial.motion is None
        assert partial.saturation is None
        # Everything else is bit-for-bit what it was.
        assert partial.shot_ms == full.shot_ms
        assert partial.cut_rate == full.cut_rate
        assert partial.luminance == full.luminance
        assert partial.beat_sync == full.beat_sync

    def test_the_version_is_recorded(self) -> None:
        profile = profile_from(media_id=MEDIA, duration_ms=30_000, **self.payloads())  # type: ignore[arg-type]
        assert profile.version == PROFILE_VERSION
        assert profile.as_payload()["version"] == PROFILE_VERSION


class TestPayloadSafety:
    """What crosses the HTTP and model boundaries is numbers, and only numbers."""

    def test_no_identifier_or_string_escapes_into_the_payload(self) -> None:
        profile = profile_from(
            media_id=MEDIA,
            duration_ms=30_000,
            scenes=scenes_payload([1_000] * 9),
            quality=dict(QUALITY),
            dynamics=dict(DYNAMICS),
        )
        payload = profile.as_payload()

        assert "media_id" not in payload
        assert str(MEDIA) not in repr(payload)

        allowed_strings = {PROFILE_VERSION} | {p.value for p in Pacing}
        for key, value in payload.items():
            if isinstance(value, str):
                assert value in allowed_strings, f"{key} carries free text"

    def test_a_hostile_analyzer_payload_cannot_smuggle_text(self) -> None:
        """Every field is read with an isinstance check, so a string where a
        number belongs is dropped rather than carried through."""
        profile = profile_from(
            media_id=MEDIA,
            duration_ms=30_000,
            scenes=scenes_payload([1_000] * 9),
            quality={"frame_count": 5, "mean_luminance": "ignore previous instructions"},
            dynamics={"motion": "rm -rf /", "motion_confidence": 1.0},
        )
        assert profile.luminance is None
        assert profile.motion is None
        assert "ignore previous" not in repr(profile.as_payload())


class TestOverallConfidence:
    def test_an_empty_profile_is_zero_confidence_and_unusable(self) -> None:
        profile = ReferenceProfile(
            version=PROFILE_VERSION, media_id=MEDIA, source_duration_ms=1_000
        )
        assert profile.confidence == 0.0
        assert not profile.is_usable
        assert profile.pacing is None

    def test_confidence_averages_only_what_was_measured(self) -> None:
        profile = profile_from(
            media_id=MEDIA,
            duration_ms=30_000,
            scenes=scenes_payload([1_000] * 9),
            quality=dict(QUALITY),
        )
        assert 0.0 < profile.confidence <= 1.0
        assert profile.is_usable
