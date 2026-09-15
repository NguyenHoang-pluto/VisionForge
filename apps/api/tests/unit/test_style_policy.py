"""Style strength: what the dial does, and what it must never do.

The single most important test in this file is
``test_zero_strength_is_the_preset_untouched``. Everything else describes how
the reference gets its influence; that one says the feature can be turned off
completely, which is what makes it safe to ship on by default in the UI.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.ids import MediaId
from visionforge.domain.policy import (
    MAX_AFFINITY,
    StylePolicy,
    StyleStrength,
    StyleTarget,
    blend,
    policy_for,
)
from visionforge.domain.reference import PROFILE_VERSION, Measurement, ReferenceProfile
from visionforge.domain.style import EditStyle, profile_for

MEDIA = MediaId(uuid.UUID("22222222-2222-2222-2222-222222222222"))

STRENGTHS = list(StyleStrength)


def reference(
    shot_ms: float = 700.0,
    confidence: float = 1.0,
    **overrides: object,
) -> ReferenceProfile:
    """A confidently-measured, fast-cutting, bright, busy reference."""
    defaults: dict[str, object] = {
        "shot_ms": Measurement(value=shot_ms, confidence=confidence),
        "shot_ms_p25": int(shot_ms * 0.7),
        "shot_ms_p75": int(shot_ms * 1.4),
        "cut_rate": Measurement(value=60.0, confidence=confidence),
        "scene_count": 40,
        "luminance": Measurement(value=0.7, confidence=confidence),
        "contrast": Measurement(value=0.6, confidence=confidence),
        "saturation": Measurement(value=0.8, confidence=confidence),
        "motion": Measurement(value=0.65, confidence=confidence),
        "beat_sync": Measurement(value=0.9, confidence=confidence),
        "bpm": 128.0,
        "beat_confidence": 0.9,
    }
    defaults.update(overrides)
    return ReferenceProfile(
        version=PROFILE_VERSION,
        media_id=MEDIA,
        source_duration_ms=60_000,
        **defaults,  # type: ignore[arg-type]
    )


class TestTheDial:
    def test_the_stops_are_the_five_the_product_offers(self) -> None:
        assert {s.value for s in StyleStrength} == {"0", "25", "50", "75", "100"}

    @pytest.mark.parametrize("strength", STRENGTHS)
    def test_fraction_matches_the_label(self, strength: StyleStrength) -> None:
        assert strength.fraction == int(strength.value) / 100


class TestZeroIsOff:
    @pytest.mark.parametrize("style", [None, *EditStyle])
    def test_zero_strength_is_the_preset_untouched(self, style: EditStyle | None) -> None:
        """The load-bearing guarantee.

        A plan made at strength 0 with a reference attached must be the same
        plan as one made with no reference at all -- for every style, not just
        the neutral one.
        """
        preset = profile_for(style)
        policy = blend(preset, reference(), StyleStrength.ZERO)

        assert policy.weights == preset.weights
        assert policy.weights.affinity == 0.0
        assert policy.min_clip_ms == preset.min_clip_ms
        assert policy.max_clip_ms == preset.max_clip_ms
        assert policy.target_clip_ms == preset.target_clip_ms
        assert policy.target.axes == {}
        assert policy.influence == {}
        assert not policy.is_styled
        assert not policy.suggests_beat_sync

    def test_no_reference_is_the_preset_untouched_at_any_strength(self) -> None:
        preset = profile_for(EditStyle.CINEMATIC)
        for strength in STRENGTHS:
            policy = blend(preset, None, strength)
            assert policy.target_clip_ms == preset.target_clip_ms
            assert policy.weights == preset.weights
            assert not policy.is_styled

    def test_an_unusable_reference_is_ignored_at_full_strength(self) -> None:
        """A profile with no pacing has nothing to steer towards."""
        empty = ReferenceProfile(version=PROFILE_VERSION, media_id=MEDIA, source_duration_ms=1_000)
        preset = profile_for(EditStyle.SOCIAL)
        policy = blend(preset, empty, StyleStrength.FULL)
        assert policy.target_clip_ms == preset.target_clip_ms
        assert not policy.is_styled


class TestPacingPull:
    def test_strength_moves_pacing_monotonically_toward_the_reference(self) -> None:
        preset = profile_for(EditStyle.CINEMATIC)  # long takes
        fast = reference(shot_ms=600.0)
        targets = [blend(preset, fast, s).target_clip_ms for s in STRENGTHS]

        assert targets[0] == preset.target_clip_ms
        assert targets == sorted(targets, reverse=True), "each stop must move further"
        assert targets[-1] == pytest.approx(600, abs=2)

    def test_full_strength_lands_on_the_measured_shot_length(self) -> None:
        policy = blend(
            profile_for(EditStyle.NATURE), reference(shot_ms=1_500.0), StyleStrength.FULL
        )
        assert policy.target_clip_ms == pytest.approx(1_500, abs=2)

    def test_low_confidence_pulls_proportionally_less(self) -> None:
        preset = profile_for(EditStyle.CINEMATIC)
        certain = blend(preset, reference(confidence=1.0), StyleStrength.FULL)
        unsure = blend(preset, reference(confidence=0.25), StyleStrength.FULL)

        # Both move toward the reference; the unsure one moves a quarter as far.
        assert certain.target_clip_ms < unsure.target_clip_ms < preset.target_clip_ms

    def test_the_window_always_contains_the_target(self) -> None:
        for style in EditStyle:
            for strength in STRENGTHS:
                for shot in (300.0, 900.0, 5_000.0, 20_000.0):
                    policy = blend(profile_for(style), reference(shot_ms=shot), strength)
                    assert policy.min_clip_ms <= policy.target_clip_ms <= policy.max_clip_ms
                    assert policy.clamp_clip_ms(policy.target_clip_ms) == policy.target_clip_ms

    def test_a_degenerate_window_is_refused_outright(self) -> None:
        with pytest.raises(ValueError):
            StylePolicy(
                strength=StyleStrength.HALF,
                style=None,
                weights=profile_for(None).weights,
                min_clip_ms=5_000,
                max_clip_ms=1_000,
                target_clip_ms=2_000,
                prefer_sequence=False,
                target=StyleTarget(),
            )


class TestRankingWeights:
    def test_affinity_is_taken_from_the_others_not_added_on_top(self) -> None:
        """The score has to stay inside 0..1, and the usability components have
        to keep their relative balance."""
        preset = profile_for(None)
        policy = blend(preset, reference(), StyleStrength.FULL)
        w = policy.weights

        total = w.sharpness + w.exposure + w.contrast + w.resolution + w.duration + w.affinity
        assert total == pytest.approx(1.0, abs=1e-9)

        base = preset.weights
        # Ratios preserved: only the scale changed.
        assert w.sharpness / w.exposure == pytest.approx(base.sharpness / base.exposure)
        assert w.contrast / w.duration == pytest.approx(base.contrast / base.duration)

    @pytest.mark.parametrize("strength", STRENGTHS)
    def test_affinity_never_exceeds_its_ceiling(self, strength: StyleStrength) -> None:
        policy = blend(profile_for(None), reference(), strength)
        assert 0.0 <= policy.weights.affinity <= MAX_AFFINITY

    def test_affinity_grows_with_strength(self) -> None:
        weights = [blend(profile_for(None), reference(), s).weights.affinity for s in STRENGTHS]
        assert weights[0] == 0.0
        assert weights == sorted(weights)
        assert weights[-1] == pytest.approx(MAX_AFFINITY)

    def test_a_reference_with_no_look_gets_no_affinity(self) -> None:
        """Pacing alone is a usable reference; it is not a palette."""
        pacing_only = reference(luminance=None, contrast=None, saturation=None, motion=None)
        policy = blend(profile_for(None), pacing_only, StyleStrength.FULL)
        assert policy.weights.affinity == 0.0
        assert policy.target.axes == {}
        assert policy.is_styled  # pacing still moved


class TestLookTarget:
    def test_the_target_carries_only_measured_axes(self) -> None:
        partial = reference(saturation=None, motion=None)
        policy = blend(profile_for(None), partial, StyleStrength.FULL)
        assert set(policy.target.axes) == {"luminance", "contrast"}

    def test_affinity_is_one_for_an_identical_candidate(self) -> None:
        target = StyleTarget(luminance=0.5, contrast=0.4, saturation=0.6, motion=0.3)
        assert target.affinity(luminance=0.5, contrast=0.4, saturation=0.6, motion=0.3) == 1.0

    def test_affinity_falls_with_distance(self) -> None:
        target = StyleTarget(luminance=0.5, motion=0.5)
        near = target.affinity(luminance=0.5, motion=0.45)
        far = target.affinity(luminance=0.0, motion=1.0)
        assert near is not None and far is not None
        assert near > far
        # Mean distance over the two axes is 0.5, so affinity is 0.5 -- the
        # floor at zero is reached only by a target at one end of every axis and
        # a candidate at the other.
        assert far == pytest.approx(0.5)
        assert StyleTarget(luminance=1.0).affinity(luminance=0.0) == 0.0

    def test_a_candidate_is_compared_only_on_shared_axes(self) -> None:
        """An unanalysed axis must not count as maximum distance -- that would
        rank footage lower for the analyser not having reached it."""
        target = StyleTarget(luminance=0.5, motion=0.9)
        only_luma = target.affinity(luminance=0.5, motion=None)
        assert only_luma == 1.0

    def test_no_shared_axis_is_unknown_rather_than_zero(self) -> None:
        assert StyleTarget(motion=0.5).affinity(luminance=0.5) is None
        assert StyleTarget().affinity(luminance=0.5) is None


class TestBeatSyncIsAdvisory:
    def test_a_beat_cut_reference_suggests_but_does_not_decide(self) -> None:
        policy = blend(profile_for(None), reference(), StyleStrength.FULL)
        assert policy.suggests_beat_sync is True
        # The policy has no field with which to turn it on: the request owns it.
        assert not hasattr(policy, "beat_sync")

    def test_a_reference_not_cut_to_music_suggests_nothing(self) -> None:
        loose = reference(beat_sync=Measurement(value=0.1, confidence=1.0))
        assert blend(profile_for(None), loose, StyleStrength.FULL).suggests_beat_sync is False

    def test_an_unmeasured_tendency_suggests_nothing(self) -> None:
        unknown = reference(beat_sync=None)
        assert blend(profile_for(None), unknown, StyleStrength.FULL).suggests_beat_sync is False


class TestPayload:
    def test_the_payload_explains_what_moved(self) -> None:
        policy = blend(profile_for(EditStyle.SPORTS_HIGHLIGHT), reference(), StyleStrength.HALF)
        payload = policy.as_payload()
        assert payload["strength"] == "50"
        assert payload["style"] == "sports_highlight"
        assert set(payload["influence"]) >= {"shot_ms", "luminance", "motion"}  # type: ignore[arg-type]

    def test_the_payload_carries_no_reference_identity(self) -> None:
        policy = blend(profile_for(None), reference(), StyleStrength.FULL)
        assert str(MEDIA) not in repr(policy.as_payload())


class TestEntryPoint:
    def test_policy_for_defaults_to_no_reference_and_no_strength(self) -> None:
        assert (
            policy_for(EditStyle.ANIME).target_clip_ms
            == profile_for(EditStyle.ANIME).target_clip_ms
        )
        assert policy_for(None).weights == profile_for(None).weights
