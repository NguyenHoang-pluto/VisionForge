"""Style-aware planning, through the rules engine.

The reference is allowed to change which clips are chosen and how long they run.
It is never allowed to change what a plan *is*: the same validation, the same
bounds, the same structure, and -- at strength zero -- the same bytes.

The LLM is absent from this file on purpose. The rules engine has to honour a
reference on its own, because that is what makes the fallback a fallback rather
than a downgrade, and because a deployment with no model configured is the
deployment this runs on.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.editplan import MAX_SEGMENT_MS, MIN_SEGMENT_MS
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind
from visionforge.domain.planner import PlanRequest, RulesEnginePlanner
from visionforge.domain.policy import StyleStrength, blend, policy_for
from visionforge.domain.reference import PROFILE_VERSION, Measurement, ReferenceProfile
from visionforge.domain.selection import Candidate
from visionforge.domain.style import EditStyle, profile_for

PROJECT = ProjectId(uuid.UUID("33333333-3333-3333-3333-333333333333"))
REFERENCE_MEDIA = MediaId(uuid.UUID("44444444-4444-4444-4444-444444444444"))


def candidates(count: int = 6, duration_ms: int = 20_000) -> list[Candidate]:
    """Clips good enough to survive selection, distinguishable by hash.

    The hashes are 16 bits apart: the selector drops anything within five bits
    of a clip it already took, so sequential counters would deduplicate the
    whole set down to one and every assertion below would be about a single
    clip.
    """
    return [
        Candidate(
            media_id=MediaId(uuid.uuid4()),
            kind=MediaKind.VIDEO,
            is_ready=True,
            duration_ms=duration_ms,
            width=1920,
            height=1080,
            blur_score=400.0,
            contrast=50.0,
            mean_luminance=128.0,
            clipped_ratio=0.01,
            phash=f"{(i * 0x1111111111111111) & 0xFFFFFFFFFFFFFFFF:016x}",
            sequence=i,
        )
        for i in range(1, count + 1)
    ]


def fast_reference(**overrides: object) -> ReferenceProfile:
    defaults: dict[str, object] = {
        "shot_ms": Measurement(value=800.0, confidence=1.0),
        "shot_ms_p25": 600,
        "shot_ms_p75": 1_200,
        "cut_rate": Measurement(value=70.0, confidence=1.0),
        "scene_count": 50,
        "luminance": Measurement(value=0.5, confidence=1.0),
        "contrast": Measurement(value=0.4, confidence=1.0),
        "saturation": Measurement(value=0.7, confidence=1.0),
        "motion": Measurement(value=0.8, confidence=1.0),
    }
    defaults.update(overrides)
    return ReferenceProfile(
        version=PROFILE_VERSION,
        media_id=REFERENCE_MEDIA,
        source_duration_ms=120_000,
        **defaults,  # type: ignore[arg-type]
    )


def request(**overrides: object) -> PlanRequest:
    base: dict[str, object] = {
        "project_id": PROJECT,
        "target_duration_ms": 24_000,
        "max_clips": 6,
        "min_clips": 2,
    }
    base.update(overrides)
    return PlanRequest(**base)  # type: ignore[arg-type]


class TestStrengthZeroChangesNothing:
    """The guarantee that lets this ship: a reference attached at zero is a
    reference that is not attached."""

    @pytest.mark.parametrize("style", [None, EditStyle.CINEMATIC, EditStyle.SOCIAL])
    def test_the_plan_is_identical_with_and_without_a_reference(
        self, style: EditStyle | None
    ) -> None:
        clips = candidates()
        planner = RulesEnginePlanner()

        without = planner.plan(request(style=style), list(clips)).plan
        with_reference = planner.plan(
            request(
                style=style,
                style_policy=blend(profile_for(style), fast_reference(), StyleStrength.ZERO),
            ),
            list(clips),
        ).plan

        assert [s.media_id for s in with_reference.segments] == [
            s.media_id for s in without.segments
        ]
        assert [(s.source_in_ms, s.source_out_ms) for s in with_reference.segments] == [
            (s.source_in_ms, s.source_out_ms) for s in without.segments
        ]
        assert with_reference.total_duration_ms == without.total_duration_ms
        assert with_reference.metadata["per_clip_ms"] == without.metadata["per_clip_ms"]
        assert with_reference.metadata["style_policy"] is None

    def test_an_explicitly_weighted_planner_still_means_what_it_says(self) -> None:
        """With no style and no reference, the injected weights win -- the Phase
        4 contract, which the policy must not quietly take over."""
        from visionforge.domain.selection import SelectionWeights

        weights = SelectionWeights(
            sharpness=0.9, exposure=0.1, contrast=0.0, resolution=0.0, duration=0.0
        )
        planner = RulesEnginePlanner(weights)
        plan = planner.plan(request(), candidates()).plan
        assert plan.metadata["weights"]["sharpness"] == pytest.approx(0.9)


class TestPacingFollowsTheReference:
    def test_a_fast_reference_shortens_the_clips(self) -> None:
        clips = candidates()
        planner = RulesEnginePlanner()

        plain = planner.plan(request(style=EditStyle.CINEMATIC), list(clips)).plan
        styled = planner.plan(
            request(
                style=EditStyle.CINEMATIC,
                style_policy=blend(
                    profile_for(EditStyle.CINEMATIC), fast_reference(), StyleStrength.FULL
                ),
            ),
            list(clips),
        ).plan

        assert styled.metadata["per_clip_ms"] < plain.metadata["per_clip_ms"]

    @pytest.mark.parametrize("strength", list(StyleStrength))
    def test_every_strength_still_produces_a_renderable_plan(self, strength: StyleStrength) -> None:
        """A reference may move the pacing; it may never move it outside what
        the renderer accepts."""
        plan = (
            RulesEnginePlanner()
            .plan(
                request(
                    style=EditStyle.SPORTS_HIGHLIGHT,
                    style_policy=blend(
                        profile_for(EditStyle.SPORTS_HIGHLIGHT),
                        fast_reference(shot_ms=Measurement(value=120.0, confidence=1.0)),
                        strength,
                    ),
                ),
                candidates(),
            )
            .plan
        )
        for segment in plan.segments:
            length = segment.source_out_ms - segment.source_in_ms
            assert MIN_SEGMENT_MS <= length <= MAX_SEGMENT_MS

    def test_the_dial_moves_the_pacing_at_every_stop(self) -> None:
        """A dial, not a switch.

        The first implementation only clamped to the policy's bounds, so a
        reference did nothing until a bound narrowed past the requested pacing
        and then did all of it at once -- 0% and 50% produced identical clips
        and 100% produced something seven times shorter. Each stop must move.
        """
        clips = candidates()
        planner = RulesEnginePlanner()
        preset = profile_for(None)

        lengths = [
            planner.plan(
                request(style_policy=blend(preset, fast_reference(), strength)), list(clips)
            ).plan.metadata["per_clip_ms"]
            for strength in StyleStrength
        ]

        # Strictly decreasing: the reference cuts faster than the request asks
        # for, so every step toward it shortens the clips.
        assert lengths == sorted(lengths, reverse=True)
        assert len(set(lengths)) == len(lengths), f"a stop did nothing: {lengths}"

    def test_a_low_confidence_measurement_moves_the_pacing_less(self) -> None:
        """Confidence gates the pull, not just the bounds."""
        clips = candidates()
        planner = RulesEnginePlanner()
        preset = profile_for(None)

        certain = planner.plan(
            request(
                style_policy=blend(
                    preset,
                    fast_reference(shot_ms=Measurement(value=800.0, confidence=1.0)),
                    StyleStrength.FULL,
                ),
            ),
            list(clips),
        ).plan.metadata["per_clip_ms"]
        unsure = planner.plan(
            request(
                style_policy=blend(
                    preset,
                    fast_reference(shot_ms=Measurement(value=800.0, confidence=0.25)),
                    StyleStrength.FULL,
                ),
            ),
            list(clips),
        ).plan.metadata["per_clip_ms"]

        assert certain < unsure

    def test_an_absurd_reference_cannot_widen_the_plan_bounds(self) -> None:
        """A reference measuring hour-long takes is clamped by the plan, not
        honoured by it."""
        plan = (
            RulesEnginePlanner()
            .plan(
                request(
                    target_duration_ms=600_000,
                    max_clips=2,
                    style_policy=blend(
                        profile_for(None),
                        fast_reference(
                            shot_ms=Measurement(value=3_600_000.0, confidence=1.0),
                            shot_ms_p25=3_000_000,
                            shot_ms_p75=4_000_000,
                        ),
                        StyleStrength.FULL,
                    ),
                ),
                candidates(count=2),
            )
            .plan
        )
        for segment in plan.segments:
            assert segment.source_out_ms - segment.source_in_ms <= MAX_SEGMENT_MS


class TestRankingFollowsTheReference:
    def test_a_high_motion_reference_prefers_high_motion_clips(self) -> None:
        """The style term's whole purpose, demonstrated end to end.

        Two clips identical in every usability signal, differing only in motion.
        With no reference the tie breaks on upload order; with a high-motion
        reference the busy one wins.
        """
        shared = {
            "kind": MediaKind.VIDEO,
            "is_ready": True,
            "duration_ms": 20_000,
            "width": 1920,
            "height": 1080,
            "blur_score": 400.0,
            "contrast": 50.0,
            "mean_luminance": 128.0,
            "clipped_ratio": 0.01,
        }
        calm = Candidate(
            media_id=MediaId(uuid.uuid4()),
            phash="0000000000000000",
            sequence=1,
            motion=0.05,
            saturation=0.7,
            **shared,  # type: ignore[arg-type]
        )
        busy = Candidate(
            media_id=MediaId(uuid.uuid4()),
            phash="ffffffffffffffff",
            sequence=2,
            motion=0.85,
            saturation=0.7,
            **shared,  # type: ignore[arg-type]
        )

        planner = RulesEnginePlanner()
        plain = planner.plan(request(max_clips=1, min_clips=1), [calm, busy]).plan
        styled = planner.plan(
            request(
                max_clips=1,
                min_clips=1,
                style_policy=blend(profile_for(None), fast_reference(), StyleStrength.FULL),
            ),
            [calm, busy],
        ).plan

        assert plain.segments[0].media_id == calm.media_id
        assert styled.segments[0].media_id == busy.media_id

    def test_unanalysed_footage_is_not_punished_for_being_unanalysed(self) -> None:
        """A clip with no dynamics row must not rank below an analysed one purely
        because the analyser had not reached it."""
        shared = {
            "kind": MediaKind.VIDEO,
            "is_ready": True,
            "duration_ms": 20_000,
            "width": 1920,
            "height": 1080,
            "blur_score": 400.0,
            "contrast": 50.0,
            "mean_luminance": 128.0,
            "clipped_ratio": 0.01,
        }
        unanalysed = Candidate(
            media_id=MediaId(uuid.uuid4()),
            phash="0000000000000000",
            sequence=1,
            **shared,  # type: ignore[arg-type]
        )
        # Analysed, and a poor match for the reference's look.
        mismatched = Candidate(
            media_id=MediaId(uuid.uuid4()),
            phash="ffffffffffffffff",
            sequence=2,
            motion=0.0,
            saturation=0.0,
            **shared,  # type: ignore[arg-type]
        )

        plan = (
            RulesEnginePlanner()
            .plan(
                request(
                    max_clips=1,
                    min_clips=1,
                    style_policy=blend(profile_for(None), fast_reference(), StyleStrength.FULL),
                ),
                [unanalysed, mismatched],
            )
            .plan
        )
        assert plan.segments[0].media_id == unanalysed.media_id


class TestProvenance:
    def test_the_plan_records_what_the_reference_moved(self) -> None:
        plan = (
            RulesEnginePlanner()
            .plan(
                request(
                    style=EditStyle.GAMING,
                    style_policy=blend(
                        profile_for(EditStyle.GAMING), fast_reference(), StyleStrength.HALF
                    ),
                ),
                candidates(),
            )
            .plan
        )
        recorded = plan.metadata["style_policy"]
        assert recorded is not None
        assert recorded["strength"] == "50"
        assert "shot_ms" in recorded["influence"]
        assert recorded["affinity_weight"] > 0

    def test_the_plan_never_records_the_reference_media_id(self) -> None:
        """The plan is what the renderer resolves. A reference id in it would be
        an id the renderer could be asked to fetch."""
        plan = (
            RulesEnginePlanner()
            .plan(
                request(
                    style_policy=blend(profile_for(None), fast_reference(), StyleStrength.FULL)
                ),
                candidates(),
            )
            .plan
        )
        assert str(REFERENCE_MEDIA) not in repr(plan.metadata)
        assert REFERENCE_MEDIA not in [s.media_id for s in plan.segments]
        assert REFERENCE_MEDIA not in plan.referenced_media_ids


class TestPolicyDefaulting:
    def test_a_request_with_no_policy_builds_one_from_the_style(self) -> None:
        """Every caller from Phase 4 onwards omits the policy, and must keep
        getting the style they asked for."""
        clips = candidates()
        planner = RulesEnginePlanner()

        implicit = planner.plan(request(style=EditStyle.ANIME), list(clips)).plan
        explicit = planner.plan(
            request(style=EditStyle.ANIME, style_policy=policy_for(EditStyle.ANIME)),
            list(clips),
        ).plan

        assert implicit.metadata["per_clip_ms"] == explicit.metadata["per_clip_ms"]
        assert [s.media_id for s in implicit.segments] == [s.media_id for s in explicit.segments]
