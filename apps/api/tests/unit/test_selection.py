"""Selection scoring, usability rejection and duplicate suppression.

Pure functions over constructed candidates. No database, no media, no GPU.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.domain.selection import (
    Candidate,
    RejectionReason,
    SelectionWeights,
    contrast_component,
    duration_component,
    exposure_component,
    group_duplicates,
    resolution_component,
    score_candidate,
    select,
    sharpness_component,
    usability_rejection,
)

W = SelectionWeights()


def candidate(**overrides: object) -> Candidate:
    """A usable 1080p video clip. Override one field to make it interesting."""
    defaults: dict[str, object] = {
        "media_id": MediaId(uuid.uuid4()),
        "kind": MediaKind.VIDEO,
        "is_ready": True,
        "duration_ms": 5000,
        "width": 1920,
        "height": 1080,
        "blur_score": 300.0,
        "contrast": 60.0,
        "mean_luminance": 128.0,
        "clipped_ratio": 0.01,
        "phash": f"{uuid.uuid4().int & ((1 << 64) - 1):016x}",
        "sequence": 0,
    }
    defaults.update(overrides)
    return Candidate(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ components
class TestComponents:
    def test_sharpness_rises_with_blur_score(self) -> None:
        low = sharpness_component(candidate(blur_score=50.0), W)
        high = sharpness_component(candidate(blur_score=500.0), W)
        assert low < high

    def test_sharpness_saturates_rather_than_exceeding_one(self) -> None:
        """A single freakishly sharp clip must not dominate every other signal."""
        assert sharpness_component(candidate(blur_score=100_000.0), W) == 1.0

    def test_sharpness_is_normalised_against_a_fixed_reference(self) -> None:
        """Not against the batch.

        Batch normalisation would make a clip's score depend on what it was
        uploaded alongside -- the same footage would rank differently in a
        different folder, and results would stop being reproducible.
        """
        alone = score_candidate(candidate(blur_score=300.0), W).components["sharpness"]
        crowd = [candidate(blur_score=300.0), candidate(blur_score=5000.0)]
        together = score_candidate(crowd[0], W).components["sharpness"]
        assert alone == together

    def test_exposure_peaks_at_mid_grey(self) -> None:
        mid = exposure_component(candidate(mean_luminance=128.0, clipped_ratio=0.0), W)
        dark = exposure_component(candidate(mean_luminance=20.0, clipped_ratio=0.0), W)
        bright = exposure_component(candidate(mean_luminance=240.0, clipped_ratio=0.0), W)

        assert mid > dark
        assert mid > bright
        assert mid == pytest.approx(1.0)

    def test_exposure_penalises_both_extremes_equally(self) -> None:
        """A crushed frame and a blown one are equally unusable."""
        dark = exposure_component(candidate(mean_luminance=28.0, clipped_ratio=0.0), W)
        bright = exposure_component(candidate(mean_luminance=228.0, clipped_ratio=0.0), W)
        assert dark == pytest.approx(bright)

    def test_clipping_reduces_exposure_independently_of_the_mean(self) -> None:
        """Why clipping is a separate term: the mean alone cannot see it."""
        clean = exposure_component(candidate(mean_luminance=128.0, clipped_ratio=0.0), W)
        clipped = exposure_component(candidate(mean_luminance=128.0, clipped_ratio=0.5), W)

        assert clipped < clean
        assert clipped == pytest.approx(clean * 0.5)

    def test_contrast_rises_and_saturates(self) -> None:
        assert contrast_component(candidate(contrast=10.0), W) < contrast_component(
            candidate(contrast=70.0), W
        )
        assert contrast_component(candidate(contrast=500.0), W) == 1.0

    def test_resolution_full_marks_at_1080p(self) -> None:
        assert resolution_component(candidate(width=1920, height=1080), W) == 1.0
        assert resolution_component(candidate(width=640, height=360), W) < 0.2

    def test_missing_dimensions_score_zero_not_an_error(self) -> None:
        assert resolution_component(candidate(width=None, height=None), W) == 0.0

    def test_duration_saturates_at_ten_seconds(self) -> None:
        """Past that, more footage says nothing about whether it is good."""
        assert duration_component(candidate(duration_ms=10_000), W) == 1.0
        assert duration_component(candidate(duration_ms=60_000), W) == 1.0


# ---------------------------------------------------------------------- score
class TestScoring:
    def test_score_is_bounded(self) -> None:
        perfect = score_candidate(
            candidate(blur_score=1e6, contrast=1e6, mean_luminance=128.0, clipped_ratio=0.0), W
        )
        assert 0.0 <= perfect.score <= 1.0

    def test_better_footage_outranks_worse(self) -> None:
        good = score_candidate(candidate(blur_score=500.0, contrast=70.0), W)
        poor = score_candidate(candidate(blur_score=60.0, contrast=20.0), W)
        assert good.score > poor.score

    def test_components_are_reported_with_the_score(self) -> None:
        """A score without its parts cannot be argued with."""
        scored = score_candidate(candidate(), W)
        assert set(scored.components) == {
            "sharpness",
            "exposure",
            "contrast",
            "resolution",
            "duration",
        }

    def test_is_deterministic(self) -> None:
        one = candidate()
        assert score_candidate(one, W).score == score_candidate(one, W).score

    def test_weights_change_the_ranking(self) -> None:
        """The point of making weights configurable."""
        sharp_dull = candidate(blur_score=600.0, contrast=15.0)
        soft_punchy = candidate(blur_score=60.0, contrast=80.0)

        sharpness_first = SelectionWeights(
            sharpness=0.9, exposure=0.04, contrast=0.02, resolution=0.02, duration=0.02
        )
        contrast_first = SelectionWeights(
            sharpness=0.02, exposure=0.04, contrast=0.9, resolution=0.02, duration=0.02
        )

        assert (
            score_candidate(sharp_dull, sharpness_first).score
            > score_candidate(soft_punchy, sharpness_first).score
        )
        assert (
            score_candidate(soft_punchy, contrast_first).score
            > score_candidate(sharp_dull, contrast_first).score
        )


class TestWeightValidation:
    def test_weights_must_sum_to_one(self) -> None:
        """Otherwise scores are not comparable between weight sets."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SelectionWeights(sharpness=0.9, exposure=0.9)

    def test_normalisation_references_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SelectionWeights(sharpness_reference=0.0)


# ----------------------------------------------------------------- usability
class TestUsability:
    def test_a_good_clip_passes(self) -> None:
        assert usability_rejection(candidate(), W) is None

    def test_audio_is_rejected_as_not_visual(self) -> None:
        rejection = usability_rejection(candidate(kind=MediaKind.AUDIO), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.NOT_VISUAL

    def test_media_still_ingesting_is_rejected(self) -> None:
        rejection = usability_rejection(candidate(is_ready=False), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.NOT_READY

    def test_unanalysed_media_is_rejected_distinctly_from_bad_media(self) -> None:
        """ "Never analysed" is a different problem from "analysed and poor"."""
        rejection = usability_rejection(candidate(blur_score=None, contrast=None), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.NO_ANALYSIS

    def test_out_of_focus_is_rejected(self) -> None:
        rejection = usability_rejection(candidate(blur_score=5.0), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.TOO_BLURRY

    def test_heavily_clipped_is_rejected(self) -> None:
        rejection = usability_rejection(candidate(clipped_ratio=0.9), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.BADLY_EXPOSED

    def test_flat_is_rejected(self) -> None:
        rejection = usability_rejection(candidate(contrast=2.0), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.LOW_CONTRAST

    def test_a_video_too_short_to_cut_is_rejected(self) -> None:
        rejection = usability_rejection(candidate(duration_ms=100), W)
        assert rejection is not None
        assert rejection.reason is RejectionReason.TOO_SHORT

    def test_images_are_exempt_from_the_duration_floor(self) -> None:
        """A still has no duration; that is not a defect."""
        assert usability_rejection(candidate(kind=MediaKind.IMAGE, duration_ms=None), W) is None

    def test_every_rejection_explains_itself(self) -> None:
        rejection = usability_rejection(candidate(blur_score=5.0), W)
        assert rejection is not None and rejection.detail
        assert "5.0" in rejection.detail


# -------------------------------------------------------------- deduplication
class TestDuplicateGrouping:
    def test_identical_hashes_group(self) -> None:
        a = candidate(phash="0000000000000000")
        b = candidate(phash="0000000000000000")
        groups = group_duplicates([a, b], max_distance=5)
        assert len(groups) == 1 and len(groups[0]) == 2

    def test_near_identical_hashes_group(self) -> None:
        a = candidate(phash="0000000000000000")
        b = candidate(phash="0000000000000003")
        assert len(group_duplicates([a, b], max_distance=5)) == 1

    def test_distant_hashes_do_not_group(self) -> None:
        a = candidate(phash="0000000000000000")
        b = candidate(phash="ffffffffffffffff")
        assert group_duplicates([a, b], max_distance=5) == []

    def test_candidates_without_a_hash_are_never_grouped(self) -> None:
        """An unknown hash is not evidence of similarity."""
        a = candidate(phash=None)
        b = candidate(phash=None)
        assert group_duplicates([a, b], max_distance=5) == []

    def test_transitive_chains_form_one_group(self) -> None:
        chain = [
            candidate(phash="0000000000000000"),
            candidate(phash="0000000000000001"),
            candidate(phash="0000000000000003"),
        ]
        groups = group_duplicates(chain, max_distance=5)
        assert len(groups) == 1 and len(groups[0]) == 3


# ------------------------------------------------------------------- selector
class TestSelect:
    def test_returns_at_most_the_limit(self) -> None:
        result = select([candidate() for _ in range(10)], limit=3, weights=W)
        assert len(result.selected) == 3

    def test_orders_by_score_descending(self) -> None:
        pool = [
            candidate(blur_score=100.0, sequence=0),
            candidate(blur_score=500.0, sequence=1),
            candidate(blur_score=300.0, sequence=2),
        ]
        scores = [s.score for s in select(pool, limit=3, weights=W).selected]
        assert scores == sorted(scores, reverse=True)

    def test_unusable_candidates_are_rejected_with_reasons(self) -> None:
        pool = [candidate(), candidate(blur_score=2.0), candidate(kind=MediaKind.AUDIO)]
        result = select(pool, limit=5, weights=W)

        assert len(result.selected) == 1
        reasons = {r.reason for r in result.rejected}
        assert reasons == {RejectionReason.TOO_BLURRY, RejectionReason.NOT_VISUAL}

    def test_duplicate_group_keeps_only_its_strongest_member(self) -> None:
        weak = candidate(phash="0000000000000000", blur_score=80.0, sequence=0)
        strong = candidate(phash="0000000000000001", blur_score=550.0, sequence=1)
        result = select([weak, strong], limit=5, weights=W)

        assert result.selected_ids == (strong.media_id,)
        assert any(r.reason is RejectionReason.NEAR_DUPLICATE for r in result.rejected)

    def test_a_blurry_clip_cannot_win_its_duplicate_group(self) -> None:
        """Usability is checked before deduplication, deliberately.

        If dedupe ran first, an unusable clip could be chosen to represent a
        group and take a usable one down with it.
        """
        unusable = candidate(phash="0000000000000000", blur_score=2.0, sequence=0)
        usable = candidate(phash="0000000000000001", blur_score=300.0, sequence=1)
        result = select([unusable, usable], limit=5, weights=W)

        assert result.selected_ids == (usable.media_id,)
        assert {r.reason for r in result.rejected} == {RejectionReason.TOO_BLURRY}

    def test_is_deterministic_across_runs(self) -> None:
        pool = [candidate(blur_score=100.0 + i * 10, sequence=i) for i in range(8)]
        first = select(pool, limit=4, weights=W).selected_ids
        second = select(pool, limit=4, weights=W).selected_ids
        assert first == second

    def test_equal_scores_break_on_upload_order(self) -> None:
        """Without a stable tie-break the same folder could yield two edits."""
        pool = [candidate(sequence=i) for i in range(4)]
        result = select(pool, limit=2, weights=W)
        assert [s.candidate.sequence for s in result.selected] == [0, 1]

    def test_duplicate_groups_are_reported(self) -> None:
        a = candidate(phash="0000000000000000")
        b = candidate(phash="0000000000000001")
        result = select([a, b], limit=5, weights=W)
        assert len(result.duplicate_groups) == 1

    def test_empty_input_selects_nothing(self) -> None:
        result = select([], limit=5, weights=W)
        assert result.selected == () and result.rejected == ()

    def test_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="limit must be positive"):
            select([candidate()], limit=0, weights=W)
