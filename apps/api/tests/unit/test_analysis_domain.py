"""Analysis domain rules: sampling, thresholds, hashing distance, versioning.

Pure functions, no OpenCV, no files.
"""

from __future__ import annotations

import pytest

from visionforge.domain.analysis import (
    CLIP_EMBEDDING_DIM,
    FRAME_SAMPLE_FRACTIONS,
    PHASH_DUPLICATE_MAX_DISTANCE,
    AnalysisOutcome,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
    QualityThresholds,
    are_near_duplicates,
    hamming_distance,
    sample_timestamps_ms,
)


class TestFrameSampling:
    def test_is_deterministic(self) -> None:
        """The same duration must always yield the same timestamps.

        This is what makes an analysis reproducible: a changed score is then
        attributable to the model, not to which frames happened to be picked.
        """
        assert sample_timestamps_ms(10_000) == sample_timestamps_ms(10_000)

    def test_samples_across_the_whole_clip(self) -> None:
        stamps = sample_timestamps_ms(10_000)
        assert stamps == (500, 2500, 5000, 7500, 9500)

    def test_avoids_the_very_first_and_last_frames(self) -> None:
        """Real footage opens and closes on black or a fade far too often."""
        stamps = sample_timestamps_ms(10_000)
        assert stamps[0] > 0
        assert stamps[-1] < 10_000
        assert 0.0 not in FRAME_SAMPLE_FRACTIONS
        assert 1.0 not in FRAME_SAMPLE_FRACTIONS

    def test_timestamps_are_ordered_and_unique(self) -> None:
        stamps = sample_timestamps_ms(60_000)
        assert list(stamps) == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    @pytest.mark.parametrize("duration", [None, 0, -1])
    def test_missing_duration_falls_back_to_a_single_frame(self, duration: int | None) -> None:
        assert sample_timestamps_ms(duration) == (0,)

    def test_very_short_clip_collapses_without_duplicates(self) -> None:
        """A 10 ms clip cannot yield five distinct points; it must not try."""
        stamps = sample_timestamps_ms(10)
        assert len(set(stamps)) == len(stamps)
        assert all(0 <= t < 10 for t in stamps)

    def test_never_samples_past_the_end(self) -> None:
        assert all(t < 1000 for t in sample_timestamps_ms(1000))


class TestQualityThresholds:
    def test_defaults_are_coherent(self) -> None:
        t = QualityThresholds()
        assert t.underexposed_below < t.overexposed_above
        assert 0.0 <= t.max_clipped_ratio <= 1.0
        low, high = t.good_luma_range
        assert low < high

    def test_rejects_an_impossible_clipped_ratio(self) -> None:
        with pytest.raises(ValueError, match="max_clipped_ratio"):
            QualityThresholds(max_clipped_ratio=1.5)

    def test_rejects_inverted_exposure_bounds(self) -> None:
        with pytest.raises(ValueError, match="underexposed_below"):
            QualityThresholds(underexposed_below=200, overexposed_above=100)

    def test_is_immutable(self) -> None:
        """Thresholds are recorded with the scores; they must not drift midway."""
        with pytest.raises(AttributeError):
            QualityThresholds().blur_min_variance = 1.0  # type: ignore[misc]


class TestHammingDistance:
    def test_identical_hashes_are_zero_apart(self) -> None:
        assert hamming_distance("ffffffffffffffff", "ffffffffffffffff") == 0

    def test_counts_differing_bits(self) -> None:
        assert hamming_distance("0000000000000000", "0000000000000001") == 1
        assert hamming_distance("0000000000000000", "000000000000000f") == 4

    def test_opposite_hashes_are_maximally_apart(self) -> None:
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_is_symmetric(self) -> None:
        a, b = "8133add5bbc433c4", "a319a5e61be611e6"
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            hamming_distance("ff", "ffff")


class TestNearDuplicateThreshold:
    def test_identical_is_a_duplicate(self) -> None:
        assert are_near_duplicates("8133add5bbc433c4", "8133add5bbc433c4")

    def test_small_differences_still_count_as_duplicates(self) -> None:
        """A resize or re-compression perturbs a few low-frequency bits."""
        assert are_near_duplicates("0000000000000000", "0000000000000003")

    def test_large_differences_do_not(self) -> None:
        assert not are_near_duplicates("0000000000000000", "ffffffffffffffff")

    def test_threshold_is_configurable(self) -> None:
        pair = ("0000000000000000", "00000000000000ff")  # 8 bits apart
        assert not are_near_duplicates(*pair)
        assert are_near_duplicates(*pair, max_distance=8)

    def test_default_threshold_is_conservative(self) -> None:
        assert 0 < PHASH_DUPLICATE_MAX_DISTANCE <= 10


class TestAnalysisOutcome:
    def test_ok_outcome(self) -> None:
        outcome = AnalysisOutcome(
            analyzer=AnalyzerName.QUALITY, version="1", status=AnalysisStatus.OK
        )
        assert outcome.is_ok

    def test_unsupported_is_not_ok_but_is_not_a_failure_either(self) -> None:
        """Audio has no blur score. That is a fact, not an error."""
        outcome = AnalysisOutcome(
            analyzer=AnalyzerName.QUALITY, version="1", status=AnalysisStatus.UNSUPPORTED
        )
        assert not outcome.is_ok
        assert outcome.status is not AnalysisStatus.FAILED

    def test_payload_and_metrics_default_to_empty(self) -> None:
        outcome = AnalysisOutcome(analyzer=AnalyzerName.CLIP, version="1", status=AnalysisStatus.OK)
        assert outcome.payload == {}
        assert outcome.metrics == {}
        assert outcome.embedding is None


class TestAnalyzerIdentity:
    def test_analyzer_names_are_stable_strings(self) -> None:
        """These values are persisted; renaming one orphans every existing row."""
        assert {a.value for a in AnalyzerName} == {
            "quality",
            "scenes",
            "phash",
            "clip",
            "faces",
        }

    def test_analyzer_kinds_match_the_queues(self) -> None:
        assert {k.value for k in AnalyzerKind} == {"cpu", "gpu"}

    def test_clip_dimension_is_pinned(self) -> None:
        """The pgvector column type must match; changing this needs a migration."""
        assert CLIP_EMBEDDING_DIM == 512
