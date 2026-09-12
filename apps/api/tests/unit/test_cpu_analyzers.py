"""CPU analyzers against synthetic images built to have known properties.

Synthetic rather than fixture-based on purpose: a generated checkerboard is
*provably* sharp and a Gaussian blur of it is *provably* softer, so these tests
assert the metric actually measures what it claims rather than merely that it
returns a number.

No GPU, no network, no media files.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest

from visionforge.domain.analysis import (
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
    QualityThresholds,
)
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import ANALYSIS_LONG_EDGE, normalise, to_grayscale
from visionforge.infra.analysis.phash import (
    PerceptualHashAnalyzer,
    average_hash,
    cluster_by_hash,
    perceptual_hash,
)
from visionforge.infra.analysis.quality import (
    QualityAnalyzer,
    blur_score,
    contrast_score,
    exposure_stats,
)
from visionforge.infra.analysis.scenes import SceneAnalyzer

THRESHOLDS = QualityThresholds()


# ------------------------------------------------------------------- builders
def checkerboard(size: int = 256, square: int = 8) -> np.ndarray:
    """A high-frequency pattern: maximally sharp and high contrast."""
    rows = np.arange(size) // square
    cols = np.arange(size) // square
    pattern = ((rows[:, None] + cols[None, :]) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


def broadband(seed: int, size: int = 256) -> np.ndarray:
    """A grayscale image with energy across low frequencies, like a photograph.

    Smoothed noise rather than raw noise: raw noise is all high-frequency, which
    the low-frequency DCT block discards, making it as degenerate an input as a
    checkerboard.
    """
    rng = np.random.RandomState(seed)
    coarse = rng.rand(8, 8).astype(np.float32)
    upscaled = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    scaled = cv2.normalize(upscaled, None, 0, 255, cv2.NORM_MINMAX)
    return scaled.astype(np.uint8)


def flat(size: int = 256, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def write(tmp_path, image: np.ndarray, name: str = "img.png") -> str:  # type: ignore[no-untyped-def]
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return str(path)


def image_source(path: str) -> AnalysisSource:
    return AnalysisSource(
        media_id=MediaId(uuid.uuid4()),
        kind=MediaKind.IMAGE,
        local_path=path,
        used_proxy=False,
    )


# ---------------------------------------------------------------------- blur
class TestBlurScore:
    def test_sharp_scores_far_above_blurred(self) -> None:
        sharp = to_grayscale(checkerboard())
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

        assert blur_score(sharp) > blur_score(blurred) * 10

    def test_flat_image_has_almost_no_edge_energy(self) -> None:
        assert blur_score(to_grayscale(flat())) < 1.0

    def test_increasing_blur_monotonically_lowers_the_score(self) -> None:
        sharp = to_grayscale(checkerboard())
        scores = [blur_score(cv2.GaussianBlur(sharp, (k, k), 0)) for k in (3, 9, 21)]
        assert scores == sorted(scores, reverse=True)

    def test_is_deterministic(self) -> None:
        gray = to_grayscale(checkerboard())
        assert blur_score(gray) == blur_score(gray)


# ------------------------------------------------------------------ exposure
class TestExposureStats:
    def test_black_frame_is_fully_underexposed(self) -> None:
        stats = exposure_stats(np.zeros((64, 64), dtype=np.uint8), THRESHOLDS)

        assert stats["mean_luminance"] == 0.0
        assert stats["underexposed_ratio"] == 1.0
        assert stats["overexposed_ratio"] == 0.0

    def test_white_frame_is_fully_overexposed(self) -> None:
        stats = exposure_stats(np.full((64, 64), 255, dtype=np.uint8), THRESHOLDS)

        assert stats["overexposed_ratio"] == 1.0
        assert stats["underexposed_ratio"] == 0.0

    def test_mid_grey_clips_at_neither_end(self) -> None:
        stats = exposure_stats(np.full((64, 64), 128, dtype=np.uint8), THRESHOLDS)

        assert stats["mean_luminance"] == 128.0
        assert stats["underexposed_ratio"] == 0.0
        assert stats["overexposed_ratio"] == 0.0

    def test_half_black_half_white_averages_neutral_but_is_fully_clipped(self) -> None:
        """Why the clipped ratios exist: the mean alone would call this fine."""
        gray = np.vstack(
            [np.zeros((32, 64), dtype=np.uint8), np.full((32, 64), 255, dtype=np.uint8)]
        )
        stats = exposure_stats(gray, THRESHOLDS)

        assert stats["mean_luminance"] == pytest.approx(127.5, abs=1)
        assert stats["underexposed_ratio"] + stats["overexposed_ratio"] == 1.0


# ------------------------------------------------------------------ contrast
class TestContrastScore:
    def test_flat_image_has_zero_contrast(self) -> None:
        assert contrast_score(to_grayscale(flat())) == 0.0

    def test_checkerboard_has_maximal_contrast(self) -> None:
        assert contrast_score(to_grayscale(checkerboard())) > 120

    def test_ranks_gradients_between_the_extremes(self) -> None:
        low = np.full((64, 64), 128, dtype=np.uint8)
        low[:, 32:] = 138
        high = np.full((64, 64), 0, dtype=np.uint8)
        high[:, 32:] = 255

        assert contrast_score(low) < contrast_score(high)


# ------------------------------------------------------- quality analyzer
class TestQualityAnalyzer:
    def test_identity(self) -> None:
        analyzer = QualityAnalyzer()
        assert analyzer.name is AnalyzerName.QUALITY
        assert analyzer.kind is AnalyzerKind.CPU
        assert analyzer.version

    def test_audio_is_unsupported_not_failed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The Phase 3 rule: audio must degrade cleanly, never fail a job."""
        source = AnalysisSource(
            media_id=MediaId(uuid.uuid4()),
            kind=MediaKind.AUDIO,
            local_path=str(tmp_path / "nothing.mp3"),
            used_proxy=False,
        )
        outcome = QualityAnalyzer().analyze(source)

        assert outcome.status is AnalysisStatus.UNSUPPORTED
        assert outcome.status is not AnalysisStatus.FAILED
        assert "audio" in outcome.payload["reason"]

    def test_sharp_image_is_not_flagged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        outcome = QualityAnalyzer().analyze(image_source(write(tmp_path, checkerboard())))

        assert outcome.status is AnalysisStatus.OK
        assert outcome.payload["is_blurry"] is False
        assert outcome.payload["is_low_contrast"] is False

    def test_flat_grey_image_is_flagged_blurry_and_low_contrast(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        outcome = QualityAnalyzer().analyze(image_source(write(tmp_path, flat())))

        assert outcome.payload["is_blurry"] is True
        assert outcome.payload["is_low_contrast"] is True

    def test_black_image_is_badly_exposed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        outcome = QualityAnalyzer().analyze(image_source(write(tmp_path, black)))

        assert outcome.payload["frames"][0]["is_badly_exposed"] is True

    def test_thresholds_in_force_are_recorded_with_the_scores(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A stored row stays interpretable after the defaults are tuned."""
        custom = QualityThresholds(blur_min_variance=1e9)
        outcome = QualityAnalyzer(custom).analyze(image_source(write(tmp_path, checkerboard())))

        assert outcome.payload["thresholds"]["blur_min_variance"] == 1e9
        assert outcome.payload["is_blurry"] is True, "custom threshold must be applied"

    def test_reports_whether_a_proxy_was_analysed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write(tmp_path, checkerboard())
        source = AnalysisSource(
            media_id=MediaId(uuid.uuid4()),
            kind=MediaKind.IMAGE,
            local_path=path,
            used_proxy=True,
        )
        assert QualityAnalyzer().analyze(source).payload["used_proxy"] is True

    def test_is_deterministic(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = write(tmp_path, checkerboard())
        first = QualityAnalyzer().analyze(image_source(path))
        second = QualityAnalyzer().analyze(image_source(path))

        assert first.payload["blur_score"] == second.payload["blur_score"]
        assert first.payload["contrast"] == second.payload["contrast"]


# ------------------------------------------------------------ normalisation
class TestNormalisation:
    def test_large_images_are_downscaled_to_the_working_size(self) -> None:
        normalised = normalise(np.zeros((2000, 3000, 3), dtype=np.uint8))
        assert max(normalised.shape[:2]) == ANALYSIS_LONG_EDGE

    def test_small_images_are_left_alone(self) -> None:
        """Upscaling would invent detail and inflate the sharpness score."""
        small = np.zeros((100, 120, 3), dtype=np.uint8)
        assert normalise(small).shape == small.shape

    def test_aspect_ratio_is_preserved(self) -> None:
        normalised = normalise(np.zeros((1000, 2000, 3), dtype=np.uint8))
        height, width = normalised.shape[:2]
        assert width / height == pytest.approx(2.0, abs=0.02)

    def test_normalisation_makes_resolutions_comparable(self) -> None:
        """The reason normalisation exists at all.

        Laplacian variance scales with pixel count, so without a common working
        size the same picture at two resolutions would score differently and the
        stored numbers would be meaningless across a library.
        """
        big = checkerboard(1024, 32)
        small = cv2.resize(big, (512, 512), interpolation=cv2.INTER_AREA)

        big_score = blur_score(to_grayscale(normalise(big)))
        small_score = blur_score(to_grayscale(normalise(small)))

        assert big_score == pytest.approx(small_score, rel=0.35)


# ---------------------------------------------------------------- pHash
class TestPerceptualHash:
    def test_hash_is_16_hex_characters(self) -> None:
        assert len(perceptual_hash(to_grayscale(checkerboard()))) == 16

    def test_is_deterministic(self) -> None:
        gray = to_grayscale(checkerboard())
        assert perceptual_hash(gray) == perceptual_hash(gray)

    def test_survives_resizing(self) -> None:
        """The whole point: a resized copy must still hash close to the original."""
        from visionforge.domain.analysis import hamming_distance

        original = broadband(seed=7, size=512)
        resized = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)

        assert hamming_distance(perceptual_hash(original), perceptual_hash(resized)) <= 5

    def test_survives_jpeg_recompression(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from visionforge.domain.analysis import hamming_distance

        original = broadband(seed=11, size=512)
        path = str(tmp_path / "q30.jpg")
        cv2.imwrite(path, original, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        recompressed = to_grayscale(cv2.imread(path, cv2.IMREAD_COLOR))

        assert hamming_distance(perceptual_hash(original), perceptual_hash(recompressed)) <= 5

    def test_distinguishes_genuinely_different_images(self) -> None:
        """Two unrelated broadband images must hash far apart.

        Broadband on purpose. A checkerboard concentrates all its energy at one
        high frequency, which the 8x8 low-frequency block discards by design, so
        it is a degenerate pHash input and says nothing about discrimination on
        real photographs.
        """
        from visionforge.domain.analysis import hamming_distance

        left = broadband(seed=1)
        right = broadband(seed=2)

        assert hamming_distance(perceptual_hash(left), perceptual_hash(right)) > 5

    def test_no_image_pins_a_constant_bit(self) -> None:
        """The DC coefficient is excluded, so no bit is shared by construction.

        DC is average brightness: positive and large for any real image, hence
        always above the AC median. Hashing it would waste a bit and force every
        pair of unrelated images to agree on it.
        """
        hashes = [perceptual_hash(broadband(seed=s)) for s in range(6)]
        top_bits = {int(h, 16) >> 63 for h in hashes}

        assert top_bits == {0}

    def test_average_hash_is_also_16_hex_characters(self) -> None:
        assert len(average_hash(to_grayscale(checkerboard()))) == 16


class TestPerceptualHashAnalyzer:
    def test_identity(self) -> None:
        analyzer = PerceptualHashAnalyzer()
        assert analyzer.name is AnalyzerName.PHASH
        assert analyzer.kind is AnalyzerKind.CPU

    def test_audio_is_unsupported(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        source = AnalysisSource(
            media_id=MediaId(uuid.uuid4()),
            kind=MediaKind.AUDIO,
            local_path=str(tmp_path / "x.mp3"),
            used_proxy=False,
        )
        assert PerceptualHashAnalyzer().analyze(source).status is AnalysisStatus.UNSUPPORTED

    def test_reports_both_hashes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        outcome = PerceptualHashAnalyzer().analyze(image_source(write(tmp_path, checkerboard())))
        assert len(outcome.payload["phash"]) == 16
        assert len(outcome.payload["ahash"]) == 16
        assert outcome.payload["hash_bits"] == 64


class TestClustering:
    def test_groups_near_duplicates(self) -> None:
        clusters = cluster_by_hash(
            {
                "a": "0000000000000000",
                "a_resized": "0000000000000001",
                "b": "ffffffffffffffff",
            }
        )
        assert clusters == [["a", "a_resized"]]

    def test_singletons_are_not_reported(self) -> None:
        """A cluster of one is not a duplicate."""
        assert cluster_by_hash({"a": "0000000000000000", "b": "ffffffffffffffff"}) == []

    def test_transitive_chains_join_one_cluster(self) -> None:
        clusters = cluster_by_hash(
            {
                "a": "0000000000000000",
                "b": "0000000000000001",
                "c": "0000000000000003",
            }
        )
        assert clusters == [["a", "b", "c"]]

    def test_nothing_is_deleted_only_reported(self) -> None:
        """Clustering informs a later decision; it never removes media itself."""
        hashes = {"a": "0000000000000000", "a2": "0000000000000001"}
        before = dict(hashes)
        cluster_by_hash(hashes)
        assert hashes == before

    def test_empty_input(self) -> None:
        assert cluster_by_hash({}) == []


# --------------------------------------------------------------- scenes
class TestSceneAnalyzer:
    def test_identity(self) -> None:
        analyzer = SceneAnalyzer()
        assert analyzer.name is AnalyzerName.SCENES
        assert analyzer.kind is AnalyzerKind.CPU

    @pytest.mark.parametrize("kind", [MediaKind.IMAGE, MediaKind.AUDIO])
    def test_only_video_is_supported(self, kind: MediaKind, tmp_path) -> None:  # type: ignore[no-untyped-def]
        source = AnalysisSource(
            media_id=MediaId(uuid.uuid4()),
            kind=kind,
            local_path=str(tmp_path / "x"),
            used_proxy=False,
        )
        outcome = SceneAnalyzer().analyze(source)

        assert outcome.status is AnalysisStatus.UNSUPPORTED
        assert not SceneAnalyzer().supports(kind)

    def test_detector_settings_are_configurable(self) -> None:
        analyzer = SceneAnalyzer(threshold=40.0, min_scene_len=30, frame_skip=2)
        assert analyzer._threshold == 40.0
        assert analyzer._min_scene_len == 30
