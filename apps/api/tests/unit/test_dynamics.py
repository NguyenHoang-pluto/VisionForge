"""The dynamics analyzer: energy and palette, measured on synthetic frames.

Synthetic rather than real footage on purpose. The arithmetic is what is under
test -- that a static pair reads as no motion, that an inverted pair reads as
maximum, that grey reads as unsaturated -- and a real clip would make those
assertions approximate for no gain. Real footage is exercised by the
integration tests, where FFmpeg is present.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from visionforge.domain.analysis import AnalysisSource, AnalysisStatus, AnalyzerKind, AnalyzerName
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.dynamics import (
    MOTION_INTERVAL_MS,
    MOTION_REFERENCE,
    DynamicsAnalyzer,
    motion_between,
    saturation_of,
)


def solid(value: int, size: int = 64) -> np.ndarray:
    """A flat BGR frame."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def colour(b: int, g: int, r: int, size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


class TestAnalyzerContract:
    def test_it_is_a_cpu_analyzer_with_a_pinned_version(self) -> None:
        analyzer = DynamicsAnalyzer()
        assert analyzer.name is AnalyzerName.DYNAMICS
        assert analyzer.kind is AnalyzerKind.CPU
        assert analyzer.version == "1"

    def test_it_applies_to_pictures_and_not_to_sound(self) -> None:
        analyzer = DynamicsAnalyzer()
        assert analyzer.supports(MediaKind.VIDEO)
        assert analyzer.supports(MediaKind.IMAGE)
        assert not analyzer.supports(MediaKind.AUDIO)

    def test_audio_is_unsupported_rather_than_failed(self) -> None:
        outcome = DynamicsAnalyzer().analyze(
            AnalysisSource(
                media_id=MediaId(uuid.uuid4()),
                kind=MediaKind.AUDIO,
                local_path="/does/not/matter.wav",
                used_proxy=False,
            )
        )
        assert outcome.status is AnalysisStatus.UNSUPPORTED
        assert "reason" in outcome.payload


class TestMotion:
    def test_an_unchanged_pair_has_no_motion(self) -> None:
        frame = solid(120)
        assert motion_between(frame, frame) == 0.0

    def test_a_fully_inverted_pair_saturates_at_one(self) -> None:
        """Black to white is more change than any footage contains, and the
        normalisation must clamp rather than exceed its own scale."""
        assert motion_between(solid(0), solid(255)) == 1.0

    def test_motion_rises_with_the_size_of_the_change(self) -> None:
        small = motion_between(solid(100), solid(105))
        large = motion_between(solid(100), solid(160))
        assert 0.0 < small < large <= 1.0

    def test_the_reference_scale_is_what_it_claims(self) -> None:
        """A difference of exactly MOTION_REFERENCE reads as full energy."""
        assert motion_between(solid(0), solid(int(MOTION_REFERENCE))) == pytest.approx(1.0)

    def test_mismatched_shapes_report_no_motion_rather_than_raising(self) -> None:
        """Two frames of different sizes cannot be differenced. That is a decode
        oddity, not an analysis failure, and it must not take down a job."""
        assert motion_between(solid(100, size=64), solid(200, size=32)) == 0.0

    def test_it_is_deterministic(self) -> None:
        a, b = solid(40), solid(90)
        assert motion_between(a, b) == motion_between(a, b)


class TestSaturation:
    def test_grey_is_unsaturated(self) -> None:
        for level in (0, 64, 128, 200, 255):
            assert saturation_of(solid(level)) == 0.0

    def test_a_pure_hue_is_fully_saturated(self) -> None:
        assert saturation_of(colour(0, 0, 255)) == pytest.approx(1.0)
        assert saturation_of(colour(255, 0, 0)) == pytest.approx(1.0)

    def test_a_muted_colour_sits_between(self) -> None:
        muted = saturation_of(colour(120, 140, 160))
        assert 0.0 < muted < 1.0

    def test_it_is_deterministic(self) -> None:
        frame = colour(30, 90, 150)
        assert saturation_of(frame) == saturation_of(frame)


class TestStills:
    def test_a_still_reports_colour_and_declines_to_report_motion(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The distinction the profile depends on: absent, not zero.

        A still that reported ``motion: 0`` would tell the policy layer it had
        measured a completely static clip, which is a claim about footage rather
        than about the medium.
        """
        import cv2

        path = tmp_path / "still.png"
        cv2.imwrite(str(path), colour(0, 0, 255))

        outcome = DynamicsAnalyzer().analyze(
            AnalysisSource(
                media_id=MediaId(uuid.uuid4()),
                kind=MediaKind.IMAGE,
                local_path=str(path),
                used_proxy=False,
            )
        )
        assert outcome.status is AnalysisStatus.OK
        assert outcome.payload["motion"] is None
        assert outcome.payload["motion_confidence"] == 0.0
        assert outcome.payload["saturation"] == pytest.approx(1.0)
        assert outcome.payload["colour_confidence"] == 1.0


class TestPayloadShape:
    def test_the_interval_and_scale_are_recorded_with_the_numbers(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A measurement stays interpretable after the constants are retuned."""
        import cv2

        path = tmp_path / "still.png"
        cv2.imwrite(str(path), solid(128))

        payload = (
            DynamicsAnalyzer()
            .analyze(
                AnalysisSource(
                    media_id=MediaId(uuid.uuid4()),
                    kind=MediaKind.IMAGE,
                    local_path=str(path),
                    used_proxy=False,
                )
            )
            .payload
        )
        assert payload["motion_interval_ms"] == MOTION_INTERVAL_MS
        assert payload["motion_reference"] == MOTION_REFERENCE

    def test_a_missing_file_fails_rather_than_inventing_numbers(self) -> None:
        outcome = DynamicsAnalyzer().analyze(
            AnalysisSource(
                media_id=MediaId(uuid.uuid4()),
                kind=MediaKind.VIDEO,
                local_path="/no/such/file.mp4",
                used_proxy=False,
                duration_ms=5_000,
            )
        )
        assert outcome.status is AnalysisStatus.FAILED
