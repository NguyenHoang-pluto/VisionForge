"""Deterministic image quality signals.

These are **objective measurements, not judgements of artistic quality.** A
sharp, well-exposed frame of a wall scores highly. What they are good for is the
opposite direction: reliably identifying frames that are out of focus, crushed to
black, blown out, or flat -- which is what a selection stage needs in order to
discard the obvious rejects before anything expensive looks at the rest.

Every metric is a pure function of pixels. Same input, same score, forever.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from visionforge.domain.analysis import (
    DEFAULT_QUALITY_THRESHOLDS,
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
    QualityThresholds,
)
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import SampledFrame, sample_frames, to_grayscale

#: Bumped whenever a metric's definition changes, so old rows stay interpretable
#: and a scoring change is visible as a new row rather than a silent edit.
QUALITY_ANALYZER_VERSION = "1"


def blur_score(gray: NDArray[np.uint8]) -> float:
    """Variance of the Laplacian: the standard focus measure.

    The Laplacian is a second-derivative edge operator. A sharp image has many
    strong edges and therefore high variance; a blurred one has few, so variance
    collapses. Frames are normalised to a fixed working size first (see
    ``frames.ANALYSIS_LONG_EDGE``) because this value scales with resolution and
    would otherwise be incomparable between a 4K and a 720p source.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(gray: NDArray[np.uint8], thresholds: QualityThresholds) -> dict[str, float]:
    """Mean luminance plus the fraction of pixels crushed to black or blown out.

    The clipped ratios matter more than the mean: an image can average a
    perfectly neutral 128 while being half pure black and half pure white.
    """
    total = int(gray.size)
    under = int(np.count_nonzero(gray < thresholds.underexposed_below))
    over = int(np.count_nonzero(gray > thresholds.overexposed_above))
    return {
        "mean_luminance": round(float(gray.mean()), 3),
        "underexposed_ratio": round(under / total, 5),
        "overexposed_ratio": round(over / total, 5),
    }


def contrast_score(gray: NDArray[np.uint8]) -> float:
    """Standard deviation of luminance: global contrast.

    Chosen over Michelson or RMS-on-extremes because a single hot specular
    highlight should not make a flat frame read as high contrast.
    """
    return round(float(gray.std()), 3)


def brightness_score(image: NDArray[np.uint8]) -> float:
    """Mean of the V channel in HSV -- perceived brightness, not luma."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return round(float(hsv[:, :, 2].mean()), 3)


@dataclass(frozen=True, slots=True)
class FrameScore:
    """One frame's measurements.

    A typed record rather than a bare dict so the aggregation below operates on
    known floats instead of ``object``, and so a renamed field breaks the build
    rather than silently producing a payload nobody reads.
    """

    timestamp_ms: int
    blur_score: float
    contrast: float
    brightness: float
    mean_luminance: float
    underexposed_ratio: float
    overexposed_ratio: float
    is_blurry: bool
    is_low_contrast: bool
    is_badly_exposed: bool

    def as_payload(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _score_frame(frame: SampledFrame, thresholds: QualityThresholds) -> FrameScore:
    gray = to_grayscale(frame.image)
    exposure = exposure_stats(gray, thresholds)
    blur = round(blur_score(gray), 3)
    contrast = contrast_score(gray)
    clipped = exposure["underexposed_ratio"] + exposure["overexposed_ratio"]
    low, high = thresholds.good_luma_range

    return FrameScore(
        timestamp_ms=frame.timestamp_ms,
        blur_score=blur,
        contrast=contrast,
        brightness=brightness_score(frame.image),
        mean_luminance=exposure["mean_luminance"],
        underexposed_ratio=exposure["underexposed_ratio"],
        overexposed_ratio=exposure["overexposed_ratio"],
        is_blurry=blur < thresholds.blur_min_variance,
        is_low_contrast=contrast < thresholds.contrast_min_stddev,
        is_badly_exposed=(
            clipped > thresholds.max_clipped_ratio
            or not (low <= exposure["mean_luminance"] <= high)
        ),
    )


class QualityAnalyzer:
    """CPU quality signals over representative frames.

    A video is scored per sampled frame and then aggregated, because a single
    soft frame in a ten-second clip says something different from a clip that is
    soft throughout -- and the per-frame detail is kept so a later stage can tell
    the difference.
    """

    name = AnalyzerName.QUALITY
    version = QUALITY_ANALYZER_VERSION
    kind = AnalyzerKind.CPU

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind in (MediaKind.IMAGE, MediaKind.VIDEO)

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"quality analysis does not apply to {source.kind.value}"},
            )

        frames = sample_frames(source)
        scored = [_score_frame(frame, self._thresholds) for frame in frames]

        blurs = [f.blur_score for f in scored]
        contrasts = [f.contrast for f in scored]
        lumas = [f.mean_luminance for f in scored]

        first = frames[0]
        payload: dict[str, object] = {
            "frames": [f.as_payload() for f in scored],
            "frame_count": len(scored),
            # Aggregates use the median: one black frame in a sampled set should
            # not drag the whole clip's score down.
            "blur_score": round(float(np.median(blurs)), 3),
            "contrast": round(float(np.median(contrasts)), 3),
            "mean_luminance": round(float(np.median(lumas)), 3),
            "is_blurry": bool(np.median(blurs) < self._thresholds.blur_min_variance),
            "is_low_contrast": bool(np.median(contrasts) < self._thresholds.contrast_min_stddev),
            "badly_exposed_frames": sum(1 for f in scored if f.is_badly_exposed),
            "analysis_width": first.width,
            "analysis_height": first.height,
            "aspect_ratio": round(first.width / first.height, 4) if first.height else None,
            "source_width": source.width,
            "source_height": source.height,
            # The thresholds in force are recorded with the scores, so a row
            # stays interpretable after the defaults are tuned.
            "thresholds": {
                "blur_min_variance": self._thresholds.blur_min_variance,
                "contrast_min_stddev": self._thresholds.contrast_min_stddev,
                "underexposed_below": self._thresholds.underexposed_below,
                "overexposed_above": self._thresholds.overexposed_above,
                "max_clipped_ratio": self._thresholds.max_clipped_ratio,
                "good_luma_range": list(self._thresholds.good_luma_range),
            },
            "used_proxy": source.used_proxy,
        }
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload=payload,
        )
