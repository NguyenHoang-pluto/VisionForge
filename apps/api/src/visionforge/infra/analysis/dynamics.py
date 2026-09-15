"""Motion energy and colour, measured deterministically on the CPU.

Phase 8 needs to say how *busy* and how *saturated* a video is, for two
purposes: describing a reference video's style, and ranking the user's own
footage against it. Neither is available from the analyzers that already exist.
``quality`` samples brightness from the HSV *value* channel, not saturation, and
five frames spread across a whole video say nothing about motion.

So this measures both, and nothing else:

**Motion.** At each of the deterministic sample points, a *pair* of frames a
fixed short interval apart is decoded and the mean absolute difference of their
grayscale images is taken. That is a local measurement -- how much changed in
120 ms at 25% through the video -- repeated five times. It is not optical flow
and does not claim to be: a whip pan and a cut inside the interval both read as
high energy, which is correct for "how energetic is this footage" and wrong for
"how fast is the camera moving". The profile that consumes it says energy.

**Saturation.** The mean of the HSV saturation channel per sampled frame,
normalised to 0..1. A palette measurement, not a grade.

No model, no weights, no GPU, no network -- which is the whole point on a
machine that has to run Postgres, MinIO and FFmpeg at the same time. The cost is
one extra frame read per sample point on top of what ``quality`` already does.

Determinism is the same contract as every other analyzer: same file in, same
numbers out. The sample points come from ``sample_timestamps_ms`` and the second
frame of each pair is a fixed offset from the first, so nothing depends on
decoder timing or on which frames happened to be keyframes.
"""

from __future__ import annotations

import logging
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from visionforge.domain.analysis import (
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
    sample_timestamps_ms,
)
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import Frame, normalise, read_image, to_grayscale

logger = logging.getLogger(__name__)

#: How far apart the two frames of a motion pair sit.
#:
#: Long enough that real movement registers above sensor noise, short enough to
#: stay inside one shot at any ordinary cutting rate -- a 120 ms window crosses
#: a cut only in footage that is already cutting faster than eight times a
#: second, and such footage is high-energy by any measure anyway.
MOTION_INTERVAL_MS = 120

#: Mean absolute 8-bit difference that reads as "fully energetic".
#:
#: Normalising against a fixed ceiling rather than against the batch keeps a
#: clip's energy independent of what it was uploaded alongside -- the same
#: reasoning as ``SelectionWeights.sharpness_reference``. 40/255 is a hard cut
#: or a whip pan; ordinary handheld motion lands around 8-15.
MOTION_REFERENCE = 40.0


def motion_between(first: Frame, second: Frame) -> float:
    """Mean absolute grayscale difference, normalised to 0..1.

    Both frames are already scaled to the common analysis size, so the result is
    comparable between a 4K source and a 720p proxy.
    """
    a = to_grayscale(first).astype(np.int16)
    b = to_grayscale(second).astype(np.int16)
    if a.shape != b.shape:
        return 0.0
    delta = float(np.abs(a - b).mean())
    return round(min(delta / MOTION_REFERENCE, 1.0), 4)


def saturation_of(image: Frame) -> float:
    """Mean HSV saturation, normalised to 0..1."""
    hsv = cast("NDArray[np.uint8]", cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return round(float(hsv[:, :, 1].mean()) / 255.0, 4)


def _pair_at(capture: cv2.VideoCapture, timestamp_ms: int) -> tuple[Frame, Frame] | None:
    """The frame at ``timestamp_ms`` and its partner ``MOTION_INTERVAL_MS`` later.

    Returns ``None`` when either seek misses, which happens at the very end of a
    clip and with inaccurate container durations. A missed pair is dropped, not
    substituted -- the confidence reported below is how many survived.
    """
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok_a, first = capture.read()
    if not ok_a or first is None:
        return None

    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms + MOTION_INTERVAL_MS))
    ok_b, second = capture.read()
    if not ok_b or second is None:
        return None

    return normalise(cast(Frame, first)), normalise(cast(Frame, second))


class DynamicsAnalyzer:
    """Energy and palette over deterministic sample points.

    Images are supported and report saturation with no motion: a still has a
    palette and cannot have energy, and saying so is more useful than declaring
    the whole medium unsupported.
    """

    name = AnalyzerName.DYNAMICS
    version = "1"
    kind = AnalyzerKind.CPU

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind in (MediaKind.IMAGE, MediaKind.VIDEO)

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"dynamics analysis does not apply to {source.kind.value}"},
            )

        if source.kind is MediaKind.IMAGE:
            return self._still(source)
        return self._video(source)

    # ------------------------------------------------------------------ still
    def _still(self, source: AnalysisSource) -> AnalysisOutcome:
        image = read_image(source.local_path)
        saturation = saturation_of(image)
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload={
                "samples": [{"timestamp_ms": 0, "saturation": saturation}],
                "sample_count": 1,
                "saturation": saturation,
                "saturation_spread": 0.0,
                # A still cannot move. Absent, not zero: zero would mean
                # "measured, and it was still", which is a different claim.
                "motion": None,
                "motion_spread": None,
                "motion_confidence": 0.0,
                "colour_confidence": 1.0,
                "motion_interval_ms": MOTION_INTERVAL_MS,
                "motion_reference": MOTION_REFERENCE,
                "used_proxy": source.used_proxy,
            },
        )

    # ------------------------------------------------------------------ video
    def _video(self, source: AnalysisSource) -> AnalysisOutcome:
        capture = cv2.VideoCapture(source.local_path)
        if not capture.isOpened():
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.FAILED,
                payload={"reason": "OpenCV could not open this video"},
            )

        timestamps = sample_timestamps_ms(source.duration_ms)
        samples: list[dict[str, float | int]] = []
        try:
            for timestamp_ms in timestamps:
                pair = _pair_at(capture, timestamp_ms)
                if pair is None:
                    logger.debug("motion pair missed", extra={"timestamp_ms": timestamp_ms})
                    continue
                first, second = pair
                samples.append(
                    {
                        "timestamp_ms": timestamp_ms,
                        "motion": motion_between(first, second),
                        "saturation": saturation_of(first),
                    }
                )
        finally:
            capture.release()

        if not samples:
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.FAILED,
                payload={"reason": "no decodable frame pairs in this video"},
            )

        motions = [float(s["motion"]) for s in samples]
        saturations = [float(s["saturation"]) for s in samples]

        # Confidence is the share of intended sample points that produced a
        # usable pair. A clip that yielded one pair out of five is reported, and
        # reported as weak, rather than being silently averaged into the same
        # number as one that yielded five.
        confidence = round(len(samples) / max(len(timestamps), 1), 3)

        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload={
                "samples": samples,
                "sample_count": len(samples),
                # Median, for the same reason quality uses it: one cut inside a
                # sample window should not become the clip's character.
                "motion": round(float(np.median(motions)), 4),
                "motion_spread": round(float(np.std(motions)), 4),
                "saturation": round(float(np.median(saturations)), 4),
                "saturation_spread": round(float(np.std(saturations)), 4),
                "motion_confidence": confidence,
                "colour_confidence": confidence,
                "motion_interval_ms": MOTION_INTERVAL_MS,
                "motion_reference": MOTION_REFERENCE,
                "used_proxy": source.used_proxy,
            },
        )
