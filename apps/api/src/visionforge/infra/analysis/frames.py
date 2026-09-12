"""Frame extraction for analysis.

Decoding discipline (7.4 GB RAM, 4 GB VRAM): frames are seeked to and read one
at a time, never decoded in bulk into a list of full-resolution arrays. An image
is read once; a video yields at most five frames, each downscaled before anything
expensive touches it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from visionforge.domain.analysis import AnalysisSource, sample_timestamps_ms
from visionforge.domain.errors import UnsupportedMediaError
from visionforge.domain.media import MediaKind

logger = logging.getLogger(__name__)

#: Longest edge an analysis frame is scaled to before scoring.
#:
#: Quality metrics are resolution-sensitive -- Laplacian variance in particular
#: rises with pixel count -- so every frame is normalised to the same working
#: size. That makes a 4K still and a 720p still directly comparable, which is the
#: whole point of storing a score.
ANALYSIS_LONG_EDGE = 512

Frame = NDArray[np.uint8]
#: OpenCV's type stubs describe returns as loosely-typed arrays of any dtype.
#: Every path here reads 8-bit images, so results are cast at the cv2 boundary
#: rather than weakening ``Frame`` and losing the guarantee everywhere else.


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """One decoded, normalised frame plus where it came from."""

    timestamp_ms: int
    image: Frame

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


def normalise(image: Frame) -> Frame:
    """Downscale so the longest edge is ``ANALYSIS_LONG_EDGE``. Never upscales."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= ANALYSIS_LONG_EDGE:
        return image

    scale = ANALYSIS_LONG_EDGE / longest
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cast(Frame, resized)


def read_image(path: str) -> Frame:
    """Read a still image, normalised. Raises on anything OpenCV cannot decode."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise UnsupportedMediaError(
            "OpenCV could not decode this image",
            hint="The file may be corrupt or in an unsupported pixel format.",
        )
    return normalise(cast(Frame, image))


def sample_video_frames(path: str, duration_ms: int | None) -> list[SampledFrame]:
    """Seek to the deterministic sample points and read one frame at each.

    Seeking rather than sequential decoding: reading a 10-minute video frame by
    frame to reach 75% would cost minutes for five frames. If a seek lands past
    the end -- which happens with variable frame rates and inaccurate container
    durations -- that sample is skipped rather than failing the analysis.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise UnsupportedMediaError("OpenCV could not open this video")

    try:
        frames: list[SampledFrame] = []
        for timestamp_ms in sample_timestamps_ms(duration_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, image = capture.read()
            if not ok or image is None:
                logger.debug("frame seek missed", extra={"timestamp_ms": timestamp_ms})
                continue
            frames.append(
                SampledFrame(timestamp_ms=timestamp_ms, image=normalise(cast(Frame, image)))
            )

        if not frames:
            # Every seek failed. Fall back to the first decodable frame so a
            # short or oddly-indexed clip still yields something.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = capture.read()
            if ok and image is not None:
                frames.append(SampledFrame(timestamp_ms=0, image=normalise(cast(Frame, image))))

        if not frames:
            raise UnsupportedMediaError("no decodable frames in this video")
        return frames
    finally:
        capture.release()


def sample_frames(source: AnalysisSource) -> list[SampledFrame]:
    """Representative frames for any visual medium. Audio yields none."""
    if source.kind is MediaKind.IMAGE:
        return [SampledFrame(timestamp_ms=0, image=read_image(source.local_path))]
    if source.kind is MediaKind.VIDEO:
        return sample_video_frames(source.local_path, source.duration_ms)
    return []


def to_grayscale(image: Frame) -> NDArray[np.uint8]:
    return cast("NDArray[np.uint8]", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
