"""Thumbnail and proxy generation.

Both operations are file-to-file. Nothing decodes a whole video into memory: on a
7.4 GB machine that is the difference between a working laptop and a swapping
one. FFmpeg streams from disk and writes to disk; we only ever hold paths.
"""

from __future__ import annotations

import logging

from visionforge.domain.media import MediaKind, MediaMetadata
from visionforge.infra.ffmpeg.runner import run

logger = logging.getLogger(__name__)

THUMBNAIL_TIMEOUT_S = 120.0
PROXY_TIMEOUT_S = 1800.0  # 30 min: a long 1080p source on a 6-core CPU

#: Seek this far in before grabbing a video's representative frame, so the
#: thumbnail is not the black frame most videos open on.
_THUMBNAIL_SEEK_FRACTION = 0.1
_THUMBNAIL_MAX_SEEK_S = 10.0


def make_thumbnail(
    source: str,
    destination: str,
    metadata: MediaMetadata,
    *,
    width: int,
) -> None:
    """Write a JPEG thumbnail scaled to ``width``, preserving aspect ratio.

    ``-2`` for the height keeps the value even, which several encoders require.
    """
    scale = f"scale={width}:-2:force_original_aspect_ratio=decrease"
    args: list[str] = ["-y", "-loglevel", "error"]

    if metadata.kind is MediaKind.VIDEO:
        args += ["-ss", f"{_seek_offset(metadata):.3f}"]

    args += [
        "-i",
        source,
        "-frames:v",
        "1",
        "-vf",
        scale,
        "-f",
        "image2",
        "-c:v",
        "mjpeg",
        "-q:v",
        "4",
        destination,
    ]
    run(args, timeout_s=THUMBNAIL_TIMEOUT_S)


def make_proxy(source: str, destination: str, *, height: int) -> None:
    """Write a 720p H.264 proxy for preview, editing and later AI analysis.

    ``veryfast`` and CRF 26 on purpose: a proxy is a working copy, not a
    deliverable, and encode time on this hardware matters more than the last few
    percent of quality. The original is never modified.
    """
    run(
        [
            "-y",
            "-loglevel",
            "error",
            "-i",
            source,
            "-vf",
            f"scale=-2:{height}:force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            destination,
        ],
        timeout_s=PROXY_TIMEOUT_S,
    )


def _seek_offset(metadata: MediaMetadata) -> float:
    if not metadata.duration_ms:
        return 0.0
    seconds = metadata.duration_ms / 1000.0
    return min(seconds * _THUMBNAIL_SEEK_FRACTION, _THUMBNAIL_MAX_SEEK_S)
