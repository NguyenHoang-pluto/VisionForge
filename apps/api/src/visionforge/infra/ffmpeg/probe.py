"""ffprobe: the source of truth for media metadata.

Neither the file extension nor the client's Content-Type is trusted. If ffprobe
cannot make sense of a file, the file is rejected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from visionforge.domain.errors import UnsupportedMediaError
from visionforge.domain.media import MediaKind, MediaMetadata, StreamInfo
from visionforge.infra.ffmpeg.runner import FFPROBE, run

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 60.0

#: Containers that can hold a still image. ffprobe reports single images as a
#: video stream in one of these formats, so format is what distinguishes an image
#: from a one-frame video.
_IMAGE_FORMATS = frozenset(
    {"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "gif", "bmp_pipe", "tiff_pipe", "mjpeg"}
)


def probe_media(path: str) -> MediaMetadata:
    """Run ffprobe on a local file and return structured metadata."""
    result = run(
        [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ],
        timeout_s=PROBE_TIMEOUT_S,
        binary=FFPROBE,
    )

    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("ffprobe returned unparseable output") from exc

    streams = [_parse_stream(s) for s in payload.get("streams", [])]
    fmt = payload.get("format", {})

    if not streams:
        raise UnsupportedMediaError(
            "file contains no decodable media streams",
            hint="The upload may be corrupt or may not be a media file.",
        )

    kind = _classify(streams, fmt.get("format_name", ""))
    video = next((s for s in streams if s.kind == "video"), None)
    audio = next((s for s in streams if s.kind == "audio"), None)

    # ffprobe's image2 demuxer guesses a codec from the *filename* extension, so
    # a renamed non-media file comes back as a 0x0 "mjpeg image" rather than an
    # error. Degenerate dimensions are the tell. Without this, such a file would
    # pass METADATA and fail later inside FFmpeg with a cryptic message.
    if kind is not MediaKind.AUDIO and not (video and video.width and video.height):
        raise UnsupportedMediaError(
            "media reports no usable video dimensions",
            hint="The file is corrupt, truncated, or not the type its name claims.",
        )

    duration_ms: int | None = None
    if kind is not MediaKind.IMAGE:
        raw_duration = fmt.get("duration")
        if raw_duration is not None:
            try:
                duration_ms = int(float(raw_duration) * 1000)
            except (TypeError, ValueError):
                duration_ms = None

    primary = video if kind is not MediaKind.AUDIO else audio

    return MediaMetadata(
        kind=kind,
        container_format=fmt.get("format_name"),
        duration_ms=duration_ms,
        width=video.width if video else None,
        height=video.height if video else None,
        fps=video.fps if video and kind is MediaKind.VIDEO else None,
        codec=primary.codec if primary else None,
        pix_fmt=video.pix_fmt if video else None,
        bit_rate=_as_int(fmt.get("bit_rate")),
        sample_rate=audio.sample_rate if audio else None,
        channels=audio.channels if audio else None,
        streams=tuple(streams),
    )


def _classify(streams: list[StreamInfo], format_name: str) -> MediaKind:
    has_video = any(s.kind == "video" for s in streams)
    has_audio = any(s.kind == "audio" for s in streams)

    if has_video:
        formats = set(format_name.split(","))
        if formats & _IMAGE_FORMATS:
            return MediaKind.IMAGE
        return MediaKind.VIDEO
    if has_audio:
        return MediaKind.AUDIO

    raise UnsupportedMediaError(f"unsupported stream layout (format: {format_name or 'unknown'})")


def _parse_stream(raw: dict[str, Any]) -> StreamInfo:
    return StreamInfo(
        index=int(raw.get("index", 0)),
        kind=str(raw.get("codec_type", "unknown")),
        codec=raw.get("codec_name"),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        fps=_parse_fraction(raw.get("avg_frame_rate") or raw.get("r_frame_rate")),
        pix_fmt=raw.get("pix_fmt"),
        sample_rate=_as_int(raw.get("sample_rate")),
        channels=_as_int(raw.get("channels")),
        bit_rate=_as_int(raw.get("bit_rate")),
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fraction(value: Any) -> float | None:
    """ffprobe reports frame rates as ``"30000/1001"``."""
    if not value or not isinstance(value, str) or "/" not in value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        den = float(denominator)
        if den == 0:
            return None
        return round(float(numerator) / den, 4)
    except ValueError:
        return None
