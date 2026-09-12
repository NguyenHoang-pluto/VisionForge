"""Media domain model.

Deliberately free of ORM and storage concerns: these types describe what a media
asset *is*, not how it is persisted or where its bytes live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB -- a product limit, not a guess
PROXY_HEIGHT = 720
THUMBNAIL_WIDTH = 480


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MediaStatus(StrEnum):
    """Lifecycle of a media asset.

    ``PENDING_UPLOAD`` exists because the row is created *before* the bytes
    arrive: the client needs a server-generated object key to upload to.
    """

    PENDING_UPLOAD = "pending_upload"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DerivativeKind(StrEnum):
    THUMBNAIL = "thumbnail"
    PROXY = "proxy"


# --------------------------------------------------------------------- formats
#: Container/codec allow-list, keyed by the extension the client claims. The claim
#: is only used to pick a candidate object key -- ffprobe decides what the file
#: actually is.
SUPPORTED_EXTENSIONS: dict[str, MediaKind] = {
    # images
    ".jpg": MediaKind.IMAGE,
    ".jpeg": MediaKind.IMAGE,
    ".png": MediaKind.IMAGE,
    ".webp": MediaKind.IMAGE,
    ".gif": MediaKind.IMAGE,
    ".bmp": MediaKind.IMAGE,
    ".tif": MediaKind.IMAGE,
    ".tiff": MediaKind.IMAGE,
    ".heic": MediaKind.IMAGE,
    # video
    ".mp4": MediaKind.VIDEO,
    ".mov": MediaKind.VIDEO,
    ".mkv": MediaKind.VIDEO,
    ".webm": MediaKind.VIDEO,
    ".avi": MediaKind.VIDEO,
    ".m4v": MediaKind.VIDEO,
    # audio
    ".mp3": MediaKind.AUDIO,
    ".wav": MediaKind.AUDIO,
    ".flac": MediaKind.AUDIO,
    ".aac": MediaKind.AUDIO,
    ".m4a": MediaKind.AUDIO,
    ".ogg": MediaKind.AUDIO,
}

#: Magic-byte signatures checked server-side after upload. The client's
#: Content-Type header is never trusted.
_MAGIC_SIGNATURES: tuple[tuple[bytes, int, MediaKind], ...] = (
    (b"\xff\xd8\xff", 0, MediaKind.IMAGE),  # JPEG
    (b"\x89PNG\r\n\x1a\n", 0, MediaKind.IMAGE),  # PNG
    (b"GIF87a", 0, MediaKind.IMAGE),
    (b"GIF89a", 0, MediaKind.IMAGE),
    (b"BM", 0, MediaKind.IMAGE),  # BMP
    (b"II*\x00", 0, MediaKind.IMAGE),  # TIFF LE
    (b"MM\x00*", 0, MediaKind.IMAGE),  # TIFF BE
    (b"RIFF", 0, MediaKind.IMAGE),  # WEBP/WAV -- refined below
    (b"ftyp", 4, MediaKind.VIDEO),  # MP4/MOV/M4A family
    (b"\x1aE\xdf\xa3", 0, MediaKind.VIDEO),  # Matroska / WebM
    (b"ID3", 0, MediaKind.AUDIO),  # MP3 with tag
    (b"\xff\xfb", 0, MediaKind.AUDIO),  # MP3 frame sync
    (b"\xff\xf3", 0, MediaKind.AUDIO),
    (b"fLaC", 0, MediaKind.AUDIO),
    (b"OggS", 0, MediaKind.AUDIO),
)

#: Bytes needed to evaluate every signature above.
MAGIC_PROBE_BYTES = 16


def sniff_kind(header: bytes) -> MediaKind | None:
    """Guess the media kind from a file header. ``None`` means unrecognised.

    Advisory only: ffprobe is the source of truth. This exists to reject obvious
    junk before spending a subprocess on it.
    """
    for signature, offset, kind in _MAGIC_SIGNATURES:
        if header[offset : offset + len(signature)] == signature:
            if signature == b"RIFF":
                # RIFF is a container: WEBP and WAV share the prefix.
                fourcc = header[8:12]
                if fourcc == b"WEBP":
                    return MediaKind.IMAGE
                if fourcc == b"WAVE":
                    return MediaKind.AUDIO
                return None
            return kind
    return None


# --------------------------------------------------------------------- metadata
@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One elementary stream, as reported by ffprobe."""

    index: int
    kind: str  # "video" | "audio" | "subtitle" | ...
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pix_fmt: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """Structured ffprobe output. The probe is the source of truth for all of it."""

    kind: MediaKind
    container_format: str | None = None
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    pix_fmt: str | None = None
    bit_rate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    streams: tuple[StreamInfo, ...] = field(default_factory=tuple)

    @property
    def needs_proxy(self) -> bool:
        """Video taller than the proxy height gets a 720p proxy."""
        return self.kind is MediaKind.VIDEO and (self.height or 0) > PROXY_HEIGHT
