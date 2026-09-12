"""The media ingest pipeline.

Steps: VALIDATE -> METADATA -> HASH -> THUMBNAIL -> PROXY -> FINALIZE.

Every step is a plain function taking a ``JobContext``. Retry, cancellation,
progress and state transitions belong to ``JobRunner`` -- this module only knows
how to ingest media.

Memory discipline (7.4 GB host): the original is downloaded to a temp file once
and every operation is file-to-file. Hashing streams in 1 MiB chunks. Nothing
here holds a video in memory.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import select

from visionforge.domain.errors import (
    DuplicateMediaError,
    PermanentError,
    UnsupportedMediaError,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import (
    MAGIC_PROBE_BYTES,
    PROXY_HEIGHT,
    THUMBNAIL_WIDTH,
    DerivativeKind,
    MediaKind,
    MediaStatus,
    sniff_kind,
)
from visionforge.domain.storage import derivative_key
from visionforge.infra.db.models import MediaAsset, MediaDerivative
from visionforge.infra.ffmpeg import make_proxy, make_thumbnail, probe_media
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers.runtime import JobContext

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES = 1024 * 1024


def _media(ctx: JobContext) -> MediaAsset:
    media = ctx.session.get(MediaAsset, ctx.job.media_id)
    if media is None:
        raise PermanentError("media row disappeared")
    return media


def _store(ctx: JobContext) -> S3ObjectStore:
    store = ctx.data.get("store")
    if store is None:
        store = S3ObjectStore()
        ctx.data["store"] = store
    return store


def _local_copy(ctx: JobContext) -> str:
    """Path to the original on local disk, downloading it if this attempt has none.

    ``ctx.data`` is per-attempt and deliberately not persisted, but a retry
    resumes at the first unfinished step -- which may be long after the step that
    originally downloaded the file. Every step that touches bytes therefore goes
    through here rather than assuming an earlier step left a path behind.

    Idempotent and lazy: the download happens at most once per attempt, and not
    at all for a job that never gets past its cheap validation checks.
    """
    cached = ctx.data.get("local_path")
    if cached and os.path.exists(cached):
        return str(cached)

    media = _media(ctx)
    workdir = ctx.data.get("workdir")
    if not workdir or not os.path.isdir(workdir):
        workdir = tempfile.mkdtemp(prefix=f"vf-{ctx.job_id}-")
        ctx.data["workdir"] = workdir

    local = os.path.join(workdir, "original" + Path(media.storage_key).suffix)
    _store(ctx).download_to(media.storage_key, local)
    ctx.data["local_path"] = local
    return local


# --------------------------------------------------------------------- VALIDATE
def step_validate(ctx: JobContext) -> None:
    """Confirm the object exists and looks like media, then pull it local.

    Magic bytes are read with a *ranged* GET so validating a 2 GB upload does not
    cost 2 GB of transfer before we know it is junk.
    """
    media = _media(ctx)
    store = _store(ctx)

    info = store.stat(media.storage_key)
    if info is None:
        raise PermanentError(
            "no object at the media's storage key",
            hint="The upload never completed.",
        )
    if info.size_bytes == 0:
        raise UnsupportedMediaError("uploaded object is empty")

    header = store.read_range(media.storage_key, length=MAGIC_PROBE_BYTES)
    if sniff_kind(header) is None:
        raise UnsupportedMediaError(
            "file signature is not a recognised media format",
            hint="The extension may not match the real contents of the file.",
        )

    _local_copy(ctx)

    media.bytes_size = info.size_bytes
    media.status = MediaStatus.PROCESSING


# --------------------------------------------------------------------- METADATA
def step_metadata(ctx: JobContext) -> None:
    """ffprobe is the source of truth, overriding whatever the extension claimed."""
    media = _media(ctx)
    metadata = probe_media(_local_copy(ctx))
    ctx.data["metadata"] = metadata

    media.kind = metadata.kind
    media.container_format = metadata.container_format
    media.duration_ms = metadata.duration_ms
    media.width = metadata.width
    media.height = metadata.height
    media.fps = metadata.fps
    media.codec = metadata.codec
    media.pix_fmt = metadata.pix_fmt
    media.bit_rate = metadata.bit_rate
    media.sample_rate = metadata.sample_rate
    media.channels = metadata.channels
    media.mime_type = _mime_for(metadata.kind, media.storage_key)
    media.probe = {
        "container_format": metadata.container_format,
        "streams": [
            {
                "index": s.index,
                "kind": s.kind,
                "codec": s.codec,
                "width": s.width,
                "height": s.height,
                "fps": s.fps,
                "pix_fmt": s.pix_fmt,
                "sample_rate": s.sample_rate,
                "channels": s.channels,
                "bit_rate": s.bit_rate,
            }
            for s in metadata.streams
        ],
    }


# ------------------------------------------------------------------------- HASH
def step_hash(ctx: JobContext) -> None:
    """Stream a SHA-256 and enforce per-project deduplication.

    The unique constraint on ``(project_id, sha256)`` is the real guarantee; this
    lookup turns the common case into a ``DuplicateMediaError`` that the task
    layer reports as a duplicate rather than a failure.
    """
    media = _media(ctx)
    digest = hashlib.sha256()
    with open(_local_copy(ctx), "rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    sha256 = digest.hexdigest()

    existing = ctx.session.execute(
        select(MediaAsset).where(
            MediaAsset.project_id == media.project_id,
            MediaAsset.sha256 == sha256,
            MediaAsset.id != media.id,
        )
    ).scalar_one_or_none()

    if existing is not None:
        raise DuplicateMediaError(
            f"identical bytes already ingested as media {existing.id}",
            duplicate_of=existing.id,
        )

    media.sha256 = sha256
    ctx.data["sha256"] = sha256


# -------------------------------------------------------------------- THUMBNAIL
def step_thumbnail(ctx: JobContext) -> None:
    media = _media(ctx)
    metadata = ctx.data["metadata"]

    if metadata.kind is MediaKind.AUDIO:
        ctx.data["thumbnail_skipped"] = True
        return

    source = _local_copy(ctx)
    destination = os.path.join(ctx.data["workdir"], "thumb.jpg")
    make_thumbnail(source, destination, metadata, width=THUMBNAIL_WIDTH)

    key = derivative_key(
        ProjectId(media.project_id),
        MediaId(media.id),
        DerivativeKind.THUMBNAIL,
        "default",
        ".jpg",
    )
    _store(ctx).upload_file(destination, key, content_type="image/jpeg")
    _upsert_derivative(
        ctx,
        kind=DerivativeKind.THUMBNAIL,
        variant="default",
        key=key,
        path=destination,
        mime="image/jpeg",
        width=THUMBNAIL_WIDTH,
    )


# ------------------------------------------------------------------------ PROXY
def step_proxy(ctx: JobContext) -> None:
    """Generate a 720p proxy for video taller than 720p. Originals are untouched."""
    media = _media(ctx)
    metadata = ctx.data["metadata"]

    if not metadata.needs_proxy:
        ctx.data["proxy_skipped"] = True
        return

    source = _local_copy(ctx)
    destination = os.path.join(ctx.data["workdir"], "proxy.mp4")
    make_proxy(source, destination, height=PROXY_HEIGHT)

    key = derivative_key(
        ProjectId(media.project_id), MediaId(media.id), DerivativeKind.PROXY, "720p", ".mp4"
    )
    _store(ctx).upload_file(destination, key, content_type="video/mp4")
    _upsert_derivative(
        ctx,
        kind=DerivativeKind.PROXY,
        variant="720p",
        key=key,
        path=destination,
        mime="video/mp4",
        height=PROXY_HEIGHT,
    )


# --------------------------------------------------------------------- FINALIZE
def step_finalize(ctx: JobContext) -> None:
    media = _media(ctx)
    media.status = MediaStatus.READY
    media.error = None
    ctx.job.result = {
        "media_id": str(media.id),
        "sha256": ctx.data.get("sha256"),
        "kind": str(media.kind),
        "duration_ms": media.duration_ms,
        "width": media.width,
        "height": media.height,
        "thumbnail": not ctx.data.get("thumbnail_skipped", False),
        "proxy": not ctx.data.get("proxy_skipped", False),
    }
    cleanup(ctx)


def cleanup(ctx: JobContext) -> None:
    """Remove the scratch directory. Safe to call more than once."""
    workdir = ctx.data.pop("workdir", None)
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


STEPS = {
    "VALIDATE": step_validate,
    "METADATA": step_metadata,
    "HASH": step_hash,
    "THUMBNAIL": step_thumbnail,
    "PROXY": step_proxy,
    "FINALIZE": step_finalize,
}


# ----------------------------------------------------------------------- helpers
def _upsert_derivative(
    ctx: JobContext,
    *,
    kind: DerivativeKind,
    variant: str,
    key: str,
    path: str,
    mime: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Idempotent by ``(media_id, kind, variant)`` so a retry cannot duplicate."""
    media = _media(ctx)
    existing = ctx.session.execute(
        select(MediaDerivative).where(
            MediaDerivative.media_id == media.id,
            MediaDerivative.kind == kind,
            MediaDerivative.variant == variant,
        )
    ).scalar_one_or_none()

    size = os.path.getsize(path)
    if existing is not None:
        existing.storage_key = key
        existing.bytes_size = size
        existing.mime_type = mime
        existing.width = width
        existing.height = height
        return

    ctx.session.add(
        MediaDerivative(
            media_id=media.id,
            kind=kind,
            variant=variant,
            storage_key=key,
            bytes_size=size,
            mime_type=mime,
            width=width,
            height=height,
        )
    )


_MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


def _mime_for(kind: MediaKind, storage_key: str) -> str:
    extension = Path(storage_key).suffix.lower()
    return _MIME_BY_EXTENSION.get(extension, f"{kind.value}/octet-stream")
