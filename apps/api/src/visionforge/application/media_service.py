"""Media ingest use cases.

The invariant this module exists to protect: **no media byte passes through the
API process**. The API creates a row, hands back a presigned URL, and is told
when the upload finished. Bytes go browser -> object storage, directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from visionforge.domain.errors import (
    MediaTooLargeError,
    NotFoundError,
    UnsupportedMediaError,
    ValidationError,
)
from visionforge.domain.ids import MediaId, ProjectId, new_uuid
from visionforge.domain.jobs import JobType
from visionforge.domain.media import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    MediaKind,
    MediaStatus,
)
from visionforge.domain.ports import EventRepositoryPort, MediaRepositoryPort, UnitOfWork
from visionforge.domain.storage import ObjectStore, PresignedUpload, original_key

logger = logging.getLogger(__name__)

UPLOAD_URL_TTL_S = 15 * 60
DOWNLOAD_URL_TTL_S = 5 * 60


@dataclass(frozen=True, slots=True)
class UploadTicket:
    media_id: MediaId
    object_key: str
    upload: PresignedUpload
    duplicate_of: MediaId | None = None


class MediaService:
    def __init__(
        self,
        session: UnitOfWork,
        media: MediaRepositoryPort,
        events: EventRepositoryPort,
        store: ObjectStore,
    ) -> None:
        self._session = session
        self._media = media
        self._events = events
        self._store = store

    # ------------------------------------------------------------- upload start
    async def create_upload_ticket(
        self, *, project_id: ProjectId, filename: str, declared_size: int | None
    ) -> UploadTicket:
        """Create a pending media row and presign a PUT for its bytes.

        The object key is built from server-side identifiers. The client's
        filename is stored for display and never becomes a path component, which
        removes path traversal as a category rather than filtering for it.
        """
        extension = self._safe_extension(filename)
        kind = SUPPORTED_EXTENSIONS[extension]

        if declared_size is not None:
            if declared_size <= 0:
                raise ValidationError("declared size must be positive")
            if declared_size > MAX_UPLOAD_BYTES:
                raise MediaTooLargeError(
                    f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
                )

        media_id = MediaId(new_uuid())
        key = original_key(project_id, media_id, extension)

        await self._media.create(
            id=media_id,
            project_id=project_id,
            original_filename=os.path.basename(filename)[:512],
            storage_key=key,
            kind=kind,
            status=MediaStatus.PENDING_UPLOAD,
            bytes_size=declared_size,
        )
        await self._events.record(
            kind="media.upload_requested",
            actor="dev-user",
            project_id=project_id,
            payload={"media_id": str(media_id), "filename": filename},
        )
        await self._session.commit()

        upload = self._store.presign_put(
            key,
            content_type=None,  # the client's Content-Type is not trusted anyway
            expires_in_s=UPLOAD_URL_TTL_S,
            max_bytes=MAX_UPLOAD_BYTES,
        )
        return UploadTicket(media_id=media_id, object_key=key, upload=upload)

    # ---------------------------------------------------------- upload complete
    async def complete_upload(
        self, *, project_id: ProjectId, media_id: MediaId
    ) -> tuple[Any, bool]:
        """Confirm the bytes landed and mark the asset ready for processing.

        Returns ``(media, should_process)``. Verification here is deliberately
        cheap -- existence and size. The expensive truth (ffprobe, hashing) is the
        worker's job, because it needs the file on local disk.
        """
        media = await self._media.get_in_project(media_id, project_id)
        if media is None:
            raise NotFoundError("media not found in this project")

        if media.status not in (MediaStatus.PENDING_UPLOAD, MediaStatus.FAILED):
            return media, False  # already completed; idempotent replay

        info = self._store.stat(media.storage_key)
        if info is None:
            raise ValidationError(
                "no object at the expected key",
                hint="The upload did not complete. Request a new upload URL.",
            )
        if info.size_bytes == 0:
            raise UnsupportedMediaError("uploaded object is empty")
        if info.size_bytes > MAX_UPLOAD_BYTES:
            raise MediaTooLargeError(f"uploaded object is {info.size_bytes} bytes, over the limit")

        media.bytes_size = info.size_bytes
        media.status = MediaStatus.UPLOADED
        await self._events.record(
            kind="media.uploaded",
            actor="dev-user",
            project_id=project_id,
            payload={"media_id": str(media_id), "bytes": info.size_bytes},
        )
        await self._session.commit()
        return media, True

    # ----------------------------------------------------------------- download
    def presign_download(self, storage_key: str) -> str:
        return self._store.presign_get(storage_key, expires_in_s=DOWNLOAD_URL_TTL_S)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _safe_extension(filename: str) -> str:
        """Map a client filename to an allow-listed extension.

        The extension selects a *candidate* kind only; ffprobe decides what the
        file really is during ingest. Rejecting here just avoids spending a
        subprocess on obvious junk.
        """
        if not filename or len(filename) > 512:
            raise ValidationError("filename is missing or too long")

        extension = os.path.splitext(filename)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedMediaError(
                f"unsupported file type '{extension or filename}'",
                hint=f"Supported extensions: {supported}",
            )
        return extension


INGEST_JOB_TYPE = JobType.MEDIA_INGEST
__all__ = ["INGEST_JOB_TYPE", "MediaKind", "MediaService", "UploadTicket"]
