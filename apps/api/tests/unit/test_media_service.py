"""Upload ticket creation and completion, against an in-memory object store.

This is what the ``ObjectStore`` port buys: the upload rules are tested with no
MinIO, no network and no bytes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from visionforge.application.media_service import MediaService
from visionforge.domain.errors import (
    MediaTooLargeError,
    NotFoundError,
    UnsupportedMediaError,
    ValidationError,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MAX_UPLOAD_BYTES, MediaKind, MediaStatus
from visionforge.domain.storage import ObjectInfo, PresignedUpload

PROJECT_ID = ProjectId(uuid.uuid4())


class FakeStore:
    """An in-memory object store. Records presign calls; holds no bytes."""

    def __init__(self) -> None:
        self.objects: dict[str, ObjectInfo] = {}
        self.presigned_puts: list[str] = []

    def put_object(self, key: str, size: int, content_type: str | None = None) -> None:
        self.objects[key] = ObjectInfo(key=key, size_bytes=size, content_type=content_type)

    def presign_put(
        self, key: str, *, content_type: str | None, expires_in_s: int, max_bytes: int
    ) -> PresignedUpload:
        self.presigned_puts.append(key)
        return PresignedUpload(
            url=f"https://storage.test/{key}?sig=x",
            object_key=key,
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_s),
            max_bytes=max_bytes,
        )

    def presign_get(self, key: str, *, expires_in_s: int) -> str:
        return f"https://storage.test/{key}?sig=get"

    def stat(self, key: str) -> ObjectInfo | None:
        return self.objects.get(key)

    def download_to(self, key: str, destination: str) -> None: ...
    def upload_file(self, source: str, key: str, *, content_type: str | None = None) -> None: ...
    def read_range(self, key: str, *, length: int) -> bytes:
        return b""

    def delete_prefix(self, prefix: str) -> int:
        return 0


class FakeMediaRow:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeMediaRepo:
    def __init__(self) -> None:
        self.rows: dict[UUID, FakeMediaRow] = {}

    async def create(self, **kwargs: Any) -> FakeMediaRow:
        row = FakeMediaRow(**kwargs)
        self.rows[kwargs["id"]] = row
        return row

    async def get_in_project(self, media_id: UUID, project_id: UUID) -> FakeMediaRow | None:
        row = self.rows.get(media_id)
        return row if row and row.project_id == project_id else None

    async def find_by_sha256(self, project_id: UUID, sha256: str) -> None:
        return None

    async def list_in_project(self, project_id: UUID, **kw: Any) -> list[FakeMediaRow]:
        return []

    async def count_in_project(self, project_id: UUID) -> int:
        return len(self.rows)

    async def set_status(self, media_id: UUID, status: MediaStatus, **kw: Any) -> None: ...
    async def upsert_derivative(self, **kwargs: Any) -> None: ...


class FakeEvents:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class FakeSession:
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def flush(self) -> None: ...


@pytest.fixture
def service() -> tuple[MediaService, FakeStore, FakeMediaRepo, FakeEvents]:
    store, media, events = FakeStore(), FakeMediaRepo(), FakeEvents()
    return MediaService(FakeSession(), media, events, store), store, media, events


Wiring = tuple[MediaService, FakeStore, FakeMediaRepo, FakeEvents]


class TestUploadTicket:
    @pytest.mark.asyncio
    async def test_creates_pending_row_and_presigns(self, service: Wiring) -> None:
        svc, store, media, events = service

        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="holiday.jpg", declared_size=1024
        )

        row = media.rows[ticket.media_id]
        assert row.status is MediaStatus.PENDING_UPLOAD
        assert row.kind is MediaKind.IMAGE
        assert store.presigned_puts == [ticket.object_key]
        assert events.records[0]["kind"] == "media.upload_requested"

    @pytest.mark.asyncio
    async def test_object_key_is_server_generated(self, service: Wiring) -> None:
        svc, _store, _media, _events = service

        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="holiday.jpg", declared_size=None
        )

        assert ticket.object_key == (f"projects/{PROJECT_ID}/media/{ticket.media_id}/original.jpg")

    @pytest.mark.asyncio
    async def test_path_traversal_in_filename_cannot_escape(self, service: Wiring) -> None:
        """A traversal attempt is stored as a display name; the key is unaffected."""
        svc, _store, media, _events = service

        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID,
            filename="../../../../etc/passwd.jpg",
            declared_size=10,
        )

        assert ".." not in ticket.object_key
        assert "etc" not in ticket.object_key
        assert media.rows[ticket.media_id].original_filename == "passwd.jpg"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename", ["evil.exe", "script.php", "archive.zip", "noextension", "doc.pdf"]
    )
    async def test_unsupported_extensions_are_rejected(
        self, service: Wiring, filename: str
    ) -> None:
        svc, _store, _media, _events = service
        with pytest.raises(UnsupportedMediaError):
            await svc.create_upload_ticket(
                project_id=PROJECT_ID, filename=filename, declared_size=10
            )

    @pytest.mark.asyncio
    async def test_oversized_declaration_is_rejected_before_presigning(
        self, service: Wiring
    ) -> None:
        svc, store, _media, _events = service
        with pytest.raises(MediaTooLargeError):
            await svc.create_upload_ticket(
                project_id=PROJECT_ID, filename="huge.mp4", declared_size=MAX_UPLOAD_BYTES + 1
            )
        assert store.presigned_puts == [], "no URL should be issued for a rejected upload"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [0, -1])
    async def test_nonpositive_size_is_rejected(self, service: Wiring, size: int) -> None:
        svc, _store, _media, _events = service
        with pytest.raises(ValidationError):
            await svc.create_upload_ticket(
                project_id=PROJECT_ID, filename="x.jpg", declared_size=size
            )

    @pytest.mark.asyncio
    async def test_absurd_filename_length_is_rejected(self, service: Wiring) -> None:
        svc, _store, _media, _events = service
        with pytest.raises(ValidationError):
            await svc.create_upload_ticket(
                project_id=PROJECT_ID, filename="a" * 600 + ".jpg", declared_size=10
            )

    @pytest.mark.asyncio
    async def test_extension_matching_is_case_insensitive(self, service: Wiring) -> None:
        svc, _store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="PHOTO.JPG", declared_size=10
        )
        assert ticket.object_key.endswith(".jpg")


class TestCompleteUpload:
    @pytest.mark.asyncio
    async def test_marks_uploaded_when_object_present(self, service: Wiring) -> None:
        svc, store, _media, events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.jpg", declared_size=None
        )
        store.put_object(ticket.object_key, size=2048)

        row, should_process = await svc.complete_upload(
            project_id=PROJECT_ID, media_id=ticket.media_id
        )

        assert should_process is True
        assert row.status is MediaStatus.UPLOADED
        assert row.bytes_size == 2048
        assert events.records[-1]["kind"] == "media.uploaded"

    @pytest.mark.asyncio
    async def test_missing_object_is_rejected(self, service: Wiring) -> None:
        """Completing an upload that never happened must not start a job."""
        svc, _store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.jpg", declared_size=None
        )

        with pytest.raises(ValidationError, match="no object"):
            await svc.complete_upload(project_id=PROJECT_ID, media_id=ticket.media_id)

    @pytest.mark.asyncio
    async def test_empty_object_is_rejected(self, service: Wiring) -> None:
        svc, store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.jpg", declared_size=None
        )
        store.put_object(ticket.object_key, size=0)

        with pytest.raises(UnsupportedMediaError, match="empty"):
            await svc.complete_upload(project_id=PROJECT_ID, media_id=ticket.media_id)

    @pytest.mark.asyncio
    async def test_oversized_actual_upload_is_rejected(self, service: Wiring) -> None:
        """The declared size was a hint; the stored size is the fact."""
        svc, store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.mp4", declared_size=1000
        )
        store.put_object(ticket.object_key, size=MAX_UPLOAD_BYTES + 1)

        with pytest.raises(MediaTooLargeError):
            await svc.complete_upload(project_id=PROJECT_ID, media_id=ticket.media_id)

    @pytest.mark.asyncio
    async def test_unknown_media_id_is_not_found(self, service: Wiring) -> None:
        svc, _store, _media, _events = service
        with pytest.raises(NotFoundError):
            await svc.complete_upload(project_id=PROJECT_ID, media_id=MediaId(uuid.uuid4()))

    @pytest.mark.asyncio
    async def test_media_from_another_project_is_not_found(self, service: Wiring) -> None:
        """Ownership is enforced by the query, not by a separate check."""
        svc, store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.jpg", declared_size=None
        )
        store.put_object(ticket.object_key, size=10)

        with pytest.raises(NotFoundError):
            await svc.complete_upload(project_id=ProjectId(uuid.uuid4()), media_id=ticket.media_id)

    @pytest.mark.asyncio
    async def test_second_completion_is_idempotent(self, service: Wiring) -> None:
        svc, store, _media, _events = service
        ticket = await svc.create_upload_ticket(
            project_id=PROJECT_ID, filename="a.jpg", declared_size=None
        )
        store.put_object(ticket.object_key, size=10)

        await svc.complete_upload(project_id=PROJECT_ID, media_id=ticket.media_id)
        _row, should_process = await svc.complete_upload(
            project_id=PROJECT_ID, media_id=ticket.media_id
        )

        assert should_process is False, "a replay must not start a second ingest"
