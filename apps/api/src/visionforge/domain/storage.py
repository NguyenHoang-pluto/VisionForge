"""Object-key construction and the storage port.

Keys are *derived* from identifiers, never from user-supplied filenames. That one
rule removes path traversal as a category: there is no code path where a client
string reaches an object key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import DerivativeKind


def original_key(project_id: ProjectId, media_id: MediaId, extension: str) -> str:
    """Key for the uploaded bytes.

    ``extension`` is sanitised by the caller to a member of
    ``SUPPORTED_EXTENSIONS``; it is a display convenience for tooling, never a
    path component supplied by the client.
    """
    return f"projects/{project_id}/media/{media_id}/original{extension}"


def derivative_key(
    project_id: ProjectId,
    media_id: MediaId,
    kind: DerivativeKind,
    variant: str,
    extension: str,
) -> str:
    return f"projects/{project_id}/media/{media_id}/{kind.value}/{variant}{extension}"


@dataclass(frozen=True, slots=True)
class PresignedUpload:
    url: str
    object_key: str
    expires_at: datetime
    max_bytes: int


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    key: str
    size_bytes: int
    content_type: str | None


class ObjectStore(Protocol):
    """The storage port.

    The domain depends on this Protocol; ``infra.storage`` implements it with
    boto3. No SDK type crosses this boundary, so the whole media pipeline can be
    tested against an in-memory fake.
    """

    def presign_put(
        self, key: str, *, content_type: str | None, expires_in_s: int, max_bytes: int
    ) -> PresignedUpload: ...

    def presign_get(self, key: str, *, expires_in_s: int) -> str: ...

    def stat(self, key: str) -> ObjectInfo | None: ...

    def download_to(self, key: str, destination: str) -> None: ...

    def upload_file(self, source: str, key: str, *, content_type: str | None = None) -> None: ...

    def read_range(self, key: str, *, length: int) -> bytes: ...

    def delete_prefix(self, prefix: str) -> int: ...
