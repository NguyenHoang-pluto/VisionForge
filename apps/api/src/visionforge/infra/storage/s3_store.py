"""S3ObjectStore: the boto3 implementation of the ``ObjectStore`` port.

Every method here is synchronous. Callers in async contexts wrap them in
``asyncio.to_thread``; workers call them directly.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from botocore.exceptions import ClientError

from visionforge.core.config import Settings, get_settings
from visionforge.domain.errors import TransientError
from visionforge.domain.storage import ObjectInfo, PresignedUpload
from visionforge.infra.storage.client import build_s3_client

logger = logging.getLogger(__name__)

#: Errors that mean "the object is not there", as opposed to "storage is broken".
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ObjectStore:
    """Object storage over an S3-compatible endpoint (MinIO locally, S3 later)."""

    def __init__(self, settings: Settings | None = None, bucket: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._bucket = bucket or self._settings.s3_bucket_media
        self._client = build_s3_client(self._settings)

    @property
    def bucket(self) -> str:
        return self._bucket

    # ------------------------------------------------------------------ presign
    def presign_put(
        self, key: str, *, content_type: str | None, expires_in_s: int, max_bytes: int
    ) -> PresignedUpload:
        """Presign a single-object upload.

        The size limit is enforced by the *storage service*, not by application
        code: the browser uploads directly to MinIO/S3 and never touches the API,
        so an application-side check would have nothing to inspect.
        """
        params: dict[str, object] = {"Bucket": self._bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type

        url: str = self._client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=expires_in_s
        )
        return PresignedUpload(
            url=url,
            object_key=key,
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_s),
            max_bytes=max_bytes,
        )

    def presign_get(self, key: str, *, expires_in_s: int) -> str:
        url: str = self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires_in_s
        )
        return url

    # -------------------------------------------------------------------- reads
    def stat(self, key: str) -> ObjectInfo | None:
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if self._is_not_found(exc):
                return None
            raise TransientError(f"storage head_object failed for {key}") from exc
        return ObjectInfo(
            key=key,
            size_bytes=int(head["ContentLength"]),
            content_type=head.get("ContentType"),
        )

    def read_range(self, key: str, *, length: int) -> bytes:
        """Read the first ``length`` bytes. Used for magic-byte sniffing.

        A ranged GET rather than a full download: validating a 2 GB upload must
        not cost 2 GB of transfer.
        """
        try:
            response = self._client.get_object(
                Bucket=self._bucket, Key=key, Range=f"bytes=0-{length - 1}"
            )
            data: bytes = response["Body"].read()
            return data
        except ClientError as exc:
            if self._is_not_found(exc):
                return b""
            raise TransientError(f"storage read_range failed for {key}") from exc

    def download_to(self, key: str, destination: str) -> None:
        """Stream an object to a local file. Never buffers the whole object."""
        try:
            self._client.download_file(self._bucket, key, destination)
        except ClientError as exc:
            raise TransientError(f"storage download failed for {key}") from exc

    # ------------------------------------------------------------------- writes
    def upload_file(self, source: str, key: str, *, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        try:
            self._client.upload_file(source, self._bucket, key, ExtraArgs=extra)
        except ClientError as exc:
            raise TransientError(f"storage upload failed for {key}") from exc

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a prefix. Returns the number deleted."""
        deleted = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            if not contents:
                continue
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in contents]},
            )
            deleted += len(contents)
        return deleted

    @staticmethod
    def _is_not_found(exc: ClientError) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in _NOT_FOUND_CODES or status == 404
