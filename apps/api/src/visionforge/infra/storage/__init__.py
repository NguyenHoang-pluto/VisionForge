"""S3-compatible object storage adapter (MinIO locally, AWS S3 later)."""

from visionforge.infra.storage.client import build_s3_client
from visionforge.infra.storage.probe import StorageProbe
from visionforge.infra.storage.s3_store import S3ObjectStore

__all__ = ["S3ObjectStore", "StorageProbe", "build_s3_client"]
