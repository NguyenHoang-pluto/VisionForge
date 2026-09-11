"""S3 client construction and bucket bootstrap.

boto3 is synchronous, so every call from async code goes through
``asyncio.to_thread``. Phase 1 uses the client only for the readiness probe and
the local bucket bootstrap -- the presigned upload flow arrives in Phase 2.

Storage holds bytes; the database holds facts (Phase 0 rule #8). Nothing here
knows what a file means.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from visionforge.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_s3_client(settings: Settings | None = None) -> Any:
    """Create an S3 client. ``endpoint_url`` is None on AWS, set for MinIO."""
    cfg = settings or get_settings()
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3_endpoint_url,
        region_name=cfg.s3_region,
        aws_access_key_id=cfg.s3_access_key.get_secret_value(),
        aws_secret_access_key=cfg.s3_secret_key.get_secret_value(),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},  # MinIO requires path-style addressing
            retries={"max_attempts": 2, "mode": "standard"},
            connect_timeout=2,
            read_timeout=5,
        ),
    )


def ensure_buckets(settings: Settings | None = None) -> list[str]:
    """Create the configured buckets if they do not exist. Returns bucket names.

    Local convenience only. In AWS the buckets are created by infrastructure code
    with the right policies, not by the application.
    """
    cfg = settings or get_settings()
    client = build_s3_client(cfg)
    created: list[str] = []
    for bucket in cfg.buckets:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)
            created.append(bucket)
            logger.info("created bucket", extra={"bucket": bucket})
    return created
