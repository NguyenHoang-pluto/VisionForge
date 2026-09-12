"""Request and response models for projects, media and jobs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------- projects
class ProjectCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None
    created_at: datetime
    media_count: int = 0


# ---------------------------------------------------------------------- media
class UploadUrlRequest(BaseModel):
    """The client proposes a filename and size; the server decides the key."""

    filename: str = Field(min_length=1, max_length=512)
    size_bytes: int | None = Field(default=None, gt=0)


class UploadUrlResponse(BaseModel):
    media_id: UUID
    object_key: str
    upload_url: str
    expires_at: datetime
    max_bytes: int


class CompleteUploadResponse(BaseModel):
    media_id: UUID
    status: str
    job_id: UUID | None = None


class DerivativeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: str
    variant: str
    bytes_size: int | None
    width: int | None
    height: int | None


class MediaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    original_filename: str
    kind: str
    status: str
    bytes_size: int | None
    mime_type: str | None
    sha256: str | None

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

    error: dict[str, object] | None = None
    created_at: datetime
    derivatives: list[DerivativeResponse] = Field(default_factory=list)

    has_thumbnail: bool = False
    has_proxy: bool = False


class MediaListResponse(BaseModel):
    items: list[MediaResponse]
    total: int


# ----------------------------------------------------------------------- jobs
class JobCreateRequest(BaseModel):
    project_id: UUID
    type: str = Field(examples=["media_ingest"])
    media_id: UUID | None = None


class JobStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    name: str
    status: str
    attempt: int
    error: dict[str, object] | None = None
    metrics: dict[str, object] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    media_id: UUID | None
    type: str
    status: str
    attempts: int
    max_attempts: int
    cancel_requested: bool
    progress: float = 0.0
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    retry_at: datetime | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    steps: list[JobStepResponse] = Field(default_factory=list)


class SignedUrlResponse(BaseModel):
    url: str
    expires_in_s: int
