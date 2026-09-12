"""ORM models for the Phase 2 MVP tables.

Seven tables: users, projects, media_assets, media_derivatives, events, jobs,
job_steps. Nothing else, because nothing else has a caller yet.

Every media and job row carries ``project_id``. Project is the authorization
anchor (Phase 0): access is always "does this user own this project", never
"does this media id exist".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from visionforge.domain.jobs import JobStatus, JobType, StepStatus
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.infra.db.base import Base


def _uuid_pk() -> Mapped[UUID]:
    return mapped_column(primary_key=True, default=uuid4)


def _enum(enum_type: type[Any], length: int) -> Enum:
    """A StrEnum column stored as VARCHAR that loads back as the enum.

    A plain ``String`` column returns ``str``, which makes ``is`` comparisons
    against enum members silently False -- the bug that let retries redo steps
    that had already succeeded. ``native_enum=False`` emits the same VARCHAR DDL
    (so no migration is needed) while restoring type round-tripping, and
    ``create_constraint=False`` keeps a new member from requiring a migration.
    """
    return Enum(
        enum_type,
        native_enum=False,
        create_constraint=False,
        length=length,
        values_callable=lambda e: [m.value for m in e],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ------------------------------------------------------------------------ users
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Phase 2 has no authentication; the column exists so Phase 11 adds a value,
    # not a migration on a table with live foreign keys.
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    projects: Mapped[list[Project]] = relationship(back_populates="owner")


# --------------------------------------------------------------------- projects
class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[UUID] = _uuid_pk()
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    owner: Mapped[User] = relationship(back_populates="projects")
    media: Mapped[list[MediaAsset]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


# ----------------------------------------------------------------- media assets
class MediaAsset(Base, TimestampMixin):
    __tablename__ = "media_assets"
    __table_args__ = (
        # Deduplication within a project. Cross-project dedupe is deliberately
        # out of scope for Phase 2: it raises ownership and deletion questions
        # that have no answer until real auth exists.
        UniqueConstraint("project_id", "sha256", name="uq_media_project_sha256"),
        Index("ix_media_project_created", "project_id", "created_at"),
    )

    id: Mapped[UUID] = _uuid_pk()
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- identity ---
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bytes_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(160), nullable=True)

    # --- classification & lifecycle ---
    kind: Mapped[MediaKind] = mapped_column(_enum(MediaKind, 16), nullable=False)
    status: Mapped[MediaStatus] = mapped_column(
        _enum(MediaStatus, 24), default=MediaStatus.PENDING_UPLOAD, nullable=False, index=True
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # --- ffprobe-derived metadata (ffprobe is the source of truth) ---
    container_format: Mapped[str | None] = mapped_column(String(120), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pix_fmt: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bit_rate: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probe: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    project: Mapped[Project] = relationship(back_populates="media")
    derivatives: Mapped[list[MediaDerivative]] = relationship(
        back_populates="media", cascade="all, delete-orphan"
    )


class MediaDerivative(Base, TimestampMixin):
    __tablename__ = "media_derivatives"
    __table_args__ = (
        # Regenerating a derivative with identical parameters is idempotent.
        UniqueConstraint("media_id", "kind", "variant", name="uq_derivative_media_kind_variant"),
    )

    id: Mapped[UUID] = _uuid_pk()
    media_id: Mapped[UUID] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[DerivativeKind] = mapped_column(_enum(DerivativeKind, 24), nullable=False)
    variant: Mapped[str] = mapped_column(String(48), nullable=False)  # "default", "720p"
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    bytes_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(160), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    media: Mapped[MediaAsset] = relationship(back_populates="derivatives")


# ------------------------------------------------------------------------- jobs
class Job(Base, TimestampMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        # Idempotency is scoped to (project, type, key) so the same key may be
        # reused for a different operation without colliding.
        UniqueConstraint(
            "project_id", "type", "idempotency_key", name="uq_job_project_type_idempotency"
        ),
        Index("ix_jobs_status_created", "status", "created_at"),
        # Dispatch recovery: find jobs stuck in PENDING because the broker
        # publish failed after the transaction committed.
        Index("ix_jobs_dispatch_recovery", "status", "queued_at"),
        CheckConstraint("attempts >= 0", name="ck_jobs_attempts_nonneg"),
    )

    id: Mapped[UUID] = _uuid_pk()
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    media_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=True, index=True
    )

    type: Mapped[JobType] = mapped_column(_enum(JobType, 48), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, 24), default=JobStatus.PENDING, nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # Cancellation is cooperative: the API sets the flag, the worker observes it
    # at the next step boundary. Nothing is ever SIGKILLed.
    cancel_requested: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Celery's task id is recorded for operator debugging only. It is never read
    # to determine status -- this table is the source of truth (ADR-0003).
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[JobStep]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStep.seq"
    )


class JobStep(Base, TimestampMixin):
    __tablename__ = "job_steps"
    __table_args__ = (UniqueConstraint("job_id", "seq", name="uq_step_job_seq"),)

    id: Mapped[UUID] = _uuid_pk()
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(48), nullable=False)
    status: Mapped[StepStatus] = mapped_column(
        _enum(StepStatus, 16), default=StepStatus.PENDING, nullable=False
    )
    attempt: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped[Job] = relationship(back_populates="steps")


# ----------------------------------------------------------------------- events
class Event(Base):
    """Append-only activity log.

    Becomes the audit log in Phase 11 without a migration. Never updated, never
    deleted outside of project cascade.
    """

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_project_created", "project_id", "created_at"),)

    id: Mapped[UUID] = _uuid_pk()
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = mapped_column(String(80), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
