"""Repository and unit-of-work ports.

The application layer depends on these Protocols, not on ``infra``. Two things
follow: the layering is real rather than aspirational, and every use case can be
tested against in-memory fakes with no database running.

Return types are intentionally loose (``Any`` for rows). Mirroring the full ORM
model into the domain would be a second schema to keep in sync for no benefit at
this stage; what matters is that the *operations* are named and fixed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from visionforge.domain.jobs import JobType
from visionforge.domain.media import DerivativeKind, MediaStatus


class UnitOfWork(Protocol):
    """Transaction control. Implemented by a SQLAlchemy session."""

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def flush(self) -> None: ...


class MediaRepositoryPort(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...

    async def get_in_project(self, media_id: UUID, project_id: UUID) -> Any | None: ...

    async def find_by_sha256(self, project_id: UUID, sha256: str) -> Any | None: ...

    async def list_in_project(
        self, project_id: UUID, *, limit: int = ..., offset: int = ...
    ) -> Sequence[Any]: ...

    async def count_in_project(self, project_id: UUID) -> int: ...

    async def set_status(
        self, media_id: UUID, status: MediaStatus, *, error: dict[str, Any] | None = ...
    ) -> None: ...

    async def upsert_derivative(
        self,
        *,
        media_id: UUID,
        kind: DerivativeKind,
        variant: str,
        storage_key: str,
        bytes_size: int | None,
        mime_type: str | None,
        width: int | None = ...,
        height: int | None = ...,
    ) -> Any: ...


class JobRepositoryPort(Protocol):
    async def create(
        self,
        *,
        project_id: UUID,
        job_type: JobType,
        params: dict[str, Any],
        step_names: Sequence[str],
        media_id: UUID | None = ...,
        idempotency_key: str | None = ...,
        max_attempts: int = ...,
    ) -> Any: ...

    async def get(self, job_id: UUID) -> Any | None: ...

    async def find_by_idempotency_key(
        self, project_id: UUID, job_type: JobType, key: str
    ) -> Any | None: ...

    async def mark_queued(self, job_id: UUID, celery_task_id: str | None) -> None: ...

    async def request_cancel(self, job_id: UUID) -> bool: ...

    async def find_undispatched(
        self, older_than: datetime, *, limit: int = ...
    ) -> Sequence[Any]: ...


class EventRepositoryPort(Protocol):
    async def record(
        self, *, kind: str, actor: str, project_id: UUID | None, payload: dict[str, Any]
    ) -> None: ...
