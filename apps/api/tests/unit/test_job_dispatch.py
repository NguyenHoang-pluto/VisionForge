"""Job creation ordering, idempotency and publish-failure recovery.

Exercised entirely with fakes. These are the rules that are hardest to observe in
an integration test (you cannot easily make a real broker fail on demand) and the
most expensive to get wrong, so they get their own unit coverage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest

from visionforge.application.job_dispatch import JobDispatcher
from visionforge.domain.jobs import JobStatus, JobType


@dataclass
class FakeJob:
    id: UUID
    project_id: UUID
    type: str
    status: str = JobStatus.PENDING
    idempotency_key: str | None = None
    celery_task_id: str | None = None
    steps: list[str] = field(default_factory=list)


class FakeSession:
    """Records the commit sequence so ordering can be asserted."""

    def __init__(self) -> None:
        self.commits = 0
        self.log: list[str] = []

    async def commit(self) -> None:
        self.commits += 1
        self.log.append("commit")

    async def rollback(self) -> None:
        self.log.append("rollback")

    async def flush(self) -> None:
        self.log.append("flush")


class FakeJobRepo:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.jobs: dict[UUID, FakeJob] = {}

    async def create(self, **kwargs: Any) -> FakeJob:
        self._session.log.append("create_job")
        job = FakeJob(
            id=uuid.uuid4(),
            project_id=kwargs["project_id"],
            type=kwargs["job_type"],
            idempotency_key=kwargs.get("idempotency_key"),
            steps=list(kwargs["step_names"]),
        )
        self.jobs[job.id] = job
        return job

    async def get(self, job_id: UUID) -> FakeJob | None:
        return self.jobs.get(job_id)

    async def find_by_idempotency_key(
        self, project_id: UUID, job_type: JobType, key: str
    ) -> FakeJob | None:
        return next(
            (
                j
                for j in self.jobs.values()
                if j.project_id == project_id and j.type == job_type and j.idempotency_key == key
            ),
            None,
        )

    async def mark_queued(self, job_id: UUID, celery_task_id: str | None) -> None:
        self._session.log.append("mark_queued")
        job = self.jobs[job_id]
        job.status = JobStatus.QUEUED
        job.celery_task_id = celery_task_id

    async def request_cancel(self, job_id: UUID) -> bool:
        return True

    async def find_undispatched(self, older_than: Any, *, limit: int = 50) -> list[FakeJob]:
        return [j for j in self.jobs.values() if j.status == JobStatus.PENDING]


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[UUID] = []

    def publish(self, job_type: JobType, job_id: UUID) -> str | None:
        self.published.append(job_id)
        return f"celery-{job_id}"


class BrokenPublisher:
    """A broker that is down."""

    def publish(self, job_type: JobType, job_id: UUID) -> str | None:
        raise ConnectionError("broker unreachable")


@pytest.fixture
def wiring() -> tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]:
    session = FakeSession()
    jobs = FakeJobRepo(session)
    publisher = RecordingPublisher()
    return session, jobs, publisher, JobDispatcher(session, jobs, publisher)


PROJECT_ID = uuid.uuid4()


async def _create(dispatcher: JobDispatcher, **overrides: Any) -> tuple[FakeJob, bool]:
    params: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "job_type": JobType.MEDIA_INGEST,
        "params": {},
    }
    params.update(overrides)
    return await dispatcher.create_and_dispatch(**params)


class TestOrdering:
    @pytest.mark.asyncio
    async def test_publish_happens_after_commit(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        """The rule that prevents a worker claiming a job whose row is invisible."""
        session, _jobs, publisher, dispatcher = wiring
        await _create(dispatcher)

        assert session.log.index("create_job") < session.log.index("commit")
        assert session.log.index("commit") < session.log.index("mark_queued")
        assert len(publisher.published) == 1

    @pytest.mark.asyncio
    async def test_job_is_queued_after_successful_publish(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        _session, jobs, _publisher, dispatcher = wiring
        job, created = await _create(dispatcher)

        assert created is True
        assert jobs.jobs[job.id].status == JobStatus.QUEUED
        assert jobs.jobs[job.id].celery_task_id is not None

    @pytest.mark.asyncio
    async def test_steps_are_created_up_front(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        """Progress can be honest from the first poll, before the worker starts."""
        _session, _jobs, _publisher, dispatcher = wiring
        job, _ = await _create(dispatcher)

        assert job.steps == ["VALIDATE", "METADATA", "HASH", "THUMBNAIL", "PROXY", "FINALIZE"]


class TestPublishFailure:
    @pytest.mark.asyncio
    async def test_broker_failure_leaves_the_job_pending_not_lost(self) -> None:
        """The committed job survives a broker outage and stays redispatchable."""
        session = FakeSession()
        jobs = FakeJobRepo(session)
        dispatcher = JobDispatcher(session, jobs, BrokenPublisher())

        job, created = await _create(dispatcher)

        assert created is True
        assert jobs.jobs[job.id].status == JobStatus.PENDING
        assert "mark_queued" not in session.log

    @pytest.mark.asyncio
    async def test_undispatched_jobs_can_be_redispatched(self) -> None:
        session = FakeSession()
        jobs = FakeJobRepo(session)
        await JobDispatcher(session, jobs, BrokenPublisher()).create_and_dispatch(
            project_id=PROJECT_ID, job_type=JobType.MEDIA_INGEST, params={}
        )

        publisher = RecordingPublisher()
        recovered = await JobDispatcher(session, jobs, publisher).redispatch(
            await jobs.find_undispatched(None)
        )

        assert recovered == 1
        assert len(publisher.published) == 1
        assert all(j.status == JobStatus.QUEUED for j in jobs.jobs.values())


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_same_key_returns_the_same_job(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        _session, jobs, publisher, dispatcher = wiring

        first, created_first = await _create(dispatcher, idempotency_key="abc")
        second, created_second = await _create(dispatcher, idempotency_key="abc")

        assert created_first is True
        assert created_second is False
        assert first.id == second.id
        assert len(jobs.jobs) == 1
        assert len(publisher.published) == 1, "a replay must not enqueue a second time"

    @pytest.mark.asyncio
    async def test_different_keys_create_different_jobs(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        _session, jobs, publisher, dispatcher = wiring

        await _create(dispatcher, idempotency_key="one")
        await _create(dispatcher, idempotency_key="two")

        assert len(jobs.jobs) == 2
        assert len(publisher.published) == 2

    @pytest.mark.asyncio
    async def test_no_key_means_no_deduplication(
        self, wiring: tuple[FakeSession, FakeJobRepo, RecordingPublisher, JobDispatcher]
    ) -> None:
        _session, jobs, _publisher, dispatcher = wiring

        await _create(dispatcher)
        await _create(dispatcher)

        assert len(jobs.jobs) == 2
