"""Job execution runtime shared by every worker task.

Owns the parts that must behave identically for all job types: step transitions,
progress publishing, cancellation checks at step boundaries, and the retry
decision. A task implements *what* each step does; this decides *whether* it runs
and what happens when it fails.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from visionforge.domain.errors import (
    CancelledError,
    DuplicateMediaError,
    PermanentError,
    TransientError,
)
from visionforge.domain.jobs import (
    DEFAULT_RETRY_POLICY,
    JobStatus,
    StepStatus,
    assert_transition,
)
from visionforge.infra.db.models import Job, JobStep
from visionforge.infra.redis.events import publish_sync

logger = logging.getLogger(__name__)

StepFn = Callable[["JobContext"], None]


class JobContext:
    """Per-execution state handed to each step.

    ``data`` carries values between steps within one attempt (the local file
    path, the probe result). It is intentionally not persisted: anything a retry
    needs must be durable in Postgres or object storage.
    """

    def __init__(self, session: Session, job: Job) -> None:
        self.session = session
        self.job = job
        self.data: dict[str, Any] = {}

    @property
    def job_id(self) -> UUID:
        return self.job.id


class JobRunner:
    """Executes a job's steps in order, with cancellation and retry."""

    def __init__(self, session: Session, job: Job) -> None:
        self._session = session
        self._job = job
        self._ctx = JobContext(session, job)

    # ------------------------------------------------------------------ lifecycle
    def run(self, steps: Mapping[str, StepFn]) -> None:
        job = self._job
        job.attempts += 1
        self._transition(JobStatus.RUNNING)
        job.started_at = job.started_at or datetime.now(UTC)
        self._session.commit()
        self._publish("job.running")

        try:
            for step in sorted(job.steps, key=lambda s: s.seq):
                if step.status is StepStatus.SUCCEEDED:
                    continue  # resumed after a retry; do not redo completed work
                self._check_cancelled()
                self._run_step(step, steps[step.name])

            self._finish_success()

        except DuplicateMediaError:
            # Not a failure. The task layer deletes the redundant row and marks
            # the job succeeded, so the runner must not swallow it as an error.
            raise
        except CancelledError:
            self._finish_cancelled()
        except PermanentError as exc:
            self._finish_failed(exc, retryable=False)
        except TransientError as exc:
            self._handle_transient(exc)
        except Exception as exc:  # unknown failure: treat as transient once
            logger.exception("unhandled job error", extra={"job_id": str(job.id)})
            self._handle_transient(TransientError(str(exc) or type(exc).__name__))

    def _run_step(self, step: JobStep, fn: StepFn) -> None:
        step.status = StepStatus.RUNNING
        step.attempt += 1
        step.started_at = datetime.now(UTC)
        step.error = None
        self._session.commit()
        self._publish("step.started", step=step)

        started = datetime.now(UTC)
        try:
            fn(self._ctx)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.finished_at = datetime.now(UTC)
            step.error = _error_payload(exc)
            self._session.commit()
            self._publish("step.failed", step=step)
            raise

        step.status = StepStatus.SUCCEEDED
        step.finished_at = datetime.now(UTC)
        step.metrics = {"duration_ms": int((step.finished_at - started).total_seconds() * 1000)}
        self._session.commit()
        self._publish("step.succeeded", step=step)

    # ---------------------------------------------------------------- outcomes
    def _finish_success(self) -> None:
        self._transition(JobStatus.SUCCEEDED)
        self._job.finished_at = datetime.now(UTC)
        self._job.error = None
        self._session.commit()
        self._publish("job.succeeded")

    def _finish_cancelled(self) -> None:
        for step in self._job.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                step.status = StepStatus.SKIPPED
                step.finished_at = datetime.now(UTC)
        self._transition(JobStatus.CANCELLED)
        self._job.finished_at = datetime.now(UTC)
        self._session.commit()
        self._publish("job.cancelled")

    def _finish_failed(self, exc: Exception, *, retryable: bool) -> None:
        self._transition(JobStatus.FAILED)
        self._job.finished_at = datetime.now(UTC)
        self._job.error = _error_payload(exc) | {"retryable": retryable}
        self._session.commit()
        self._publish("job.failed")

    def _handle_transient(self, exc: TransientError) -> None:
        """Retry with exponential backoff and full jitter, or give up."""
        job = self._job
        policy = DEFAULT_RETRY_POLICY
        if not policy.should_retry(job.attempts) or job.attempts >= job.max_attempts:
            self._finish_failed(exc, retryable=True)
            return

        job.retry_at = policy.next_attempt_at(job.attempts)
        self._transition(JobStatus.RETRY_WAIT)
        job.error = _error_payload(exc) | {"retryable": True}
        self._session.commit()
        self._publish("job.retry_wait")

    # ----------------------------------------------------------------- helpers
    def _check_cancelled(self) -> None:
        """Cooperative cancellation, observed only at step boundaries.

        Never a SIGKILL: a half-killed FFmpeg leaves a truncated object behind
        and no record of why.
        """
        self._session.refresh(self._job, ["cancel_requested", "status"])
        if self._job.cancel_requested or self._job.status is JobStatus.CANCEL_REQUESTED:
            raise CancelledError("cancellation requested")

    def _transition(self, target: JobStatus) -> None:
        assert_transition(JobStatus(self._job.status), target)
        self._job.status = target

    def _publish(self, event: str, *, step: JobStep | None = None) -> None:
        job = self._job
        done = sum(1 for s in job.steps if s.status is StepStatus.SUCCEEDED)
        total = len(job.steps) or 1
        publish_sync(
            job.id,
            {
                "event": event,
                "job_id": str(job.id),
                "status": job.status,
                "attempt": job.attempts,
                "max_attempts": job.max_attempts,
                "progress": round(done / total, 4),
                "steps_done": done,
                "steps_total": total,
                "current_step": step.name if step is not None else None,
                "retry_at": job.retry_at.isoformat() if job.retry_at else None,
                "error": job.error,
                "ts": datetime.now(UTC).isoformat(),
            },
        )


def _error_payload(exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", type(exc).__name__)
    hint = getattr(exc, "hint", None)
    return {"code": code, "message": str(exc), "hint": hint}
