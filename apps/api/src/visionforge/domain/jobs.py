"""Job domain model: the state machine and the retry policy.

PostgreSQL owns job state (ADR-0003). These types describe the rules; the
persistence of them lives in ``infra.db``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class JobType(StrEnum):
    MEDIA_INGEST = "media_ingest"
    MEDIA_ANALYZE_CPU = "media_analyze_cpu"
    MEDIA_ANALYZE_GPU = "media_analyze_gpu"


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)

#: The legal state machine. Any transition not listed here is a bug, and
#: ``assert_transition`` turns it into a loud one.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    # PENDING -> RUNNING is legal, and deliberately so. The dispatcher commits
    # the job row, publishes to the broker, and only then writes QUEUED. A fast
    # worker can therefore claim a job that is still PENDING. That is benign:
    # the row it sees is committed and valid, and QUEUED is bookkeeping applied
    # after the fact.
    #
    # The alternative -- writing QUEUED before publishing -- would be worse: a
    # publish that then failed would leave a job marked QUEUED that no message
    # exists for, and `find_undispatched` (which looks for PENDING) could never
    # recover it. Keeping PENDING as "committed but not confirmed dispatched" is
    # what makes that recovery path work.
    JobStatus.PENDING: frozenset(
        {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.QUEUED: frozenset(
        {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.RETRY_WAIT,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.RETRY_WAIT: frozenset(
        {JobStatus.QUEUED, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.CANCEL_REQUESTED: frozenset(
        # A job already past the point of no return may still finish normally.
        {JobStatus.CANCELLED, JobStatus.SUCCEEDED, JobStatus.FAILED}
    ),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def assert_transition(current: JobStatus, target: JobStatus) -> None:
    if not can_transition(current, target):
        raise ValueError(f"illegal job transition: {current} -> {target}")


# ------------------------------------------------------------------ retry policy
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Full jitter rather than fixed backoff so that a batch of jobs failing on the
    same transient cause does not retry in lockstep and reproduce the outage.
    """

    max_attempts: int = 3
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0

    def should_retry(self, attempt: int) -> bool:
        """``attempt`` is the number of attempts already made (1-based)."""
        return attempt < self.max_attempts

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        ceiling = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return (rng or random).uniform(0.0, ceiling)

    def next_attempt_at(
        self, attempt: int, *, now: datetime | None = None, rng: random.Random | None = None
    ) -> datetime:
        return (now or datetime.now(UTC)) + timedelta(seconds=self.delay_for(attempt, rng=rng))


DEFAULT_RETRY_POLICY = RetryPolicy()


# ------------------------------------------------------------------- step plans
#: Steps of a media-ingest job, in order. Named here rather than in the worker so
#: that the API can create the step rows up front and report honest progress
#: before the worker has even started.
MEDIA_INGEST_STEPS: tuple[str, ...] = (
    "VALIDATE",
    "METADATA",
    "HASH",
    "THUMBNAIL",
    "PROXY",
    "FINALIZE",
)

#: CPU analysis: deterministic signals, no model weights, no GPU.
MEDIA_ANALYZE_CPU_STEPS: tuple[str, ...] = (
    "RESOLVE",
    "QUALITY",
    "SCENES",
    "PHASH",
    "FINALIZE",
)

#: GPU analysis: model inference. RESOLVE is repeated rather than shared with the
#: CPU job because the two run on different queues and must not depend on each
#: other's scratch state.
MEDIA_ANALYZE_GPU_STEPS: tuple[str, ...] = (
    "RESOLVE",
    "EMBED",
    "FACES",
    "FINALIZE",
)

JOB_STEP_PLANS: dict[JobType, tuple[str, ...]] = {
    JobType.MEDIA_INGEST: MEDIA_INGEST_STEPS,
    JobType.MEDIA_ANALYZE_CPU: MEDIA_ANALYZE_CPU_STEPS,
    JobType.MEDIA_ANALYZE_GPU: MEDIA_ANALYZE_GPU_STEPS,
}
