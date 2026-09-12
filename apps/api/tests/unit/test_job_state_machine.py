"""The job state machine and retry policy.

Pure domain logic: no database, no broker, no clock.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from visionforge.domain.jobs import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    JobStatus,
    RetryPolicy,
    assert_transition,
    can_transition,
)


class TestTransitions:
    def test_happy_path(self) -> None:
        for current, target in [
            (JobStatus.PENDING, JobStatus.QUEUED),
            (JobStatus.QUEUED, JobStatus.RUNNING),
            (JobStatus.RUNNING, JobStatus.SUCCEEDED),
        ]:
            assert can_transition(current, target)

    def test_retry_loop(self) -> None:
        assert can_transition(JobStatus.RUNNING, JobStatus.RETRY_WAIT)
        assert can_transition(JobStatus.RETRY_WAIT, JobStatus.QUEUED)

    def test_terminal_states_are_absorbing(self) -> None:
        for status in TERMINAL_STATUSES:
            assert ALLOWED_TRANSITIONS[status] == frozenset(), status

    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (JobStatus.SUCCEEDED, JobStatus.RUNNING),
            (JobStatus.FAILED, JobStatus.QUEUED),
            (JobStatus.CANCELLED, JobStatus.RUNNING),
            (JobStatus.PENDING, JobStatus.RUNNING),  # must be queued first
            (JobStatus.PENDING, JobStatus.SUCCEEDED),
        ],
    )
    def test_illegal_transitions_raise(self, current: JobStatus, target: JobStatus) -> None:
        assert not can_transition(current, target)
        with pytest.raises(ValueError, match="illegal job transition"):
            assert_transition(current, target)

    def test_cancel_requested_may_still_finish(self) -> None:
        """A job past the point of no return is allowed to complete normally.

        Cancellation is cooperative: if the last step already succeeded before
        the flag was observed, forcing CANCELLED would discard real work.
        """
        assert can_transition(JobStatus.CANCEL_REQUESTED, JobStatus.SUCCEEDED)
        assert can_transition(JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED)

    def test_every_status_has_a_transition_entry(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(JobStatus)


class TestRetryPolicy:
    def test_respects_max_attempts(self) -> None:
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(1)
        assert policy.should_retry(2)
        assert not policy.should_retry(3)

    def test_backoff_ceiling_grows_exponentially(self) -> None:
        policy = RetryPolicy(base_delay_s=2.0, max_delay_s=60.0)
        rng = random.Random(0)
        # Full jitter samples [0, ceiling], so assert the ceiling, not the draw.
        assert max(policy.delay_for(1, rng=rng) for _ in range(200)) <= 2.0
        assert max(policy.delay_for(2, rng=rng) for _ in range(200)) <= 4.0
        assert max(policy.delay_for(3, rng=rng) for _ in range(200)) <= 8.0

    def test_delay_is_capped(self) -> None:
        policy = RetryPolicy(base_delay_s=2.0, max_delay_s=10.0)
        rng = random.Random(1)
        assert all(policy.delay_for(20, rng=rng) <= 10.0 for _ in range(200))

    def test_jitter_spreads_retries(self) -> None:
        """Without jitter a batch of failures retries in lockstep and re-DDoSes."""
        policy = RetryPolicy()
        rng = random.Random(7)
        draws = {round(policy.delay_for(3, rng=rng), 6) for _ in range(50)}
        assert len(draws) > 40

    def test_next_attempt_at_is_in_the_future(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = RetryPolicy().next_attempt_at(1, now=now, rng=random.Random(3))
        assert result >= now
