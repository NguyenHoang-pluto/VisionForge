# ADR-0007: Job execution, retry, resume and cancellation

**Status:** Accepted · 2026-09-12 (Phase 2)

Builds on [ADR-0003](0003-job-state-ownership.md), which established that
PostgreSQL owns job state and Celery is transport.

## Steps exist before the worker starts

`jobs` and all six `job_steps` rows are created in the same transaction. The API
can therefore answer "step 0 of 6" honestly on the first poll, rather than
reporting a spinner and inventing progress later.

Progress is always `completed_steps / total_steps` computed from real rows.

## Commit, then publish

```
BEGIN -> insert job + steps -> COMMIT -> publish to broker -> mark QUEUED
```

Publishing before the commit lets a worker claim a job whose row is not yet
visible. The failure modes are deliberately asymmetric:

- **commit fails** -> nothing published; no job, no message.
- **publish fails after commit** -> the job stays `PENDING` with `queued_at`
  NULL. `find_undispatched()` locates it and `redispatch()` sends it. This is
  why `PENDING` and `QUEUED` are separate states rather than one.

## Retry

Only `TransientError` retries. `PermanentError` -- corrupt media, unsupported
format, missing object -- fails on the first attempt and consumes no retry
budget. The retry decision reads exactly one thing: which base class was raised.

Backoff is exponential with **full jitter**. Without jitter a batch of jobs
failing on one transient cause retries in lockstep and reproduces the outage.

`attempts` lives in PostgreSQL, not in Celery, so a broker flush cannot resurrect
a poison message forever. `RETRY_WAIT` is a real persisted state, so the UI can
say "Retrying in 40s · attempt 2/3" instead of showing a stalled bar.

## Resume, and the bug it hid

A retry skips steps already marked `SUCCEEDED` -- it resumes rather than
restarting. Two things had to be fixed to make that true:

1. **Enum round-tripping.** Status columns were plain `String`, so SQLAlchemy
   returned `str` and `step.status is StepStatus.SUCCEEDED` was silently always
   False, making every retry redo completed work. The columns now use
   `Enum(..., native_enum=False)`, which emits identical VARCHAR DDL (no
   migration) but converts on load.
2. **Attempt-local state.** `JobContext.data` is per-attempt and not persisted,
   but `VALIDATE` was what downloaded the file. A resume starting at `METADATA`
   therefore had no local copy. Every step that touches bytes now goes through
   `_local_copy()`, which is lazy and idempotent: it downloads at most once per
   attempt, and not at all for a job that never gets past validation.

The rule this leaves: **anything a retry needs must be durable in PostgreSQL or
object storage, or cheaply rebuildable.**

## Cancellation

Cooperative, never `SIGKILL`. The API sets `cancel_requested`; the worker checks
it at each step boundary and stops cleanly. A half-killed FFmpeg leaves a
truncated object and no record of why.

- Not yet started -> `CANCELLED` immediately; there is no worker to observe it.
- Running -> `CANCEL_REQUESTED`, then `CANCELLED` at the next boundary; remaining
  steps become `SKIPPED` and completed work is preserved, not rolled back.
- Already past the last boundary -> allowed to finish `SUCCEEDED`. Forcing
  `CANCELLED` there would discard work that is already done.

## Duplicates are a success, not a failure

`DuplicateMediaError` carries the id of the asset that already holds those bytes.
It propagates through the runner untouched (it is neither transient nor
permanent); the task layer deletes the redundant `pending_upload` row, points the
job at the original, and marks the job `SUCCEEDED` with
`{"duplicate": true, "duplicate_of": ...}`.

Deduplication is scoped to `(project_id, sha256)`. Cross-project deduplication
raises ownership and deletion questions that have no answer before real auth.
