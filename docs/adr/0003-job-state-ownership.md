# ADR-0003: PostgreSQL owns job state; Celery is transport

**Status:** Accepted · 2026-09-11 (carried forward from Phase 0)

## Context

A media pipeline runs jobs that take minutes and fail in interesting ways. The
tempting shortcut is to let Celery's result backend answer "what is job X doing?"
That backend is a cache: it expires, it loses history, it cannot be joined
against project data, and it cannot express the per-step progress a ten-minute
render needs.

## Decision

- `jobs` and `job_steps` tables in PostgreSQL are the single source of truth.
- Celery is configured with `result_backend=None` and `task_ignore_result=True`.
- Redis is broker, cache and pub/sub for progress events -- never state.
- The job row is written inside the request transaction; the Celery message is
  published only on `after_commit`.

## Queues and pools

Three queues, established in Phase 1 and used from Phase 2:

| Queue    | Work                               | Pool (local) | Pool (Linux prod) |
|----------|------------------------------------|--------------|-------------------|
| `cpu`    | probe, thumbnails, deterministic analysis | `threads` (2) | `prefork` (2) |
| `gpu`    | model inference                    | `solo` (1)   | `solo` (1)        |
| `render` | FFmpeg encoding                    | `solo` (1)   | `solo` (1)        |

`--pool=solo` on the GPU queue is the design, not a Windows workaround. With
~3.2 GB of usable VRAM on an RTX 3050, exactly one model may be resident and one
inference may run at a time, so a single-slot worker **is** the GPU mutex -- no
distributed lock, no semaphore, no VRAM race. Production scales by adding worker
*instances*, one per GPU. The concurrency model is identical in both places.

## Consequences

- Job status survives a broker flush, a Redis eviction and a worker restart.
- Job history is queryable and joinable, which is what version history and usage
  metering in later phases need.
- **Cost:** one extra write per state transition. Acceptable for jobs measured in
  seconds to minutes.
