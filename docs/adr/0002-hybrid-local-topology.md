# ADR-0002: Stateful services in Docker, application processes native

**Status:** Accepted · 2026-09-11 (carried forward from Phase 0)

## Context

The development machine has **7.4 GB of RAM total**, with roughly 0.2-0.4 GB free
under normal use. Docker Desktop is allocated 3.76 GB of that. Running the full
stack in containers -- Postgres, Redis, MinIO, API, three workers, Next.js --
alongside an IDE and a browser does not fit.

## Decision

`docker-compose.yml` runs **only** stateful infrastructure:

- PostgreSQL 16 + pgvector (`shared_buffers=128MB`, `max_connections=30`, 512 MB limit)
- Redis 7 (`maxmemory 128mb`, no persistence, 192 MB limit)
- MinIO (single-node single-drive, 384 MB limit)

The API, the Celery workers and the Next.js dev server run **natively** in the
repo's `.venv` and `node_modules`.

## Consequences

- Memory headroom: infrastructure costs ~390 MB instead of ~1.5 GB.
- The processes under active development get native reload and a native debugger.
- Workers get direct access to the host GPU and to FFmpeg without container GPU
  passthrough, which is unverified on this machine and blocked by an old driver.
- **Cost:** the application is not exercised in a container locally. Mitigated by
  building the production images in CI from Phase 2 onward, so a
  "works on my machine" gap cannot accumulate silently.
- Prometheus and Grafana are deferred to week 15 and will sit behind a compose
  profile, never in the default `up`.
