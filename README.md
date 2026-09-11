# VisionForge

AI-assisted media creation and editing platform.

Upload a folder of images, video and audio; VisionForge analyses the media,
selects the strongest assets, infers a theme, recommends a template and a
soundtrack, builds a timeline, renders a video, evaluates the result, and lets
you take over manually at any point.

> **Current phase: Phase 1 — Foundation.**
> Infrastructure and application skeleton only. No AI, media processing or
> editing features are implemented yet. See [Roadmap](#roadmap).

---

## What it is

A modular monolith with asynchronous workers, not a constellation of
microservices. One FastAPI application owns the domain; long-running media and
model work happens in Celery workers on separate queues. The API creates jobs and
returns immediately — it never decodes, infers or encodes anything itself.

Three rules shape everything else:

1. **The API process never touches media bytes.** It presigns, records and
   enqueues. Enforced mechanically by import-linter contracts in CI.
2. **PostgreSQL owns job state.** Celery is transport. Redis is broker, cache and
   progress pub/sub — never a source of truth.
3. **AI output is a validated document, never executable code.** The reasoning
   layer emits declarative plans; deterministic compilers turn them into
   timelines and FFmpeg commands.

---

## Architecture

```mermaid
graph TB
  subgraph CLIENT["Browser"]
    UI["Next.js · TypeScript · Tailwind<br/>TanStack Query + Zustand"]
  end

  subgraph APP["Application tier — modular monolith"]
    API["FastAPI<br/>REST + SSE"]
    DOM["Domain modules"]
    API --> DOM
  end

  subgraph DATA["State"]
    PG[("PostgreSQL 16<br/>+ pgvector")]
    RD[("Redis<br/>broker · cache · pubsub")]
    OBJ[("Object storage<br/>MinIO → S3")]
  end

  subgraph WORK["Worker tier"]
    WC["CPU worker<br/>queue: cpu"]
    WG["GPU worker<br/>queue: gpu"]
    WR["Render worker<br/>queue: render"]
  end

  UI -->|"HTTPS JSON"| API
  UI -.->|"SSE progress"| API
  UI -->|"presigned PUT"| OBJ

  DOM --> PG
  DOM --> RD
  DOM -->|"presign only"| OBJ
  DOM -->|"enqueue"| RD

  RD --> WC
  RD --> WG
  RD --> WR
  WC --> PG
  WG --> PG
  WR --> PG
  WC --> OBJ
  WG --> OBJ
  WR --> OBJ
```

Internally the backend is layered, with dependencies pointing inward:

```
api/            HTTP routing, schemas, middleware
  ↓
application/    use cases
  ↓
domain/         entities, value objects, ports (Protocols)  ← pure Python
  ↑
infra/          db · redis · storage · queue  (implements the ports)
```

Design decisions are recorded in [`docs/adr/`](docs/adr).

---

## Requirements

| Tool           | Version            | Notes                                       |
|----------------|--------------------|---------------------------------------------|
| Python         | **3.11**           | Pinned. 3.12+ has thinner AI wheel coverage |
| Node.js        | 20+                | Developed on 24.11                          |
| pnpm           | 10+                | `npm install -g pnpm`                       |
| Docker Desktop | with Compose v2    | For Postgres, Redis and MinIO               |
| Git            | 2.40+              |                                             |

FFmpeg and a CUDA-capable GPU are **not** needed for Phase 1. They become
requirements in Phase 2 (rendering) and Phase 3 (inference).

---

## Local setup

```powershell
git clone <repo> E:\Tool\VisionForge
cd E:\Tool\VisionForge

Copy-Item .env.example .env
Copy-Item apps\web\.env.example apps\web\.env.local

.\scripts\vf.ps1 setup      # create .venv (3.11), install backend + frontend
.\scripts\vf.ps1 infra-up   # start containers, apply migrations
```

On Linux, WSL or CI use the equivalent `make` targets (`make setup`,
`make infra-up`, …). GNU make is not installed on the Windows dev machine, which
is why `scripts/vf.ps1` is the primary entrypoint there.

---

## Environment variables

Copy `.env.example` to `.env`. It is git-ignored and must never be committed.

| Variable | Purpose | Local default |
|---|---|---|
| `ENVIRONMENT` | `local` / `ci` / `staging` / `production` | `local` |
| `LOG_LEVEL` | Root log level | `INFO` |
| `DATABASE_URL` | SQLAlchemy URL (psycopg3) | `postgresql+psycopg://visionforge:visionforge@localhost:5442/visionforge` |
| `REDIS_URL` | Broker and cache | `redis://localhost:6389/0` |
| `S3_ENDPOINT_URL` | MinIO locally; empty on AWS (use an IAM role) | `http://localhost:9000` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | Object storage credentials | dev values |
| `S3_BUCKET_MEDIA` / `_DERIVATIVES` / `_RENDERS` | Bucket names | `visionforge-*` |
| `CORS_ORIGINS` | JSON array of allowed browser origins | `["http://localhost:3000"]` |
| `POSTGRES_PORT` / `REDIS_PORT` / `MINIO_PORT` | Host ports | `5442` / `6389` / `9000` |
| `NEXT_PUBLIC_API_URL` | API base URL baked into the web bundle | `http://localhost:8000` |

> **Ports are deliberately non-default.** 5432, 5433 and 6379 are occupied on the
> development machine by unrelated projects — see
> [ADR-0001](docs/adr/0001-local-ports.md).

---

## Running infrastructure

```powershell
.\scripts\vf.ps1 infra-up       # start + migrate
.\scripts\vf.ps1 infra-status   # container health
.\scripts\vf.ps1 infra-down     # stop, keep data
.\scripts\vf.ps1 infra-reset    # stop and DELETE volumes
```

| Service  | Host port | Console                                        |
|----------|-----------|------------------------------------------------|
| Postgres | 5442      | —                                              |
| Redis    | 6389      | —                                              |
| MinIO    | 9000      | http://localhost:9001 (`visionforge` / `visionforge-dev-secret`) |

Only stateful services run in Docker. The API, workers and web app run natively —
this machine has 7.4 GB of RAM and containerising everything does not fit. See
[ADR-0002](docs/adr/0002-hybrid-local-topology.md).

---

## Running the backend

```powershell
.\scripts\vf.ps1 api          # http://localhost:8000  (docs at /docs)
.\scripts\vf.ps1 worker-cpu   # Celery worker on the cpu queue
.\scripts\vf.ps1 migrate      # alembic upgrade head
```

| Endpoint            | Purpose                                                     |
|---------------------|-------------------------------------------------------------|
| `GET /health`       | Shallow check. Touches no dependency                        |
| `GET /health/live`  | Liveness — restart the process if this fails                |
| `GET /health/ready` | Readiness — probes Postgres, Redis and storage; **503** when a required dependency is down |
| `GET /version`      | Application name, version and environment                   |

Use `python -m visionforge`, not `uvicorn` directly: on Windows the event-loop
policy must be set before uvicorn creates its loop
([ADR-0004](docs/adr/0004-windows-event-loop.md)).

---

## Running the frontend

```powershell
.\scripts\vf.ps1 web          # http://localhost:3000
```

The home page reports API connectivity, per-dependency infrastructure status with
latency, and the version of both halves of the stack.

---

## Testing

```powershell
.\scripts\vf.ps1 test               # unit tests — no infrastructure needed
.\scripts\vf.ps1 test-integration   # requires infra-up
```

Unit tests never touch Postgres, Redis or MinIO: dependency probes are defined as
`Protocol` ports in the domain layer, so readiness logic is exercised with fakes.
Integration tests are marked `@pytest.mark.integration` and excluded from the
default run, because CI must stay green on a machine with no Docker.

A `gpu` marker exists and is always excluded. CI never requires a GPU, an NVIDIA
runtime, or downloaded model weights.

---

## Linting

```powershell
.\scripts\vf.ps1 check       # everything CI runs
.\scripts\vf.ps1 lint        # ruff check + ruff format --check
.\scripts\vf.ps1 format      # ruff --fix + ruff format
.\scripts\vf.ps1 typecheck   # mypy --strict
.\scripts\vf.ps1 contracts   # import-linter architecture contracts
```

`contracts` is the interesting one. It fails the build if:

- a layer imports outward (`domain` → `infra`, for example);
- the domain layer imports SQLAlchemy, FastAPI, Redis, boto3 or Celery;
- `api/` or `application/` imports `torch`, `cv2`, `ffmpeg`, `numpy` or `PIL`.

The third contract is the architecture's central rule, enforced by CI rather than
by discipline.

---

## Project structure

```
VisionForge/
├─ apps/
│  ├─ api/                      FastAPI backend (Python 3.11)
│  │  ├─ src/visionforge/
│  │  │  ├─ core/               config, logging, request context, runtime fixes
│  │  │  ├─ api/                routers, schemas, middleware, DI
│  │  │  ├─ application/        use cases
│  │  │  ├─ domain/             entities and ports — pure Python
│  │  │  ├─ infra/              db · redis · storage · queue
│  │  │  ├─ workers/            Celery entrypoints: cpu · gpu · render
│  │  │  └─ __main__.py         dev entrypoint (python -m visionforge)
│  │  ├─ alembic/               migrations
│  │  ├─ tests/                 unit/ · integration/
│  │  ├─ pyproject.toml         deps, ruff, mypy, pytest
│  │  └─ .importlinter          architecture contracts
│  └─ web/                      Next.js frontend (TypeScript)
│     └─ src/
│        ├─ app/                App Router
│        ├─ components/
│        ├─ lib/                api client, query provider
│        └─ stores/             Zustand — ephemeral client state only
├─ docs/adr/                    architecture decision records
├─ infrastructure/docker/       production Dockerfiles (Phase 2)
├─ scripts/vf.ps1               developer commands (Windows)
├─ .github/workflows/ci.yml
├─ docker-compose.yml           postgres · redis · minio
├─ Makefile                     same targets for Linux/WSL/CI
└─ .env.example
```

Workers are modules inside `apps/api`, not separate top-level projects: in a
modular monolith they share the domain code with the API and differ only in which
queue they consume. They ship as the same image with a different command.

---

## Current phase

**Phase 1 — Foundation.** Complete when the repository is clean, reproducible and
testable end to end:

- [x] Git repository with a `.gitignore` that excludes secrets, media and weights
- [x] Python 3.11 virtual environment, no global packages
- [x] Backend skeleton with layered architecture and enforced contracts
- [x] Frontend skeleton reporting live infrastructure status
- [x] PostgreSQL 16 + pgvector, Redis and MinIO in Docker Compose
- [x] Alembic initialised with a first migration
- [x] `/health`, `/health/live`, `/health/ready`, `/version`
- [x] Structured JSON logging with request correlation IDs
- [x] ruff, mypy (strict), import-linter, pytest and frontend build all green
- [x] CI that needs no GPU, no model weights and no NVIDIA runtime

---

## Roadmap

| Phase | Weeks | Scope |
|-------|-------|-------|
| **1** | 1     | Foundation — infrastructure, skeleton, CI *(current)* |
| 2     | 2–3   | Media ingest (presigned upload, ffprobe, thumbnails) and the job system |
| 3     | 4–6   | Analysis lanes — quality, dedupe, scenes, CLIP embeddings, faces |
| 4     | 7–8   | **Vertical slice**: folder in → timeline → rendered MP4 out, plus the web UI |
| 5     | 9–10  | Music, beat synchronisation, manual and hybrid timeline editing |
| 6     | 11–12 | Subtitles, authentication, super-resolution, frame interpolation |
| 7     | 13–14 | AI copilot, evaluation loop, Asset Studio with license tracking |
| 8     | 15    | AWS deployment, observability, usage metering |

Deliberately out of scope for these fifteen weeks: personal style learning, model
training infrastructure, AI asset generation, multi-tenancy and billing.

---

## License

Not yet licensed. All rights reserved.
