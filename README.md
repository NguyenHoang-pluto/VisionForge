# VisionForge

AI-assisted media creation and editing platform.

Upload a folder of images, video and audio; VisionForge analyses the media,
selects the strongest assets, infers a theme, recommends a template and a
soundtrack, builds a timeline, renders a video, evaluates the result, and lets
you take over manually at any point.

> **Current phase: Phase 3 — Media intelligence and the GPU lane.**
> Media is analysed into structured signals: quality, scene boundaries,
> perceptual hashes, CLIP embeddings and face counts. No editing, planning or
> rendering yet. See [Roadmap](#roadmap).

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
| **FFmpeg**     | **6.0+**           | `ffmpeg` and `ffprobe` on PATH              |
| NVIDIA driver  | 525+ (developed on 616.92) | Optional — GPU analysis falls back to CPU |

A CUDA-capable GPU is optional. Without one, `select_device()` returns `cpu`, the
GPU lane still runs (slowly), and every GPU test is skipped.

### Model weights

Weights live **outside the repository** and are downloaded on first use:

```powershell
$env:VF_MODEL_CACHE = "E:\ml-cache"   # YuNet face detector (232 KB)
$env:HF_HOME        = "E:\ml-cache\huggingface"   # OpenCLIP ViT-B/32 (~600 MB)
$env:TORCH_HOME     = "E:\ml-cache	orch"
```

Nothing is committed, nothing enters a Docker build context, and the cache is
shared between projects on the machine.

### Installing FFmpeg

FFmpeg is a hard dependency: workers assert it at start-up and refuse to accept
jobs they cannot process.

**Windows** (no admin needed):

1. Download the release build from <https://www.gyan.dev/ffmpeg/builds/>
   (`ffmpeg-release-essentials.zip`).
2. Extract it, e.g. to `E:\ffmpeg`, so that `E:\ffmpeg\bin\ffmpeg.exe` exists.
3. Add `E:\ffmpeg\bin` to your user `PATH`:
   ```powershell
   [Environment]::SetEnvironmentVariable(
     "Path", [Environment]::GetEnvironmentVariable("Path","User") + ";E:\ffmpeg\bin", "User")
   ```
4. Open a new terminal and verify: `ffprobe -version`.

**Linux / WSL:** `sudo apt install ffmpeg`  ·  **macOS:** `brew install ffmpeg`

This project was developed against **9.0.1**; the minimum enforced version is 6.

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
| `API_HOST` / `API_PORT` | Where the API binds | `127.0.0.1` / `8000` |
| `API_RELOAD` | Auto-reload on code changes (local only) | `true` |
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

### Projects and media

| Endpoint | Purpose |
|---|---|
| `POST /api/projects` | Create a project |
| `GET /api/projects` | List projects |
| `GET /api/projects/{id}` | One project |
| `POST /api/projects/{id}/media/upload-url` | Reserve a media row, get a presigned `PUT` |
| `POST /api/projects/{id}/media/{media_id}/complete` | Confirm the upload, dispatch ingest |
| `GET /api/projects/{id}/media` | Media library listing |
| `GET /api/projects/{id}/media/{media_id}` | One asset with full metadata |
| `GET /api/projects/{id}/media/{media_id}/thumbnail` | Presigned thumbnail URL |
| `GET /api/projects/{id}/media/{media_id}/proxy` | Presigned 720p proxy URL |
| `GET /api/projects/{id}/jobs` | Recent jobs for a project |

### Jobs

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs` | Create a job (honours `Idempotency-Key`) |
| `GET /api/jobs/{id}` | Job with its steps and progress |
| `POST /api/jobs/{id}/cancel` | Request cooperative cancellation |
| `GET /api/jobs/{id}/events` | **SSE** progress stream |

### Analysis

| Endpoint | Purpose |
|---|---|
| `POST /api/projects/{id}/analysis` | Queue analysis for one asset or the whole project |
| `GET /api/projects/{id}/analysis` | All analysis results in a project |
| `GET /api/projects/{id}/media/{id}/analysis` | Latest result per analyzer for one asset |
| `GET /api/projects/{id}/media/{id}/similar` | Visually similar media (cosine over CLIP embeddings) |

Every media, job and analysis route resolves access through project ownership.
There is no endpoint that reaches a row by id alone, and the similarity search is
project-scoped in SQL so it cannot cross the ownership boundary.

### The ingest pipeline

```
upload  ->  VALIDATE  ->  METADATA  ->  HASH  ->  THUMBNAIL  ->  PROXY  ->  FINALIZE
            exists?       ffprobe      sha256    480px jpeg     720p      mark ready
            magic bytes   is truth     dedupe    (not audio)    (>720p
            download                                            video only)
```

Each step is a row in `job_steps` with its own status, attempt count and timing,
so reported progress is measured rather than estimated.

### The analysis lanes

```
                      RESOLVE  (720p proxy preferred over the original)
                         |
        cpu queue -------+------- gpu queue  (solo pool = the GPU mutex)
            |                         |
        QUALITY   blur, exposure,  EMBED   OpenCLIP ViT-B/32 -> vector(512)
                  contrast
        SCENES    PySceneDetect    FACES   YuNet, CPU, detection only
        PHASH     DCT + aHash
            |                         |
            +---------- FINALIZE -----+
```

Results land in `media_analysis`, keyed `(media_id, analyzer, analyzer_version)`.
A model upgrade writes a new row rather than overwriting the old one, so a
quality regression is visible instead of silent.

Media types an analyzer does not apply to are recorded as `unsupported` with a
reason — audio has no blur score, and that is a fact rather than a failure.

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
│  │  │  ├─ api/                routers, schemas, middleware, DI, serializers
│  │  │  ├─ application/        media_service · job_dispatch · health_service
│  │  │  ├─ domain/             media · jobs · storage · errors · ports — pure Python
│  │  │  ├─ infra/              db · redis · storage · queue · ffmpeg
│  │  │  ├─ workers/            runtime · media_ingest · tasks · cpu/gpu/render
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
├─ scripts/e2e_acceptance.py    end-to-end acceptance test
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

**Phase 3 — Media intelligence and the GPU lane.**

- [x] `media_analysis` keyed `(media_id, analyzer, analyzer_version)`; versions coexist
- [x] pgvector `vector(512)` column with an HNSW cosine index
- [x] CPU lane: quality (blur, exposure, contrast), scene boundaries, pHash + aHash
- [x] Perceptual near-duplicate clustering (reports only; deletes nothing)
- [x] GPU lane: OpenCLIP ViT-B/32 embeddings, YuNet face detection
- [x] `GpuLeaseManager` with a 3000 MiB budget, LRU eviction and a lease mutex
- [x] `ModelRegistry` with lazy loading and warm caching (5140 ms cold → 138 ms warm)
- [x] Proxy-first analysis; the original is never opened by the analysis lane
- [x] Deterministic frame sampling at 5/25/50/75/95% of duration
- [x] Analysis jobs on the existing job system, CPU and GPU queues
- [x] Per-job GPU metrics: model, device, VRAM peak, duration
- [x] Analysis API and a media-workstation inspector UI
- [x] 182 unit + 74 integration tests, 42 Phase 3 acceptance checks

Deliberately **not** in Phase 3: the LLM planner, `EditPlan`, timeline
generation, editing, music, subtitles, rendering, asset search, authentication.

---

## Roadmap

| Phase | Weeks | Scope |
|-------|-------|-------|
| 1     | 1     | Foundation — infrastructure, skeleton, CI |
| 2     | 2–3   | Media ingest and the job system |
| **3** | 4–6   | Media intelligence — quality, dedupe, scenes, CLIP, faces *(current)* |
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
