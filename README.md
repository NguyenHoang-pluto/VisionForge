# VisionForge

AI-assisted media creation and editing platform.

Upload a folder of images, video and audio; VisionForge analyses the media,
selects the strongest assets, infers a theme, recommends a template and a
soundtrack, builds a timeline, renders a video, evaluates the result, and lets
you take over manually at any point.

> **Current phase: Phase 6 — a real editing workstation.**
> A media browser, a preview viewer that plays the timeline, a timeline you can
> trim, reorder, split and delete on, an inspector, and an export panel. The
> automatic edit is now one tab inside the editor rather than the whole
> application. A hand-cut timeline is stored through the same validator and
> rendered by the same worker as a planned one — the backend stays
> authoritative, and the browser never gains a rendering model of its own.
> See [Roadmap](#roadmap).

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
   layer emits declarative plans over a closed vocabulary; deterministic
   compilers turn them into timelines and FFmpeg commands. A model refers to
   clips by opaque handles — `c1`, `c2` — and never sees a media id, a filename
   or a path.

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
.\scripts\vf.ps1 worker-cpu    # Celery worker on the cpu queue
.\scripts\vf.ps1 worker-gpu    # ...           on the gpu queue    (solo pool)
.\scripts\vf.ps1 worker-render # ...           on the render queue (solo pool)
.\scripts\vf.ps1 migrate       # alembic upgrade head
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

### Editing and rendering

| Endpoint | Purpose |
|---|---|
| `POST /api/projects/{id}/edit-plan` | Select media and plan an edit (deterministic or AI) |
| `POST /api/projects/{id}/edit-plan/manual` | Store a timeline the user cut by hand |
| `GET /api/projects/{id}/edit-plan` | Plans in a project, newest first |
| `GET /api/projects/{id}/edit-plan/{plan_id}` | One plan with the selection record behind it |
| `POST /api/projects/{id}/renders` | Queue a render of a plan |
| `GET /api/projects/{id}/renders` | Renders in a project |
| `GET /api/projects/{id}/renders/{render_id}` | One render, with a presigned playback URL |
| `GET /api/planner/capabilities` | Which modes, styles and presets exist, and whether AI is configured |
| `GET /api/projects/{id}/llm-runs` | Planning calls made for this project, successful or not |

A plan request carries intent, not geometry: a target duration, a clip budget, an
aspect ratio and an ordering. Output dimensions come from a closed preset map on
the server, so no request can ask for a 30 000-pixel canvas.

`edit-plan/manual` is the timeline's route into the system and carries exactly the
same intent: a sequence of cuts (`media_id`, `source_in_ms`, `source_out_ms`) plus
a shape, a frame rate and a quality *level*. It has no `order` field — position in
the array **is** the edit order, which is the only encoding that cannot express an
overlap or a gap. The cuts run through `validate_plan` against the project's own
media rows before anything is stored, so a hand-cut edit gets no weaker a gate
than a planned one: trimming past the end of a source is a 422 with the reason,
not a render that fails four minutes later.

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

### The edit lane

```
media_analysis
     |
  SELECT      score = 0.40 sharpness + 0.25 exposure + 0.20 contrast
              + 0.10 resolution + 0.05 duration; unusable clips rejected
              with a reason; pHash near-duplicates collapsed to one
     |
  PLAN        RulesEnginePlanner -> EditPlan   (closed vocabulary: media ids,
              millisecond offsets, one enum per transition -- no paths,
              no commands, no free text)
     |
  VALIDATE    every segment re-checked against the project's own media
     |
  TIMELINE    butt-joined clips on tracks         (knows nothing about FFmpeg)
     |
  RENDERSPEC  codec, CRF, preset, pixel format    (knows nothing about editing)
     |
  ARGV        compiler -> argv list -> subprocess (never a shell string)
     |
  MP4
```

The plan is the boundary, and it is structural rather than advisory: no field in
an `EditPlan` is wide enough to hold a filesystem path or an FFmpeg argument, so
a planner cannot emit one — whether it is today's rules engine or tomorrow's
language model ([ADR-0009](docs/adr/0009-edit-plan-boundary.md)). The plan is
validated again inside the render worker against the live database, because the
world can change between planning and rendering.

### The planning lane

Two planners implement one port. Which runs is decided per request, recorded on
the plan, and shown in the UI.

```
                    Planner (Protocol)
                     |              |
        RulesEnginePlanner      LlmPlanner  ──  LlmProvider
        deterministic           interprets      ├── anthropic
        always available        prose           ├── openai
                                                └── stub (local, no model)
```

An AI plan is always built as `FallbackPlanner(LlmPlanner, RulesEnginePlanner)`,
never as a bare `LlmPlanner`. There is no configuration in which a provider
failure becomes a failed request.

```
user request + analysed media
     |
  SELECT      the deterministic gates run FIRST -- a model is never offered a
              black frame, a blurred frame, or a duplicate
     |
  BRIEF       clips become handles: c1, c2, c3 ... the model's entire
              vocabulary for naming media. No ids. No filenames. No paths.
     |
  MODEL       -> EditDirective { style, pacing, clips:[{ref,duration_ms}], rationale }
     |
  RESOLVE     handles -> media ids, through a table the model never saw.
              An invented handle resolves to nothing and is refused.
     |
  CLAMP       every duration: style bounds, then plan bounds, then the source's
              own length. The total is capped by dropping whole clips.
     |
  EditPlan    -> the same Phase 4 validator -> Timeline -> RenderSpec -> argv
```

A model **cannot** reference another project's media, because that media has no
handle — the namespace does not contain it. It cannot leak a filename, because
it was never told one. Filenames are withheld for a second reason too: a file
named `ignore-previous-instructions-….mp4` is a prompt injection a user can
plant by naming a file ([ADR-0010](docs/adr/0010-llm-planner.md)).

**Styles are numbers, not prompt text.** Each of the eight styles is a
`StyleProfile` — pacing bounds, default duration and aspect, ordering, and
selection weights. So choosing "Cinematic" with the AI disabled still produces a
cinematic edit: the fallback costs you the interpretation of your sentence, not
the style you picked.

**Automatic mode does not always mean AI.** A written request goes to the model,
because interpreting prose is the one thing the rules engine cannot do. A bare
style does not: a style is already a complete, deterministic instruction, and
paying a model to restate numbers we already wrote down would be slower and less
predictable for no gain.

### Configuring a provider

```powershell
# .env -- server-side only, never sent to the browser
LLM_ENABLED=true
LLM_PROVIDER=anthropic      # anthropic | openai | stub
LLM_MODEL=claude-sonnet-5
LLM_API_KEY=...             # NEVER COMMIT THIS
```

With `LLM_ENABLED=false` (the default) every mode still plans, the AI control is
disabled in the UI, and nothing fails. `LLM_PROVIDER=stub` selects a
deterministic local provider with no model and no network, used by the tests; it
is refused outright when `ENVIRONMENT=production`, and the UI labels it
"deterministic stub, not a model" wherever it appears.

API keys are read in `infra/llm/factory.py` and nowhere else. They are never
returned by an endpoint, logged, stored in a plan, or written to a database row.

### What is recorded, and what is not

Every planning call writes an `llm_runs` row — provider, model, prompt version,
status, attempts, latency, token counts, and the fallback reason if it fell back.
The rows worth having are the ones with no plan attached, which is why this is a
table rather than a column on `edit_plans`.

It stores a truncated SHA-256 **digest** of the user's request, never the request
itself, and never the prompt or the completion. Enough to correlate a repeat or
match a support report to a row; not enough to reconstruct what somebody typed.

Use `python -m visionforge`, not `uvicorn` directly: on Windows the event-loop
policy must be set before uvicorn creates its loop
([ADR-0004](docs/adr/0004-windows-event-loop.md)).

---

## Running the frontend

```powershell
.\scripts\vf.ps1 web          # http://localhost:3000
```

The frontend is an editing workstation, not a page. It opens straight into a
fixed viewport with independently scrolling panels:

```
+----------------------------- top bar -----------------------------+
| VisionForge | project | analyse |                   panel toggles  |
+----------+-------------------------------------+-----------------+
| MEDIA    |                                     | INSPECTOR       |
| BROWSER  |            PREVIEW                  | clip . analysis |
|          |   source / program / render         | AI edit . export|
| grid or  +-------------------------------------+                 |
| list     |            TIMELINE                 |                 |
|          |  V1 --[clip][clip][clip]--          |                 |
|          |  A1 --[....][....][....]--          |                 |
+----------+-------------------------------------+-----------------+
| jobs . progress . health                                          |
+-------------------------------------------------------------------+
```

**Media browser** -- grid or list over the same library, with thumbnails,
duration, resolution, frame rate, size, ingest status and an analysis mark.
Ctrl-click and Shift-click select ranges; the list view is windowed so a library
of several hundred assets renders a screenful rather than all of it.

**Preview** -- one `<video>` element and three sources. *Source* scrubs a library
clip, *program* plays the timeline by sequencing each clip's 720p proxy and
cutting at its out point, *render* plays the finished MP4. The playhead is shared
with the timeline: one position, two views of it. The playhead is advanced from
the media element's own clock once a frame, so it tracks what is actually
playing rather than a wall-clock timer that would drift on a slow decode.

**Timeline** -- a video track and an audio representation, a ruler, a draggable
playhead, zoom (`Ctrl`+wheel zooms at the pointer) and horizontal scroll. Clips
can be trimmed by their edges, reordered by dragging, split at the playhead and
deleted. There are no gaps to drag into and no overlaps to create, because
`EditPlan` cannot describe either -- a UI that let you build one would be
offering an edit the renderer must reject.

**Inspector** -- four tabs over one selection: clip properties and trim points,
the analysis readout, the AI edit panel, and export.

**AI edit** -- mode (automatic / rules / AI), style, a free-text request, and the
target duration, aspect, frame rate and quality. What the panel offers is
whatever `/api/planner/capabilities` reports, so on a server with no API key the
AI mode is visibly disabled rather than silently falling back. A generated plan
opens in a review dialog -- clips, order, trims, transitions, output preset, the
planner that produced it, and the clips it rejected with reasons -- which can be
accepted into the timeline, adjusted and regenerated, or discarded. The raw model
output is not shown because it does not exist in the API: what the model "said"
*is* the structured plan.

**Export** -- resolution, aspect, frame rate, quality and audio, all as presets
the server declared. Rendering stores the current timeline as a plan and then
renders that plan, so what is encoded is always something the server has already
validated.

**Status bar** -- the job queue, live over the existing SSE stream. Collapsed it
is one line ("idle", or *n* jobs running with overall progress); expanded it is
every job with its step count, attempt, progress and failure reason.

Keyboard: `Space` play/pause, `S` split, `Del` delete clip, arrows nudge the
playhead (`Shift` for a second), `+`/`-` zoom, `B`/`I` toggle the side panels,
`?` for the full list. Single-key shortcuts are ignored while a text field has
focus.

Desktop-first, as an editor should be: it targets 1280x720 and up, and collapses
the media browser below 1180 px rather than squeezing four panels into a space
that fits three.

Infrastructure health is no longer its own page section -- it is the indicator at
the right of the status bar, which is where a tool that is being used rather than
inspected should put it.

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
├─ scripts/e2e_acceptance.py    Phase 2 acceptance test (ingest and jobs)
├─ scripts/e2e_analysis.py      Phase 3 acceptance test (analysis lanes)
├─ scripts/e2e_edit.py          Phase 4 acceptance test (plan and render)
├─ scripts/e2e_llm.py           Phase 5 acceptance test (LLM planning)
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

**Phase 6 — a professional editing workstation.**

- [x] Workstation shell: top bar, media browser, preview, timeline, inspector
      and a job status bar in one fixed viewport with resizable panels
- [x] Design system rebuilt on semantic tokens (`bg-panel`, `border-line`,
      `text-muted`) — four surface levels, one accent, 2 px radius, 11 px working
      type. No component spells a colour
- [x] Media browser: grid and list views, thumbnails, duration, resolution, FPS,
      size, status and analysis marks; Ctrl/Shift multi-select; windowed list
      rendering for large libraries
- [x] Preview viewer: one `<video>` across source / program / render, play,
      pause, seek, second-step navigation, volume, mute, speed, fullscreen,
      aspect-aware framing, and the name of the source clip on screen
- [x] **Program playback** — the timeline plays by sequencing 720p proxies and
      cutting at each clip's out point, driven by the element's own clock
- [x] Timeline: ruler, draggable playhead, zoom and scroll, trim handles, drag
      reorder with a drop indicator, split at playhead, delete, and an audio
      track that shows which sources actually carry audio
- [x] `POST /edit-plan/manual` — the one new contract. Cuts plus output intent,
      order implied by array position, validated by the unchanged Phase 4
      validator against the project's own media rows
- [x] AI edit as one inspector tab, driven entirely by
      `/api/planner/capabilities`; plan review dialog with segments, rejections
      and provenance, and accept / regenerate / render
- [x] Export panel that sends intent only — a shape, a frame rate, a quality
      level. No width, no CRF, no path, and no field that could carry one
- [x] Keyboard throughout, visible focus ring defined once globally, ARIA roles
      on the listbox, tablist, sliders and progress bars
- [x] 447 unit tests (+30) and no change to any Phase 1–5 contract

Deliberately **not** in Phase 6: music and beat synchronisation, audio mixing,
subtitles, transitions beyond a cut, multi-track video, keyframes, effects,
style learning, and collaboration.

<details>
<summary>Phase 5 — LLM planning behind the existing boundary</summary>

- [x] `LlmPlanner` implementing the Phase 4 `Planner` port; `RulesEnginePlanner`
      unchanged and still the default
- [x] `LlmProvider` port with two real adapters (Anthropic, OpenAI-compatible)
      plus a deterministic local stub for tests
- [x] `EditBrief` with opaque clip handles — the model never sees a media id, a
      filename or a path
- [x] `EditDirective`: a closed-vocabulary output schema, strictly parsed,
      with one repair attempt on invalid output
- [x] Three-stage clamping (style bounds → plan bounds → source length) before
      the unchanged Phase 4 validator
- [x] `FallbackPlanner` — every failure mode reaches the rules engine, and the
      reason is recorded rather than swallowed
- [x] Eight deterministic `StyleProfile`s that bind *both* planners
- [x] `resolve_mode` — automatic mode's decision as a pure, testable function
- [x] `llm_runs` table: provider, model, prompt version, latency, tokens,
      attempts, fallback reason. Request text is digested, never stored
- [x] 417 unit + 115 integration tests; Phase 5 acceptance passes with the
      provider both **disabled** and **enabled**

</details>

<details>
<summary>Phase 4 — Deterministic selection, <code>EditPlan</code>, timeline and render</summary>

- [x] Heuristic selection: a weighted score plus usability gates that reject with
      a reason, and pHash near-duplicate collapsing
- [x] `EditPlan` as a closed, frozen vocabulary — no field can carry a path or a command
- [x] Plan validation against the project's own media; an invalid plan is a
      permanent failure, never a retry
- [x] `RulesEnginePlanner` behind a `Planner` protocol, so an LLM planner is a
      swap rather than a rewrite
- [x] `Timeline` (no FFmpeg knowledge) → `RenderSpec` (no editing knowledge) → argv
- [x] Pure-argv FFmpeg compiler: one filter graph, `concat`, never a shell string
- [x] Render worker on the existing `render` queue, streaming `-progress` for
      measured progress and honouring cancellation at step boundaries
- [x] Edit and render API, all of it behind `require_project`
- [x] Workstation edit panel: settings strip, proportional timeline, rejection
      list with reasons, and an output column with the player and its real numbers
- [x] 323 unit + 88 integration tests, 40 Phase 4 acceptance checks
- [x] Verified end to end: 6 clips in → 2 rejected → 4 selected → 24.0 s
      1280×720 / 30 fps H.264 MP4, confirmed by an independent `ffprobe` and a
      full decode pass

</details>

<details>
<summary>Phase 3 — Media intelligence and the GPU lane</summary>

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
- [x] 182 unit + 74 integration tests, 52 Phase 3 acceptance checks

</details>

---

## Roadmap

| Phase | Weeks | Scope |
|-------|-------|-------|
| 1     | 1     | Foundation — infrastructure, skeleton, CI |
| 2     | 2–3   | Media ingest and the job system |
| 3     | 4–6   | Media intelligence — quality, dedupe, scenes, CLIP, faces |
| 4     | 7–8   | **Vertical slice**: folder in → timeline → rendered MP4 out, plus the web UI |
| 5     | 9–10  | LLM planning behind the `EditPlan` boundary, styles, provider abstraction |
| **6** | 11–12 | Editing workstation — media browser, preview, timeline, manual editing *(current)* |
| 7     | 13–14 | Subtitles, authentication, super-resolution, frame interpolation |
| 8     | 15    | Evaluation loop, Asset Studio with license tracking, AWS deployment |

The reasoning layer moved forward. It was planned for Phase 7, and was brought
into Phase 5 because Phase 4 had already built the boundary it has to sit behind
— once `EditPlan` existed as a validated artefact, adding a second planner was a
smaller job than the original ordering assumed.

Phase 6 then took the manual-editing half of its former slot and left the music
half behind. Two features in one phase would have meant a timeline good enough to
demonstrate beat synchronisation and not much else; the editor was the half that
everything after it has to be built inside, so it got the whole phase. Music and
beat synchronisation move to Phase 7, which now has a timeline to hang them on.

Deliberately out of scope for these fifteen weeks: personal style learning, model
training infrastructure, AI asset generation, multi-tenancy and billing.

---

## License

Not yet licensed. All rights reserved.
