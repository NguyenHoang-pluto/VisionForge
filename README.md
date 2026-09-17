# VisionForge

AI-assisted media creation and editing platform.

Upload a folder of images, video and audio; VisionForge analyses the media,
selects the strongest assets, infers a theme, recommends a template and a
soundtrack, builds a timeline, renders a video, evaluates the result, and lets
you take over manually at any point.

> **Current phase: Phase 11 — the editorial decision engine.**
> Automatic editing stopped being "select N clips and concatenate them". Media
> now becomes an edit through five stages — signals and editorial events, a
> story arc, creative selection, a pacing curve, and typed editorial decisions
> — before it reaches the unchanged `EditPlan`. Shot lengths come from an
> energy curve rather than from division; ten genre policies and three variants
> cut the same footage materially differently; and every clip kept or rejected
> says why in structured reasons. A model sets direction and never names a clip.
> See [Roadmap](#roadmap) and
> [ADR-0015](docs/adr/0015-editorial-decision-engine.md).

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
   or a path. Changing an existing edit works the same way: a model emits an
   `EditDelta` of typed operations addressed by clip position, and deterministic
   code applies it or rejects it whole.

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
        BEATS     tempo + grid
                  (audio only)
            |                         |
            +---------- FINALIZE -----+
```

Results land in `media_analysis`, keyed `(media_id, analyzer, analyzer_version)`.
A model upgrade writes a new row rather than overwriting the old one, so a
quality regression is visible instead of silent.

Media types an analyzer does not apply to are recorded as `unsupported` with a
reason — audio has no blur score, and that is a fact rather than a failure.

### The audio lane

```
audio asset  ->  ingest (no thumbnail, no proxy)  ->  BEATS
                                                        |
                          decode to mono 22050 Hz PCM (FFmpeg, argv)
                                                        |
                          STFT -> spectral flux -> onset envelope
                                                        |
                          autocorrelation + tempo prior -> BPM
                          exhaustive phase search       -> grid
                                                        |
                          media_analysis (analyzer="beats", versioned)
```

**Deterministic and local.** No model, no network, no music API: the tempo is
measured from the waveform with numpy arithmetic, so the same file yields the
same beats forever and a change in a beat-synced cut is attributable to a
parameter rather than to chance. Nothing is ever downloaded — music is a
project-owned asset the user uploaded, and the test fixtures *generate* theirs
with FFmpeg.

Confidence is periodicity times grid agreement, and both must hold. Silence,
white noise and a sustained tone all score below the threshold, because
autocorrelation alone would call an idling engine 128 BPM. Below the floor the
grid is treated as absent, and "absent", "never analysed" and "not requested"
are deliberately indistinguishable downstream: all three mean *plan the way
Phase 4 planned*.

What it assumes and where it fails — one steady tempo, no downbeats, no rubato,
±23 ms precision, a 120 BPM prior — is set out in
[ADR-0011](docs/adr/0011-audio-track-and-beats.md).

### Beat-synced cutting

The timeline is butt-joined: every clip starts where the last one ended. So if
each clip is a whole number of beats long and the first starts on a beat, *every*
cut lands on a beat — with no per-cut search, no drift, and no way to express a
gap or an overlap, because the structure that would hold one does not exist.

A clip's length is the difference of two cumulative beat positions, never one
count times a period. At 128 BPM a beat is 468.75 ms, so rounding each clip
separately and adding compounds the error at half a millisecond per cut; rounding
the running total never exceeds half a millisecond however many cuts precede it.

A source too short for its allocation gets *fewer whole beats*, not a truncated
one. The plan records the integer beat count it chose, so it can say what it
synced to.

### The audio mix

```
[clips:v]  trim -> scale -> fps -> format -> concat            -> [vout]

[clips:a]  atrim -> asetpts -> aformat  -> concat -> volume    -\
                                                                 amix -> apad
[music:a]  atrim -> asetpts -> aformat -> volume -> afade x2   -/    -> atrim
                                              -> adelay                -> [aout]
```

Four shapes — silent, source only, music only, both — and the last step is the
same in all of them: the audio is cut to exactly the video's length. `apad`
covers a bed shorter than the picture, `atrim` cuts one longer than it. FFmpeg's
`-shortest` was rejected because it decides by whichever stream ends first, which
is the right answer only by luck.

Source and music levels are independent, so dialogue can be ducked under a bed
without being turned off. `amix` runs with `normalize=0`: with normalisation on
it divides by the input count, so adding music would silently halve the dialogue
by a gain nobody set.

Every value reaching the filter graph is a number or a server-resolved path. The
music cue is a media id and six numbers — there is no field in it for a filename,
a filter or an encoder setting, and the tests pin that field set and assert every
word in the generated graph comes from an allow-list.

### Reference style

A reference video is a clip the user already owns, nominated as "cut mine like
this". Phase 8 reads it with the analyzers that already exist and turns their
rows into a small, versioned, entirely numeric description:

```
scenes  + beats  ->  shot length, quartiles, cut rate, beat-sync tendency
quality          ->  luminance, contrast
dynamics         ->  motion energy, saturation
```

`dynamics` is the one new analyzer, and it exists because nothing measured
movement or palette: `quality` samples the HSV *value* channel rather than
saturation, and five frames spread across a video say nothing about motion. At
each of the deterministic sample points it decodes a *pair* of frames 120 ms
apart and takes the mean absolute grayscale difference, plus the saturation of
the first. No model, no weights, no GPU, no network -- one extra frame read per
sample point.

**Every feature carries its own confidence**, and a signal that was not analysed
is *absent* rather than zero. "Measured, and it was still" and "nobody looked"
are different claims, and the policy layer weights by exactly this difference. A
single detected scene -- which is how the scene detector reports finding no cuts
-- yields no pacing at all, because it cannot be told apart from a genuine single
take.

**Beat-sync tendency** is the fraction of the reference's own cuts that land
within a tempo-proportional window of its own beats: the measurement behind
"this was cut to the music". It needs three cuts and a trusted grid before it
will say anything.

The profile is **derived on read**, not stored. The derivation is deterministic,
so recomputing it from the analysis rows costs a little arithmetic and buys a
profile that can never be stale against a re-analysis.

### Style strength

`StyleProfile (named preset) + ReferenceProfile (measured) -> StylePolicy -> planner`

Five stops: 0, 25, 50, 75, 100. Each measurement's own confidence multiplies the
strength before it is applied, so a shot length read from three cuts moves the
pacing about a third as far as one read from twenty -- asking for 100% is asking
for as much of the reference as the reference actually supports.

**Zero is exactly the Phase 4-7 behaviour**, for every style, and a test plans
the same footage twice and compares segments, trims, totals and metadata to
prove it. Style affinity is taken *out of* the existing selection weights rather
than added on top, so a score stays a convex combination and is capped at a
third: style decides between usable clips, it does not decide what usable means.

The rules engine honours a reference with no model involved. When an LLM
provider fails, the fallback produces the pacing the model was asked for rather
than a generic cut.

### The reference is never footage

The rendered output contains only the user's *other* media. The nominated clip
is removed from the candidate list where candidates are built, so neither
planner can select what it was never handed, and the model is additionally never
given a handle for it.

This is not a formality. A professionally-cut reference outscores phone footage
on every usability signal the ranker has, so a reference that was silently
eligible would usually *win*. Detaching it makes the clip ordinary footage
again -- the exclusion follows from being current, not from a mark on the asset.

Which clip is the reference is project state, set through its own endpoint and
resolved server-side. The plan request has no field naming one, so a caller
cannot aim a request at another project's media and read its measurements out of
the result. Nominating resolves the id through the project first, and a media id
belonging to someone else gets the answer a nonexistent one gets.

```
PUT    /api/projects/{id}/reference   {"media_id": "..."}
GET    /api/projects/{id}/reference
DELETE /api/projects/{id}/reference
```

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
fixed viewport with independently scrolling panels, behind a navigation rail
that names six places and keeps exactly one of them current:

```
+--+--------------------------- project bar ---------------------------------+
|◆ | Phase 7 Music ▾ > Editor | analyse · generate · render |        panels ⌨ |
|  +----------+---------------------------------------+--------------------+
|⌂ | MEDIA    |                                       | INSPECTOR          |
|▣ | BROWSER  |            PREVIEW                    | clip . analysis    |
|▤ |          |   source / program / render           | AI edit . audio    |
|∿ | grid     +---------------------------------------+ export             |
|⇥ | list     |            TIMELINE                   |                    |
|  | compact  |  TC |0:00    |0:02    |0:04    |0:06   |                    |
|  |          |  V1 [==clip==][==clip==][==clip==]     |                    |
|☀ |          |  A1 [--------][--------][- - - -]     |                    |
|EN|          |  M1 [========= music bed ===========]  |                    |
|⚙ +----------+---------------------------------------+--------------------+
|  | jobs . progress . health                                               |
+--+------------------------------------------------------------------------+
```

**Workspaces.** Home, Editor, Assets, Audio, Exports and Settings, on the rail
and on the digit keys. They are all the same project and the same client state
— switching to Assets does not unload the timeline — but each gives one activity
the whole viewport when that is the activity you are doing. Picking twelve clips
out of two hundred is a different job from trimming four of them, and a 260px
column is the wrong size for the first and the right size for the second. The
media browser grows from two tile columns to six; the music panel sits beside a
full-width timeline and the picture; the export settings sit beside the real
pipeline stages, walked from the render job's own step rows.

**Home** is the only workspace that works without a project, and every figure on
a project card — cover frame, footage duration, format, aspect, activity — is
derived from that project's own media listing, because the projects endpoint
carries none of it. Nothing is invented: a project with no ready footage shows
no duration rather than a zero, and the timestamp reads "active", not "last
edited", because nothing in the API records when a timeline was last touched.

### Design system

One palette, three switchable axes, all of them resolved in CSS on `<html>` so
that changing one is a repaint rather than a re-render. The tokens live in
[`globals.css`](apps/web/src/app/globals.css) and are reached only through the
semantic Tailwind names in [`tailwind.config.ts`](apps/web/tailwind.config.ts):
a component says `bg-surface`, `border-subtle`, `text-muted`, `h-control`, and
cannot spell a colour or a height of its own.

| Axis | Attribute | Values |
| --- | --- | --- |
| Theme | `data-theme` | `dark` (default), `light` |
| Accent | `data-accent` | `azure` (default), `violet`, `teal`, `amber`, `crimson` |
| Density | `data-density` | `comfortable` (default), `compact` |

**Surfaces.** `background`, `surface`, `surface-elevated`, `surface-hover` and
a `surface-sunken` well, on a warm graphite ground that is nowhere near black.
The levels are spaced far enough apart to carry hierarchy on their own, which is
what lets the layout drop almost every internal border it used to draw: depth
comes from a change of level first, a soft shadow second, and a line only where
two things genuinely abut.

That ordering is a correction. An earlier revision built every boundary out of a
hairline and said so proudly; the result was structurally correct and visually
exhausting, forty outlined rectangles of near-identical value reading as a
wiring diagram rather than as a place to work.

Light mode is not an inversion and not a white page. It runs the luminance the
other way — warm paper ground, panels lifted above it, controls lifted above
those — and nothing in it is pure white or pure black, because #FFF panels
beside #000 text is the combination that makes a light editor tiring within the
hour.

**Shape and motion.** The radius scale runs to 24px, far enough that a panel can
be visibly softer than the control sitting on it. Motion is two durations and
one ease-out curve, moving nothing but `transform` and `opacity`, and every bit
of it is disabled under `prefers-reduced-motion`.

One accent does three jobs and no more: the active state of a control, the
current selection, and the single primary action on a surface. Anything needing
a fourth meaning is a status colour (`ok` / `warn` / `danger` / `info`), and
every status colour is paired with a word — no state in the application is
carried by hue alone. The playhead keeps its own red, because "the clip I am
holding" and "the frame I am looking at" have to stay distinguishable.

Colours are stored as sRGB channel triples rather than hex, so Tailwind composes
them with an alpha channel (`bg-accent/40` → `rgb(var(--accent) / 0.4)`). Given
a bare `var()` Tailwind 3 drops the modifier silently, which is how a palette
ends up with forty hand-written translucent variants nobody can audit. The
accent's soft tint and the timeline's clip fills are mixed from the accent per
theme with `color-mix`, so adding a preset is two lines.

**Density** compacts chrome only — control heights, header strips, row rhythm,
panel padding, track height. Type size does not shrink, because a professional
tool that becomes unreadable on its densest setting effectively has one setting.

### Preferences

Language, theme, accent and density, reachable from the gear at the foot of
the navigation rail, with theme and language also one click away on the rail
itself — they are the two settings most likely to be wrong on first run, and the
two whose wrongness makes everything else harder. They are stored in
`localStorage` under `visionforge.preferences` and are never sent to the API — a
second machine is entitled to a different answer.

An inline script in `<head>` applies the three CSS axes before first paint, so
the first frame is already in the right theme; the store rehydrates in a layout
effect, which is after React has matched the server's HTML and before the
browser paints, so there is neither a hydration mismatch nor a flash.

### Language

English and Vietnamese, switchable live from the navigation rail. The
dictionary lives
in [`src/lib/i18n`](apps/web/src/lib/i18n): English is authoritative and its
keys define the message set, and `vi` is typed as `Record<MessageKey, string>`
so a missing translation is a build error rather than a stray English word in
the middle of a Vietnamese panel. Components contain no English at all — they
ask for `media.empty.title`, and the fact that there are two languages is not
something they can see. Interpolation is `{name}` slots; `t.plural(base, count)`
picks between `.one` and `.other` so the call site does not have to know which
languages inflect.

### Panels

**Project Home** — with no project open (and any time the rail is asked for
it), a grid of project cards with cover frames, footage length, format and
aspect, above the three steps and a note that everything runs locally. It
replaced a column of text in a black viewport that was correct and
indistinguishable from an error page.

**Media browser** — three views over the same library, because the questions
differ in kind: *grid* answers "which shot is this" (thumbnail, duration burnt
into the frame, kind, resolution), *list* answers "which of these is 60fps"
(columns for duration, resolution, frame rate, size, status), *compact* answers
"where is clip_047" (one line each, as many filenames on screen as fit). List
and compact are windowed, so a library of several hundred assets renders a
screenful. Selection is marked as well as tinted — filled tick for the item the
inspector is following, outlined tick for the rest of a multiple selection.
Ctrl-click and Shift-click select ranges.

**Preview** — one `<video>` element and three sources. *Source* scrubs a library
clip, *program* plays the timeline by sequencing each clip's 720p proxy and
cutting at its out point, *render* plays the finished MP4. Transport, timecode
(burnt into the frame and in the bar), scrub, speed, volume and fullscreen. The
playhead is shared with the timeline: one position, two views of it. It is
advanced from the media element's own clock once a frame, so it tracks what is
actually playing rather than a wall-clock timer that would drift on a slow
decode. The letterbox stays black in both themes — it is the surround for a
picture, and a light grey one would lie about the black level of what is inside.

**Timeline** — a ruler with two levels of tick, a draggable playhead, a video
track and an audio representation, filmstripped clip blocks with trim handles,
zoom (`Ctrl`+wheel zooms at the pointer, plus fit-to-width) and horizontal
scroll. The lanes always span the panel, so an empty sequence looks empty rather
than broken. Clips can be trimmed by their edges, reordered by dragging, split
at the playhead and deleted. There are no gaps to drag into and no overlaps to
create, because `EditPlan` cannot describe either -- a UI that let you build one
would be offering an edit the renderer must reject. For the same reason there is
no snapping indicator: cuts are butt-joined by construction, so a magnet button
here would be a light that is always on.

**Inspector** — four tabs over one selection: clip properties and trim points,
the analysis readout, the AI edit panel, and export. Label/value pairs on a
fixed column, so a column of them lines up down the panel instead of ragging
with the length of each value.

**AI edit** — mode (automatic / rules / AI), style, a free-text request, and the
target duration, aspect, frame rate and quality. What the panel offers is
whatever `/api/planner/capabilities` reports, so on a server with no API key the
AI mode is visibly disabled rather than silently falling back. A generated plan
opens in a review dialog -- clips, order, trims, transitions, output preset, the
planner that produced it, and the clips it rejected with reasons -- which can be
accepted into the timeline, adjusted and regenerated, or discarded. The raw model
output is not shown because it does not exist in the API: what the model "said"
*is* the structured plan.

**Export** — resolution, aspect, frame rate, quality and audio, all as presets
the server declared. Rendering stores the current timeline as a plan and then
renders that plan, so what is encoded is always something the server has already
validated.

**Status bar** — the job queue, live over the existing SSE stream. Collapsed it
is one line ("idle", or *n* jobs running with overall progress); expanded it is
every job with its step count, attempt, progress and failure reason.
Infrastructure health is the indicator at the right, which is where a tool that
is being used rather than inspected should put it.

### Keyboard and accessibility

`Space` play/pause, `S` split, `Del` delete clip, arrows nudge the playhead
(`Shift` for a second), `+`/`-` zoom, `B`/`I` toggle the side panels, `?` for the
full list. Single-key shortcuts are ignored while a text field has focus.

One focus ring is defined for the whole application on `:focus-visible`, so
pointer users never see it and no control can ship without it. Every icon-only
control takes a mandatory label that becomes both its accessible name and its
tooltip. Toggles carry `aria-pressed`, tabs `aria-selected`, the playhead and
trim handles `role="slider"` with live values.

Desktop-first, as an editor should be. Panel widths step at 1400 and 1800 px;
below 1180 the media browser collapses and below 1024 the inspector follows,
rather than squeezing four panels into a space that fits three. Verified at
1280×720, 1440×900 and 1920×1080, in both themes and both languages.

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

Audio fixtures are generated by FFmpeg at test time — a sine wave, a decaying
pulse train, a synthesised metronome at a known tempo. Nothing copyrighted is
committed, downloaded, or sent over the wire, and a click track built at exactly
120 BPM gives beat detection a right answer to be measured against rather than
an opinion to be agreed with.

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
├─ scripts/e2e_editor.py        Phase 6 acceptance test (timeline editing)
├─ scripts/e2e_music.py         Phase 7 acceptance test (beats and audio mix)
├─ scripts/e2e_style.py         Phase 8 acceptance test (reference style)
├─ scripts/e2e_effects.py       Phase 9 acceptance test (transitions, subtitles)
├─ scripts/e2e_coedit.py        Phase 10 acceptance test (AI co-editor)
├─ scripts/e2e_editorial.py     Phase 11 acceptance test (editorial engine)
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
- [x] Design system rebuilt on semantic tokens — four surface levels, one
      accent, one working type size. No component spells a colour or a height.
      (The token *names* were replaced in the UI redesign below; the rule that a
      component cannot invent a colour is unchanged)
- [x] **Dark and light themes**, five accent presets and two interface
      densities, all three resolved as CSS on `<html>` so a change repaints
      rather than re-renders; applied before first paint, so no flash
- [x] **English and Vietnamese**, from one typed dictionary — a missing
      translation is a build error, and no component contains English
- [x] Preferences dialog (language, theme, accent, density), persisted to this
      browser and never sent to the API
- [x] Media browser: grid, list and compact views, thumbnails, duration,
      resolution, FPS, size, status and analysis marks; Ctrl/Shift multi-select
      marked by tick as well as tint; windowed rendering for large libraries
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
      on the listbox, tablist, sliders and progress bars; no state carried by
      colour alone
- [x] 447 unit tests (+30) and no change to any Phase 1–5 contract

Deliberately **not** in Phase 6: music and beat synchronisation, audio mixing,
subtitles, transitions beyond a cut, multi-track video, keyframes, effects,
style learning, and collaboration.

---

**Phase 7 — music, beats and the audio mix.**

- [x] `MusicCue` on `EditPlan` — a media id and six numbers: a trim within the
      track, a position on the output, a linear gain and two fades. Optional, so
      a Phase 4-6 plan deserialises to `music=None` and re-renders to the same
      bytes
- [x] `OutputSpec.source_gain` — the clips' own level, independent of the bed, so
      dialogue can be ducked without being turned off
- [x] Validation mirrors the video track check for check, plus the audio-only
      rules: gain bounds, fade bounds, overlapping fades, and a cue that starts
      after the picture ends. Violations carry `is_music` so an editor can point
      at the audio lane
- [x] **Beat detection** — spectral flux, autocorrelation with a tempo prior,
      parabolic refinement and an exhaustive phase search. NumPy only, no new
      dependency, no model, no network. Versioned like every other analyzer
- [x] Confidence is periodicity times grid agreement; silence, white noise and a
      sustained tone are all rejected, and the planner then declines to move a cut
- [x] **Beat-synced cutting** — clip *duration* is quantised to whole beats, and
      because the timeline is butt-joined every cut then lands on a beat. Lengths
      are differences of cumulative positions, so rounding never accumulates
- [x] Graceful fallback: not requested, not analysed, not trusted, or no whole
      beat count that fits are all indistinguishable downstream and all mean
      "plan the way Phase 4 planned"
- [x] **Audio mix** — source and music concatenated separately then mixed with
      `normalize=0`, levels independent, fades and delay on the bed, and the
      result `apad`/`atrim`ed to exactly the video's length rather than trusting
      `-shortest`
- [x] `MediaFact.from_media` — one mapping from a media row to planning facts,
      shared by the planning service and the render worker, which sit on
      opposite sides of the validation gate
- [x] Music lane in the workstation with beat markers, a fifth inspector tab for
      the bed, levels as percentages, fades, trim and placement, and the beat
      readout with the confidence the planner actually uses
- [x] 38 new message keys in English and Vietnamese (477, exact parity), the
      same tokens, themes, density and accent system as Phase 6
- [x] 686 unit tests (+71) and 17 FFmpeg integration cases; no change to any
      Phase 1–6 contract

Deliberately **not** in Phase 7: downloading music from anywhere, an online
provider or catalogue, ducking automation, multi-track mixing, waveform display,
per-clip audio gain, and any copyrighted asset in the repository. The fixtures
and the acceptance script generate their own music with FFmpeg.

---

**UI redesign — a creative application rather than a dashboard.**

A frontend-only pass over everything Phases 1–7 built. No backend change, no new
capability, no change to any API contract: the same product, presented properly.

- [x] **Token layer rebuilt.** Warm graphite dark and warm-paper light, neither
      near black nor near white; four surface levels spaced far enough apart to
      carry hierarchy without a border; a radius scale that runs to 24px; a type
      scale raised a step throughout with display sizes above it; four shadow
      steps; a motion system of two durations and one curve
- [x] Vocabulary renamed to the semantic set it should always have used —
      `background`, `surface`, `surface-elevated`, `surface-hover`,
      `border-subtle`, `foreground`, `foreground-muted`, `accent`, `success`,
      `warning`, `danger`, `info` — mechanically, across all 233 usages, so the
      codebase has one set of names rather than two
- [x] **Six named workspaces** behind a 56px navigation rail (Home, Editor,
      Assets, Audio, Exports, Settings), on the digit keys. Switching never
      touches the draft
- [x] **Project Home** with cover frames and per-project facts, every one of
      them derived from that project's own media listing because the projects
      endpoint carries none of them. Nothing invented, nothing rounded up
- [x] Assets, Audio and Exports are the existing panels given the viewport, not
      reimplementations: the browser grows to six tile columns, the music panel
      sits beside the picture and a full-width timeline, and export settings sit
      beside the render's real pipeline stages, walked from its own step rows
- [x] Softer shapes throughout — clips, tiles, cards, controls — with depth from
      surface level and restrained shadow rather than from hairlines
- [x] The music panel now offers the project's audio assets directly, instead of
      telling the user to select one in a browser the Audio workspace does not
      show
- [x] `relativeTime` localised through `Intl.RelativeTimeFormat`: it was the one
      place in the application that returned hard-coded English
- [x] 495 message keys, exact English/Vietnamese parity, zero unreferenced — 19
      orphaned by the deleted top bar and welcome screen were removed
- [x] Verified at 1280×720, 1440×900 and 1920×1080 in both themes, both
      languages, all five accents and both densities, with a scripted
      horizontal-overflow audit on every combination

Deliberately **not** in this pass: any backend change, a waveform (the API
exposes no sample data and drawing one would be fiction), per-clip colour or
transform controls, and anything belonging to Phase 8.

---

**Phase 11 — the editorial decision engine.**

- [x] **Five stages between media and `EditPlan`**, each a pure function with its
      own version: derived signals → editorial events → story roles → creative
      selection → a pacing curve → typed decisions. Nothing downstream changed;
      the plan, the timeline, the compiler and the renderer are Phase 4's and
      Phase 9's, which is what makes the claim "the edit is different" checkable
      rather than a matter of taste
- [x] **Signals are derived, never re-measured.** Quality, visual energy, motion,
      face presence, semantic similarity, duplicate distance, beat relation and
      reference-style match all come from analyses Phase 3 and Phase 8 already
      wrote. No new model, no second pass over the footage, nothing new on the
      4 GB card
- [x] **A closed, versioned editorial event vocabulary** of twelve members plus
      `UNKNOWN` — establishing, close-up, subject entry and exit, action, peak
      motion, impact, reaction, celebration, dialogue, transition moment, ending.
      Every event carries a confidence, a clip carries at most three, and
      `UNKNOWN` is an answer rather than a failure: an unknown clip is still
      selectable, it simply argues for no particular place in the edit
- [x] **Selection is creative, and diversity can outvote quality.** Six weighted
      terms — quality, relevance, diversity, role fit, energy fit, style match —
      with a per-policy diversity floor, so the second-best angle on a goal loses
      to a merely-good shot of something else. Near-duplicates are *rejected with
      a reason*, not quietly ranked down
- [x] **CLIP embeddings tell "two angles on one goal" from "two different
      shots"** — a question pHash cannot answer, because those two frames are not
      similar in any pixel sense. The vector is lifted out of the typed pgvector
      column the GPU lane already fills
- [x] **A story arc, not an ordering:** hook, setup, build, peak, reaction,
      ending. Three arc shapes (event, observational, showcase) and the policy
      picks one. Where the arc departs from shoot order the plan says so, rather
      than presenting a reorder as chronology
- [x] **Pacing is an explicit energy curve** — ramp, build, wave, steady, decay —
      turned into per-shot lengths by shape, scale, emphasis, clamp-and-
      redistribute, then beat quantisation. An even split is what this replaces,
      and a plan whose durations are all equal is a bug the tests catch
- [x] **Ten genre policies** — football, gaming, anime, cinematic travel, nature,
      vlog, social, product, fashion, automotive, plus neutral. A policy is a
      table of numbers and weights, so a new genre is data, not a code path
- [x] **`EditorialDecisionEngine` emits eleven typed decision kinds** — keep,
      reject, trim, reorder, emphasize, slow, speed up, place on beat,
      transition, add effect, add subtitle. Planning data with bounded numbers;
      there is no field in any of them that could hold a path, a storage key, a
      filter expression or an FFmpeg argument
- [x] **The model sets direction; the engine chooses the clips.** An LLM answers
      with a validated `EditorialIntent` — a policy name, an energy, an emphasis,
      a pacing shape — and never a media id. The prompt contains no per-clip row
      and no identifier at all, so there is nothing for a model to point at.
      One repair attempt, then the deterministic engine
- [x] **Explainable, not confessional.** Every kept clip carries a role and a
      short list of structured reason codes (`high_motion`, `unique_content`,
      `peak_moment`, …), and every rejected clip carries why. Concise structured
      reasons only — no hidden chain-of-thought is stored or exposed
- [x] **Three variants that are not shuffles.** High Energy, Cinematic and Social
      Fast Cut are policy modifiers, so they differ in shot count, shot length,
      selection, emphasis and treatment. Previewed four at a time through a route
      that **stores nothing** — affordable because the engine is deterministic,
      so picking one reproduces exactly what was shown
- [x] **Eight editorial metrics** with declared directions, so a client can
      render them without knowing which way is good
- [x] **Deterministic without a provider.** No key configured means the editorial
      engine runs alone and produces the same edit every time. A model failure
      degrades to it rather than to a failed request
- [x] **Both generations stay reachable.** `engine: "rules"` selects Phase 4's
      even split with Phase 5's directive planner in front of it, which is how
      "the new engine is not the old one carrying extra metadata" is a test
      rather than an assertion: on the same footage the rules engine returns one
      distinct duration and the editorial engine returns several
- [x] The editor: an Auto Edit workflow over the existing inspector — editorial
      plan preview, roles, pacing, selection reasons, variants, regenerate,
      accept, render. No new tab, no redesign
- [x] 1371 unit tests and 243 integration tests; `scripts/e2e_editorial.py`
      drives football, nature and gaming libraries through analysis, events,
      story, selection, pacing, decisions, effects and render to an MP4 verified
      with `ffprobe` and a full decode — and then compares the three, which must
      differ in shot count, shot length and pacing shape. They do: 5 shots at a
      4.0 s mean on a ramp, 3 at 6.7 s on a wave, 6 at 3.3 s on a build

Deliberately **not** in Phase 11: any new model or weights (the machine has
7.4 GB of RAM and a 4 GB card, and every signal is derived from analyses that
already exist); audio understanding beyond beats, so `dialogue` is inferred from
framing and channel count and is the weakest claim in the vocabulary; per-shot
colour grading; and learning from what the user accepts, which needs a feedback
store that does not exist yet.

**Phase 10 — the AI co-editor.**

- [x] **`EditDelta`** — sixteen operations over a closed vocabulary, every field
      an integer, a float, a boolean or a member of an enum the codebase already
      validates. No parameter dictionary, so there is nowhere to put a path, a
      storage key, a filter expression, an FFmpeg argument or a command; the
      parser does not filter them out, it has no field to read them into
- [x] **Clips are addressed by position in the plan the user was shown**, so
      `[remove 0, remove 1]` removes those two clips rather than one of them and
      its neighbour. A clip an earlier operation removed is reported as gone
- [x] **Patching is deterministic and all-or-nothing.** A failure anywhere —
      including in the same `validate_plan` gate every plan has passed since
      Phase 4 — returns the original plan object untouched
- [x] Two normalisations run afterwards and are **reported in the diff rather
      than applied quietly**: a dissolve that lost the clip it came from, and a
      transition the clips can no longer carry. Cues past a shortened end are
      clipped or dropped, and the count is said out loud
- [x] `CHANGE_STYLE_STRENGTH` and `CHANGE_BEAT_SYNC` record planning intent, and
      the diff labels them **"next plan"** — a cut edit cannot be restyled
      without re-cutting it, and claiming otherwise would be a lie
- [x] **The rules run first, and only when every clause resolves.** "Lower the
      music to 40%" never reaches a model; "make it feel like a trailer" always
      does. Half-understanding a two-part request and silently doing one half is
      the failure this rule exists to prevent
- [x] The resolver is conservative on purpose: no number means no amount
      invented, 60% is not snapped to a dial stop that does not exist, and a clip
      the edit does not have is not guessed at
- [x] **The model is shown shape, never identity** — clip count, durations,
      transitions, volumes, cue count. No media id, no filename, no project id,
      no path. The prompt is generated from the domain's own enums
- [x] **Versions are a tree and undo moves the head.** No plan is written, so an
      undo restores the exact bytes rather than a recomputation; editing after an
      undo branches instead of erasing. One head per project, enforced by a
      partial unique index
- [x] Generated, hand-cut and AI-changed plans all record versions, which is what
      lets manual and AI editing interleave on one history
- [x] Preview writes nothing and is produced by the same code that commits, so
      the diff cannot disagree with the result. Apply carries the version it was
      previewed against and 409s if the head moved
- [x] **Applying never renders.** A plan change must not queue an encode
- [x] The editor: the AI tab gained Create / Refine rather than the application
      gaining a seventh tab. A field, a before/after, an applied-changes list,
      undo, redo, render and recent requests — an editing tool, not a chatbot
- [x] 1184 unit tests and 217 integration tests; `scripts/e2e_coedit.py` verifies
      the whole path to an MP4 with `ffprobe` and a full decode, passing both
      with an AI provider configured and without one

Deliberately **not** in Phase 10: adding new footage through a delta (that is a
timeline action, and choosing new material is planning, which has a route);
per-cue subtitle styling; conversational memory across requests; and any
operation that could name a file, a filter or an encoder setting.

**Phase 9 — subtitles, transitions and core visual effects.**

- [x] **`TransitionKind`** grew from one member to four: cut, crossfade, fade in,
      fade to black. `consumes_time` is a property of the kind, so a fifth member
      cannot silently default to "free" and make a plan's duration a lie
- [x] **Crossfades shorten the programme**, and the plan, the compiled timeline,
      the render spec and the editor all subtract the same overlap. A test pins
      the three server numbers equal; `place` makes the browser agree
- [x] Transitions bounded at 80–4000 ms and never more than half of either clip
      they join; a crossfade on the first clip is refused, because there is
      nothing to fade from
- [x] **`Effect`** — a kind from a closed enum of seven and one bounded number.
      No parameter dictionary, because a free-form bag on a renderer instruction
      is a hole in the shape of an arbitrary filter argument
- [x] Zoom is `crop` + `scale`, not `zoompan`, which regenerates timestamps and
      turned a 4 s clip into a 1,024,000 ms one. Found by rendering, not reading
- [x] Speed via `setpts` and chained `atempo`, and `output_duration_ms` is
      speed-aware everywhere the programme is measured
- [x] **Subtitles are an ASS document**, not `drawtext`. The text lives in a data
      file libass parses as text; the only thing on the command line is a
      filename the server generated. `drawtext` would put user text inside a
      filter string where `:`, `\`, `'` and `%` are syntax
- [x] **Style is a preset id, never a parameter** — five looks, five positions,
      and no font, size, colour, outline, margin or coordinate accepted from a
      client. Sizes are given at 1080 lines and scaled to the real output height
- [x] Cues in output coordinates, 400 ms–10 s, 120 characters, 300 per plan,
      ordered and non-overlapping; text stripped to display characters
- [x] The planner's directive gained `transition` and `effects`. An invented
      kind is a violation carrying the real vocabulary; a number is clamped
- [x] **AI subtitles against a stored plan** — the model is told how long the
      edit is and where the cuts fall, and nothing else. A failure is a named
      state with no cues, never invented subtitle text. The route writes nothing
- [x] Reference style suggests a subtitle preset by id; the server's preset table
      still decides what that id means
- [x] The editor: transition and effect sections under the selected clip, a Text
      tab, crossfades drawn as the overlap they are, speed and effect badges, an
      effect-range bar, and a fourth lane for cues. 93 new message keys in
      English and Vietnamese (619, exact parity)
- [x] `/planner/capabilities` declares every transition, effect bound, preset and
      cue bound, asserted equal to the domain's own tables
- [x] 1018 unit tests and 190 integration tests, including real-FFmpeg renders
      that decode fully and a subtitle burn proven visible by frame comparison

Deliberately **not** in Phase 9: wipes, irises and other transitions that are
each a filter plus a timing rule; a keyframe editor; shaders, particles, 3D,
motion tracking or lip sync; per-cue styling; overlapping cues; arbitrary pixel
positions; and any path by which a client could name a font file.

**Phase 8 — reference video style intelligence.**

- [x] **`dynamics`** — motion energy and saturation, from a pair of frames at
      each existing sample point. CPU only, no model, no weights, no network
- [x] `BeatAnalyzer` widened to video, so a reference's cutting rhythm can be
      read against its own music. A file with no audio stream is `unsupported`,
      not a failure — which had broken every silent video until a real run
      caught it
- [x] **`ReferenceProfile`** — versioned, pure, derived on read from analysis
      rows. Every feature carries its own confidence and an unmeasured signal is
      absent rather than zero
- [x] **Beat-sync tendency** — the fraction of a reference's own cuts landing on
      its own beats, needing three cuts and a trusted grid before it will speak
- [x] **`StylePolicy.blend`** at 0/25/50/75/100, each measurement scaled by its
      own confidence; affinity taken out of the selection weights rather than
      added to them, and capped at a third
- [x] Zero is byte-identical to no reference, for every style, pinned by a test
- [x] **The reference is never footage** — excluded where candidates are built,
      so neither the rules engine nor a model can select it, and the model is
      given no handle for it
- [x] `projects.reference_media_id` with `ON DELETE SET NULL`; three
      project-scoped routes; the plan request has no field naming a reference
- [x] The model receives the profile as numbers, enums and nulls only, labelled
      as data — a profile whose every string field is an injection reaches the
      prompt as nothing
- [x] Reference section in the AI Edit tab: the project's videos offered
      directly, every measurement drawn with its confidence, unmeasured features
      saying so, and the dial disabled above 0% while the profile is unusable
- [x] 31 new message keys in English and Vietnamese (526, exact parity, none
      unreferenced), same tokens, themes, density and accent system
- [x] 830 unit tests and 150 integration tests; no change to any Phase 1–7
      contract

Deliberately **not** in Phase 8: transition-type detection (the detector reports
cuts, not dissolves), composition analysis beyond face presence, colour grading,
style transfer of any kind that would put the reference's own frames in the
output, and any new model or downloaded weights.

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
| 6     | 11–12 | Editing workstation — media browser, preview, timeline, manual editing |
| 7     | 13–14 | Music, deterministic beat detection, beat-synced cutting, audio mixing |
| 8     | 15    | Reference video style intelligence |
| 9     | 16    | Subtitles, transitions and core visual effects |
| **10** | 17   | AI co-editor — edit deltas, plan versions, undo/redo *(current)* |
| 11    | 18    | Evaluation loop, Asset Studio with license tracking, AWS deployment |

The reasoning layer moved forward. It was planned for Phase 7, and was brought
into Phase 5 because Phase 4 had already built the boundary it has to sit behind
— once `EditPlan` existed as a validated artefact, adding a second planner was a
smaller job than the original ordering assumed.

Phase 6 then took the manual-editing half of its former slot and left the music
half behind, which Phase 7 has now picked up. Subtitles, authentication and the
upscaling work move to Phase 8. Two features in one phase would have meant a timeline good enough to
demonstrate beat synchronisation and not much else; the editor was the half that
everything after it has to be built inside, so it got the whole phase. Music and
beat synchronisation move to Phase 7, which now has a timeline to hang them on.

Deliberately out of scope for these fifteen weeks: personal style learning, model
training infrastructure, AI asset generation, multi-tenancy and billing.

---

## License

Not yet licensed. All rights reserved.
