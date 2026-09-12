# ADR-0008: The media intelligence layer

**Status:** Accepted · 2026-09-12 (Phase 3)

## Analyzers are versioned, and versions coexist

A result is identified by `(media_id, analyzer, analyzer_version)`. Re-running the
same version replaces its row; a new version writes a new one.

The alternative -- one row per `(media, analyzer)`, overwritten on each run --
makes a model upgrade silently destructive. When a new CLIP version scores worse,
you want to be able to see that it scored worse, and that requires the old
numbers still being there. The quality analyzer records the thresholds in force
alongside its scores for the same reason: a row stays interpretable after the
defaults are tuned.

## "Unsupported" is a success, not a gap

An audio file has no blur score. That is a fact about the medium, not a failure,
so it is persisted as `status=unsupported` with a reason. Two consequences: an
audio file never fails an analysis job, and the pipeline does not re-attempt an
analysis it has already determined is inapplicable.

## Proxy-first

Analysis reads the 720p proxy wherever one exists. Decoding a 1080p or 4K master
to compute a score that is then normalised to a 512px working size is wasted
decode, wasted memory, and wasted time -- on a 7.4 GB machine, meaningfully so.

Every result records `used_proxy`, because a score computed on a proxy is not
interchangeable with one computed on a master. The original is never opened by
the analysis lane and remains the source of truth for rendering.

## Frame sampling is deterministic, and avoids the edges

Videos are sampled at 5/25/50/75/95% of duration -- not 0% and 100%. The first
and last frames of real footage are routinely black, a fade, or a slate, and
sampling them would bias every quality score and embedding in the library.

Determinism is what makes a changed score attributable to the model rather than
to which frames happened to be picked.

## Normalisation before measurement

Every frame is scaled to a 512px long edge before scoring. Laplacian variance
scales with pixel count, so without a common working size the same photograph at
two resolutions produces two different sharpness scores and the stored numbers
are not comparable across a library.

## Model sizing is driven by the card, not by benchmarks

Measured on this RTX 3050 (4096 MiB total, ~3305 MiB reachable under WDDM):

| Model | Resident | Peak | Notes |
|---|---|---|---|
| OpenCLIP ViT-B/32 fp16 | 352 MiB | 584 MiB @ batch 8 | 512-dim |
| YuNet face detector | 0 (CPU) | 0 | 232 KB ONNX |

ViT-L/14 would roughly quadruple CLIP's footprint for a retrieval gain nothing in
VisionForge has yet shown it needs. On a 4 GiB card, a model that leaves room for
a second one is worth more than a marginally better benchmark.

The lease budget is **3000 MiB**, not 4096. WDDM holds several hundred MiB for
the desktop compositor, and PyTorch's caching allocator reserves beyond what it
reports allocated. Budgeting the full card is how you get an OOM at 95%
utilisation.

## Face detection: YuNet, not SCRFD

A deliberate deviation from the Phase 3 brief. YuNet is from the same family of
lightweight anchor-based detectors and ships in the OpenCV Zoo:

- **232 KB** versus SCRFD's multi-megabyte weights;
- **no new runtime** -- SCRFD in practice means `insightface` plus
  `onnxruntime-gpu`, and no onnxruntime-gpu build targets CUDA 13 yet, while
  OpenCV's DNN module already loads YuNet's ONNX graph;
- **runs on CPU in milliseconds**, leaving the whole card to CLIP.

On this hardware, spending VRAM on face detection would be the wrong trade. The
`Analyzer` port means a GPU SCRFD adapter later is a new file and a config
change, not a refactor.

**Detection only.** No recognition, no identity, no face embeddings. The payload
records `identity_stored: false` so that property is verifiable rather than
merely claimed.

## HNSW over IVFFlat

IVFFlat needs representative data present before its lists are meaningful, and
the `media_analysis` table starts empty. HNSW builds incrementally and is correct
from the first row. `vector_cosine_ops` because CLIP embeddings are compared by
cosine similarity, and the analyzer L2-normalises before storing so cosine and
inner product agree.

`m=16, ef_construction=64` are pgvector's defaults, kept deliberately: a larger
graph costs build memory for accuracy no one has yet measured a need for.

## PENDING -> RUNNING is a legal transition

Found by the Phase 3 acceptance run, which stranded a job with
`illegal job transition: pending -> running`.

`create_and_dispatch` commits the job row, publishes to the broker, and only then
writes `QUEUED`. A fast worker can claim a job inside that window. The row it
sees is committed and valid, so running it is correct.

Writing `QUEUED` *before* publishing would be worse: a publish that then failed
would leave a job marked `QUEUED` with no message behind it, and
`find_undispatched` -- which looks for `PENDING` -- could never recover it.
Keeping `PENDING` as "committed but not confirmed dispatched" is what makes that
recovery path work, so the state machine accommodates the race instead.
