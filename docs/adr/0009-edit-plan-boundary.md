# ADR-0009: The EditPlan boundary

**Status:** Accepted · 2026-09-12 (Phase 4)

## The plan is a structural boundary, not a convention

Everything upstream of rendering decides *what* the video should be; everything
downstream decides *how* to produce it. `EditPlan` is where those two halves
meet, and it is deliberately the narrowest thing that can carry the decision: a
list of segments, each holding a media id, two millisecond offsets, an integer
order, and one enum for the transition. Output is an aspect ratio, width, height,
fps, a fit mode and an audio mode -- all enums or bounded integers.

There is no string field. Not because a planner would be careless with one, but
because a field that *can* hold a path or a flag will eventually hold one, and at
that point the only thing standing between a planner and `subprocess` is
review discipline. Removing the field removes the class of bug.

This matters most for the planner that does not exist yet. Phase 4 ships
`RulesEnginePlanner`, which is a deterministic scorer -- it could not emit a shell
command if it tried. The boundary is built for its replacement: an LLM planner
that produces plausible-looking text on demand and has no notion of what is
dangerous. Both implement the same `Planner` protocol and both are constrained by
the same vocabulary, so swapping them changes which plans get made and nothing
about what a plan is allowed to say.

## Validation runs twice, on purpose

A plan is validated when it is created and again inside the render worker,
immediately before compilation. This is not belt-and-braces; the two checks are
answering different questions at different times.

Creation-time validation asks whether the plan is internally coherent. Render-time
validation asks whether it still agrees with the world: the media may have been
deleted, replaced, or shortened in the minutes between planning and rendering, and
a segment trimming to 6 000 ms of a clip that is now 2 000 ms long is an FFmpeg
error at best. The render worker re-reads the media rows and checks every
reference against them, including project ownership -- a plan cannot reach into
another project even if a stored row says it should.

## An invalid plan fails permanently

`PlanInvalidError` extends `PermanentError`, and this was a bug before it was a
decision: it originally extended `ValueError`, so the job runner's generic
handler treated it as transient and burned the full retry budget re-validating a
plan that referenced deleted media. Three attempts with backoff, several minutes,
and the same answer each time.

A plan is a fixed document and so is its disagreement with the database. Fixing it
means planning again, which produces a *different* plan; this one never becomes
valid. The failure is reported immediately, with every violation rather than just
the first, so one round trip tells the user everything wrong with the plan.

## Two compilation stages, neither of which knows the other's domain

`EditPlan → Timeline → RenderSpec → argv`.

`Timeline` resolves the plan into clips laid butt-joined on tracks with absolute
start and end times. It contains no FFmpeg concept whatsoever -- no filters, no
codecs, no flags -- and is the structure a future manual editor will manipulate
directly.

`RenderSpec` is the opposite: codec, CRF, preset, pixel format, container, and
the resolved local input paths. It knows nothing about editorial intent.

Only `infra/ffmpeg/compiler.py` knows FFmpeg syntax, and it produces an argv
list -- never a string. No `shell=True`, no concatenation, no interpolation of
anything a user supplied. A filename containing a space, a quote or a semicolon is
one element of a list and cannot become two.

The split earns its keep in tests: timeline logic is checked without invoking
FFmpeg, and the filter graph is checked by asserting on the argv list without
encoding anything. The integration suite then renders for real and verifies the
output with `ffprobe`, because an argv list that looks right and an MP4 that plays
are different claims.

## Selection reports why, and the reason has to be the useful one

Rejected media is returned with a `RejectionReason` and a human-readable detail.
A clip that vanishes from the edit with no explanation is indistinguishable from a
bug, and a library owner needs to know whether to re-shoot or re-encode.

The usability gates run in a specific order -- exposure, then contrast, then
sharpness -- and the order is the point. A frame crushed to black has almost no
edge energy, so its Laplacian variance is near zero and a sharpness-first check
reports "out of focus" for a frame whose actual problem is that it is black.
Sharpness is only a meaningful measurement on a frame that has detail to measure,
so the checks that establish that run first. Both orderings reject the same
clips; only one of them explains itself.

## Output dimensions come from a closed server-side map

The API accepts an aspect ratio, not a width and height. `16:9`, `9:16` and `1:1`
map to fixed dimensions on the server. Requests carry intent -- how long, how many
clips, what shape, what order -- and never geometry, so no request can ask for a
30 000-pixel canvas or a 1 fps 4K encode that pins the machine.

## Rendering is one job at a time

The render worker runs `--pool=solo`. FFmpeg already saturates the cores it is
given, so a second concurrent encode buys nothing and costs contention on the same
scratch disk and the same limited memory. The render timeout is 1 800 s, generous
enough for a long timeline on a laptop and short enough that a wedged process is
eventually reaped rather than holding the queue forever.

Progress is read from `-progress pipe:1` and converted to a fraction of the
planned duration. It is measured, not estimated -- the same principle as the
ingest pipeline's per-step rows.
