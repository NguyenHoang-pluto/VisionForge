# ADR-0010: The LLM planner

**Status:** Accepted · 2026-09-12 (Phase 5)

## The model never sees, and never emits, a media id

This is the decision everything else follows from.

An `EditPlan` refers to media by UUID. Asking a model to produce one directly
would mean asking it to produce UUIDs — and a model that can produce a UUID can
produce the *wrong* UUID: one belonging to another project, one that was
deleted, one it invented. Every defence against that would then be a check
someone has to remember to write.

So the model does not produce a plan. It is given a **brief** in which every clip
carries a short handle — `c1`, `c2`, `c3` — assigned by us, and it answers with
an **EditDirective** that refers to those handles. Our code resolves handles back
to media ids through a table the model never sees.

```
selection → brief(c1..cN) → model → directive(c3, c1, c7) → resolve → EditPlan → the Phase 4 gate
```

The consequences are structural rather than procedural:

- A model **cannot** reference another project's media, because that media has no
  handle. This is not a check that could be skipped; the namespace does not
  contain it.
- An invented handle resolves to nothing and is rejected, with no fallback that
  guesses what was meant.
- The model cannot leak a filename, because it was never told one.

The last point is also why filenames are withheld. A file named
`ignore-previous-instructions-and-….mp4` is a prompt injection a user can plant
just by naming a file, and a media library is exactly where names arrive from
outside. Excluding them costs a little context about intent and removes the
entire class of attack.

## Three narrowings before the Phase 4 gate

The directive is not trusted arithmetic either. Compilation clamps every number
the model produced:

1. to the style's pacing bounds (a cinematic clip is 2.5–8 s);
2. to the plan's segment bounds (`MIN_SEGMENT_MS`…`MAX_SEGMENT_MS`);
3. to the source's actual length, so a trim cannot run past the end of a file.

The total is then clamped to `MAX_OUTPUT_MS` by dropping whole clips from the
tail rather than rescaling everything — a plan with two fewer clips is the same
edit, shorter, while a silently rescaled one is an edit nobody asked for.

Only then does the plan meet `validate_plan`, the same gate the rules engine
passes through, unchanged from Phase 4. A model asking for a nine-hour edit of a
four-second clip produces a short plan, not an error and not a nine-hour render.

## The compiler is total, and that is load-bearing

`compile_directive` never raises on an in-range directive. When a directive
clamps down to nothing legal — one clip at the 300 ms floor cannot reach the
1 000 ms output minimum — it returns *no segments* rather than something the
validator would reject. The planner reads an empty result as "the model produced
nothing usable" and falls back.

This came out of a test that asserted the wrong thing. It is worth recording
because the alternative design, raising, would have turned a recoverable
situation into a failed request.

## Validation runs inside the planner, not only in the service

Phase 4 validated in `EditService`. The LLM planner validates there *too*, but
also validates its own output before returning it — against media facts derived
from the same candidates the selection was built from.

The reason is the fallback. By the time the service sees an outcome, the decision
about which planner produced it has already been made and recorded. Validating
only in the service would mean an invalid model plan reaches a point where the
rules engine is no longer available, turning a recoverable failure into a 422.

## The fallback is the planner, not an error path

`FallbackPlanner(LlmPlanner, RulesEnginePlanner)` is what "AI planning" means
here. There is no configuration in which a provider failure becomes a failed
request, because the AI planner is never wired in bare.

Every fallback records its reason — provider unavailable, provider error,
invalid output, invalid plan, no usable media, unexpected error — into the plan
metadata, the `llm_runs` table and the UI. A fallback nobody can see is
indistinguishable from an AI planner that quietly does nothing, and that is the
failure mode this feature is most likely to have in production.

## Styles are numbers, so the fallback is not a downgrade

A style could have been a phrase in a prompt. Instead each is a `StyleProfile`:
pacing bounds, a default duration and aspect, an ordering preference, and a set
of selection weights.

Three things follow. The rules engine can honour a style with no model involved,
so choosing "Cinematic" with the AI disabled still produces a cinematic edit —
the user loses the interpretation of their sentence, not the style they picked.
The model is given the profile as bounds to work within, so its pacing is
checkable against a number rather than against taste. And a style change is a
reviewable diff rather than a reworded prompt whose effect nobody can predict.

The weights are not arbitrary. Animation has large flat regions, so Laplacian
variance under-reads good frames and the anime profile leans on contrast
instead; screen capture is already sharp, so the gaming profile weights
resolution; at one second a clip, softness is the first thing the eye catches,
so fast montage weights sharpness hardest.

## Automatic mode does not always mean AI

`resolve_mode` is a pure function, and its third rule is the interesting one:

- no provider configured → rules, because there is no choice to make;
- the user wrote a request → AI, because interpreting prose is the one thing the
  rules engine genuinely cannot do;
- **a style was picked but nothing was written → rules**, because a style is
  already a complete instruction and the profile behind it is deterministic.
  Paying a model to re-derive numbers we already wrote down would be slower,
  costlier and less predictable for no gain.

It is tempting to route everything through the model. "Cinematic, 30 seconds"
contains no ambiguity for a model to resolve.

When no provider is available and the user wrote something, automatic mode
matches a style by keyword. That is a lookup table, not language understanding,
and it is labelled as one: the UI reports "matched: cinematic" and the plan
metadata records the match rather than implying comprehension.

## Providers are HTTP, not SDKs

`LlmProvider` is text in, text out. No tools, no streaming, no conversation
state, no images — every one of those is a capability an attacker would rather
the model had, and none is needed. Tools are not merely disabled: the field is
not in the request body, so the API cannot return a tool call.

Two real providers ship (Anthropic Messages, OpenAI-compatible Chat Completions)
because a port with one implementation is a guess about what varies, and these
two actually differ: a separate `system` field versus a system message,
`max_tokens` versus `max_completion_tokens`, `input_tokens` versus
`prompt_tokens`. The Chat Completions shape is also what most gateways and
self-hosted runtimes speak, so a base-URL change points this at a local model
with no new code.

Raw HTTP rather than vendor SDKs: an SDK adds a dependency tree to a process
that already runs FFmpeg and PyTorch, and its retry and streaming surface is
exactly what this code deliberately does not use. Writing the call out also
makes it obvious what is sent, which matters when the claim being made is that
the model receives no files, no tools and no paths.

**API keys are read in `infra/llm/factory.py` and nowhere else.** They are never
returned by an endpoint, logged, placed in plan metadata, or written to a
database row. `/api/planner/capabilities` reports whether a provider is
configured plus its name and model — a model name is not a secret, and there is
no code path that serialises a key.

## The stub is a stub everywhere it appears

`StubProvider` answers prompts with arithmetic so the entire path — brief,
prompt, provider, parse, resolution, clamping, compilation, validation,
persistence — runs in CI and on a laptop with no key, no network and no cost.

It is named `stub` in configuration, in plan metadata, in `llm_runs`, and in the
UI, which says "deterministic stub, not a model" in plain words. Selecting it
when `environment == "production"` raises at construction rather than degrading
quietly: a test double that can reach real users is not a test double, and
telling someone an edit was AI-planned when it was arithmetic is a lie the
system should be incapable of.

It also reports **no token usage**, rather than a plausible-looking count that
would put fiction into the usage table.

## What is stored, and what is not

`llm_runs` holds provider, model, prompt version, status, attempts, latency,
token counts, the provider's request id, and the fallback reason. It is a
separate table rather than columns on `edit_plans` because the rows worth having
are the ones with no plan attached.

It does **not** hold the prompt, the completion, or the user's request text —
only a truncated SHA-256 digest of the request and its length. That is enough to
tell two runs apart, correlate a repeat, or match a support report to a row, and
not enough to reconstruct what somebody typed. Retaining user content to debug a
token count is not a trade worth making.

The one free-text field that *is* stored is the model's `rationale`, bounded at
400 characters and treated as display text: shown to the user, never parsed,
never matched against, never used to make a decision. It exists because an
automatic edit that cannot say why it cut the way it did is not reviewable.

## Planning stays synchronous, on a worker thread

Phase 4's `POST /edit-plan` returns 201 with the plan. An LLM plan takes seconds
rather than milliseconds, which raised the question of making planning a job.

It stays synchronous. Making it a job would add a queue round trip, a progress
bar and a polling client to something the user is already waiting on, and would
break a contract Phase 4 clients depend on. Instead `EditService` runs the
planner through `anyio.to_thread.run_sync`, so a slow plan costs one thread
rather than the event loop, and the provider's own timeout bounds it.

## Two bugs this phase found

**The capabilities endpoint read settings, not the provider.** It built a fresh
provider from configuration while the route planned with an injected one. Those
can differ, and a capabilities endpoint that disagrees with the planner is worse
than none: the UI would offer AI that does not run, or hide AI that does. It now
receives the same provider the planner will use.

**`request_digest` was computed but never persisted.** It was set on the run
record and omitted from `as_payload()`, which is what both the plan metadata and
the database row are built from — so the column was always null. Anything not in
that payload is simply never stored, which is a sharp enough edge to be worth
writing down.
