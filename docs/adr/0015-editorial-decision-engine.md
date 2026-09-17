# ADR-0015: The editorial decision engine — signals, events, story, pacing

**Status:** Accepted · Phase 11
**Supersedes:** nothing. Extends [ADR-0008](0008-media-intelligence.md) (what the
analyzers measure), [ADR-0009](0009-edit-plan-boundary.md) (the plan boundary),
[ADR-0010](0010-llm-planner.md) (the model's role) and
[ADR-0012](0012-reference-style.md) (reference style as a dial).

---

## Context

Through Phase 10 the automatic editing path was, accurately described:

```
score every clip  ->  take the best N  ->  divide the duration evenly
                  ->  trim each from the centre  ->  concatenate
```

Every part of that is defensible on its own. The scoring rejects footage nobody
can watch. The even division is honest about not knowing better. The centre trim
avoids the focus hunt at the head of a handheld clip. Phases 5 through 10 added a
great deal around it — styles, references, music, transitions, subtitles, a
co-editor — but none of them changed the five lines above.

The result is that a generated edit reads as a **slideshow**. It has no opening,
no build and no ending; every shot is the same length; and when five clips show
the same moment from five angles, all five score well and all five are in it.

Three specific failures follow from the shape, not from the tuning:

1. **Ranking is not selection.** Taking the top *n* by a technical score cannot
   express "one of these five, not all five", and cannot express "this shot is
   soft and it is the only record of the goal".
2. **Even division is not pacing.** A real cut accelerates into its strongest
   moment and breathes afterwards. Clip length is the mechanism, and an even
   split throws the mechanism away.
3. **A list of clips is not a story.** Nothing in the plan said which shot was
   the opening or which was the point — so nothing downstream could emphasise
   one, the editor could not explain one, and the co-editor could not be asked
   about "the climax" because no such thing was recorded.

Phase 11 inserts an **editorial decision layer** between analysis and the plan.

---

## Decision

### 1. Five stages, each a pure function, each versioned

```
  analysis rows
      │
      ▼
  EditorialSignals   normalised 0..1 per clip          domain/editorial.py
      │              (+ SignalBoard: the same clips compared to each other)
      ▼
  DetectedEvent[]    a closed vocabulary, with confidence and evidence
      │
      ▼
  EditorialPolicy    genre table: arc, pacing, weights  domain/story.py
      │              (+ reference blend, + variant, + model intent)
      ▼
  PacingPlan         an energy curve -> per-slot lengths  domain/pacing.py
      │
      ▼
  EditorialPlan      roles, trims, treatment, reasons   domain/decisions.py
      │              (+ EditMetrics                      domain/metrics.py)
      ▼
  EditPlan           the unchanged Phase 4 document     domain/editorial_planner.py
      │
      ▼
  validate -> Timeline -> RenderSpec -> argv[]          all unchanged
```

`SIGNAL_VERSION`, `EVENT_VOCABULARY_VERSION`, `POLICY_VERSION` and
`DECISION_ENGINE_VERSION` are recorded on every plan, for the reason the
analyzers are versioned: an edit that looks different next month must be
attributable to a change somebody made rather than to drift.

### 2. Signals are derived; events carry confidence; unknown is an answer

`EditorialSignals` reduces one clip to comparable 0..1 axes — sharpness,
exposure, contrast, brightness, motion, motion variation, saturation, shot
density, face presence, face scale, style match — plus the selection score the
ranker already computed, carried rather than recomputed so the two layers cannot
disagree about which clip is sharper.

A signal whose analyzer did not run is `None`. Not zero, not the population
mean, not filled in from the style the user picked.

`EditorialEvent` is a closed vocabulary of twelve observations plus `UNKNOWN`:

| Framing | Movement | People | Structure |
|---|---|---|---|
| `establishing` | `action` | `subject_entry` | `transition_moment` |
| `close_up` | `peak_motion` | `subject_exit` | `ending` |
| | `impact` | `reaction` | |
| | | `celebration` | |
| | | `dialogue` | |

Each detection carries a **confidence** and a tuple of **evidence tokens**, and
the confidences are deliberately unequal:

- `close_up` rests on a measured face-box area and reaches 0.9;
- `peak_motion` is *relative to the project* — a percentile, not a threshold,
  because a nature project's busiest shot and a football project's are nothing
  alike and one threshold finds eleven peaks in one and none in the other;
- `dialogue` is capped at 0.45 and carries the evidence token
  `no_speech_detection`, because nothing in this product listens to audio. What
  is actually known is that faces are on screen, the camera is still and the
  file has an audio track. That is a hypothesis, and the cap is what stops the
  rest of the pipeline treating it as an observation.

A clip whose signals support nothing gets `UNKNOWN`, which is usable: it can
still be selected on quality and given a role, it simply contributes no
event-based argument for where it belongs.

### 3. One pipeline, many policies

A genre is a **table of numbers**, not a code path:

```python
EditorialPolicy(
    id=PolicyId.FOOTBALL,
    arc=EVENT_ARC,                 # hook, setup, build, peak, reaction, ending
    pacing=PacingShape.RAMP,
    min_clip_ms=1_000, target_clip_ms=2_200, max_clip_ms=6_000,
    weights=CreativeWeights(quality=0.22, energy_fit=0.18, diversity=0.28,
                            role_fit=0.24, style_match=0.04, relevance=0.04),
    min_diversity=0.12, prefer_chronological=True,
    transition_appetite=0.1, effect_appetite=0.5, slow_motion_peak=True,
    beat_sync_preference=0.5,
)
```

Eleven ship: football, gaming, anime, cinematic travel, nature, vlog, social,
product, fashion, automotive, neutral. They share **three arcs** between them —
event, observational, showcase — which is itself the evidence that the per-genre
differences live in pacing, holding and selection rather than in structure, and
therefore that a per-genre pipeline would have been nine copies of one algorithm.

Adding "wedding" later is an entry in a dict, not a module.

### 4. Selection is creative, and diversity can outvote quality

A clip competes for a **role**, against the clips already chosen, on six weighted
axes that sum to one:

| quality | energy fit | diversity | role fit | style match | relevance |
|---|---|---|---|---|---|

`quality` is below 0.5 in every shipped policy, and a test pins that. If quality
could outvote everything else combined, five excellent shots of one wall would
still become the edit.

Two mechanics make the difference concrete:

- **The diversity gate runs before scoring.** A clip whose cosine similarity to
  something already chosen is above the policy's floor is not "a slightly worse
  choice", it is a repeat, and it is refused. Similarity is measured on the CLIP
  vector, which answers a question pHash cannot: two angles on one goal are not
  near-duplicates by any pixel measure and are the same moment.
- **Slots fill in priority order, not narrative order.** The peak picks first,
  then the ending, then the hook; the setup picks last from what is left. Filling
  left to right gives the least important role first refusal on the best footage,
  which is exactly how a peak ends up being whatever the setup did not want.

### 5. Pacing is an explicit energy curve

```
PacingShape.RAMP   (0.15, 0.35, 0.60, 0.85, 1.00, 0.45)   slow -> peak -> release
PacingShape.BUILD  (0.45, 0.65, 0.85, 1.00)               no preamble
PacingShape.WAVE   (0.20, 0.55, 0.85, 0.55, 0.20)         breathing room at both ends
PacingShape.STEADY (0.50, 0.50, 0.50, 0.50)               a deliberate even rhythm
PacingShape.DECAY  (1.00, 0.75, 0.50, 0.30, 0.20)         strongest first
```

Clip length is *derived* from the curve — high energy means a short shot — and
then put through four more steps, in this order, each of which matters:

1. **scale** so the lengths sum to the target the user asked for;
2. **emphasis**, which moves time *between* shots rather than adding it, because
   the user stated a length;
3. **clamp and redistribute**, iteratively. Clamping one slot changes what the
   others should be, and without the redistribution the curve survives only
   until the first bound bites — which in practice means it survives on paper
   and not in any real project;
4. **quantise** to whole beats, measured as the difference of two *cumulative*
   positions. Rounding each length independently accumulates up to half a
   millisecond of error per cut; rounding the running total cannot.

### 6. Decisions are typed planning data, and carry their reasons

`EditorialDecision` has eleven kinds — `KEEP`, `REJECT`, `TRIM`, `REORDER`,
`EMPHASIZE`, `SLOW`, `SPEED_UP`, `PLACE_ON_BEAT`, `TRANSITION`, `ADD_EFFECT`,
`ADD_SUBTITLE` — and every field is an integer, a float, a media id or a member
of an enum this codebase already validates. **There is no options dictionary and
no free string**, which is the same structural guarantee `Effect` and `MusicCue`
give: there is nowhere here to put a filter expression, so one cannot arrive.

The engine has no idea what FFmpeg is. `domain/editorial_planner.py` compiles its
decisions into the existing `EditPlan`, and that plan goes through the same
`validate_plan` every plan has gone through since Phase 4.

Trimming is where the difference from a centre trim is most visible. Three
strategies, tried in order and each reported:

- **centred on the action**, for roles that want movement, when the dynamics
  analyzer recorded which of its five sample points was busiest. It names a
  fifth of the clip rather than a frame — coarse, and still the difference
  between cutting the goal and cutting the run-up to it;
- **avoiding an internal cut**, where the source contains a scene boundary that a
  window would otherwise straddle;
- **the centre**, which remains right for an unexamined clip.

Every decision carries a tuple of `ReasonCode` — short tokens from a closed enum,
rendered by the UI in the user's own language, asserted on exactly by tests, and
addressable by the co-editor. **No model chain-of-thought exists anywhere in this
path**, because no model produces these decisions.

### 7. The model sets direction, not clips

For automatic editing the model is no longer asked *which clips*. It is shown an
**aggregate** summary of the footage — counts, medians, an energy distribution
and a histogram of observed events — and returns an `EditorialIntent`: a policy
name, a curve shape, an emphasis role and six clamped numbers.

```
footage summary (no ids)  ->  model  ->  EditorialIntent  ->  EditorialPolicy'
                                                          ->  the same engine
```

This is a narrower boundary than Phase 5's, and a stronger one. There it was
given handles for the user's clips and told not to invent any; here **there is no
reference to any clip at all**, so there is nothing to leak, misname or be talked
into naming. A file called `ignore-previous-instructions.mp4` contributes a
duration to a median and nothing else.

Choosing clips is the one thing the deterministic engine does *better*: it has
the embeddings, the motion measurements and the face boxes, and it applies them
identically every time. What a model genuinely contributes is the judgement above
that — what kind of edit this should be, where it should peak, whether it should
breathe or drive.

`FallbackPlanner` is unchanged in shape and now wraps either model planner, so
there is still no configuration in which VisionForge cannot produce an edit.

### 8. Variants are policy modifiers, not shuffles

Three ship: **High energy**, **Cinematic**, **Social fast cut**. Each is applied
to the policy *before anything is decided*, so it changes which clips are chosen,
how many, how long each is held and what treatment the seams get.

That is the difference between offering real alternatives and offering a shuffle.
Shuffling is what a variant system degenerates into when it is bolted on after
selection: the clips are already chosen, so the only freedom left is their order,
and three orders of the same six clips is one edit wearing three hats.

A variant has deliberately **no duration multiplier**. It changes how the footage
is cut, never how long the result runs — the user asked for a length, and
silently returning a shorter video because they clicked "social" would be
answering a question they did not ask.

`POST /projects/{id}/edit-plan/variants` returns the base edit and all three,
**stores none of them**, and is affordable to offer because the engine is
deterministic: asking for the chosen variant through the ordinary plan route
reproduces exactly what was previewed.

### 9. Eight metrics, and no ninth

`content_diversity`, `repetition`, `pacing_consistency`, `energy_progression`,
`beat_alignment`, `style_adherence`, `story_completeness`, `quality`.

**They are deliberately not combined.** A single number would be the most
requested feature here and the least defensible one: "Edit quality: 0.72" cannot
be acted on, it hides a catastrophic beat alignment behind an excellent
diversity, and the weights that produced it would be an invented claim about how
much a repeated shot costs relative to an off-beat cut — a claim nobody can
justify and everybody would then tune against.

Each metric reports its direction (half are better low), its sample size, and
whether it could be measured at all. `beat_alignment` over an edit with no music
is *absent*, not zero: reporting zero would read as "every cut missed".

### 10. The co-editor gains a vocabulary, and bypasses nothing

A plan now records which segment is the peak, so "give the climax more time" has
something to refer to. The `PlanShape` the co-editor builds gains a `role` and an
`energy` per clip — a role name says what a clip is *for*, not which file it is,
so it carries no more information than "clip 3" does.

Three deterministic rules are added: hold a role longer, cut it shorter, and use
fewer shots. They resolve to a **documented step** (`EDITORIAL_STEP = 1.35`)
rather than an invented amount, and the rationale says a step was applied.

Every one of them emits operations from the **unchanged Phase 10 vocabulary**,
applied by the unchanged patcher, recorded as an unchanged version. There is no
editorial side-channel into a stored plan. A plan with no roles — hand-cut, or
made by the Phase 4 rules engine — is not guessed at: the rules decline and the
model is asked.

---

## Consequences

**Automatic editing is no longer describable as "pick N and concatenate".** The
count comes from the pacing rather than from `max_clips`; the lengths come from a
curve; the order comes from an arc; the trims come from where the action was
measured; and clips are declined for editorial reasons with the clip they repeat
named.

**Every edit can explain itself.** Roles, energies, beat relationships, reasons
and eight metrics are recorded on the plan and rendered in the inspector. The
`selection` the API returns now reports what the *edit* used rather than what was
merely usable.

**Nothing downstream changed.** `EditPlan`, `validate_plan`, `compile_timeline`,
the render worker, the version history and `EditDelta` are byte-for-byte the
interfaces they were. `RulesEnginePlanner` is untouched and remains the last
resort; `LlmPlanner` is untouched and remains reachable with `editorial=False`.

**Two analyzer outputs stopped being write-only.** Motion and saturation have
been written by the dynamics analyzer since Phase 8 and read only by reference
style; the CLIP vector has been in a pgvector column since Phase 3 and read only
by the similarity endpoint. Both are now load-bearing for every automatic edit.

**Costs.** The engine runs the selection loop up to five times per plan (arc
refitting, then a pacing pass with real source lengths), and the semantic
comparison is O(n²) — capped at `MAX_CONSIDERED = 60` clips, which is 1,770 dot
products over 512 floats and a few milliseconds. The variants route does that
four times. All of it is CPU arithmetic on data already in memory; no model runs
and nothing is decoded.

---

## Limitations

Stated plainly, because a system that oversells its perception is a system whose
outputs cannot be trusted.

- **Nothing here listens to audio.** `dialogue` is a framing hypothesis capped at
  0.45 confidence. There is no speech detection, no transcription and no voice
  activity measurement, so a `ADD_SUBTITLE` decision is a recommendation with no
  text in it.
- **`impact` is not a collision detector.** What is measured is that the
  inter-frame difference at one sample point is very unlike the others. A whip
  pan qualifies, and so does a cut the scene detector missed. The evidence token
  says `no_object_tracking`.
- **Motion is sampled at five points.** `motion_peak_ms` names a fifth of a clip,
  not a frame. It is enough to cut the goal rather than the run-up; it is not
  enough to cut on the strike.
- **`subject_entry` and `subject_exit` rest on five booleans** and are reported
  at 0.45 confidence with the evidence token `sparse_sampling`.
- **There is no recognition of anything.** No identity, no objects, no scene
  classification, no tracking. "Someone is on screen" and "the largest face box
  covers this fraction of the frame" are the strongest claims available, which is
  the boundary ADR-0008 drew and this phase does not move.
- **Chronology is the upload order.** There is no timecode reading and no
  multi-camera sync, so "the order it happened" means "the order the files
  arrived".
- **Semantic similarity is CLIP's opinion.** Two shots CLIP considers the same
  are declined as repeats even when a person would keep both, and the threshold
  is one number for every genre.
- **A policy is a guess about a genre, not about a project.** The numbers are
  defensible defaults chosen by reading how those genres are cut; they are not
  learned from anything, and no telemetry tunes them.
- **The acceptance libraries are synthetic, and narrower than they look.**
  `scripts/e2e_editorial.py` generates its footage with FFmpeg, and CLIP -- which
  keys on structure rather than colour -- resolves everything `lavfi` can draw
  into roughly four kinds of picture. A generated library therefore offers four
  or five *editorially distinct* shots however many files it holds, so those
  scenarios exercise the engine's reasoning honestly but understate how many
  shots it would find in real footage. Two consequences are worth knowing before
  reading a scenario's numbers: the diversity term does most of the rejecting
  there, and a target duration that happens to equal *shots x the policy's
  ceiling* pins every shot to that ceiling, where a correct pacing model and a
  naive even split are indistinguishable. The script picks its target to avoid
  that degenerate point and says so.

---

## Alternatives considered

**Keep ranking and add a diversity penalty to the score.** Rejected: a penalty is
still a ranking, and the case that matters — "one of these five, not all five" —
depends on what has *already been chosen*, which a per-clip score cannot express.

**Let the model choose clips and roles.** Rejected on both capability and safety.
The engine has the embeddings, the motion measurements and the face boxes and
applies them identically every run; a model has a summary. And a model choosing
clips needs handles for the user's media, which is the whole attack surface this
phase removed.

**A pipeline per genre.** Rejected. Nine of the ten would never be fixed when the
tenth was. The three shared arcs are the evidence that the differences are
parametric.

**One quality score.** Rejected; see §9. Eight numbers can each be argued with.

**Global optimisation over all clip-to-role assignments.** Rejected: exponential,
not more correct, and — worse — not explainable, because no clip's presence would
have a reason that did not depend on every other clip's.

**Store all four variants as plans.** Rejected: three plan rows and three
versions per click, two of which nobody looks at again. Determinism makes
previewing free.

---

## Verification

- **Unit** (`tests/unit/test_editorial_signals.py`, `test_story_and_pacing.py`,
  `test_decision_engine.py`, `test_editorial_planner.py`,
  `test_editorial_intent.py`, `test_coedit_editorial.py`): signal derivation,
  event confidence and evidence, arc expansion and degradation, policy tables,
  pacing curves and beat quantisation drift, creative selection, the diversity
  gate, trim placement, treatment, variants, metrics, intent parsing and
  clamping, prompt content, fallback behaviour, and the co-editor's new rules.
- **Integration** (`tests/integration/test_editorial_lane.py`): real Postgres,
  real pgvector embeddings, real analysis rows, the real router — the editorial
  document surviving JSONB, two policies producing different edits of one
  library, the variants route storing nothing, and both engines reachable: the
  Phase 4 rules engine returns one distinct shot length and the editorial engine
  returns several, over the same footage.
- **Acceptance** (`scripts/e2e_editorial.py`): three scenarios — football and
  nature over twelve generated clips each, gaming over sixteen — driven through
  ingest, analysis, editorial planning, variants and render to an MP4 verified
  with `ffprobe` and a full decode, with assertions that fail if the output is a
  concatenation. The run ends by comparing the three: they must differ in shot
  count, mean shot length and pacing shape, which is the claim the phase turns
  on.
