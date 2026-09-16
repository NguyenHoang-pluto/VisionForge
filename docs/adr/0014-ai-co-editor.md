# ADR-0014: The AI co-editor — edit deltas, plan versions, and undo

**Status:** Accepted · Phase 10
**Supersedes:** nothing. Extends [ADR-0009](0009-edit-plan-boundary.md) (the plan
boundary) and [ADR-0010](0010-llm-planner.md) (the model's role).

---

## Context

Through Phase 9, AI in VisionForge was a **one-shot generator**. A user described
an edit, a planner produced an `EditPlan`, and that was the end of the model's
involvement. Wanting a change meant re-planning: a fresh selection, fresh trims,
fresh everything — throwing away every hand edit and re-deciding everything the
user had already accepted.

That is the wrong shape for editing. Editing is iterative. "Lower the music",
"drop the third shot", "make the opening snappier" are changes *to an edit that
exists*, and an assistant that answers them by producing a different edit has not
understood the request.

Phase 10 makes AI a **co-editor**: it modifies the current plan.

The problem this creates is that a model now has an opinion about something the
user has already approved. Three things had to be true before that was safe:

1. what the model may say must be a closed vocabulary, as narrow as the plan
   schema itself;
2. applying what it says must be deterministic and all-or-nothing, so a bad
   proposal cannot leave a half-edited plan;
3. the user must be able to get their edit back, exactly, without trusting that
   an inverse operation composes.

---

## Decision

### 1. The model emits an `EditDelta`, not a plan and not a directive

Phase 5 established that a model may not emit an `EditPlan`, because a plan
carries UUIDs and a model that emits UUIDs can emit the wrong one. It emits an
`EditDirective` — handles and enums — which deterministic code compiles into a
plan.

A directive is the wrong shape for a *change*, because compiling one produces a
whole plan. So Phase 10 adds a third document:

```
EditDelta
  ├── operations   [ {kind: CHANGE_MUSIC_VOLUME, value: 0.4},
  │                  {kind: TRIM_SEGMENT, segment: 0, source_out_ms: 3200} ]
  ├── rationale    display text, never interpreted
  └── source       rules | llm | client
```

**Sixteen operation kinds, and no seventeenth arrives by accident.**

| Structure | Look | Text | Sound | Output |
|---|---|---|---|---|
| `REMOVE_SEGMENT` | `CHANGE_STYLE_STRENGTH` | `ADD_SUBTITLE` | `CHANGE_MUSIC_VOLUME` | `CHANGE_OUTPUT_PRESET` |
| `REORDER_SEGMENT` | `CHANGE_TRANSITION` | `MODIFY_SUBTITLE` | `CHANGE_MUSIC_FADE` | |
| `TRIM_SEGMENT` | `ADD_EFFECT` | `REMOVE_SUBTITLE` | `CHANGE_BEAT_SYNC` | |
| `CHANGE_DURATION` | `REMOVE_EFFECT` | | | |
| | `MODIFY_EFFECT` | | | |

Every operation is a frozen dataclass whose fields are integers, floats,
booleans, or members of an enum this codebase already validates. **There is no
parameter dictionary**, and the only free string in the whole vocabulary is
subtitle text — which goes through the same `clean_text` a hand-typed cue does
and ends up in an ASS document, never in a filter expression.

So there is nowhere in this schema to put a filesystem path, a storage key, an
FFmpeg argument, a filter string, a URL or a shell command. The parser does not
*reject* those things; it has no field to read them into. An operation carrying
`{"kind": "ADD_EFFECT", "effect": "brightness", "amount": 0.2, "filter":
"drawtext=…", "path": "/etc/passwd"}` yields a brightness effect, and the other
two keys are not read, not stored and not seen again.

### 2. Segments are addressed by position, against the plan the user was shown

A delta needs to name a clip. It does not name it by media id — a model never
sees one — but by **its index in the plan being edited**. This gives the Phase 5
handle property without needing a handle table: an index out of range is a
violation, and an index in range can only ever mean the user's own clip in the
user's own plan.

The subtlety is that indices must mean the same thing throughout one delta.
`[remove 0, remove 1]` must remove the first two clips, not the first and then
whatever slid into position 1. So every clip in the working copy carries the
index it had in the *original* plan, and operations resolve through that. A clip
an earlier operation removed is reported as gone, never silently re-pointed at
its neighbour.

### 3. Patching is deterministic and atomic

```
current EditPlan + validated EditDelta  ->  new EditPlan
                                        ->  or violations, and nothing else
```

Operations apply to a working copy. If any operation fails, or if the finished
plan fails the same `validate_plan` gate every plan has passed since Phase 4,
`apply_delta` returns the **original plan object** untouched. There is no code
path that writes a partially patched plan, because the patched plan does not
exist until every operation has succeeded.

Two normalisations run after the operations and before validation, and **both are
reported in the diff rather than applied quietly**:

- a first clip left carrying a crossfade — which has nothing to dissolve from —
  becomes a fade in;
- a transition the clips can no longer carry is shortened to what they can
  spare, or dropped to a cut.

Doing neither would mean rejecting "remove the first clip" because the second one
happened to arrive on a dissolve, which is a correct-but-useless editor. Doing it
silently would be the editor telling the user it did something it did not.

The same reasoning governs subtitles on a shortened edit: cues past the new end
are clipped or dropped, and the count is reported. Refusing instead would mean
"you cannot shorten a subtitled edit".

### 4. Two operations describe planning inputs, and say so

`CHANGE_STYLE_STRENGTH` and `CHANGE_BEAT_SYNC` are different from the other
fourteen. Reference strength decides *selection weights and pacing bounds*, and
beat sync decides *where cuts fall* — both are inputs to planning, and a plan
that has already been cut cannot be restyled or re-timed without re-cutting it,
which is exactly the regeneration this phase exists to avoid.

So they record the intent on the plan where the planner reads it, and the diff
labels them **"Reference style (next plan)"** and **"Beat sync (next plan)"**. A
user who asks for less reference influence and sees the cuts unchanged has been
told the truth about what happened.

### 5. The rules run first, and only when they understand the whole request

```
"lower the music to 40%"       -> rules  -> CHANGE_MUSIC_VOLUME 0.4
"use bold subtitles"           -> rules  -> MODIFY_SUBTITLE style=bold
"make it feel like a trailer"  -> LLM    -> parsed strictly, previewed first
```

A request naming a number and a thing has no ambiguity for a model to resolve.
Sending it to one would cost a network round trip and a dependency on a third
party in order to reproduce a lookup table — and would be *less* predictable,
since the same sentence could produce a different edit next week.

The safety property is that **the rules only fire when every clause resolves**. A
sentence is split on connectors and each clause must match a rule; if one does
not, the entire request goes to the model. Half-understanding "make the opening
faster and lower the music to 40%" and silently doing only the second half would
be the worst outcome available — the user sees a change, believes they were
understood, and does not notice what was dropped.

The rules are conservative by construction. "Lower the music" with no number does
not resolve, because this resolver does not invent amounts. "Set the reference
strength to 60%" does not resolve, because 60 is not a stop on the dial and
snapping it to 50 would be the editor quietly doing something else. A clip number
the edit does not have does not resolve. Each of those goes to the model, which
is shown the real clip count and the real bounds.

### 6. The model is shown shape, never identity

What reaches a model is a `PlanShape`: clip count, per-clip played duration,
transitions and effects by name, whether there is music and at what percentage,
how many subtitles there are, and the output settings. **No media id, no
filename, no storage key, no project id, no path.** The prompt is generated from
the domain's own enums, so it cannot advertise an operation the parser lacks or
omit one it has.

The user's words travel inside a delimited block. As in Phase 5, the delimiters
are the cheap half of the defence: the half that holds is that the only thing
read back out of the completion is a closed vocabulary of operations over
integers and enums, so the worst a successful injection achieves is a
differently-edited version of the user's own plan — which they then see in a diff
before it is applied.

### 7. Versions are a tree, and undo moves the head

```
v1  generated          the plan the planner produced
 └─ v2  manual         subtitles added on the timeline
     ├─ v3  co_edit    "lower the music to 40%"      <- undone
     └─ v4  co_edit    "use bold subtitles"          <- current
```

A version is a **pointer to a plan plus the story of how it got there**. Plan rows
stay append-only exactly as Phase 4 made them: patching writes a new `edit_plans`
row and a version pointing at it, so a render is always traceable to the
immutable plan it was built from.

**Undo writes no plan.** It moves `is_current` back to the parent. That makes
undoing instant, and — more importantly — it restores the *exact bytes* that were
there rather than a recomputed approximation. The alternative, applying an
inverse delta, requires every operation to have an inverse that composes, which
`CHANGE_DURATION` on a whole edit plainly does not.

Redo moves forward to the most recently created child. Editing while two versions
back adds a new child rather than erasing the abandoned branch — which is what
every editor does, and why `parent_version_id` makes this a tree rather than a
list. The abandoned branch stays in the history and stays reachable.

Exactly one head per project, enforced by a **partial unique index** on
`(project_id) WHERE is_current` rather than by whoever remembers to clear the old
one.

Generated and hand-cut plans record versions too. That is what lets manual and AI
editing interleave: the co-editor always patches whatever is current, whoever
last changed it.

### 8. Preview, then apply — and applying never renders

`POST …/co-edit/preview` runs the whole pipeline and throws everything away but
the diff. Nothing is written, so a preview the user cancels leaves no trace. The
same code produces the committed version, so the diff shown cannot disagree with
what applying it does.

Apply carries `base_version_id`. If the head has moved since the preview, the
change is refused with a 409: a change approved against version 3 must not land
on version 5, where the clip the user meant may be a different clip.

A change **does not trigger a render**. The user asks for that separately through
the route that already exists. A model that could queue FFmpeg jobs by being
asked nicely would be a cost bug at best.

---

## The validation chain

Every co-edit passes four gates before it is a plan, in this order:

| # | Gate | Rejects |
|---|---|---|
| 1 | **Schema** — `parse_operations` | unknown kinds, wrong types, out-of-range numbers, `true` where a number belongs |
| 2 | **Operation** — `apply_delta` | a clip that does not exist, a crossfade on the first clip, an effect that is not there to modify, music operations on a silent edit |
| 3 | **Ownership + facts** — `MediaFact` | a trim past the real source length, media belonging to another project |
| 4 | **Plan** — `validate_plan` | anything the Phase 4 gate has always rejected: durations, bounds, cue overlap, geometry |

A failure at any gate returns violations and writes nothing.

---

## Consequences

**Good.**

- An edit can be changed by asking, without losing the work already in it.
- Most changes never call a model, so they are free, instant and reproducible.
- A failed or unavailable model is harmless: the edit is exactly as it was, and
  the panel says which failure happened.
- Undo is exact and server-authoritative, so a second browser sees the same head.
- The security posture is unchanged from Phase 5: the model still cannot name a
  file, and now cannot name a clip outside the plan it was shown.

**Costs, accepted.**

- The deterministic resolver is English-only and pattern-based. It is not
  language understanding and does not claim to be; what it does not recognise
  goes to a model, or is refused when there is none.
- `CHANGE_STYLE_STRENGTH` and `CHANGE_BEAT_SYNC` record intent rather than
  re-cutting. Users who expect the cuts to move must regenerate.
- Branching keeps abandoned versions forever. Plans are small JSON documents and
  a project accumulates tens of them, not thousands; pruning can be added when
  something is actually large.
- There is no way to add a *new* clip through a delta. Adding footage is a
  timeline action, deliberately: the vocabulary of "which clip, trimmed where"
  belongs to the editor, and a model choosing new material from the library is
  planning, which already has a route.

---

## Alternatives considered

**Regenerate with the request appended to the brief.** The obvious approach, and
the one this phase exists to replace. It discards hand edits, re-decides accepted
choices, and gives the user no way to see what changed.

**Let the model return a whole `EditPlan` and diff it.** Every argument from
ADR-0009 applies, plus a new one: diffing two plans to find out what a model
meant is guesswork, where a delta says it.

**A JSON Patch / RFC 6902 document.** Generic, well specified, and exactly wrong
here: `{"op": "replace", "path": "/output/width", "value": 4096}` is a hole in
the shape of arbitrary state mutation. The whole value of a closed vocabulary is
that it cannot address fields the user is not allowed to choose.

**Client-side undo.** Rejected: the server owns the plans, and a local stack
would disagree with it the first time two tabs were open or a render referenced a
version the client had forgotten.

**Inverse deltas for undo.** Requires every operation to have a composable
inverse. `CHANGE_DURATION` scaling an edit, and the normalisations that refit
transitions, do not. Moving a pointer to an already-stored plan is exact.

---

## Verification

- `tests/unit/test_editdelta.py` — the vocabulary, and that paths, filters,
  commands and unknown kinds have nowhere to land.
- `tests/unit/test_plan_patch.py` — every operation, atomicity, addressing
  against the plan the user saw, and the diff.
- `tests/unit/test_coeditor.py` — the rules, the line between rules and model,
  strict parsing of model output, and every failure state.
- `tests/unit/test_coedit_route.py` — the API boundary.
- `tests/integration/test_coedit_lane.py` — real Postgres: versions, undo, redo,
  branching, the one-head index, cross-project refusal.
- `scripts/e2e_coedit.py` — the whole thing against a running stack, ending in an
  MP4 verified with `ffprobe` and a full decode.
