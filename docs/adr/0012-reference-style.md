# ADR-0012: Reference video style intelligence

**Status:** Accepted · 2026-09-16 (Phase 8)

Phase 8 lets a user point at a clip they like and say "cut mine like that". Four
decisions carry the design: what a reference is allowed to contribute, where the
numbers come from, how much they are allowed to matter, and what stops "style
transfer" from meaning "use that footage".

## A reference contributes numbers, never frames

The output must contain only the user's *other* media. That is the whole
security posture of the feature, and it is enforced structurally rather than by
convention: `without_reference` removes the nominated clip from the candidate
list where candidates are built, so neither the rules engine nor a model can
select what it was never handed, and the model is additionally never given a
handle for it.

This matters more than it looks. A professionally-cut reference outscores a
user's phone footage on every usability signal the ranker has -- sharpness,
exposure, contrast. A reference that was silently eligible would not merely be
*able* to appear in the edit; it would usually win.

Detaching the reference makes that clip ordinary footage again. The exclusion
follows from being *current*, not from a mark on the asset, because a user who
stops styling after a clip has not thereby forbidden themselves from using it.

## Every feature is measured, and carries its own confidence

`ReferenceProfile` is built from analyzer rows and nothing else:

| feature | source |
| --- | --- |
| shot length, quartiles, cut rate | `scenes` |
| luminance, contrast | `quality` |
| motion energy, saturation | `dynamics` (new) |
| tempo, beat-sync tendency | `beats` + `scenes` |

Three of those already existed. `dynamics` is new because nothing measured
movement or palette: `quality` samples the HSV *value* channel, not saturation,
and five frames spread across a whole video say nothing about motion. It decodes
a *pair* of frames 120 ms apart at each existing sample point and takes the mean
absolute difference. That is a local measurement repeated five times, not
optical flow, and it does not claim to be -- a whip pan and a cut inside the
window both read as high energy, which is right for "how energetic is this" and
wrong for "how fast is the camera moving".

**No model, no weights, no network.** One extra frame read per sample point.

**A signal that was not analysed is absent, not zero.** "Measured, and it was
still" and "nobody looked" are different claims, and the policy layer treats them
differently -- an absent feature pulls nothing, where a zero would pull hard
toward stillness. A single detected scene, which is how the scene analyzer
reports finding no cuts, yields no pacing at all: it is indistinguishable from a
genuine single take.

**Beat-sync tendency** is the fraction of the reference's own cuts that land
within a tempo-proportional window of its own beats -- the measurement behind
"this was cut to the music". It needs three cuts and a trusted grid before it
will say anything, because two cuts coincide with a beat by chance often enough
to be worthless. Finding it required widening the beat analyzer to video, which
in turn required admitting that a file with **no audio stream** is not a failure:
it gets an `unsupported` row, and the rest of the analysis lane is unaffected.

### What is deliberately not measured

Transition *type* -- the detector reports cuts, not dissolves. Composition
beyond face presence. Colour grading beyond luminance, contrast and saturation.
Each of these would need analysis that does not exist, and a profile field that
was always the same value would be worse than no field.

## The profile is derived on read, not stored

There is no `style_profiles` table. The profile is recomputed from the
reference's analysis rows every time it is asked for, which is safe only because
the derivation is deterministic -- and the determinism tests exist to keep it
that way.

The cost is a little arithmetic. What it buys: a profile that can never be stale
against a re-analysis, and a `PROFILE_VERSION` bump that re-derives every
project's profile with no migration.

## Strength is a dial, and zero is exactly the old behaviour

`StylePolicy.blend(preset, reference, strength)` at 0/25/50/75/100. The named
style preset supplies the baseline; the reference pulls it.

**Each measurement's own confidence multiplies the strength before it is
applied.** A shot length read from three cuts moves the pacing about a third as
far as one read from twenty. A user asking for 100% is asking for as much of the
reference as the reference actually supports, not for a confident answer to be
invented.

**Style affinity is taken out of the existing selection weights, not added on
top.** A score stays a convex combination, so a styled clip cannot outrank an
unstyled one by inflation, and the five usability components keep their relative
balance. Affinity is capped at a third: selection's first job is to reject
footage nobody can watch, and a style term that could outvote sharpness would let
a reference's palette put an out-of-focus shot in the edit.

**Zero changes nothing, for every style**, and a test plans the same footage
twice and compares segments, trims, totals and metadata to prove it. That
guarantee is what makes the feature safe to offer in the default tab.

The first implementation got the dial wrong in a way worth recording: it blended
the target pacing, recorded it on the plan, and then only ever *clamped* to the
policy's bounds. The reference therefore did nothing until a bound happened to
bite and then did everything at once -- 0% and 50% produced identical edits. A
strength control that is inert for half its travel is a switch wearing a dial's
clothes. `StylePolicy.pace` now interpolates the per-clip duration toward the
measured shot length by the same strength-times-confidence quantity, and the
clamps still run afterwards.

## The model gets the profile, and no vocabulary for the reference

The Phase 5 boundary is unchanged: handles, numbers and bounded enums, never a
uuid, a filename, a path or a storage key. The profile crosses it as the same
numeric payload the HTTP API returns, labelled as data and explicitly not as
instructions.

The argument that holds is structural rather than textual. Every field of the
profile is a number, a bounded enum or null, so there is nothing for an
instruction to travel in -- and a test builds a profile whose every string field
is an injection and shows none of it survives into the prompt.

**The rules engine honours a reference with no model involved**, which is what
keeps the fallback a fallback rather than a downgrade. When a provider fails, the
fallback produces the pacing the model was asked for, not a generic cut.

## Beat sync stays the user's decision

A reference that cuts to its own music is a reason to *offer* beat sync, never a
reason to switch it on. Phase 7 kept "add music" and "re-time the edit" as
separate decisions; a reference does not get to merge them behind the user's
back. `suggests_beat_sync` is advisory and the UI surfaces it as a sentence.
