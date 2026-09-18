# ADR-0016: Stills in the edit, and templates

**Status:** Accepted · Phase 12
**Extends:** [ADR-0009](0009-edit-plan-boundary.md) (the plan boundary),
[ADR-0012](0012-reference-style.md) (reference style) and
[ADR-0015](0015-editorial-decision-engine.md) (the editorial engine).

---

## Context

Two gaps, both visible the moment someone drops a real folder into a project.

1. **Photos were uploaded, analysed and then ignored.** Ingest accepted images
   and the quality, pHash and dynamics analyzers ran on them. But
   `MediaFact.is_renderable` was `kind is VIDEO`, so no planner could place one
   and the validator refused one placed by hand. A trip folder is photos *and*
   videos, and the output contained only the videos.
2. **"Cut it like this" only copied a mood.** The Phase 8 reference measures
   pacing, colour and beat tendency, then blends them into a policy. It does not
   copy an edit's *structure*: how many shots, how long each is, where the peak
   falls, how the shots join. It also belongs to one project, so an edit someone
   likes cannot be reused anywhere else.

A defect found while doing this is fixed here too. Phase 12's resolution presets
(up to 2160p) rendered from the 720-line proxy, so a 4K export was an upscale of
720p. The render worker now reads the original whenever the output's short side
is larger than the proxy's.

## Decision

### Stills are holds, not trims

A photo has no length. A segment of one starts at zero and runs for as long as
the edit wants, up to `MAX_STILL_MS` (8 s). It may not change speed. The
validator enforces all three (`still_not_from_zero`, `still_too_long`,
`still_speed`) and ignores the one-frame "duration" ffprobe reports for a JPEG.

`domain/stills.py` is the single place that answers "how much may a planner take
from this source, and from where". The rules engine, the editorial engine and
the directive compiler all ask it, so they cannot disagree.

A still is given motion by default: a slow zoom or pan, cycling direction so
that neighbouring photos do not all move the same way, and vertical for a
portrait photo. A photo held motionless for three seconds reads as a stalled
render, so this happens with or without the policy's treatments. There are four
new effects (`pan_left/right/up/down`). They are the zoom's `crop` technique with
the window sliding instead of tightening, for the same reason zoom avoids
`zoompan`: timestamps.

At render time a still is an input with `-loop 1 -framerate <fps> -t <longest
hold>`, so `trim` cuts it like any other stream. Where source audio is kept, a
still contributes `anullsrc` silence of its own length, which keeps the clips'
audio in step across the join. The worker, not the plan, knows which media are
stills: it reads that from the media rows it already loads.

### A template is an edit with the footage taken out

`EditTemplate` is a row of `TemplateSlot`s. Each slot has a length, an energy,
a story role, how it is entered, how a photo placed in it moves, a media
preference (any, video or still) and, optionally, a length in beats. A template
is validated against the plan validator's own bounds, so a valid template cannot
produce a plan that is refused for structural reasons.

There are two sources, and one type:

- **Library:** six templates declared as data in `domain/template_library.py`,
  each counted in beats at 120 BPM.
- **Measured:** `domain/template_extract.py` builds a template from the
  analysis every video already has. Each detected shot becomes a slot. The median
  motion inside a shot becomes its energy. A shot that spans a whole number of
  trusted beats records that count. Roles follow position and energy. Shots too
  short to be clips are folded into a neighbour, shots too long are split, and a
  video with more shots than a plan may hold is truncated. The extraction report
  says how many of each.

**Measured templates are owned by the user, not the project** (`edit_templates`,
migration 0008). Reuse in another project is the whole point, and a template
tied to its project could not offer it. The video a template was measured from
is referenced `ON DELETE SET NULL`: deleting the video leaves the template.

### Filling a template is the editorial engine, given different slots

The engine already fills a row of pacing slots with clips. Filling a template
hands it the template's row instead of the one it would compute:

- the pacing plan is the template's slots, re-timed to the music's beats when a
  trusted grid is present and every slot has a beat count;
- the roles are the template's, one per slot;
- selection scores each slot against *its* energy, and multiplies in a media
  preference (a photo in a video slot keeps 0.55 of its score, a video in a
  photo slot 0.8). This is a factor rather than a seventh weighted axis because
  the six weights are the policy's and sum to one;
- every slot is filled. With more slots than clips, clips are reused, each reuse
  costing 0.7 of the score, and the plan says `reused_for_template`;
- joins come from the template, re-checked against the real neighbours with the
  validator's own half-a-clip rule, so a dissolve that no longer fits becomes a
  cut rather than a rejected plan.

None of the arc re-fitting applies, because the point of a template is that its
structure is not renegotiated against the footage. The request names a template
by id and the server resolves it; a user's template that is not theirs is a 404.
A templated edit is stored, versioned, patched and rendered exactly like any
other.

## What is deliberately not done

- **Joins in a measured template are all cuts.** Scene detection finds cuts.
  Telling a dissolve from a cut, or reading a zoom added in the edit, needs
  analysis this build does not have, and guessing would make templates lie
  about what they copied. The UI says so.
- **No music ships with a template.** Licensing, and the user's own track
  re-times the template anyway.
- **No text or titles in templates.** A later vocabulary, not a field today.
- **No 8K or 16K.** On the reference machine (8 GB RAM, RTX 3050) neither x264
  nor NVENC could allocate an 8K encoder, and 16K is past what H.264 describes.
  Resolutions stop at 2160p.

## Consequences

- `SIGNAL_VERSION` and `DECISION_ENGINE_VERSION` are 2, and `PROMPT_VERSION` is 3.
  Plans made before this phase are unchanged. Nothing in the untemplated path
  moves except for projects that contain photos.
- A render of a 1080p or larger output decodes originals rather than proxies,
  which is slower and uses more memory. Encoder threads are capped above 1440p,
  and the render timeout scales with pixel rate up to six hours.
- The program preview shows a still as its thumbnail, timed by the wall clock,
  because there is no proxy to play.
