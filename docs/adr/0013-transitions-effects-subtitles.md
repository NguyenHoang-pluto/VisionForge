# ADR-0013: Transitions, effects and subtitles

**Status:** Accepted · 2026-09-16 (Phase 9)

Phase 9 adds three things a video editor is expected to have and this one did
not: a way to join two shots other than cutting, a small set of adjustments to a
shot, and text on the picture. Each of the three would be easy to add badly, and
each is added badly in the same way — by letting the client describe what the
renderer should do. The decisions below are mostly about not doing that.

The pipeline is unchanged. `EditPlan → Timeline → RenderSpec → argv[]` still
holds, `infra/ffmpeg/compiler.py` is still the only module that knows FFmpeg
syntax, and nothing added here introduces a second path to the renderer.

## Transitions are four members, and two of them change the arithmetic

`TransitionKind` grew from one member to four: `CUT`, `CROSSFADE`, `FADE_IN`,
`FADE_TO_BLACK`. A wipe, an iris and a page curl are each a filter, a parameter
set and a timing rule, and shipping twelve of them badly is worse than shipping
four that are exact.

They divide in two, and the division *is* the timing model:

| kind | behaviour | effect on duration |
| --- | --- | --- |
| `CUT` | instant | none |
| `CROSSFADE` | the tail of one clip and the head of the next play together | **shortens** by the overlap |
| `FADE_IN` | this clip fades up from black | none |
| `FADE_TO_BLACK` | this clip fades down to black at its end | none |

`TransitionKind.consumes_time` answers that as a property of the kind rather
than as a set each caller keeps. A fifth member added without answering it would
otherwise silently default to "no" and produce a plan whose reported duration is
wrong — which is the failure mode this feature has, and it is silent.

### A + B with a 500 ms crossfade is not A + B

The arithmetic, in one place, in `EditPlan.total_duration_ms`:

```
total = Σ segment.output_duration_ms  −  Σ overlap_ms (segments after the first)
```

and in `compile_timeline`, which places each clip at
`max(0, offset − overlap)`. The compiler joins clips pairwise with `xfade`,
whose output is `a + b − overlap` by construction, so the three agree because
they are the same subtraction written three times — and a unit test asserts
that `plan.total_duration_ms`, the last clip's `timeline_end_ms` and
`RenderSpec.duration_ms` are equal, so they cannot drift apart quietly.

### The bounds, and why a transition cannot be long

A transition must be at least `MIN_TRANSITION_MS` (80), at most
`MAX_TRANSITION_MS` (4000), and never more than `MAX_TRANSITION_SHARE` (half)
of *either* clip it joins. The share bound is the interesting one: a crossfade
longer than a clip does not dissolve, it deletes — the shorter shot never plays
alone. Half is where a dissolve stops reading as a join between two shots.

`CROSSFADE` on the first segment is rejected: there is nothing to fade *from*.
A directive that asks for one is compiled to `FADE_IN` instead, because that is
the nearest thing in the vocabulary to what was asked for and a 422 the user
cannot act on is worse.

### A timebase note, found by running FFmpeg

`concat` emits a 1/1000000 timebase and a filtered segment emits 1/fps. `xfade`
refuses inputs whose timebases differ. The failure appeared only when a cut
*preceded* a crossfade, which no unit test would have produced. Every join now
normalises with `settb=1/fps` first. This is recorded because it is the kind of
thing that is rediscovered expensively.

## Effects are a kind and a number

`Effect` is `(kind, amount, start_ms, end_ms)` and `EffectKind` is a closed enum
of seven. There is deliberately **no parameter dictionary**. A free-form bag on
a renderer instruction is a hole in the shape of an arbitrary filter argument,
and the moment one exists somebody puts a string in it.

| kind | range | neutral | spans |
| --- | --- | --- | --- |
| `zoom_in` / `zoom_out` | 0.0 – 0.3 | 0.0 | whole segment |
| `slow_motion` | 0.25 – 1.0 | 1.0 | whole segment |
| `speed_up` | 1.0 – 4.0 | 1.0 | whole segment |
| `brightness` | −1.0 – 1.0 | 0.0 | a window |
| `contrast` | 0.5 – 1.5 | 1.0 | a window |
| `saturation` | 0.0 – 2.0 | 1.0 | a window |

Only the colour effects accept a window, and the reason is structural rather
than a limitation of effort: a zoom ramp or a speed change over *part* of a clip
needs a second segment, and the plan has nowhere to put one. Offering the field
and ignoring it would be worse than not offering it.

An effect at its neutral value is dropped rather than stored. A plan is a
description of changes, and `brightness: 0` is an instruction to do nothing
dressed up as an edit.

### `fade` is not an effect

It is a transition. Having both would mean two vocabularies that produce the
same picture with different timing rules, and a plan where a `fade` effect and a
`FADE_IN` transition disagree about the first 500 ms.

### Zoom is `crop` + `scale`, not `zoompan`

`zoompan` regenerates presentation timestamps from its own frame counter. Used
on a trimmed segment it produced a 1,024,000 ms output from a 4,000 ms clip.
The replacement ramps a `crop` window with a `t`-based expression and scales
back up, which leaves timestamps alone. Discovered by rendering, not by reading.

### Speed changes duration, everywhere

`setpts=PTS/rate` for video, `atempo` for audio — chained, because `atempo`
accepts 0.5–2.0 and the vocabulary goes to 0.25 and 4.0. `output_duration_ms`
divides the trim by the rate, and every place that measures the programme uses
it: the plan total, the timeline placement, the budget walk in
`_fit_total_duration`, and the editor's `place`.

## Subtitles: text stays text

This is the first user-supplied **string** to reach the renderer. Everything
before it was a number or a member of an enum. Two decisions answer it.

### The text never enters a filter graph

The obvious implementation is `drawtext`, and it is the wrong one. The text
becomes part of a filter string, where `:` separates options, `\` escapes, `'`
quotes and `%` expands — so a subtitle reading `12:30` is a syntax error and a
subtitle reading `':drawbox=...` is something worse. Escaping that correctly is
possible and is the wrong shape of solution: it makes safety depend on a
function that has to stay right forever.

Instead the server writes an **ASS document** into the render's own scratch
directory and passes `subtitles=filename=<path it generated>`. The text lives in
a data file that libass parses as text. The only thing on the command line is a
filename nobody outside the server chose.

`clean_text` still strips `[\x00-\x1f\x7f{}\\]` and collapses whitespace — not
as an escape, but because those characters carry meaning to the *document*
format regardless of intent. A newline inside a dialogue line changes the
structure rather than the text.

### Style is a preset id, never a parameter

A client sends `cinematic`. It does not send a font, a size, a colour, an
outline width, a shadow, a margin or a coordinate — because a font path is a
filesystem path, and eight phases have gone into making sure those do not
arrive from outside. `STYLE_PRESETS` decides all of it server-side; presets name
a font *family*, which libass resolves through fontconfig and falls back on if
the machine lacks it.

Positions are the same: a closed set of five names, not pixels. Pixel positions
would have to be validated against an output geometry the client does not
choose, would break the moment an edit was re-rendered at another aspect ratio,
and are the field through which "put the text at −10000" arrives. Sizes are
given at 1080 lines and scaled to the real output height, so a preset looks the
same on a 720p preview and a 1080p master.

### Cues are in output coordinates

A cue belongs to the programme, not to the clip under it. This matters as soon
as a crossfade shortens the timeline: a cue pinned to a segment would move when
the edit was re-paced, and a viewer would see it drift off the line it was
written for.

Bounds: 400 ms to 10 s per cue, 120 characters, 300 cues, ordered, non
overlapping, inside the programme.

## The model may ask for these, in the same way it asks for clips

Phase 5 established the shape: a model returns a **directive** in a closed
vocabulary, deterministic code narrows it, and what survives is a plan that is
valid by construction. Phase 9 widened the vocabulary by two words and not one
character more.

- `transition` and `transition_ms` on a clip. An unknown name is a violation
  carrying the real list; a number is clamped.
- `effects`, an array of `{kind, amount}`. An unknown kind is a violation; an
  amount outside the bounds is clamped; nothing else in the object is read, so
  `{"kind": "brightness", "amount": 0.2, "filter": "drawbox=c=red"}` yields a
  brightness effect and the `filter` key is never seen again.

The asymmetry is deliberate. A made-up *kind* means the model has described a
capability this renderer does not have, and silently delivering something else
would be the pipeline claiming it did what it did not. A number out of range
means the model was wrong about a quantity, which is ordinary and clampable.

### Subtitles get their own call, because timing is not knowable in advance

Cue times depend on the compiled programme, so the route runs against a
**stored** plan: it compiles that plan's timeline and tells the model how long
the edit is and where the cuts fall. Nothing else — no media ids, no filenames,
no storage keys, no project identity, no text from a reference video.

What comes back is read as three keys per cue. A completion carrying a font, a
position, a filter or a path loses them at the parser, because the parser never
looks at a fourth key.

**Failure is a state, not a guess.** A provider that is down, a completion that
will not parse, or cues that all fall outside the edit produce `ok: false` with
a named reason and no cues. Writing plausible subtitles for footage nobody
transcribed would be putting words in a speaker's mouth — a worse outcome than
an empty panel with an explanation.

The route writes nothing. A model proposes, a person accepts, and the cues reach
the database only when the user submits a plan containing them.

### Reference style influences the *choice* of preset

`subtitle_style_for(EditStyle)` maps a cinematic edit to the cinematic preset, a
social edit to the social one. That is the whole of Phase 8's influence on
subtitles, and note what it maps *to*: an id. A measurement of somebody else's
video has no path by which it could become a font size here. The user's own
choice overrides the suggestion; the preset table overrides both, because
neither of them says what "cinematic" means.

## The editor draws what the compiler will render

The frontend had its own timeline arithmetic and Phase 9 made it wrong: summing
trims no longer gives the programme's length. `place` now makes the same two
adjustments the server makes — overlap for a crossfade, played length for a
speed change — and every position drawn comes from it.

A crossfade is drawn as the overlap it is, sized to its duration, rather than as
an icon on the join: the mark is the measurement. Fades are a gradient at the
edge of the clip they belong to, because they consume no time and moving
anything for them would be a lie. Subtitles get a fourth lane, positioned by
timeline time.

Every control is a slider over a closed vocabulary. There is no font field, no
colour picker, no x/y and no filter box in the UI, and their absence is the
feature — the panel *displays* what a preset resolves to, read from the server's
own table, so the choice is informed without being editable. `/planner/
capabilities` declares the transitions, effect bounds, presets and cue bounds,
so the browser's copy cannot drift from the validator that enforces them.

## Known limitations

- **One subtitle track, one style.** Mixed typography inside one edit is a
  design mistake more often than a decision. Per-cue styling is a vocabulary
  that can be added later without changing what is stored now.
- **No overlapping cues.** Two lines on screen at once is not something the
  single-track document expresses. A second track would be the way to do it.
- **Colour windows only.** Zoom and speed span a whole segment; see above.
- **One effect of each kind per segment**, at most four. Two brightness effects
  on one clip is either a sum nobody can see or a fight the renderer arbitrates.
- **Pairwise crossfades.** Long chains build a deep filter graph. It is correct
  and it is not free; a forty-clip edit with forty crossfades has not been
  profiled.
- **No transition between a clip and the music.** Audio crossfades follow the
  video crossfade with `acrossfade`; they are not independently timed.
- **Font availability is not verified.** Presets name families that exist nearly
  everywhere and libass falls back gracefully, but a machine with no serif font
  renders the cinematic preset in something else. The render succeeds; it looks
  different.
- **The model writes captions, not a transcript.** It is given the shape of the
  edit and the user's description. It has no audio, so AI subtitles are display
  text for the described edit and should never be presented as speech.
