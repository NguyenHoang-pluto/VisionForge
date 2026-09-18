# ADR-0011: The audio track, beat detection and the mix

**Status:** Accepted · 2026-09-15 (Phase 7)

Phase 7 adds music to an editor that, until now, could only keep or discard the
sound its clips came with. Three decisions carry the design: what a music bed is
allowed to be, how a tempo is found, and where the two streams meet.

## A music cue is one optional field on the plan, not a new audio mode

`EditPlan` gains `music: MusicCue | None`. `OutputSpec.audio` keeps its old
meaning -- whether the *clips'* audio is kept -- and gains `source_gain`
alongside it.

The obvious alternative was to extend `AudioMode` to `none | source | music |
source_and_music`. It was rejected because those four values are two independent
questions wearing one name. "Keep the source audio" and "lay music under it" are
decided separately, by different controls, at different points in an edit; fusing
them means every future combination multiplies the enum, and it means a plan
written before Phase 7 has to be *migrated* rather than merely read.

As an optional field, a Phase 4-6 plan deserialises to `music=None` and
`source_gain=1.0` and re-renders to the same bytes. That property is asserted in
the tests, not assumed.

**One cue, not a list.** A montage with two beds and a crossfade between them is
a real edit, and it is a *different* edit: it needs overlap rules, relative
ordering and a mix policy, none of which the video track has either. Phase 7
ships the case a highlight reel actually has and leaves the vocabulary
extendable rather than pre-emptively general.

## The cue vocabulary is closed, for the same reason the segment vocabulary is

A `MusicCue` is a media id and six numbers: a trim within the track, a position
on the output timeline, a linear gain and two fade durations. There is no
filename, no filter string, no codec and no place to put an FFmpeg argument.

This matters more than it does for video, not less. An audio filter graph runs
commands exactly as readily as a video one, and the audio path is the newer and
less-travelled of the two. So it gets the same structural guarantee and the same
tests: the request schema's field set is pinned, every field but the id is
asserted numeric, and the generated filter graph is checked against an
allow-list of every word it may contain.

The cue is validated twice, deliberately. Pydantic rejects what is wrong on its
face and returns a field path; `validate_plan` then checks it against the real
media rows and remains the authority. A value that slipped past the first is
still refused before anything is stored.

## `MediaFact.from_media` is the single mapping

Both the planning service and the render worker turn a media row into the facts
the validator may see. Before Phase 7 each wrote the rule inline. That was
survivable with one derived flag (`is_renderable`) and a guarantee of divergence
with two -- and those two callers sit on *opposite sides of the gate*, so a
divergence means a plan that validates when created and fails when rendered.

A cue may name a ready **audio asset** in its own project. Not a video with a
soundtrack: that is a different feature, and refusing it is better than
half-implementing it.

## Beat detection is deterministic arithmetic, and it is allowed to be wrong

Detection is spectral flux, autocorrelation and an exhaustive phase search, in
numpy, with no model, no network and no music API. Same file in, same beats out,
forever -- which is what makes a beat-synced edit reproducible and a change in
the cut attributable to a parameter rather than to chance.

### Assumptions

- **One steady tempo per track.** The grid is a single period and phase,
  extrapolated across the file.
- **A pulse exists and is percussive.** Spectral flux measures energy *arriving*
  across many bins at once. That is what a drum hit is.
- **60-200 BPM.** Below 60 a beat is longer than most clips in a highlight cut;
  above 200 the grid is finer than the minimum segment length.
- **Only the first five minutes are analysed.** Beyond that the grid is
  extrapolated, which is cheaper and no less accurate for music that holds a
  tempo -- and no more wrong for music that does not, since the estimate was
  already wrong.

### Known limitations

- **No tempo changes.** A track that accelerates, or has a half-time section,
  gets one average answer. Cuts in the other section will drift.
- **No downbeats.** Beat 1 is indistinguishable from beat 3. Cuts land on beats,
  not on bars.
- **Rubato and swing are not modelled.** Expressive timing reads as low
  confidence, which is the correct outcome: the grid is then not used.
- **Precision is one STFT hop, ~23 ms.** Finer would cost time for accuracy no
  cut uses.
- **A tempo prior biases towards 120 BPM.** Autocorrelation cannot distinguish a
  tempo from half of it, and for a period that is not a whole number of frames
  the doubled lag often correlates *better* -- at 120 BPM the period is 21.5
  frames, so lag 43 lands on an integer while the fundamental straddles one.
  Something must break the tie. The prior is stated in the payload rather than
  hidden, and a genuine 60 BPM track needs its fundamental to out-correlate the
  120 BPM lag by about a third, which a real half-time groove does.

### Confidence, and why it has two independent parts

Confidence is periodicity multiplied by grid agreement, and both must hold.
Autocorrelation alone calls an idling engine 128 BPM, because its hum really is
periodic. Coincidence alone is satisfied by any grid fine enough to hit
everything. Silence, white noise and a sustained tone all score below the
threshold, and the planner then declines to move a single cut.

Below `MIN_BEAT_CONFIDENCE` (0.35) the grid is treated as absent. "Absent",
"never analysed" and "not requested" are deliberately indistinguishable
downstream: all three mean *plan the way Phase 4 planned*. A spoken-word bed
must not silently re-time an edit.

## Beat sync quantises duration, not position

The load-bearing observation: **the timeline is butt-joined.** Every clip starts
where the last one ended. So if each clip is a whole number of beats long and the
first starts on a beat, *every* cut lands on a beat -- with no per-cut search, no
drift to accumulate, and no way to express a gap or an overlap, because the
structure that would hold one does not exist.

Two corrections were needed to make that true in practice, and both are worth
recording because both looked like they worked:

- A clip's length is the **difference of two cumulative positions**,
  `span_for_beats(placed + n) - span_for_beats(placed)`, never `round(n *
  period)`. A beat period is rarely a whole number of milliseconds (128 BPM is
  468.75), so rounding each clip separately and adding compounds the error at up
  to half a millisecond per cut -- past tolerance by the fourth cut and 2 ms out
  by the eighth. Cumulative rounding never exceeds half a millisecond, however
  many cuts precede it.
- A source too short for its allocation is given **fewer whole beats**, not a
  truncated one. Truncating is right when nothing is quantised, and is exactly
  what pushes a clip and everything after it off the grid when something is.

The plan records the integer beat count it chose. Deriving it back out of a
rounded duration is what made an early version claim "10.999 beats per clip"; a
plan whose own account of itself cannot be checked is not reviewable.

## The mix happens after the concat, which required splitting it

Phase 4 concatenated video and audio in one `concat=n=N:v=1:a=1[vout][aout]`.
That cannot survive a music bed: a single filter emitting both streams leaves
nowhere to put a mix. Video and source audio are now concatenated separately.
This changes no output when there is no music -- a test asserts the video chains
are identical with and without audio -- and makes the mix expressible when there
is.

The music chain's order is not arbitrary and each step earns its place:

```
[N:a] atrim -> asetpts -> aformat -> volume -> afade in -> afade out -> adelay
```

- `atrim` then `asetpts`: take the passage and rebase it to zero, because every
  filter below measures from the start of *this* stream, not of the file.
- `aformat` early: make it mixable before anything else touches it.
- `volume` before the fades, so a fade ramps to the level the user chose rather
  than to unity and then gets scaled.
- `adelay` last, with `all=1`: it prepends silence, so every offset above would
  otherwise have to account for it -- and without `all=1` only one channel is
  delayed and the stereo image tears in half.

`amix` runs with `normalize=0`. With normalisation on, amix divides by the input
count, so adding a bed would silently halve the dialogue by a gain nobody set
and nobody can see in the plan.

The finished audio is `apad` then `atrim` to the video's exact length. `apad`
covers a bed shorter than the picture, `atrim` cuts one longer than it, and
together they make the audio exactly as long as the video whichever was longer.
`-shortest` was rejected: it decides by whichever stream happens to end first,
which is the right answer only by luck.

## Everything stays local

The music is a project-owned audio asset that the user uploaded. There is no
download, no provider, no catalogue and no network call anywhere in the audio
path. The test fixtures and the acceptance script *generate* their music with
FFmpeg -- a synthesised metronome at a known tempo -- so nothing copyrighted
enters the repository, the test run, or the wire.
