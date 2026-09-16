/**
 * The editor's timeline, and its relationship to the server's EditPlan.
 *
 * The rule this module exists to enforce: **the frontend does not have its own
 * rendering model.** A timeline here is a sequence of cuts into project media,
 * which is exactly what `EditPlan.segments` is. Dragging a trim handle changes
 * one number in one cut; it does not produce a description of a render.
 *
 *   EditPlan (server) --toDraft--> TimelineDraft (client, editable)
 *   TimelineDraft --toManualCuts--> POST /edit-plan/manual --> EditPlan (server)
 *
 * The round trip is lossless for everything Phase 6 can edit, and the server
 * re-validates on the way back in. A draft is a *proposal*; nothing renders
 * until the server has accepted it as a plan.
 */

import {
  CUE_BOUNDS,
  EFFECT_BOUNDS,
  MAX_TRANSITION_SHARE,
  MIN_CUE_MS,
  OVERLAPPING,
  TRANSITION_BOUNDS,
  type EffectKind,
  type EffectRequest,
  type PlanSubtitles,
  type SubtitlePosition,
  type SubtitleStyle,
  type SubtitleTrackRequest,
  type TransitionKind,
} from "@/lib/api";
import type {
  AspectRatio,
  EditPlan,
  ManualCut,
  MediaAsset,
  MusicRequest,
  QualityPreset,
} from "@/lib/api";

/**
 * The bounds a drag is clamped against.
 *
 * These are *defaults*, used before `/planner/capabilities` has answered and if
 * it ever cannot. The live values come from the server through `applyBounds`,
 * because the server is what actually enforces them -- a copy in the browser
 * that drifts would mean the editor permits edits the validator rejects, and
 * the user would find out at render time.
 */
export let MIN_CLIP_MS = 300;
export let MAX_CLIP_MS = 30_000;
export let MAX_CLIPS = 40;
export let MIN_TIMELINE_MS = 1_000;
export let MAX_TIMELINE_MS = 10 * 60 * 1000;

/** Audio bounds, same story: defaults until `/planner/capabilities` answers. */
export let MIN_GAIN = 0;
export let MAX_GAIN = 2;
export let MIN_MUSIC_MS = 500;
export let MAX_FADE_MS = 30_000;

/** Adopt the bounds the server declared. Called once, when capabilities load. */
export function applyBounds(bounds: {
  min_clip_ms: number;
  max_clip_ms: number;
  max_clips: number;
  min_total_ms: number;
  max_total_ms: number;
}): void {
  MIN_CLIP_MS = bounds.min_clip_ms;
  MAX_CLIP_MS = bounds.max_clip_ms;
  MAX_CLIPS = bounds.max_clips;
  MIN_TIMELINE_MS = bounds.min_total_ms;
  MAX_TIMELINE_MS = bounds.max_total_ms;
}

/** Adopt the audio bounds the server declared. */
export function applyAudioBounds(bounds: {
  min_gain: number;
  max_gain: number;
  min_music_ms: number;
  max_fade_ms: number;
}): void {
  MIN_GAIN = bounds.min_gain;
  MAX_GAIN = bounds.max_gain;
  MIN_MUSIC_MS = bounds.min_music_ms;
  MAX_FADE_MS = bounds.max_fade_ms;
}

/** One clip on the timeline. `id` is local and never reaches the server. */
export interface TimelineClip {
  id: string;
  mediaId: string;
  inMs: number;
  outMs: number;
  /** How this clip enters (Phase 9). `cut` is every clip before it. */
  transition?: TransitionKind;
  transitionMs?: number;
  effects?: EffectRequest[];
}

/**
 * How long a clip *plays*, which is not how much source it reads.
 *
 * A clip at half speed reads two seconds and plays for four. Every position the
 * editor draws is computed from this, exactly as the server computes it -- the
 * two must agree or the timeline shows an edit that will not be rendered.
 */
export function clipPlaybackMs(clip: TimelineClip): number {
  const rate = clipSpeed(clip);
  return Math.round((clip.outMs - clip.inMs) / rate);
}

/** The playback rate a clip's effects imply. 1 when it has none. */
export function clipSpeed(clip: TimelineClip): number {
  let rate = 1;
  for (const effect of clip.effects ?? []) {
    if (effect.kind === "slow_motion" || effect.kind === "speed_up") rate *= effect.amount;
  }
  return rate > 0 ? rate : 1;
}

/** How much of this clip plays over the previous one. */
export function clipOverlapMs(clip: TimelineClip): number {
  return clip.transition && OVERLAPPING.includes(clip.transition) ? (clip.transitionMs ?? 0) : 0;
}

/**
 * The music bed on the editor's timeline.
 *
 * The same shape the API takes, with a local `id` so the lane can have a
 * selection like the video lane does. Deliberately *not* a `TimelineClip`: a
 * clip has no gain and no envelope, and a bed has no position in a sequence,
 * so one type carrying both would be two types with half its fields ignored.
 */
export interface MusicBed {
  mediaId: string;
  /** Trim within the track. */
  inMs: number;
  outMs: number;
  /** Where the bed starts in the output. */
  startMs: number;
  /** Linear gain; 1 is unity. */
  gain: number;
  fadeInMs: number;
  fadeOutMs: number;
}

export interface TimelineDraft {
  clips: TimelineClip[];
  /** The plan this draft was seeded from, if any. Travels back as provenance. */
  sourcePlanId: string | null;
  music: MusicBed | null;
  /** The cue list the plan carried, or `null` when it had none (Phase 9). */
  subtitles: SubtitleDraft | null;
}

export function musicDuration(bed: MusicBed): number {
  return bed.outMs - bed.inMs;
}

/**
 * A bed from a freshly chosen track.
 *
 * Takes the whole track up to the timeline's length, so dropping a five-minute
 * song under a twenty-second cut produces twenty seconds of music rather than a
 * cue the server would trim anyway.
 */
export function bedFromMedia(
  asset: MediaAsset,
  timelineMs: number,
  defaults: { gain?: number; fadeInMs?: number; fadeOutMs?: number } = {},
): MusicBed | null {
  const duration = asset.duration_ms;
  if (asset.kind !== "audio" || asset.status !== "ready" || !duration) return null;

  const wanted = Math.max(MIN_MUSIC_MS, timelineMs || MIN_MUSIC_MS);
  const outMs = Math.min(duration, wanted);
  if (outMs < MIN_MUSIC_MS) return null;

  const fadeOutMs = Math.min(defaults.fadeOutMs ?? 1500, Math.max(0, outMs - 1));
  return {
    mediaId: asset.id,
    inMs: 0,
    outMs: Math.round(outMs),
    startMs: 0,
    gain: defaults.gain ?? 0.7,
    fadeInMs: defaults.fadeInMs ?? 0,
    fadeOutMs: Math.round(fadeOutMs),
  };
}

/** The wire form of the bed. `gain` is called `volume` on the API. */
export function toMusicRequest(bed: MusicBed | null): MusicRequest | null {
  if (!bed) return null;
  return {
    media_id: bed.mediaId,
    source_in_ms: Math.round(bed.inMs),
    source_out_ms: Math.round(bed.outMs),
    timeline_start_ms: Math.round(bed.startMs),
    volume: bed.gain,
    fade_in_ms: Math.round(bed.fadeInMs),
    fade_out_ms: Math.round(bed.fadeOutMs),
  };
}

/**
 * Keep a bed inside what the server will accept.
 *
 * Applied on every edit rather than on save, so a slider cannot be dragged into
 * a state the render would refuse -- the same reason a trim handle is clamped
 * while the pointer moves.
 */
export function clampBed(bed: MusicBed, sourceDurationMs: number | null): MusicBed {
  const limit = sourceDurationMs ?? Number.MAX_SAFE_INTEGER;
  const inMs = Math.max(0, Math.min(bed.inMs, Math.max(0, limit - MIN_MUSIC_MS)));
  const outMs = Math.max(inMs + MIN_MUSIC_MS, Math.min(bed.outMs, limit));
  const length = outMs - inMs;

  const fadeInMs = Math.max(0, Math.min(bed.fadeInMs, Math.min(MAX_FADE_MS, length)));
  const fadeOutMs = Math.max(
    0,
    Math.min(bed.fadeOutMs, Math.min(MAX_FADE_MS, length - fadeInMs)),
  );

  return {
    ...bed,
    inMs: Math.round(inMs),
    outMs: Math.round(outMs),
    startMs: Math.max(0, Math.round(bed.startMs)),
    gain: Math.max(MIN_GAIN, Math.min(MAX_GAIN, bed.gain)),
    fadeInMs: Math.round(fadeInMs),
    fadeOutMs: Math.round(fadeOutMs),
  };
}

let counter = 0;
/** Local identity for a clip. Two cuts of one source must be distinguishable. */
export function clipId(): string {
  counter += 1;
  return `c${counter}`;
}

export function clipDuration(clip: TimelineClip): number {
  return clip.outMs - clip.inMs;
}

export function totalDuration(clips: readonly TimelineClip[]): number {
  const placed = place(clips);
  return placed.length ? placed[placed.length - 1].endMs : 0;
}

/** Where each clip starts on the timeline. Position is implied by order. */
export function clipStart(clips: readonly TimelineClip[], index: number): number {
  return place(clips)[index]?.startMs ?? 0;
}

export interface PlacedClip extends TimelineClip {
  index: number;
  startMs: number;
  endMs: number;
}

/** Lay the sequence out on a time axis. The one place offsets are computed. */
export function place(clips: readonly TimelineClip[]): PlacedClip[] {
  const placed: PlacedClip[] = [];
  let cursor = 0;
  clips.forEach((clip, index) => {
    // Two adjustments, and they are the same two the server makes: a clip
    // entered by a crossfade starts *before* the previous one ends, and a clip
    // with a speed effect occupies its played length rather than its trim.
    // The editor draws what will be rendered or it draws a lie.
    const overlap = index > 0 ? clipOverlapMs(clip) : 0;
    const start = Math.max(0, cursor - overlap);
    const duration = clipPlaybackMs(clip);
    placed.push({ ...clip, index, startMs: start, endMs: start + duration });
    cursor = start + duration;
  });
  return placed;
}

/** Which clip is under the playhead, and how far into it. */
export function clipAt(
  clips: readonly TimelineClip[],
  timeMs: number,
): { clip: PlacedClip; offsetMs: number } | null {
  for (const clip of place(clips)) {
    if (timeMs < clip.endMs || clip.index === clips.length - 1) {
      if (timeMs < clip.startMs) return null;
      return { clip, offsetMs: Math.max(0, Math.min(timeMs - clip.startMs, clipDuration(clip))) };
    }
  }
  return null;
}

// ------------------------------------------------------------------ plan <-> draft
/** Seed an editable timeline from a plan the server produced. */
export function toDraft(plan: EditPlan): TimelineDraft {
  const segments = [...(plan.plan?.segments ?? [])].sort((a, b) => a.order - b.order);
  const music = plan.plan?.music ?? null;
  return {
    sourcePlanId: plan.id,
    clips: segments.map((segment) => ({
      id: clipId(),
      mediaId: segment.media_id,
      inMs: segment.source_in_ms,
      outMs: segment.source_out_ms,
      // Carried back so accepting a plan does not silently drop the
      // transitions and effects it chose. Defaulted rather than assumed: a
      // plan stored before Phase 9 has neither field.
      transition: (segment.transition_in as TransitionKind) ?? "cut",
      transitionMs: segment.transition_ms ?? 0,
      effects: segment.effects ?? [],
    })),
    // A generated plan may have chosen a bed and aligned it to the first beat.
    // Accepting the plan has to bring that with it, or the edit the user
    // reviewed is not the edit they get.
    subtitles: subtitlesFromPlan(plan.plan?.subtitles ?? null),
    music: music
      ? {
          mediaId: music.media_id,
          inMs: music.source_in_ms,
          outMs: music.source_out_ms,
          startMs: music.timeline_start_ms,
          gain: music.gain,
          fadeInMs: music.fade_in_ms,
          fadeOutMs: music.fade_out_ms,
        }
      : null,
  };
}

/** The wire form. Array position carries the order, as the route requires. */
export function toManualCuts(clips: readonly TimelineClip[]): ManualCut[] {
  return clips.map((clip) => ({
    media_id: clip.mediaId,
    source_in_ms: Math.round(clip.inMs),
    source_out_ms: Math.round(clip.outMs),
    transition_in: clip.transition ?? "cut",
    // A cut has no duration, and the server refuses one that claims otherwise.
    transition_ms: clip.transition && clip.transition !== "cut" ? (clip.transitionMs ?? 0) : 0,
    effects: clip.effects ?? [],
  }));
}

/**
 * A new clip from a media asset, trimmed to something renderable.
 *
 * Long sources are capped rather than added whole: dropping a ten-minute file
 * onto the timeline and having it exceed the per-segment limit is a rejection
 * the user cannot act on, where a visible five-second clip with trim handles is
 * an edit they can extend.
 */
export function clipFromMedia(asset: MediaAsset, defaultMs = 5000): TimelineClip | null {
  const duration = asset.duration_ms;
  if (asset.kind !== "video" || asset.status !== "ready" || !duration) return null;
  const length = Math.min(duration, Math.max(MIN_CLIP_MS, Math.min(defaultMs, MAX_CLIP_MS)));
  return { id: clipId(), mediaId: asset.id, inMs: 0, outMs: Math.round(length) };
}

// ----------------------------------------------------------------- local checks
export interface DraftProblem {
  clipId: string | null;
  message: string;
}

/**
 * What the server will say, said early.
 *
 * Not a substitute for validation -- the route runs the real thing against the
 * real rows. This exists so the Render button can be disabled with a reason
 * instead of enabled into a 422.
 */
export function draftProblems(
  clips: readonly TimelineClip[],
  media: Map<string, MediaAsset>,
): DraftProblem[] {
  const problems: DraftProblem[] = [];

  if (clips.length === 0) {
    problems.push({ clipId: null, message: "The timeline is empty." });
    return problems;
  }
  if (clips.length > MAX_CLIPS) {
    problems.push({ clipId: null, message: `More than ${MAX_CLIPS} clips.` });
  }

  for (const clip of clips) {
    const duration = clipDuration(clip);
    const asset = media.get(clip.mediaId);
    const name = asset?.original_filename ?? clip.mediaId.slice(0, 8);

    if (duration < MIN_CLIP_MS) {
      problems.push({ clipId: clip.id, message: `${name}: shorter than ${MIN_CLIP_MS} ms.` });
    }
    if (duration > MAX_CLIP_MS) {
      problems.push({
        clipId: clip.id,
        message: `${name}: longer than ${MAX_CLIP_MS / 1000} s.`,
      });
    }
    if (asset && asset.duration_ms !== null && clip.outMs > asset.duration_ms) {
      problems.push({ clipId: clip.id, message: `${name}: trimmed past the end of the source.` });
    }
    if (asset && asset.status !== "ready") {
      problems.push({ clipId: clip.id, message: `${name}: not ready to render.` });
    }
  }

  const total = totalDuration(clips);
  if (total < MIN_TIMELINE_MS) {
    problems.push({ clipId: null, message: `Total under ${MIN_TIMELINE_MS / 1000} s.` });
  }
  if (total > MAX_TIMELINE_MS) {
    problems.push({ clipId: null, message: `Total over ${MAX_TIMELINE_MS / 60_000} minutes.` });
  }
  return problems;
}

// --------------------------------------------------------------------- editing
/** Trim a clip, clamped to what the source and the renderer allow. */
export function trimClip(
  clip: TimelineClip,
  edge: "in" | "out",
  valueMs: number,
  sourceDurationMs: number | null,
): TimelineClip {
  const limit = sourceDurationMs ?? Number.MAX_SAFE_INTEGER;
  if (edge === "in") {
    const lowest = Math.max(0, clip.outMs - MAX_CLIP_MS);
    const highest = clip.outMs - MIN_CLIP_MS;
    return { ...clip, inMs: Math.round(Math.min(Math.max(valueMs, lowest), highest)) };
  }
  const lowest = clip.inMs + MIN_CLIP_MS;
  const highest = Math.min(limit, clip.inMs + MAX_CLIP_MS);
  return { ...clip, outMs: Math.round(Math.min(Math.max(valueMs, lowest), highest)) };
}

/** Move a clip to a new index, preserving every other clip's relative order. */
export function reorder(
  clips: readonly TimelineClip[],
  from: number,
  to: number,
): TimelineClip[] {
  if (from === to || from < 0 || from >= clips.length) return [...clips];
  const next = [...clips];
  const [moved] = next.splice(from, 1);
  next.splice(Math.max(0, Math.min(to, next.length)), 0, moved);
  return next;
}

export interface OutputIntent {
  aspect: AspectRatio;
  fps: number;
  quality: QualityPreset;
  audio: "none" | "source";
  /** Linear gain on the clips' own audio. Unity is the Phase 6 behaviour. */
  sourceGain: number;
}

export const DEFAULT_OUTPUT: OutputIntent = {
  aspect: "16:9",
  fps: 30,
  quality: "balanced",
  audio: "none",
  sourceGain: 1,
};

// ------------------------------------------------------------------ subtitles
/**
 * The cue list, as the editor holds it.
 *
 * A cue gets a client id so React can key it and so the inspector can address
 * one without relying on an index that shifts when an earlier cue is deleted.
 * Everything else is exactly the wire shape -- and note what is *not* here: no
 * font, no colour, no coordinate. A cue says when and what; the server preset
 * table says how, which is the whole reason a client cannot smuggle a font path
 * into a render.
 */
export interface SubtitleCueDraft {
  id: string;
  startMs: number;
  endMs: number;
  text: string;
}

export interface SubtitleDraft {
  cues: SubtitleCueDraft[];
  style: SubtitleStyle;
  position: SubtitlePosition;
}

export const EMPTY_SUBTITLES: SubtitleDraft = {
  cues: [],
  style: "clean",
  position: "bottom",
};

export function cueId(): string {
  return `cue_${Math.random().toString(36).slice(2, 10)}`;
}

function subtitlesFromPlan(subtitles: PlanSubtitles | null): SubtitleDraft | null {
  if (!subtitles) return null;
  return {
    cues: subtitles.cues.map((cue) => ({
      id: cueId(),
      startMs: cue.start_ms,
      endMs: cue.end_ms,
      text: cue.text,
    })),
    style: subtitles.style,
    position: subtitles.position,
  };
}

/** Cues in time order. The server requires it; sorting here is not a substitute
 *  for that, it is so the list the user reads matches the list that is sent. */
export function sortedCues(cues: readonly SubtitleCueDraft[]): SubtitleCueDraft[] {
  return [...cues].sort((a, b) => a.startMs - b.startMs || a.endMs - b.endMs);
}

/**
 * Where a new cue should go.
 *
 * At the playhead if there is room after it, otherwise after the last cue.
 * Clamped to the programme, because a cue past the end of the video is one the
 * server rejects and one nobody would ever see.
 */
export function nextCueWindow(
  cues: readonly SubtitleCueDraft[],
  playheadMs: number,
  totalMs: number,
): { startMs: number; endMs: number } | null {
  const sorted = sortedCues(cues);
  const wanted = Math.min(3000, Math.max(MIN_CUE_MS, Math.round(totalMs / 6)));

  for (const raw of [playheadMs, sorted.length ? sorted[sorted.length - 1].endMs : 0, 0]) {
    const start = Math.max(0, Math.round(raw));
    // Butt up against the next cue rather than overlapping it: the server
    // rejects overlapping cues, and two lines on screen at once is not
    // something the single-track document can express anyway.
    const nextStart = sorted.find((cue) => cue.startMs >= start)?.startMs ?? totalMs;
    const collides = sorted.some((cue) => start < cue.endMs && start >= cue.startMs);
    if (collides) continue;
    const end = Math.min(start + wanted, nextStart, totalMs);
    if (end - start >= MIN_CUE_MS) return { startMs: start, endMs: end };
  }
  return null;
}

/** Local cue checks, mirroring the server. Same purpose as `draftProblems`:
 *  say it before the 422, not instead of it. */
export function cueProblems(subtitles: SubtitleDraft | null, totalMs: number): DraftProblem[] {
  const problems: DraftProblem[] = [];
  if (!subtitles || subtitles.cues.length === 0) return problems;

  if (subtitles.cues.length > CUE_BOUNDS.maxCues) {
    problems.push({ clipId: null, message: `More than ${CUE_BOUNDS.maxCues} subtitle cues.` });
  }

  const sorted = sortedCues(subtitles.cues);
  sorted.forEach((cue, index) => {
    const length = cue.endMs - cue.startMs;
    const label = `Cue ${index + 1}`;
    if (!cue.text.trim()) problems.push({ clipId: null, message: `${label}: no text.` });
    if (cue.text.length > CUE_BOUNDS.maxChars) {
      problems.push({ clipId: null, message: `${label}: over ${CUE_BOUNDS.maxChars} characters.` });
    }
    if (length < CUE_BOUNDS.minMs) {
      problems.push({ clipId: null, message: `${label}: shorter than ${CUE_BOUNDS.minMs} ms.` });
    }
    if (length > CUE_BOUNDS.maxMs) {
      problems.push({
        clipId: null,
        message: `${label}: longer than ${CUE_BOUNDS.maxMs / 1000} s.`,
      });
    }
    const previous = sorted[index - 1];
    if (previous && cue.startMs < previous.endMs) {
      problems.push({ clipId: null, message: `${label}: overlaps the cue before it.` });
    }
    if (totalMs > 0 && cue.endMs > totalMs) {
      problems.push({ clipId: null, message: `${label}: ends after the video does.` });
    }
  });

  return problems;
}

export function toSubtitleRequest(subtitles: SubtitleDraft | null): SubtitleTrackRequest | null {
  if (!subtitles || subtitles.cues.length === 0) return null;
  return {
    cues: sortedCues(subtitles.cues).map((cue) => ({
      start_ms: Math.round(cue.startMs),
      end_ms: Math.round(cue.endMs),
      text: cue.text,
    })),
    style: subtitles.style,
    position: subtitles.position,
  };
}

// -------------------------------------------------------------------- effects
/**
 * Set one effect on a clip, replacing any existing effect of that kind.
 *
 * One of each kind, because two brightness effects on one clip is either a sum
 * the user cannot see or a fight the renderer arbitrates -- and the domain
 * rejects the pair regardless. Setting a kind back to its neutral value removes
 * it rather than sending `brightness: 0`, which would be an instruction to do
 * nothing dressed up as an edit.
 */
export function withEffect(clip: TimelineClip, effect: EffectRequest): TimelineClip {
  const others = (clip.effects ?? []).filter((item) => item.kind !== effect.kind);
  // Speed is exclusive with itself in both directions: a clip is not both
  // slowed and sped up, it has one rate.
  const speedy = effect.kind === "slow_motion" || effect.kind === "speed_up";
  const kept = speedy
    ? others.filter((item) => item.kind !== "slow_motion" && item.kind !== "speed_up")
    : others;
  if (effect.amount === EFFECT_BOUNDS[effect.kind].neutral) return { ...clip, effects: kept };
  return { ...clip, effects: [...kept, effect] };
}

export function withoutEffect(clip: TimelineClip, kind: EffectKind): TimelineClip {
  return { ...clip, effects: (clip.effects ?? []).filter((item) => item.kind !== kind) };
}

/** The amount set for a kind, or its neutral value when it is not set. */
export function effectAmount(clip: TimelineClip, kind: EffectKind): number {
  const found = (clip.effects ?? []).find((item) => item.kind === kind);
  return found ? found.amount : EFFECT_BOUNDS[kind].neutral;
}

/** How this clip enters, and for how long. A cut always reports zero. */
export function transitionOf(clip: TimelineClip): { kind: TransitionKind; ms: number } {
  const kind = clip.transition ?? "cut";
  return { kind, ms: kind === "cut" ? 0 : (clip.transitionMs ?? 0) };
}

/**
 * The longest transition this clip may carry.
 *
 * Mirrors the domain: a transition may not consume more than half of either
 * side it touches, and never more than the hard ceiling. Returning zero means
 * the clips are too short for one, which the inspector shows as a disabled
 * control rather than as a slider that produces a 422.
 */
export function maxTransitionMs(clips: readonly TimelineClip[], index: number): number {
  const clip = clips[index];
  if (!clip) return 0;
  const own = clipPlaybackMs(clip);
  const previous = index > 0 ? clipPlaybackMs(clips[index - 1]) : own;
  const share = Math.floor(Math.min(own, previous) * MAX_TRANSITION_SHARE);
  const limit = Math.min(share, TRANSITION_BOUNDS.max);
  return limit >= TRANSITION_BOUNDS.min ? limit : 0;
}
