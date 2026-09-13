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

import type { AspectRatio, EditPlan, ManualCut, MediaAsset, QualityPreset } from "@/lib/api";

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

/** One clip on the timeline. `id` is local and never reaches the server. */
export interface TimelineClip {
  id: string;
  mediaId: string;
  inMs: number;
  outMs: number;
}

export interface TimelineDraft {
  clips: TimelineClip[];
  /** The plan this draft was seeded from, if any. Travels back as provenance. */
  sourcePlanId: string | null;
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
  return clips.reduce((sum, clip) => sum + clipDuration(clip), 0);
}

/** Where each clip starts on the timeline. Position is implied by order. */
export function clipStart(clips: readonly TimelineClip[], index: number): number {
  let start = 0;
  for (let i = 0; i < index; i += 1) start += clipDuration(clips[i]);
  return start;
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
    const duration = clipDuration(clip);
    placed.push({ ...clip, index, startMs: cursor, endMs: cursor + duration });
    cursor += duration;
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
  return {
    sourcePlanId: plan.id,
    clips: segments.map((segment) => ({
      id: clipId(),
      mediaId: segment.media_id,
      inMs: segment.source_in_ms,
      outMs: segment.source_out_ms,
    })),
  };
}

/** The wire form. Array position carries the order, as the route requires. */
export function toManualCuts(clips: readonly TimelineClip[]): ManualCut[] {
  return clips.map((clip) => ({
    media_id: clip.mediaId,
    source_in_ms: Math.round(clip.inMs),
    source_out_ms: Math.round(clip.outMs),
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
}

export const DEFAULT_OUTPUT: OutputIntent = {
  aspect: "16:9",
  fps: 30,
  quality: "balanced",
  audio: "none",
};
