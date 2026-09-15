import { create } from "zustand";

import type { AspectRatio, MediaAsset, QualityPreset } from "@/lib/api";
import {
  clipFromMedia,
  clipId,
  clipDuration,
  DEFAULT_OUTPUT,
  MAX_CLIPS,
  reorder as reorderClips,
  totalDuration,
  trimClip,
  type TimelineClip,
} from "@/lib/timeline";

/**
 * Client-only editor state.
 *
 * The Phase 0 rule still holds: server state lives in TanStack Query and is
 * never copied here. What this store owns is what the *user* is doing -- which
 * clip is selected, where the playhead is, how far the timeline is zoomed, and
 * the timeline draft itself, which is client state precisely because it has not
 * been committed to the server yet.
 *
 * The one place the line blurs is `committedPlanId`: a server id, kept so the
 * store can tell whether the draft still matches what was stored. It is an id,
 * not a copy of the plan.
 */

export type PreviewSource = "source" | "program" | "render";
/**
 * How the media browser draws the library.
 *
 * `grid` answers "which shot is this", `list` answers "which of these is
 * 60fps", `compact` answers "where is clip_047". Three views rather than two
 * because those are three different questions, not three densities of one.
 */
export type BrowserView = "grid" | "list" | "compact";
export type InspectorTab = "clip" | "analysis" | "ai" | "export";

interface EditorState {
  // --- project ---
  projectId: string | null;
  openProject: (projectId: string) => void;

  // --- media browser ---
  browserView: BrowserView;
  setBrowserView: (view: BrowserView) => void;
  /** Every selected asset. Drives "add to timeline" and bulk analysis. */
  selectedMediaIds: string[];
  /** The asset the inspector and the source viewer are showing. */
  activeMediaId: string | null;
  selectMedia: (mediaId: string, modifier?: "replace" | "toggle" | "range", all?: string[]) => void;
  clearMediaSelection: () => void;

  // --- timeline ---
  clips: TimelineClip[];
  selectedClipId: string | null;
  sourcePlanId: string | null;
  committedPlanId: string | null;
  /** True when the timeline has been edited since it was last stored. */
  dirty: boolean;
  playheadMs: number;
  pxPerSecond: number;
  playing: boolean;

  setClips: (clips: TimelineClip[], sourcePlanId: string | null, committed?: string | null) => void;
  addMedia: (assets: MediaAsset[]) => number;
  selectClip: (clipId: string | null) => void;
  removeClip: (clipId: string) => void;
  moveClip: (from: number, to: number) => void;
  trim: (clipId: string, edge: "in" | "out", valueMs: number, sourceMs: number | null) => void;
  splitAtPlayhead: () => void;
  clearTimeline: () => void;
  markCommitted: (planId: string) => void;

  setPlayhead: (ms: number) => void;
  nudgePlayhead: (deltaMs: number) => void;
  setPlaying: (playing: boolean) => void;
  togglePlaying: () => void;
  zoom: (factor: number) => void;
  setPxPerSecond: (value: number) => void;

  // --- output intent ---
  aspect: AspectRatio;
  fps: number;
  quality: QualityPreset;
  audio: "none" | "source";
  setOutput: (patch: Partial<Pick<EditorState, "aspect" | "fps" | "quality" | "audio">>) => void;

  // --- panels ---
  previewSource: PreviewSource;
  setPreviewSource: (source: PreviewSource) => void;
  inspectorTab: InspectorTab;
  setInspectorTab: (tab: InspectorTab) => void;
  browserOpen: boolean;
  inspectorOpen: boolean;
  toggleBrowser: () => void;
  toggleInspector: () => void;
  timelineHeight: number;
  setTimelineHeight: (height: number) => void;
}

export const MIN_PX_PER_SECOND = 4;
export const MAX_PX_PER_SECOND = 400;

export const useEditorStore = create<EditorState>((set, get) => ({
  projectId: null,
  openProject: (projectId) =>
    set({
      projectId,
      // A timeline belongs to the project it was cut in. Carrying clips across
      // would produce a draft referencing media the new project does not have.
      clips: [],
      selectedClipId: null,
      sourcePlanId: null,
      committedPlanId: null,
      dirty: false,
      playheadMs: 0,
      selectedMediaIds: [],
      activeMediaId: null,
      playing: false,
    }),

  // ------------------------------------------------------------- browser
  browserView: "grid",
  setBrowserView: (browserView) => set({ browserView }),
  selectedMediaIds: [],
  activeMediaId: null,

  selectMedia: (mediaId, modifier = "replace", all = []) =>
    set((state) => {
      if (modifier === "toggle") {
        const has = state.selectedMediaIds.includes(mediaId);
        const next = has
          ? state.selectedMediaIds.filter((id) => id !== mediaId)
          : [...state.selectedMediaIds, mediaId];
        return { selectedMediaIds: next, activeMediaId: has ? state.activeMediaId : mediaId };
      }
      if (modifier === "range" && state.activeMediaId && all.length > 0) {
        const from = all.indexOf(state.activeMediaId);
        const to = all.indexOf(mediaId);
        if (from >= 0 && to >= 0) {
          const [low, high] = from < to ? [from, to] : [to, from];
          return { selectedMediaIds: all.slice(low, high + 1), activeMediaId: mediaId };
        }
      }
      return { selectedMediaIds: [mediaId], activeMediaId: mediaId };
    }),

  clearMediaSelection: () => set({ selectedMediaIds: [], activeMediaId: null }),

  // ------------------------------------------------------------ timeline
  clips: [],
  selectedClipId: null,
  sourcePlanId: null,
  committedPlanId: null,
  dirty: false,
  playheadMs: 0,
  pxPerSecond: 60,
  playing: false,

  setClips: (clips, sourcePlanId, committed = sourcePlanId) =>
    set({
      clips,
      sourcePlanId,
      committedPlanId: committed,
      dirty: false,
      selectedClipId: clips[0]?.id ?? null,
      playheadMs: 0,
      playing: false,
    }),

  addMedia: (assets) => {
    const state = get();
    const room = MAX_CLIPS - state.clips.length;
    if (room <= 0) return 0;

    const added = assets
      .map((asset) => clipFromMedia(asset))
      .filter((clip): clip is TimelineClip => clip !== null)
      .slice(0, room);
    if (added.length === 0) return 0;

    set({
      clips: [...state.clips, ...added],
      selectedClipId: added[added.length - 1].id,
      dirty: true,
    });
    return added.length;
  },

  selectClip: (selectedClipId) => set({ selectedClipId }),

  removeClip: (id) =>
    set((state) => {
      const index = state.clips.findIndex((clip) => clip.id === id);
      if (index < 0) return state;
      const clips = state.clips.filter((clip) => clip.id !== id);
      return {
        clips,
        dirty: true,
        // Keep a neighbour selected so repeated deletes need no re-aiming.
        selectedClipId: clips[Math.min(index, clips.length - 1)]?.id ?? null,
        playheadMs: Math.min(state.playheadMs, totalDuration(clips)),
      };
    }),

  moveClip: (from, to) =>
    set((state) => ({ clips: reorderClips(state.clips, from, to), dirty: true })),

  trim: (id, edge, valueMs, sourceMs) =>
    set((state) => ({
      clips: state.clips.map((clip) =>
        clip.id === id ? trimClip(clip, edge, valueMs, sourceMs) : clip,
      ),
      dirty: true,
    })),

  /**
   * Cut the clip under the playhead in two.
   *
   * Both halves keep their source trim, so a split is non-destructive in the
   * sense that matters: the media is untouched and the two cuts together still
   * describe exactly the footage the one cut did.
   */
  splitAtPlayhead: () =>
    set((state) => {
      let cursor = 0;
      for (const [index, clip] of state.clips.entries()) {
        const length = clipDuration(clip);
        const offset = state.playheadMs - cursor;
        if (offset > 0 && offset < length) {
          const cutAt = clip.inMs + offset;
          const left = { ...clip, outMs: Math.round(cutAt) };
          const right = { ...clip, id: clipId(), inMs: Math.round(cutAt) };
          const clips = [...state.clips];
          clips.splice(index, 1, left, right);
          return { clips, selectedClipId: right.id, dirty: true };
        }
        cursor += length;
      }
      return state;
    }),

  clearTimeline: () =>
    set({
      clips: [],
      selectedClipId: null,
      sourcePlanId: null,
      committedPlanId: null,
      dirty: false,
      playheadMs: 0,
      playing: false,
    }),

  markCommitted: (planId) => set({ committedPlanId: planId, dirty: false }),

  setPlayhead: (ms) => set({ playheadMs: Math.max(0, ms) }),
  nudgePlayhead: (deltaMs) =>
    set((state) => ({
      playheadMs: Math.max(
        0,
        Math.min(state.playheadMs + deltaMs, totalDuration(state.clips)),
      ),
    })),
  setPlaying: (playing) => set({ playing }),
  togglePlaying: () => set((state) => ({ playing: !state.playing })),

  zoom: (factor) =>
    set((state) => ({
      pxPerSecond: Math.max(
        MIN_PX_PER_SECOND,
        Math.min(MAX_PX_PER_SECOND, state.pxPerSecond * factor),
      ),
    })),
  setPxPerSecond: (value) =>
    set({
      pxPerSecond: Math.max(MIN_PX_PER_SECOND, Math.min(MAX_PX_PER_SECOND, value)),
    }),

  // -------------------------------------------------------------- output
  aspect: DEFAULT_OUTPUT.aspect,
  fps: DEFAULT_OUTPUT.fps,
  quality: DEFAULT_OUTPUT.quality,
  audio: DEFAULT_OUTPUT.audio,
  setOutput: (patch) => set(patch),

  // -------------------------------------------------------------- panels
  previewSource: "program",
  setPreviewSource: (previewSource) => set({ previewSource }),
  inspectorTab: "ai",
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
  browserOpen: true,
  inspectorOpen: true,
  toggleBrowser: () => set((state) => ({ browserOpen: !state.browserOpen })),
  toggleInspector: () => set((state) => ({ inspectorOpen: !state.inspectorOpen })),
  timelineHeight: 220,
  setTimelineHeight: (height) => set({ timelineHeight: Math.max(140, Math.min(520, height)) }),
}));

/** Total timeline length. A selector so components re-render on clip changes only. */
export const selectTotalMs = (state: EditorState): number => totalDuration(state.clips);
