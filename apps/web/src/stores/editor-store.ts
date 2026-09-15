import { create } from "zustand";

import type { AspectRatio, MediaAsset, QualityPreset, StyleStrength } from "@/lib/api";
import {
  bedFromMedia,
  clampBed,
  clipFromMedia,
  clipId,
  clipDuration,
  DEFAULT_OUTPUT,
  MAX_CLIPS,
  reorder as reorderClips,
  totalDuration,
  trimClip,
  type MusicBed,
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
export type InspectorTabName = "clip" | "analysis" | "ai" | "audio" | "export";
/**
 * How the media browser draws the library.
 *
 * `grid` answers "which shot is this", `list` answers "which of these is
 * 60fps", `compact` answers "where is clip_047". Three views rather than two
 * because those are three different questions, not three densities of one.
 */
export type BrowserView = "grid" | "list" | "compact";
export type InspectorTab = InspectorTabName;

/**
 * The workspace the navigation rail is pointing at.
 *
 * Six concepts rather than one screen with panels toggled on and off. They are
 * all the *same* project and the same client state -- switching to Assets does
 * not unload the timeline -- but they give each activity the whole viewport
 * when it is the activity you are doing. Picking twelve clips out of two
 * hundred is a different job from trimming four of them, and a 260px column is
 * the wrong size for the first and the right size for the second.
 *
 * `home` is the only one that works without a project open.
 */
export type AppView = "home" | "editor" | "assets" | "audio" | "export" | "settings";

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

  setClips: (
    clips: TimelineClip[],
    sourcePlanId: string | null,
    committed?: string | null,
    music?: MusicBed | null,
  ) => void;
  addMedia: (assets: MediaAsset[]) => number;
  selectClip: (clipId: string | null) => void;
  removeClip: (clipId: string) => void;
  moveClip: (from: number, to: number) => void;
  trim: (clipId: string, edge: "in" | "out", valueMs: number, sourceMs: number | null) => void;
  splitAtPlayhead: () => void;
  clearTimeline: () => void;
  markCommitted: (planId: string) => void;

  // --- music ---
  /**
   * The bed under the timeline, or ``null``.
   *
   * One bed, matching what ``EditPlan`` can express. A second would need
   * overlap rules and a mix policy that the plan has nowhere to put, so the
   * editor does not offer one.
   */
  music: MusicBed | null;
  /** Whether the next generated plan should cut to the track's beats. */
  beatSync: boolean;
  /** Whether the timeline draws detected beats on the music lane. */
  showBeatMarkers: boolean;
  setShowBeatMarkers: (show: boolean) => void;
  setMusic: (asset: MediaAsset | null) => void;
  updateMusic: (patch: Partial<MusicBed>, sourceDurationMs: number | null) => void;
  clearMusic: () => void;
  setBeatSync: (beatSync: boolean) => void;

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
  /** Gain on the clips' own audio, so dialogue can sit under a bed. */
  sourceGain: number;
  setOutput: (
    patch: Partial<
      Pick<EditorState, "aspect" | "fps" | "quality" | "audio" | "sourceGain">
    >,
  ) => void;

  // --- reference style (Phase 8) ---
  /**
   * How much of the project's reference video to apply.
   *
   * Client state, and "0" by default: attaching a reference must not change
   * anyone's edit until they ask it to. Which clip *is* the reference is server
   * state and deliberately not mirrored here -- one copy, on the side that
   * enforces ownership.
   */
  styleStrength: StyleStrength;
  setStyleStrength: (strength: StyleStrength) => void;

  // --- navigation ---
  view: AppView;
  setView: (view: AppView) => void;

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
      // Opening a project is a request to edit it; landing back on Home with a
      // project loaded would make the click look like it had failed.
      view: "editor",
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
      // A bed belongs to the project it was chosen in, for the same reason the
      // clips do: carrying it across would reference media the new project has
      // never heard of.
      music: null,
      // The reference belongs to the project that was open, and so does the
      // dial: carrying a strength across would apply one project's measurements
      // to another's footage.
      styleStrength: "0",
    }),

  // ----------------------------------------------------------------- style
  styleStrength: "0",
  setStyleStrength: (styleStrength) => set({ styleStrength }),

  // ---------------------------------------------------------------- navigation
  view: "home",
  /**
   * Switching view never clears the draft. The timeline the user is part-way
   * through is the same timeline whether they are looking at the library or at
   * the export settings, and losing it on a navigation click would be the
   * single most expensive bug this store could ship.
   */
  setView: (view) => set({ view }),

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

  setClips: (clips, sourcePlanId, committed = sourcePlanId, music) =>
    set((state) => ({
      clips,
      sourcePlanId,
      committedPlanId: committed,
      dirty: false,
      selectedClipId: clips[0]?.id ?? null,
      playheadMs: 0,
      playing: false,
      // `undefined` means the caller is not speaking about music; `null` means
      // it is saying there is none. Accepting a plan replaces the bed with
      // whatever that plan chose, including nothing.
      music: music === undefined ? state.music : music,
    })),

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
      music: null,
    }),

  markCommitted: (planId) => set({ committedPlanId: planId, dirty: false }),

  // ---------------------------------------------------------------- music
  music: null,
  beatSync: false,
  showBeatMarkers: true,
  setShowBeatMarkers: (showBeatMarkers) => set({ showBeatMarkers }),

  setMusic: (asset) =>
    set((state) => {
      if (asset === null) return { music: null, dirty: true };
      const bed = bedFromMedia(asset, totalDuration(state.clips));
      // `bedFromMedia` refuses anything that is not ready audio, so a video
      // dropped on the lane changes nothing rather than producing a cue the
      // server would reject.
      return bed ? { music: bed, dirty: true } : state;
    }),

  updateMusic: (patch, sourceDurationMs) =>
    set((state) =>
      state.music
        ? { music: clampBed({ ...state.music, ...patch }, sourceDurationMs), dirty: true }
        : state,
    ),

  clearMusic: () => set({ music: null, dirty: true }),
  setBeatSync: (beatSync) => set({ beatSync }),

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
  sourceGain: DEFAULT_OUTPUT.sourceGain,
  // Output intent is not the draft, so changing it does not dirty the
  // timeline -- except that it does reach the stored plan, so it does.
  setOutput: (patch) => set({ ...patch, dirty: true }),

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
