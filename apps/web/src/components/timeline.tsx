"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useQuery } from "@tanstack/react-query";

import { api, type BeatsPayload, type MediaAsset } from "@/lib/api";
import { timecode } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { useThumbnailUrl } from "@/lib/media-urls";
import {
  MAX_CLIP_MS,
  MIN_CLIP_MS,
  clipDuration,
  clipPlaybackMs,
  clipSpeed,
  musicDuration,
  place,
  sortedCues,
  transitionOf,
  type MusicBed,
  type PlacedClip,
  type SubtitleCueDraft,
} from "@/lib/timeline";
import {
  MAX_PX_PER_SECOND,
  MIN_PX_PER_SECOND,
  useEditorStore,
} from "@/stores/editor-store";
import { Badge, Divider, Glyph, IconButton, Slider } from "@/components/ui";

/**
 * The timeline.
 *
 * Four lanes, which is what the EditPlan schema describes: a sequence of trims
 * with position implied by order, the source audio that comes with them, one
 * music bed, and one subtitle track. The timeline draws exactly that and
 * nothing more -- no gaps to drag into, no second bed, no per-cue styling --
 * because the plan has no way to express any of them and a UI that let you
 * build one would be offering an edit the renderer must reject.
 *
 * For the same reason there is no snapping indicator, no magnet toggle and no
 * ripple mode. Clips are joined by construction, so there is nothing for a clip
 * to snap *to*; a magnet button here would be a light that is always on.
 *
 * The one place a clip is *not* butt-joined is a crossfade, and Phase 9 draws
 * that as the overlap it is: the incoming clip starts before the outgoing one
 * ends, exactly as `place` computes it and exactly as the compiler renders it.
 * Drawing butt joints and reporting a shorter duration would make the editor
 * disagree with its own export.
 *
 * Interaction is pointer-event based with pointer capture, so a drag that
 * leaves the element still tracks, and one that ends outside still commits.
 */

const RULER_HEIGHT = 26;
const GUTTER = 76;
const HANDLE_WIDTH = 8;

/** Labelled ticks, spaced so labels never collide at any zoom. */
function tickInterval(pxPerSecond: number): number {
  const candidates = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const minimumPx = 64;
  return (
    candidates.find((seconds) => seconds * pxPerSecond >= minimumPx) ??
    candidates[candidates.length - 1]
  );
}

type Drag =
  | { kind: "playhead" }
  | { kind: "trim"; clipId: string; edge: "in" | "out"; originMs: number; startX: number }
  | { kind: "move"; clipId: string; index: number; startX: number; moved: boolean };

// ------------------------------------------------------------------- the clip
function Clip({
  clip,
  asset,
  projectId,
  pxPerSecond,
  selected,
  invalid,
  onPointerDown,
}: {
  clip: PlacedClip;
  asset: MediaAsset | undefined;
  projectId: string;
  pxPerSecond: number;
  selected: boolean;
  invalid: boolean;
  onPointerDown: (event: React.PointerEvent, kind: "move" | "trim", edge?: "in" | "out") => void;
}) {
  const t = useT();
  const thumbnail = useThumbnailUrl(projectId, asset ?? null);
  // Played length, not trim length: a clip at half speed occupies twice its
  // trim on the timeline, and drawing the trim would put every later clip in
  // the wrong place.
  const width = (clipPlaybackMs(clip) / 1000) * pxPerSecond;
  const left = (clip.startMs / 1000) * pxPerSecond;
  const name = asset?.original_filename ?? clip.mediaId.slice(0, 8);
  const speed = clipSpeed(clip);
  const effects = clip.effects ?? [];

  // Three widths of clip: one that can carry a name and a duration, one that
  // can carry a number, and one that is a sliver. Deciding here rather than
  // letting text overflow is what keeps a dense timeline readable.
  const compact = width < 64;
  const sliver = width < 18;

  return (
    <div
      role="option"
      aria-selected={selected}
      aria-label={t("timeline.clip", {
        index: clip.index + 1,
        name,
        duration: timecode(clipDuration(clip)),
      })}
      tabIndex={0}
      onPointerDown={(event) => onPointerDown(event, "move")}
      style={{ left, width: Math.max(width, 3) }}
      className={`group absolute inset-y-[4px] select-none overflow-hidden rounded-md shadow-raised transition-[box-shadow,background-color] duration-fast ${
        invalid
          ? "bg-danger/25 ring-1 ring-danger"
          : selected
            ? "bg-track-video-selected ring-2 ring-inset ring-accent-strong"
            : "bg-track-video ring-1 ring-inset ring-black/15 hover:shadow-panel"
      }`}
    >
      {/* The filmstrip: one thumbnail tiled along the clip. A texture that says
          "footage", not a claim about the frames at those positions -- the
          client has exactly one frame per asset and inventing the rest would be
          a lie the renderer never told. */}
      {thumbnail.data?.url && !sliver && (
        <span
          aria-hidden
          className="vf-filmstrip pointer-events-none absolute inset-0 opacity-30"
          style={{ ["--strip" as string]: `url("${thumbnail.data.url}")` }}
        />
      )}

      {!sliver && (
        <div className="pointer-events-none absolute inset-0 flex flex-col justify-between px-1.5 py-0.5">
          {/* A name band rather than bare text on the strip: at 30% opacity a
              thumbnail still eats 11px type without it. */}
          <span className="truncate rounded bg-black/35 px-1.5 text-2xs font-medium leading-[16px] text-white/95">
            {compact ? clip.index + 1 : name}
          </span>
          {!compact && (
            <span className="flex items-center gap-1 truncate font-mono text-2xs tabular-nums text-white/70">
              {timecode(clipPlaybackMs(clip), false)}
              {/* Two marks, and only when there is something to mark. A clip
                  with no effects carries no badge, so a badge means something
                  was done rather than being permanent furniture. */}
              {speed !== 1 && (
                <span
                  title={t("timeline.speed", { rate: speed.toFixed(2) })}
                  className="rounded bg-black/45 px-1 text-white/90"
                >
                  {speed.toFixed(2)}×
                </span>
              )}
              {effects.length > 0 && (
                <span
                  title={effects.map((effect) => effect.kind).join(", ")}
                  className="rounded bg-black/45 px-1 text-white/90"
                >
                  fx{effects.length > 1 ? ` ${effects.length}` : ""}
                </span>
              )}
            </span>
          )}
        </div>
      )}

      {/* The effect range. Only the colour effects can be windowed, so this is
          drawn from the first one that has a window and covers the clip
          otherwise -- a bar across the whole clip would say nothing. */}
      {!sliver &&
        effects
          .filter((effect) => effect.start_ms != null || effect.end_ms != null)
          .slice(0, 1)
          .map((effect) => {
            const length = Math.max(1, clipDuration(clip));
            const from = (effect.start_ms ?? 0) / length;
            const to = (effect.end_ms ?? length) / length;
            return (
              <span
                key={effect.kind}
                aria-hidden
                className="pointer-events-none absolute bottom-0 h-[3px] rounded-full bg-info/80"
                style={{ left: `${from * 100}%`, width: `${Math.max(0, to - from) * 100}%` }}
              />
            );
          })}

      {/* Trim handles. Wide enough to hit, quiet until the clip is pointed at,
          and carrying a grip so that "this edge is draggable" is visible rather
          than something you have to already know. */}
      {(["in", "out"] as const).map((edge) => (
        <div
          key={edge}
          role="slider"
          tabIndex={-1}
          aria-label={t(edge === "in" ? "timeline.inPoint" : "timeline.outPoint", {
            index: clip.index + 1,
          })}
          aria-valuenow={edge === "in" ? clip.inMs : clip.outMs}
          aria-valuemin={0}
          aria-valuemax={asset?.duration_ms ?? clip.outMs}
          onPointerDown={(event) => {
            event.stopPropagation();
            onPointerDown(event, "trim", edge);
          }}
          style={{ width: HANDLE_WIDTH }}
          className={`absolute inset-y-0 flex cursor-ew-resize items-center justify-center bg-transparent transition-colors duration-fast hover:bg-accent-strong/70 ${
            edge === "in" ? "left-0 rounded-l-md" : "right-0 rounded-r-md"
          }`}
        >
          <span
            aria-hidden
            className={`h-4 w-[2px] rounded-full bg-white/80 transition-opacity duration-fast ${
              selected ? "opacity-60" : "opacity-0"
            } group-hover:opacity-80`}
          />
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ the bed
/**
 * The music bed, drawn on its own lane.
 *
 * Positioned by its own start rather than by a sequence, because that is what
 * it has: a bed is not the *n*th thing on a track, it is a block that begins at
 * a time. Beat markers are drawn inside it in *source* coordinates offset by
 * the trim, so a cue taken from 30 s into a track still shows that passage's
 * beats in the right places.
 */
function MusicBlock({
  bed,
  asset,
  pxPerSecond,
  beats,
  showBeats,
  selected,
  onSelect,
}: {
  bed: MusicBed;
  asset: MediaAsset | undefined;
  pxPerSecond: number;
  beats: readonly number[];
  showBeats: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const t = useT();
  const length = musicDuration(bed);
  const width = (length / 1000) * pxPerSecond;
  const left = (bed.startMs / 1000) * pxPerSecond;
  const name = asset?.original_filename ?? bed.mediaId.slice(0, 8);

  // Only the beats inside the trimmed passage, rebased so that the cue's own
  // start is x=0. Capped: a five-minute track at 200 BPM is a thousand markers,
  // and past a few hundred they are a grey wash rather than information.
  const visible = showBeats
    ? beats
        .filter((beat) => beat >= bed.inMs && beat < bed.outMs)
        .slice(0, 400)
        .map((beat) => ((beat - bed.inMs) / 1000) * pxPerSecond)
    : [];

  const fadeInPx = (bed.fadeInMs / 1000) * pxPerSecond;
  const fadeOutPx = (bed.fadeOutMs / 1000) * pxPerSecond;

  return (
    <div
      role="option"
      aria-selected={selected}
      aria-label={t("timeline.music.clip", { name, duration: timecode(length) })}
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      style={{ left, width: Math.max(width, 3) }}
      className={`absolute inset-y-[3px] cursor-default select-none overflow-hidden rounded-md shadow-raised transition-[box-shadow,background-color] duration-fast ${
        selected
          ? "bg-track-audio-selected ring-2 ring-inset ring-accent-strong"
          : "bg-track-audio ring-1 ring-inset ring-black/12 hover:shadow-panel"
      }`}
    >
      {/* Beat markers, behind the label. Hairlines rather than ticks: they are
          a rhythm to read at a glance, not values to measure against. */}
      {visible.map((x, index) => (
        <span
          key={index}
          aria-hidden
          className="absolute inset-y-1 w-px bg-black/25"
          style={{ left: x }}
        />
      ))}

      {/* The envelope, drawn as the ramp it is. */}
      {fadeInPx > 1 && (
        <span
          aria-hidden
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-ground/70 to-transparent"
          style={{ width: Math.min(fadeInPx, width) }}
        />
      )}
      {fadeOutPx > 1 && (
        <span
          aria-hidden
          className="absolute inset-y-0 right-0 bg-gradient-to-l from-ground/70 to-transparent"
          style={{ width: Math.min(fadeOutPx, width) }}
        />
      )}

      {width > 40 && (
        <span className="pointer-events-none absolute inset-y-0 left-2 flex items-center gap-1.5 truncate font-mono text-2xs text-fg/85">
          <Glyph name="audio" size={10} />
          {width > 90 && name}
        </span>
      )}
    </div>
  );
}


// ------------------------------------------------------------- transitions
/**
 * The mark between two clips.
 *
 * A crossfade is drawn as the *overlap it is* -- a hatched band spanning the
 * region where both clips play -- rather than as an icon sitting on the join.
 * The band is the edit: it is how much shorter the programme is, and how much
 * of each shot the viewer sees twice. A symbol would have looked tidier and
 * told the editor nothing about the length they had chosen.
 *
 * The fades are overlaid rather than overlapping. They consume no time, so they
 * get a gradient at the edge of the clip they belong to and nothing between.
 */
function TransitionMark({
  clip,
  pxPerSecond,
}: {
  clip: PlacedClip;
  pxPerSecond: number;
}) {
  const t = useT();
  const { kind, ms } = transitionOf(clip);
  if (kind === "cut" || ms <= 0) return null;

  const width = (ms / 1000) * pxPerSecond;
  const left = (clip.startMs / 1000) * pxPerSecond;

  if (kind === "crossfade") {
    return (
      <span
        aria-hidden
        title={t("transition.overlap", { ms })}
        className="vf-crossfade pointer-events-none absolute inset-y-[4px] z-10 rounded-sm ring-1 ring-inset ring-white/25"
        style={{ left, width: Math.max(width, 2) }}
      />
    );
  }

  // `fade_in` opens the clip; `fade_to_black` closes it.
  const closing = kind === "fade_to_black";
  return (
    <span
      aria-hidden
      title={t(closing ? "transition.fade_to_black" : "transition.fade_in")}
      className="pointer-events-none absolute inset-y-[4px] z-10 rounded-sm"
      style={{
        left: closing
          ? ((clip.endMs - ms) / 1000) * pxPerSecond
          : left,
        width: Math.max(width, 2),
        background: closing
          ? "linear-gradient(to right, transparent, rgba(0,0,0,0.85))"
          : "linear-gradient(to right, rgba(0,0,0,0.85), transparent)",
      }}
    />
  );
}

// --------------------------------------------------------------- subtitles
/** One cue, on the text lane. Positioned by its own timeline window. */
function CueBlock({
  cue,
  pxPerSecond,
  selected,
  onSelect,
}: {
  cue: SubtitleCueDraft;
  pxPerSecond: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const left = (cue.startMs / 1000) * pxPerSecond;
  const width = ((cue.endMs - cue.startMs) / 1000) * pxPerSecond;

  return (
    <button
      type="button"
      title={cue.text}
      aria-label={cue.text}
      onPointerDown={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      style={{ left, width: Math.max(width, 4) }}
      className={`absolute inset-y-[3px] flex items-center overflow-hidden rounded px-1.5 text-left transition-[background-color,box-shadow] duration-fast ${
        selected
          ? "bg-track-subtitle-selected ring-2 ring-inset ring-accent-strong"
          : "bg-track-subtitle ring-1 ring-inset ring-black/12 hover:shadow-panel"
      }`}
    >
      <span className="truncate text-2xs leading-none text-white/95">{cue.text}</span>
    </button>
  );
}

// --------------------------------------------------------------- track header
function TrackLabel({
  glyph,
  name,
  detail,
  height,
}: {
  glyph: "video" | "audio" | "layers";
  name: string;
  detail: string;
  height: string;
}) {
  return (
    <div
      style={{ height }}
      className="flex items-center gap-2 px-panel"
    >
      <span className="text-faint">
        <Glyph name={glyph} size={12} />
      </span>
      <span className="text-2xs font-medium text-muted">{name}</span>
      <span className="ml-auto truncate font-mono text-2xs tabular-nums text-faint">{detail}</span>
    </div>
  );
}

// --------------------------------------------------------------- the timeline
export function Timeline({
  projectId,
  media,
  invalidClipIds,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  invalidClipIds: Set<string>;
}) {
  const t = useT();
  const scrollRef = useRef<HTMLDivElement>(null);
  const laneRef = useRef<HTMLDivElement>(null);
  const subtitles = useEditorStore((s) => s.subtitles);
  const selectedCueId = useEditorStore((s) => s.selectedCueId);
  const selectCue = useEditorStore((s) => s.selectCue);
  const dragRef = useRef<Drag | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const [viewportWidth, setViewportWidth] = useState(0);

  const clips = useEditorStore((s) => s.clips);
  const music = useEditorStore((s) => s.music);
  const showBeatMarkers = useEditorStore((s) => s.showBeatMarkers);
  const setInspectorTab = useEditorStore((s) => s.setInspectorTab);
  const selectedClipId = useEditorStore((s) => s.selectedClipId);
  const selectClip = useEditorStore((s) => s.selectClip);
  const removeClip = useEditorStore((s) => s.removeClip);
  const moveClip = useEditorStore((s) => s.moveClip);
  const trim = useEditorStore((s) => s.trim);
  const split = useEditorStore((s) => s.splitAtPlayhead);
  const playheadMs = useEditorStore((s) => s.playheadMs);
  const setPlayhead = useEditorStore((s) => s.setPlayhead);
  const pxPerSecond = useEditorStore((s) => s.pxPerSecond);
  const setPxPerSecond = useEditorStore((s) => s.setPxPerSecond);
  const zoom = useEditorStore((s) => s.zoom);
  const dirty = useEditorStore((s) => s.dirty);

  // The bed's beats. Same query key as the audio panel's, so the two share one
  // request rather than each asking.
  const beatAnalysis = useQuery({
    queryKey: ["analysis", music?.mediaId],
    queryFn: () => api.mediaAnalysis(projectId, music!.mediaId),
    enabled: Boolean(music?.mediaId),
    staleTime: 60_000,
  });
  const beats = useMemo(() => {
    const record = beatAnalysis.data?.items.find((item) => item.analyzer === "beats");
    if (!record || record.status !== "ok") return [] as number[];
    return (record.payload as unknown as BeatsPayload).beats_ms ?? [];
  }, [beatAnalysis.data]);

  const placed = useMemo(() => place(clips), [clips]);
  const totalMs = placed.length > 0 ? placed[placed.length - 1].endMs : 0;

  /**
   * How wide the lanes are drawn.
   *
   * At least the width of the panel, so the ruler and the tracks reach the
   * right edge instead of stopping where the footage happens to end -- an
   * empty timeline that draws four seconds of ruler and then bare panel reads
   * as a rendering fault rather than as an empty sequence. Beyond that, the
   * length of the cut plus a little runway, so the last clip's out handle is
   * reachable and the playhead can sit at the very end.
   */
  const contentWidth = Math.max(
    ((totalMs + 4000) / 1000) * pxPerSecond,
    viewportWidth,
    400,
  );

  const msAtClientX = useCallback(
    (clientX: number): number => {
      const lane = laneRef.current;
      if (!lane) return 0;
      const x = clientX - lane.getBoundingClientRect().left;
      return Math.max(0, (x / pxPerSecond) * 1000);
    },
    [pxPerSecond],
  );

  // ----------------------------------------------------------------- dragging
  const onLanePointerDown = useCallback(
    (event: React.PointerEvent) => {
      // A click on empty lane moves the playhead and clears the selection.
      if (event.target === event.currentTarget) {
        selectClip(null);
        setPlayhead(Math.min(msAtClientX(event.clientX), totalMs));
        dragRef.current = { kind: "playhead" };
        (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
      }
    },
    [msAtClientX, selectClip, setPlayhead, totalMs],
  );

  const onClipPointerDown = useCallback(
    (
      event: React.PointerEvent,
      clip: PlacedClip,
      kind: "move" | "trim",
      edge?: "in" | "out",
    ) => {
      event.preventDefault();
      selectClip(clip.id);
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
      document.body.classList.add("vf-dragging");

      dragRef.current =
        kind === "trim"
          ? {
              kind: "trim",
              clipId: clip.id,
              edge: edge ?? "out",
              originMs: edge === "in" ? clip.inMs : clip.outMs,
              startX: event.clientX,
            }
          : {
              kind: "move",
              clipId: clip.id,
              index: clip.index,
              startX: event.clientX,
              moved: false,
            };
    },
    [selectClip],
  );

  useEffect(() => {
    function onMove(event: PointerEvent) {
      const drag = dragRef.current;
      if (!drag) return;

      if (drag.kind === "playhead") {
        setPlayhead(Math.min(msAtClientX(event.clientX), totalMs));
        return;
      }

      if (drag.kind === "trim") {
        const deltaMs = ((event.clientX - drag.startX) / pxPerSecond) * 1000;
        const clip = clips.find((item) => item.id === drag.clipId);
        if (!clip) return;
        const asset = media.get(clip.mediaId);
        trim(drag.clipId, drag.edge, drag.originMs + deltaMs, asset?.duration_ms ?? null);
        return;
      }

      // Moving: work out which slot the pointer is over. The clip lands where
      // the drop indicator says it will, which is the only honest way to show
      // a reorder that has no free positioning.
      if (Math.abs(event.clientX - drag.startX) > 4) drag.moved = true;
      const pointerMs = msAtClientX(event.clientX);
      let index = placed.length;
      for (const item of placed) {
        if (pointerMs < item.startMs + clipDuration(item) / 2) {
          index = item.index;
          break;
        }
      }
      setDropIndex(index);
    }

    function onUp() {
      const drag = dragRef.current;
      dragRef.current = null;
      document.body.classList.remove("vf-dragging");

      if (drag?.kind === "move" && drag.moved && dropIndex !== null) {
        // Removing the clip first shifts every later slot down by one.
        const target = dropIndex > drag.index ? dropIndex - 1 : dropIndex;
        moveClip(drag.index, target);
      }
      setDropIndex(null);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [clips, placed, dropIndex, media, msAtClientX, moveClip, pxPerSecond, setPlayhead, totalMs, trim]);

  /** Cues in time order, so the lane reads the way the list does. */
  const cues = useMemo(() => sortedCues(subtitles?.cues ?? []), [subtitles]);

  /** The lanes fill the panel, so their width depends on the panel's. */
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setViewportWidth(element.clientWidth));
    observer.observe(element);
    setViewportWidth(element.clientWidth);
    return () => observer.disconnect();
  }, []);

  /** Keep the playhead on screen while the program plays past the right edge. */
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const x = (playheadMs / 1000) * pxPerSecond;
    const left = element.scrollLeft;
    const right = left + element.clientWidth - GUTTER;
    if (x < left || x > right - 40) {
      element.scrollLeft = Math.max(0, x - element.clientWidth / 2);
    }
  }, [playheadMs, pxPerSecond]);

  /** Ctrl+wheel zooms about the pointer, as every timeline in this category does. */
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    function onWheel(event: WheelEvent) {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const state = useEditorStore.getState();
      state.zoom(event.deltaY < 0 ? 1.15 : 1 / 1.15);
    }

    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, []);

  /** Zoom so the whole cut is on screen. */
  const fitToWidth = useCallback(() => {
    const element = scrollRef.current;
    if (!element || totalMs <= 0) return;
    setPxPerSecond(((element.clientWidth - 24) / (totalMs + 1000)) * 1000);
  }, [setPxPerSecond, totalMs]);

  const selected = placed.find((clip) => clip.id === selectedClipId) ?? null;
  const selectedAsset = selected ? media.get(selected.mediaId) : undefined;

  // Two levels of tick: labelled majors, and unlabelled minors at a fifth of
  // the interval, dropped when they would be closer together than 7px. A ruler
  // with only one level reads as a grid rather than as a scale.
  const interval = tickInterval(pxPerSecond);
  const minorStep = interval / 5;
  const showMinor = minorStep * pxPerSecond >= 7;
  const ticks: number[] = [];
  for (let second = 0; second * pxPerSecond <= contentWidth; second += interval) {
    ticks.push(second);
  }

  const playheadX = (playheadMs / 1000) * pxPerSecond;

  return (
    <section
      className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl bg-surface shadow-panel"
      aria-label={t("timeline.title")}
    >
      {/* ---------------- toolbar ---------------- */}
      <header className="flex h-header shrink-0 items-center gap-1.5 px-panel">
        <h2 className="select-none text-xs font-semibold tracking-tight text-fg">
          {t("timeline.title")}
        </h2>
        {dirty && (
          <Badge tone="warn" title={t("timeline.editedHint")}>
            {t("timeline.edited")}
          </Badge>
        )}

        <Divider vertical />

        <IconButton
          label={t("timeline.split")}
          size="sm"
          disabled={clips.length === 0}
          onClick={split}
        >
          <Glyph name="split" size={12} />
        </IconButton>
        <IconButton
          label={t("timeline.delete")}
          size="sm"
          disabled={!selectedClipId}
          onClick={() => selectedClipId && removeClip(selectedClipId)}
        >
          <Glyph name="trash" size={12} />
        </IconButton>

        <Divider vertical />

        <IconButton label={t("timeline.zoomOut")} size="sm" onClick={() => zoom(1 / 1.4)}>
          <Glyph name="zoom-out" size={12} />
        </IconButton>
        <Slider
          min={MIN_PX_PER_SECOND}
          max={MAX_PX_PER_SECOND}
          step={1}
          value={pxPerSecond}
          aria-label={t("timeline.zoom")}
          onChange={(event) => setPxPerSecond(Number(event.target.value))}
          className="w-20"
        />
        <IconButton label={t("timeline.zoomIn")} size="sm" onClick={() => zoom(1.4)}>
          <Glyph name="zoom-in" size={12} />
        </IconButton>
        <IconButton
          label={t("timeline.zoomFit")}
          size="sm"
          disabled={totalMs === 0}
          onClick={fitToWidth}
        >
          <Glyph name="fit" size={12} />
        </IconButton>

        {/* The current time, at the size an editor actually reads it. */}
        <span className="ml-auto flex items-center gap-2.5 font-mono text-2xs tabular-nums text-faint">
          <span className="rounded-md bg-sunken/70 px-2 py-0.5 text-sm text-fg">
            {timecode(playheadMs)}
          </span>
          <span>/ {timecode(totalMs)}</span>
          <span>{t.plural("timeline.clipCount", clips.length)}</span>
        </span>
      </header>

      {/* ---------------- tracks ---------------- */}
      <div className="flex min-h-0 flex-1">
        {/* Track headers stay put while the lanes scroll. */}
        <div className="shrink-0 bg-surface" style={{ width: GUTTER }}>
          <div
            style={{ height: RULER_HEIGHT }}
            className="flex items-center px-panel font-mono text-2xs uppercase tracking-wider text-faint"
          >
            {/* The ruler's own column heading: what the numbers to the right
                are. Better than an empty box the eye has to explain to itself. */}
            TC
          </div>
          <TrackLabel
            glyph="video"
            name={t("timeline.track.videoShort")}
            detail={clips.length > 0 ? String(clips.length) : ""}
            height="var(--h-track)"
          />
          <TrackLabel
            glyph="audio"
            name={t("timeline.track.audioShort")}
            detail=""
            height="var(--h-track-audio)"
          />
          <TrackLabel
            glyph="audio"
            name={t("timeline.track.musicShort")}
            detail={music ? timecode(musicDuration(music), false) : ""}
            height="var(--h-track-audio)"
          />
          <TrackLabel
            glyph="layers"
            name={t("timeline.track.textShort")}
            detail={cues.length ? String(cues.length) : ""}
            height="var(--h-track-text)"
          />
        </div>

        <div ref={scrollRef} className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
          <div ref={laneRef} className="relative" style={{ width: contentWidth }}>
            {/* ---- ruler ---- */}
            <div
              role="slider"
              tabIndex={0}
              aria-label={t("timeline.playhead")}
              aria-valuenow={Math.round(playheadMs)}
              aria-valuemin={0}
              aria-valuemax={Math.round(totalMs)}
              aria-valuetext={timecode(playheadMs)}
              onPointerDown={(event) => {
                setPlayhead(Math.min(msAtClientX(event.clientX), totalMs));
                dragRef.current = { kind: "playhead" };
                (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
              }}
              onKeyDown={(event) => {
                const step = event.shiftKey ? 1000 : 100;
                if (event.key === "ArrowLeft") setPlayhead(Math.max(0, playheadMs - step));
                if (event.key === "ArrowRight") setPlayhead(Math.min(totalMs, playheadMs + step));
                if (event.key === "Home") setPlayhead(0);
                if (event.key === "End") setPlayhead(totalMs);
              }}
              style={{ height: RULER_HEIGHT }}
              className="relative cursor-pointer select-none bg-ruler"
            >
              {ticks.map((second) => (
                <div key={second}>
                  <span
                    aria-hidden
                    className="absolute bottom-0 top-1.5 w-px bg-strong/70"
                    style={{ left: second * pxPerSecond }}
                  />
                  <span
                    className="absolute top-[4px] ml-1.5 font-mono text-2xs tabular-nums text-faint"
                    style={{ left: second * pxPerSecond }}
                  >
                    {timecode(second * 1000, false)}
                  </span>
                  {showMinor &&
                    [1, 2, 3, 4].map((n) => (
                      <span
                        key={n}
                        aria-hidden
                        className="absolute bottom-0 h-[5px] w-px bg-strong/50"
                        style={{ left: (second + n * minorStep) * pxPerSecond }}
                      />
                    ))}
                </div>
              ))}
            </div>

            {/* ---- video track ---- */}
            <div
              role="listbox"
              aria-label={t("timeline.track.video")}
              onPointerDown={onLanePointerDown}
              style={{ height: "var(--h-track)" }}
              className="relative bg-lane"
            >
              {placed.map((clip) => (
                <Clip
                  key={clip.id}
                  clip={clip}
                  asset={media.get(clip.mediaId)}
                  projectId={projectId}
                  pxPerSecond={pxPerSecond}
                  selected={clip.id === selectedClipId}
                  invalid={invalidClipIds.has(clip.id)}
                  onPointerDown={(event, kind, edge) =>
                    onClipPointerDown(event, clip, kind, edge)
                  }
                />
              ))}

              {/* Drawn after the clips, so a crossfade band sits over both
                  sides of the join it describes rather than under one. */}
              {placed.map((clip) => (
                <TransitionMark key={`fx-${clip.id}`} clip={clip} pxPerSecond={pxPerSecond} />
              ))}

              {dropIndex !== null && (
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-y-1 z-20 w-[3px] rounded-full bg-accent-strong shadow-raised"
                  style={{
                    left:
                      ((dropIndex < placed.length ? placed[dropIndex].startMs : totalMs) / 1000) *
                      pxPerSecond,
                  }}
                />
              )}

              {clips.length === 0 && (
                <p className="pointer-events-none absolute inset-0 flex items-center pl-3 text-xs text-faint">
                  {t("timeline.empty")}
                </p>
              )}
            </div>

            {/* ---- audio track ----
                A representation, not an editable track: Phase 6 renders source
                audio with its clip or renders none, and showing a lane that
                could be cut separately would promise an edit the plan has no
                way to describe. A clip whose source is silent is drawn as an
                outline, so "no audio here" and "audio not drawn yet" are not
                the same picture. */}
            <div
              aria-label={t("timeline.track.audio")}
              style={{ height: "var(--h-track-audio)" }}
              className="relative bg-lane/60"
            >
              {placed.map((clip) => {
                const asset = media.get(clip.mediaId);
                const hasAudio = Boolean(asset?.channels);
                return (
                  <div
                    key={clip.id}
                    title={
                      hasAudio
                        ? t("timeline.audioChannels", { count: asset?.channels ?? 0 })
                        : t("timeline.noAudio")
                    }
                    style={{
                      left: (clip.startMs / 1000) * pxPerSecond,
                      width: Math.max((clipPlaybackMs(clip) / 1000) * pxPerSecond, 3),
                    }}
                    className={`absolute inset-y-[3px] overflow-hidden rounded ${
                      hasAudio
                        ? "bg-track-audio/80 ring-1 ring-inset ring-black/10"
                        : "border border-dashed border-strong/50 bg-transparent"
                    }`}
                  >
                    {hasAudio && (
                      <span
                        aria-hidden
                        className="absolute inset-x-1 top-1/2 h-px -translate-y-1/2 bg-success/50"
                      />
                    )}
                  </div>
                );
              })}
            </div>

            {/* ---- music track ----
                Its own lane rather than a second audio row: a bed has a start
                and an envelope that the clips' audio does not, and drawing
                them on one lane would mean one of the two lying about what it
                is. */}
            <div
              aria-label={t("timeline.track.music")}
              style={{ height: "var(--h-track-audio)" }}
              className="relative bg-lane/50"
            >
              {music ? (
                <MusicBlock
                  bed={music}
                  asset={media.get(music.mediaId)}
                  pxPerSecond={pxPerSecond}
                  beats={beats}
                  showBeats={showBeatMarkers}
                  selected={false}
                  onSelect={() => setInspectorTab("audio")}
                />
              ) : (
                clips.length > 0 && (
                  <p className="pointer-events-none absolute inset-0 flex items-center pl-3 text-2xs text-faint">
                    {t("timeline.music.empty")}
                  </p>
                )
              )}
            </div>

            {/* ---- text track ----
                Cue positions are timeline positions, which is what the plan
                stores: a cue belongs to the programme, not to whichever shot is
                under it. Drawing it against the clips would make it appear to
                move when an earlier crossfade was lengthened, which is the one
                thing that does *not* happen. */}
            <div
              aria-label={t("timeline.track.text")}
              style={{ height: "var(--h-track-text)" }}
              className="relative bg-lane/40"
            >
              {cues.map((cue) => (
                <CueBlock
                  key={cue.id}
                  cue={cue}
                  pxPerSecond={pxPerSecond}
                  selected={cue.id === selectedCueId}
                  onSelect={() => {
                    selectCue(cue.id);
                    setInspectorTab("subtitles");
                  }}
                />
              ))}
              {cues.length === 0 && clips.length > 0 && (
                <p className="pointer-events-none absolute inset-0 flex items-center pl-3 text-2xs text-faint">
                  {t("timeline.text.empty")}
                </p>
              )}
            </div>

            {/* ---- playhead ----
                Its own colour, not the accent. The accent means "selected", and
                an editor has to be able to tell the clip it is holding from the
                frame it is looking at. */}
            <div
              aria-hidden
              className="pointer-events-none absolute bottom-0 top-0 z-30 w-[2px] -translate-x-[0.5px] bg-playhead"
              style={{ left: playheadX }}
            >
              {/* A rounded head rather than a triangle: it reads as a handle,
                  and it is the one shape in the timeline allowed to be sharp
                  about where it is. */}
              <span className="absolute -left-[5px] top-0 h-3 w-3 rounded-b-full rounded-t-sm bg-playhead shadow-raised" />
            </div>
          </div>
        </div>
      </div>

      {/* ---------------- selected clip readout ---------------- */}
      <footer className="flex h-row shrink-0 items-center gap-3 px-panel pb-1 font-mono text-2xs tabular-nums text-faint">
        {selected ? (
          <>
            <span className="truncate text-muted">
              {selectedAsset?.original_filename ?? selected.mediaId.slice(0, 8)}
            </span>
            <span>
              {t("timeline.in")} {timecode(selected.inMs)}
            </span>
            <span>
              {t("timeline.out")} {timecode(selected.outMs)}
            </span>
            <span className="text-fg" title={t("timeline.duration")}>
              {timecode(clipPlaybackMs(selected))}
            </span>
            {transitionOf(selected).kind !== "cut" && (
              <span title={t("timeline.transitionLabel")}>
                {t(
                  transitionOf(selected).kind === "crossfade"
                    ? "transition.crossfade"
                    : transitionOf(selected).kind === "fade_in"
                      ? "transition.fade_in"
                      : "transition.fade_to_black",
                )}{" "}
                {transitionOf(selected).ms} ms
              </span>
            )}
            <span className="ml-auto hidden truncate 2xl:inline">{t("timeline.hint")}</span>
          </>
        ) : (
          <span className="ml-auto truncate">
            {t("timeline.limits", { min: MIN_CLIP_MS, max: MAX_CLIP_MS / 1000 })}
          </span>
        )}
      </footer>
    </section>
  );
}
