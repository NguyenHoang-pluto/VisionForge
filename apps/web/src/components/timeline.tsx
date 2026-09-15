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
  musicDuration,
  place,
  type MusicBed,
  type PlacedClip,
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
 * One video track and one audio track, which is what the EditPlan schema
 * describes: a sequence of trims with no gaps, position implied by order. The
 * timeline draws exactly that and nothing more -- no gaps to drag into, no
 * overlaps to create, because the plan has no way to express either and a UI
 * that let you build one would be offering an edit the renderer must reject.
 *
 * For the same reason there is no snapping indicator, no magnet toggle and no
 * ripple mode. Cuts are butt-joined by construction, so there is nothing for a
 * clip to snap *to*; a magnet button here would be a light that is always on.
 *
 * Interaction is pointer-event based with pointer capture, so a drag that
 * leaves the element still tracks, and one that ends outside still commits.
 */

const RULER_HEIGHT = 22;
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
  const width = (clipDuration(clip) / 1000) * pxPerSecond;
  const left = (clip.startMs / 1000) * pxPerSecond;
  const name = asset?.original_filename ?? clip.mediaId.slice(0, 8);

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
      className={`group absolute inset-y-[3px] select-none overflow-hidden rounded-sm border transition-colors ${
        invalid
          ? "border-danger bg-danger/25"
          : selected
            ? "border-accent-strong bg-track-video-selected ring-1 ring-inset ring-accent-strong/60"
            : "border-strong bg-track-video hover:border-strong"
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
          <span className="truncate rounded-sm bg-black/35 px-1 text-2xs font-medium leading-[13px] text-white/95">
            {compact ? clip.index + 1 : name}
          </span>
          {!compact && (
            <span className="truncate font-mono text-2xs tabular-nums text-white/70">
              {timecode(clipDuration(clip), false)}
            </span>
          )}
        </div>
      )}

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
          className={`absolute inset-y-0 flex cursor-ew-resize items-center justify-center bg-transparent transition-colors hover:bg-accent-strong/80 ${
            edge === "in" ? "left-0" : "right-0"
          }`}
        >
          <span
            aria-hidden
            className={`h-3 w-[2px] rounded-full bg-white/70 transition-opacity ${
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
      className={`absolute inset-y-[2px] cursor-default select-none overflow-hidden rounded-sm border transition-colors ${
        selected
          ? "border-accent-strong bg-track-audio ring-1 ring-inset ring-accent-strong/60"
          : "border-strong/70 bg-track-audio hover:border-strong"
      }`}
    >
      {/* Beat markers, behind the label. Hairlines rather than ticks: they are
          a rhythm to read at a glance, not values to measure against. */}
      {visible.map((x, index) => (
        <span
          key={index}
          aria-hidden
          className="absolute inset-y-0 w-px bg-fg/25"
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
        <span className="pointer-events-none absolute inset-y-0 left-1 flex items-center gap-1 truncate font-mono text-2xs text-fg/85">
          <Glyph name="audio" size={9} />
          {width > 90 && name}
        </span>
      )}
    </div>
  );
}


// --------------------------------------------------------------- track header
function TrackLabel({
  glyph,
  name,
  detail,
  height,
}: {
  glyph: "video" | "audio";
  name: string;
  detail: string;
  height: string;
}) {
  return (
    <div
      style={{ height }}
      className="flex items-center gap-1.5 border-b border-subtle px-2 last:border-b-0"
    >
      <span className="text-faint">
        <Glyph name={glyph} size={11} />
      </span>
      <span className="font-mono text-2xs font-medium text-muted">{name}</span>
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
    <section className="flex min-h-0 flex-1 flex-col bg-surface" aria-label={t("timeline.title")}>
      {/* ---------------- toolbar ---------------- */}
      <header className="flex h-header shrink-0 items-center gap-1.5 border-b border-subtle bg-elevated px-2">
        <h2 className="select-none font-mono text-2xs font-medium uppercase tracking-wider text-faint">
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
          <span className="text-sm text-fg">{timecode(playheadMs)}</span>
          <span>/ {timecode(totalMs)}</span>
          <span className="border-l border-subtle pl-2.5">
            {t.plural("timeline.clipCount", clips.length)}
          </span>
        </span>
      </header>

      {/* ---------------- tracks ---------------- */}
      <div className="flex min-h-0 flex-1">
        {/* Track headers stay put while the lanes scroll. */}
        <div className="shrink-0 border-r border-subtle bg-elevated" style={{ width: GUTTER }}>
          <div
            style={{ height: RULER_HEIGHT }}
            className="flex items-center border-b border-subtle px-2 font-mono text-2xs uppercase tracking-wider text-faint"
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
              className="relative cursor-pointer select-none border-b border-subtle bg-ruler"
            >
              {ticks.map((second) => (
                <div key={second}>
                  <span
                    aria-hidden
                    className="absolute bottom-0 top-0 w-px bg-strong"
                    style={{ left: second * pxPerSecond }}
                  />
                  <span
                    className="absolute top-[3px] ml-1 font-mono text-2xs tabular-nums text-faint"
                    style={{ left: second * pxPerSecond }}
                  >
                    {timecode(second * 1000, false)}
                  </span>
                  {showMinor &&
                    [1, 2, 3, 4].map((n) => (
                      <span
                        key={n}
                        aria-hidden
                        className="absolute bottom-0 h-[5px] w-px bg-strong/60"
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
              className="relative border-b border-subtle bg-lane"
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

              {dropIndex !== null && (
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-y-0 z-20 w-0.5 bg-accent-strong"
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
                      width: Math.max((clipDuration(clip) / 1000) * pxPerSecond, 3),
                    }}
                    className={`absolute inset-y-[2px] overflow-hidden rounded-sm border ${
                      hasAudio
                        ? "border-strong/70 bg-track-audio"
                        : "border-dashed border-strong/50 bg-transparent"
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
              className="relative border-t border-subtle bg-lane/40"
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

            {/* ---- playhead ----
                Its own colour, not the accent. The accent means "selected", and
                an editor has to be able to tell the clip it is holding from the
                frame it is looking at. */}
            <div
              aria-hidden
              className="pointer-events-none absolute bottom-0 top-0 z-30 w-px bg-playhead"
              style={{ left: playheadX }}
            >
              <span className="absolute -left-[5px] top-0 h-0 w-0 border-x-[5px] border-t-[7px] border-x-transparent border-t-playhead" />
            </div>
          </div>
        </div>
      </div>

      {/* ---------------- selected clip readout ---------------- */}
      <footer className="flex h-row shrink-0 items-center gap-3 border-t border-subtle bg-elevated px-2 font-mono text-2xs tabular-nums text-faint">
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
              {timecode(clipDuration(selected))}
            </span>
            <span className="ml-auto hidden truncate xl:inline">{t("timeline.hint")}</span>
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
