"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { MediaAsset } from "@/lib/api";
import { timecode } from "@/lib/format";
import { useThumbnailUrl } from "@/lib/media-urls";
import {
  MAX_CLIP_MS,
  MIN_CLIP_MS,
  clipDuration,
  place,
  type PlacedClip,
} from "@/lib/timeline";
import {
  MAX_PX_PER_SECOND,
  MIN_PX_PER_SECOND,
  useEditorStore,
} from "@/stores/editor-store";
import { Badge, Divider, Glyph, IconButton } from "@/components/ui";

/**
 * The timeline.
 *
 * One video track and one audio track, which is what the EditPlan schema
 * describes: a sequence of trims with no gaps, position implied by order. The
 * timeline draws exactly that and nothing more -- no gaps to drag into, no
 * overlaps to create, because the plan has no way to express either and a UI
 * that let you build one would be offering an edit the renderer must reject.
 *
 * Interaction is pointer-event based with pointer capture, so a drag that
 * leaves the element still tracks, and one that ends outside still commits.
 */

const RULER_HEIGHT = 20;
const TRACK_HEIGHT = 46;
const AUDIO_HEIGHT = 26;
const GUTTER = 44;
const HANDLE_WIDTH = 7;

/** Tick spacing, chosen so labels never collide at any zoom. */
function tickInterval(pxPerSecond: number): number {
  const candidates = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  const minimumPx = 56;
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
  const thumbnail = useThumbnailUrl(projectId, asset ?? null);
  const width = (clipDuration(clip) / 1000) * pxPerSecond;
  const left = (clip.startMs / 1000) * pxPerSecond;
  const compact = width < 56;

  return (
    <div
      role="option"
      aria-selected={selected}
      aria-label={`Clip ${clip.index + 1}: ${asset?.original_filename ?? clip.mediaId} , ${timecode(clipDuration(clip))}`}
      tabIndex={0}
      onPointerDown={(event) => onPointerDown(event, "move")}
      style={{ left, width: Math.max(width, 3), height: TRACK_HEIGHT - 6 }}
      className={`absolute top-[3px] select-none overflow-hidden rounded border ${
        invalid
          ? "border-danger bg-danger/20"
          : selected
            ? "border-accent-strong bg-track-video-selected"
            : "border-line-strong bg-track-video hover:border-line-strong/90"
      }`}
    >
      {thumbnail.data?.url && width > 26 && (
        /* eslint-disable-next-line @next/next/no-img-element -- presigned URL */
        <img
          src={thumbnail.data.url}
          alt=""
          loading="lazy"
          draggable={false}
          className="pointer-events-none absolute inset-y-0 left-0 h-full w-[34px] object-cover opacity-55"
        />
      )}

      <div
        className="pointer-events-none absolute inset-0 flex flex-col justify-between px-1 py-0.5"
        style={{ paddingLeft: thumbnail.data?.url && width > 26 ? 37 : 5 }}
      >
        <span className="truncate text-2xs text-fg/90">
          {compact ? clip.index + 1 : (asset?.original_filename ?? clip.mediaId.slice(0, 8))}
        </span>
        {!compact && (
          <span className="truncate font-mono text-2xs text-fg/60 tabular-nums">
            {timecode(clipDuration(clip), false)} · in {timecode(clip.inMs, false)}
          </span>
        )}
      </div>

      {/* Trim handles. Wide enough to hit, quiet enough to ignore. */}
      {(["in", "out"] as const).map((edge) => (
        <div
          key={edge}
          role="slider"
          tabIndex={-1}
          aria-label={`${edge === "in" ? "In" : "Out"} point of clip ${clip.index + 1}`}
          aria-valuenow={edge === "in" ? clip.inMs : clip.outMs}
          aria-valuemin={0}
          aria-valuemax={asset?.duration_ms ?? clip.outMs}
          onPointerDown={(event) => {
            event.stopPropagation();
            onPointerDown(event, "trim", edge);
          }}
          style={{ width: HANDLE_WIDTH }}
          className={`absolute inset-y-0 cursor-ew-resize bg-accent-strong/0 transition-colors hover:bg-accent-strong/70 ${
            edge === "in" ? "left-0" : "right-0"
          }`}
        />
      ))}
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
  const scrollRef = useRef<HTMLDivElement>(null);
  const laneRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);

  const clips = useEditorStore((s) => s.clips);
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

  const placed = useMemo(() => place(clips), [clips]);
  const totalMs = placed.length > 0 ? placed[placed.length - 1].endMs : 0;
  // Always a little runway past the end, so the last clip's out handle is
  // reachable and the playhead can sit at the very end.
  const contentWidth = Math.max(((totalMs + 4000) / 1000) * pxPerSecond, 400);

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
          : { kind: "move", clipId: clip.id, index: clip.index, startX: event.clientX, moved: false };
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

  const selected = placed.find((clip) => clip.id === selectedClipId) ?? null;
  const ticks: number[] = [];
  const interval = tickInterval(pxPerSecond);
  for (let t = 0; t * 1000 <= totalMs + 4000; t += interval) ticks.push(t);

  return (
    <section className="flex min-h-0 flex-1 flex-col bg-panel" aria-label="Timeline">
      {/* ---------------- toolbar ---------------- */}
      <header className="flex h-[30px] shrink-0 items-center gap-1.5 border-b border-line bg-raised px-2">
        <h2 className="select-none font-mono text-2xs uppercase tracking-wider text-dim">
          Timeline
        </h2>
        {dirty && <Badge tone="warn" title="Edited since it was last stored">edited</Badge>}

        <Divider vertical />

        <IconButton
          label="Split clip at playhead (S)"
          disabled={clips.length === 0}
          onClick={split}
        >
          <Glyph name="split" />
        </IconButton>
        <IconButton
          label="Delete selected clip (Del)"
          disabled={!selectedClipId}
          onClick={() => selectedClipId && removeClip(selectedClipId)}
        >
          <Glyph name="trash" />
        </IconButton>

        <Divider vertical />

        <IconButton label="Zoom out (-)" onClick={() => zoom(1 / 1.4)}>
          <Glyph name="zoom-out" />
        </IconButton>
        <input
          type="range"
          min={MIN_PX_PER_SECOND}
          max={MAX_PX_PER_SECOND}
          step={1}
          value={pxPerSecond}
          aria-label="Timeline zoom"
          onChange={(event) => setPxPerSecond(Number(event.target.value))}
          className="h-1 w-20 cursor-pointer appearance-none rounded bg-control accent-[var(--accent)]"
        />
        <IconButton label="Zoom in (+)" onClick={() => zoom(1.4)}>
          <Glyph name="zoom-in" />
        </IconButton>

        <span className="ml-auto flex items-center gap-3 font-mono text-2xs text-dim tabular-nums">
          <span>
            <span className="text-fg">{timecode(playheadMs)}</span> / {timecode(totalMs)}
          </span>
          <span>
            {clips.length} clip{clips.length === 1 ? "" : "s"}
          </span>
          {selected && (
            <span className="text-muted">
              #{selected.index + 1} · {timecode(clipDuration(selected), false)}
            </span>
          )}
        </span>
      </header>

      {/* ---------------- tracks ---------------- */}
      <div className="flex min-h-0 flex-1">
        {/* Track headers stay put while the lanes scroll. */}
        <div
          className="shrink-0 border-r border-line bg-raised"
          style={{ width: GUTTER }}
          aria-hidden
        >
          <div style={{ height: RULER_HEIGHT }} className="border-b border-line" />
          <div
            style={{ height: TRACK_HEIGHT }}
            className="flex items-center justify-center border-b border-line font-mono text-2xs text-dim"
          >
            V1
          </div>
          <div
            style={{ height: AUDIO_HEIGHT }}
            className="flex items-center justify-center font-mono text-2xs text-dim"
          >
            A1
          </div>
        </div>

        <div ref={scrollRef} className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
          <div ref={laneRef} className="relative" style={{ width: contentWidth }}>
            {/* ---- ruler ---- */}
            <div
              role="slider"
              tabIndex={0}
              aria-label="Playhead"
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
              className="relative cursor-pointer border-b border-line bg-ruler"
            >
              {ticks.map((second) => (
                <div
                  key={second}
                  className="absolute bottom-0 top-0 border-l border-line-strong/60"
                  style={{ left: second * pxPerSecond }}
                >
                  <span className="ml-1 select-none font-mono text-2xs text-dim tabular-nums">
                    {timecode(second * 1000, false)}
                  </span>
                </div>
              ))}
            </div>

            {/* ---- video track ---- */}
            <div
              role="listbox"
              aria-label="Video track V1"
              onPointerDown={onLanePointerDown}
              style={{ height: TRACK_HEIGHT }}
              className="relative border-b border-line bg-ground/50"
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
                  className="pointer-events-none absolute inset-y-0 w-0.5 bg-accent-strong"
                  style={{
                    left:
                      ((dropIndex < placed.length
                        ? placed[dropIndex].startMs
                        : totalMs) /
                        1000) *
                      pxPerSecond,
                  }}
                />
              )}

              {clips.length === 0 && (
                <p className="pointer-events-none absolute inset-0 flex items-center pl-3 text-xs text-dim">
                  Empty. Generate an edit, or select clips in the media browser and press
                  “To timeline”.
                </p>
              )}
            </div>

            {/* ---- audio track ----
                A representation, not an editable track: Phase 6 renders source
                audio with its clip or renders none, and showing a lane that
                cannot be cut separately would promise an edit the plan has no
                way to describe. */}
            <div
              aria-label="Audio track A1"
              style={{ height: AUDIO_HEIGHT }}
              className="relative bg-ground/30"
            >
              {placed.map((clip) => {
                const asset = media.get(clip.mediaId);
                const hasAudio = Boolean(asset?.channels);
                return (
                  <div
                    key={clip.id}
                    title={
                      hasAudio
                        ? `${asset?.channels} channel(s)`
                        : "This source has no audio stream"
                    }
                    style={{
                      left: (clip.startMs / 1000) * pxPerSecond,
                      width: Math.max((clipDuration(clip) / 1000) * pxPerSecond, 3),
                    }}
                    className={`absolute inset-y-[3px] overflow-hidden rounded-sm border ${
                      hasAudio
                        ? "border-line-strong/70 bg-track-audio"
                        : "border-dashed border-line-strong/40 bg-transparent"
                    }`}
                  >
                    {hasAudio && (
                      <span
                        aria-hidden
                        className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-ok/40"
                      />
                    )}
                  </div>
                );
              })}
            </div>

            {/* ---- playhead ---- */}
            <div
              aria-hidden
              className="pointer-events-none absolute bottom-0 top-0 z-10 w-px bg-playhead"
              style={{ left: (playheadMs / 1000) * pxPerSecond }}
            >
              <span className="absolute -left-[4px] top-0 h-0 w-0 border-x-[4px] border-t-[6px] border-x-transparent border-t-playhead" />
            </div>
          </div>
        </div>
      </div>

      {/* ---------------- selected clip readout ---------------- */}
      {selected && (
        <footer className="flex h-[22px] shrink-0 items-center gap-3 border-t border-line bg-raised px-2 font-mono text-2xs text-dim tabular-nums">
          <span className="truncate text-muted">
            {media.get(selected.mediaId)?.original_filename ?? selected.mediaId.slice(0, 8)}
          </span>
          <span>in {timecode(selected.inMs)}</span>
          <span>out {timecode(selected.outMs)}</span>
          <span>dur {timecode(clipDuration(selected))}</span>
          <span className="ml-auto text-dim/70">
            drag body to reorder · drag edges to trim · {MIN_CLIP_MS}–{MAX_CLIP_MS / 1000}s
          </span>
        </footer>
      )}
    </section>
  );
}
