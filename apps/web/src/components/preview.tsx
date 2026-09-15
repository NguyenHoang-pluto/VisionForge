"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AspectRatio, MediaAsset, Render } from "@/lib/api";
import { timecode } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { useProxyUrl, useProxyUrls } from "@/lib/media-urls";
import { clipAt, clipDuration, place } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import { Badge, Divider, Glyph, IconButton, SegmentedControl, Slider } from "@/components/ui";

/**
 * The preview viewer.
 *
 * One `<video>` element serves three sources, because the alternative -- three
 * players, one per mode -- means three decoders resident at once on a machine
 * that is also running FFmpeg.
 *
 *   source   one clip from the library, scrubbed freely
 *   program  the timeline, played through by sequencing the proxies
 *   render   the finished MP4, once one exists
 *
 * Program playback is the interesting one. There is no client-side compositor
 * and there deliberately is not going to be: the timeline is a sequence of cuts
 * into sources, so playing it is a matter of seeking the right proxy to the
 * right offset and swapping at each boundary. What you see is what the render
 * will contain, frame accuracy aside, and nothing about the preview constitutes
 * a second rendering model the backend would have to agree with.
 */

const ASPECT_CLASS: Record<AspectRatio, string> = {
  "16:9": "aspect-video",
  "9:16": "aspect-[9/16]",
  "1:1": "aspect-square",
};

const SPEEDS = [0.5, 1, 1.5, 2] as const;

/** How close to a clip's out point counts as reaching it. One frame at 30fps. */
const BOUNDARY_EPSILON_MS = 34;

export function Preview({
  projectId,
  media,
  render,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  render: Render | null;
}) {
  const t = useT();
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  const source = useEditorStore((s) => s.previewSource);
  const setSource = useEditorStore((s) => s.setPreviewSource);
  const clips = useEditorStore((s) => s.clips);
  const playing = useEditorStore((s) => s.playing);
  const setPlaying = useEditorStore((s) => s.setPlaying);
  const playheadMs = useEditorStore((s) => s.playheadMs);
  const setPlayhead = useEditorStore((s) => s.setPlayhead);
  const activeMediaId = useEditorStore((s) => s.activeMediaId);
  const aspect = useEditorStore((s) => s.aspect);

  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [speed, setSpeed] = useState<number>(1);
  const [sourceTimeMs, setSourceTimeMs] = useState(0);
  const [stalled, setStalled] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  const activeAsset = activeMediaId ? (media.get(activeMediaId) ?? null) : null;
  const sourceProxy = useProxyUrl(projectId, activeAsset);

  const placed = useMemo(() => place(clips), [clips]);
  const timelineMs = placed.length > 0 ? placed[placed.length - 1].endMs : 0;
  const mediaIds = useMemo(() => clips.map((clip) => clip.mediaId), [clips]);
  const programUrls = useProxyUrls(projectId, source === "program" ? mediaIds : []);

  const current = source === "program" ? clipAt(clips, Math.min(playheadMs, timelineMs)) : null;

  // What the element should be playing right now.
  const wantedUrl =
    source === "render"
      ? (render?.playback_url ?? null)
      : source === "source"
        ? (sourceProxy.data?.url ?? null)
        : current
          ? (programUrls.get(current.clip.mediaId) ?? null)
          : null;

  const durationMs =
    source === "program"
      ? timelineMs
      : source === "render"
        ? (render?.duration_ms ?? 0)
        : (activeAsset?.duration_ms ?? 0);

  const positionMs = source === "program" ? Math.min(playheadMs, timelineMs) : sourceTimeMs;

  // ---------------------------------------------------------------- transport
  const seek = useCallback(
    (ms: number) => {
      const clamped = Math.max(0, Math.min(ms, durationMs));
      if (source === "program") {
        setPlayhead(clamped);
      } else {
        setSourceTimeMs(clamped);
        const element = videoRef.current;
        if (element) element.currentTime = clamped / 1000;
      }
    },
    [durationMs, source, setPlayhead],
  );

  /**
   * Load the right media and put the head in the right place.
   *
   * Keyed on the URL so that a program cut swaps `src` exactly once, and a
   * re-render that changes nothing does not reload the decoder mid-playback.
   */
  useEffect(() => {
    const element = videoRef.current;
    if (!element || !wantedUrl) return;

    if (element.getAttribute("data-src") !== wantedUrl) {
      element.setAttribute("data-src", wantedUrl);
      element.src = wantedUrl;
      element.load();
    }
  }, [wantedUrl]);

  /** In program mode the element's time is derived from the playhead. */
  useEffect(() => {
    const element = videoRef.current;
    if (!element || source !== "program" || !current) return;

    const target = (current.clip.inMs + current.offsetMs) / 1000;
    // Only correct real drift. Writing currentTime every frame fights the
    // decoder and makes playback stutter.
    if (Math.abs(element.currentTime - target) > 0.25) {
      element.currentTime = target;
    }
  }, [source, current]);

  useEffect(() => {
    const element = videoRef.current;
    if (!element) return;
    element.muted = muted;
    element.volume = volume;
    element.playbackRate = speed;
  }, [muted, volume, speed]);

  useEffect(() => {
    const element = videoRef.current;
    if (!element) return;

    if (playing) {
      // A play() rejection is normal (autoplay policy, src swapped mid-call);
      // it must not leave the UI claiming to be playing.
      void element.play().catch(() => setPlaying(false));
    } else {
      element.pause();
    }
  }, [playing, wantedUrl, setPlaying]);

  /**
   * Advance the playhead from the element's own clock.
   *
   * Driven by `requestAnimationFrame` rather than
   * `requestVideoFrameCallback`. rVFC is the more precise of the two -- it
   * fires on presented frames -- but it fires *only* on presented frames, and
   * a cut swaps `src`, during which none are presented. The loop would die at
   * the first clip boundary and playback would stop one clip in. rAF keeps
   * ticking across the swap, and reading `currentTime` once a frame is as
   * accurate as a playhead needs to be.
   */
  useEffect(() => {
    const element = videoRef.current;
    if (!element || source !== "program" || !playing) return;

    let frame = 0;

    const tick = () => {
      const store = useEditorStore.getState();
      const at = clipAt(store.clips, store.playheadMs);
      if (!at) {
        store.setPlaying(false);
        return;
      }

      const elapsed = element.currentTime * 1000 - at.clip.inMs;
      const length = clipDuration(at.clip);

      if (elapsed >= length - BOUNDARY_EPSILON_MS) {
        // Reached this clip's out point: cut to the next one, or stop at the
        // end of the timeline.
        if (at.clip.endMs >= timelineMs - BOUNDARY_EPSILON_MS) {
          store.setPlayhead(timelineMs);
          store.setPlaying(false);
          return;
        }
        store.setPlayhead(at.clip.endMs);
      } else {
        store.setPlayhead(at.clip.startMs + Math.max(0, elapsed));
      }

      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [source, playing, timelineMs]);

  // Fall back to program when the render being previewed disappears.
  useEffect(() => {
    if (source === "render" && !render?.playback_url) setSource("program");
  }, [source, render, setSource]);

  /** Track fullscreen rather than assume it: Escape leaves it without asking. */
  useEffect(() => {
    function onChange() {
      setFullscreen(document.fullscreenElement === frameRef.current);
    }
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  function toggleFullscreen() {
    const element = frameRef.current;
    if (!element) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void element.requestFullscreen().catch(() => undefined);
  }

  const empty =
    (source === "program" && clips.length === 0) ||
    (source === "source" && !activeAsset?.has_proxy) ||
    (source === "render" && !render?.playback_url);

  const emptyMessage =
    source === "program"
      ? t("preview.empty.program")
      : source === "source"
        ? activeAsset
          ? t("preview.empty.noProxy")
          : t("preview.empty.noSelection")
        : t("preview.empty.render");

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-ground" aria-label={t("preview.title")}>
      {/* ---------------- viewer header ---------------- */}
      <header className="flex h-header shrink-0 items-center gap-2 border-b border-line bg-raised px-2">
        <SegmentedControl
          label={t("preview.source")}
          value={source}
          onChange={setSource}
          options={[
            {
              value: "source",
              label: t("preview.source.source"),
              title: t("preview.source.sourceHint"),
            },
            {
              value: "program",
              label: t("preview.source.program"),
              title: t("preview.source.programHint"),
            },
            {
              value: "render",
              label: t("preview.source.render"),
              title: render?.playback_url
                ? t("preview.source.renderHint")
                : t("preview.source.renderNone"),
              disabled: !render?.playback_url,
            },
          ]}
        />

        <span className="min-w-0 truncate text-2xs text-dim" title={activeAsset?.original_filename}>
          {source === "source"
            ? (activeAsset?.original_filename ?? t("preview.nothingSelected"))
            : source === "render"
              ? `${render?.width ?? "—"}×${render?.height ?? "—"}`
              : t.plural("preview.clipCount", clips.length)}
        </span>

        {source === "render" && render && (
          <Badge tone={render.status === "ready" ? "ok" : "warn"}>
            {t(`status.render.${render.status}`)}
          </Badge>
        )}

        {/* The output shape, at the right, where an NLE puts it. */}
        <span className="ml-auto shrink-0 font-mono text-2xs tabular-nums text-dim">
          {source === "program" ? aspect : ""}
        </span>
      </header>

      {/* ---------------- viewport ----------------
          The letterbox stays black in both themes: it is the surround for a
          picture, and a light grey one would lie about the black level of
          whatever is being graded inside it. */}
      <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden p-3">
        <div
          ref={frameRef}
          className={`relative flex max-h-full max-w-full items-center justify-center bg-black shadow-[0_0_0_1px_rgb(var(--line-strong))] ${
            source === "program" ? ASPECT_CLASS[aspect] : "aspect-video"
          }`}
          style={{ height: "100%" }}
        >
          {/* User media: there is no caption track and none can be invented. */}
          <video
            ref={videoRef}
            playsInline
            preload="metadata"
            aria-label={t("preview.player")}
            className={`h-full w-full ${source === "program" ? "object-cover" : "object-contain"} ${empty ? "invisible" : ""}`}
            onTimeUpdate={(event) => {
              if (source !== "program") {
                setSourceTimeMs(event.currentTarget.currentTime * 1000);
              }
            }}
            onEnded={() => {
              if (source !== "program") setPlaying(false);
            }}
            onWaiting={() => setStalled(true)}
            onPlaying={() => setStalled(false)}
            onCanPlay={() => setStalled(false)}
          />

          {empty && (
            <p className="absolute inset-0 flex items-center justify-center px-8 text-center text-xs leading-relaxed text-white/45">
              {emptyMessage}
            </p>
          )}

          {stalled && !empty && (
            <span className="absolute left-2 top-2 rounded-sm bg-black/70 px-1.5 py-0.5 font-mono text-2xs text-white/80">
              {t("preview.buffering")}
            </span>
          )}

          {/* A burnt-in timecode, as a viewer in this category has. Over the
              picture rather than under it, because it belongs to the frame. */}
          {!empty && (
            <span className="pointer-events-none absolute right-2 top-2 rounded-sm bg-black/65 px-1.5 py-0.5 font-mono text-2xs tabular-nums text-white/85">
              {timecode(positionMs)}
            </span>
          )}

          {/* Which source clip is on screen right now. The one thing that is
              invisible during program playback and matters most. */}
          {source === "program" && current && (
            <span className="pointer-events-none absolute bottom-2 left-2 max-w-[70%] truncate rounded-sm bg-black/65 px-1.5 py-0.5 font-mono text-2xs text-white/85">
              {current.clip.index + 1}.{" "}
              {media.get(current.clip.mediaId)?.original_filename ??
                current.clip.mediaId.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      {/* ---------------- transport ---------------- */}
      <div className="flex h-strip shrink-0 items-center gap-1 border-t border-line bg-raised px-2">
        <IconButton label={t("preview.toStart")} size="sm" onClick={() => seek(0)}>
          <Glyph name="skip-back" size={12} />
        </IconButton>
        <IconButton
          label={t("preview.back")}
          size="sm"
          onClick={() => seek(positionMs - 1000)}
        >
          <Glyph name="step-back" size={12} />
        </IconButton>

        {/* Play is the one control in the bar that is filled rather than
            ghosted: it is pressed more than the rest put together. */}
        <button
          type="button"
          aria-label={playing ? t("preview.pause") : t("preview.play")}
          title={playing ? t("preview.pause") : t("preview.play")}
          aria-pressed={playing}
          disabled={empty}
          onClick={() => setPlaying(!playing)}
          className="inline-flex h-control w-[30px] shrink-0 items-center justify-center rounded border border-line-strong bg-control text-fg transition-colors hover:bg-control-hover disabled:pointer-events-none disabled:opacity-35"
        >
          <Glyph name={playing ? "pause" : "play"} />
        </button>

        <IconButton
          label={t("preview.forward")}
          size="sm"
          onClick={() => seek(positionMs + 1000)}
        >
          <Glyph name="step-forward" size={12} />
        </IconButton>
        <IconButton label={t("preview.toEnd")} size="sm" onClick={() => seek(durationMs)}>
          <Glyph name="skip-forward" size={12} />
        </IconButton>

        <Divider vertical />

        {/* Position reads at full strength, duration at half: they are the same
            kind of number but only one of them changes. */}
        {/* Named, because "0:04.120 / 0:25.000" read aloud is two numbers with
            no nouns. Sighted users get the same nouns as a tooltip. */}
        <span className="shrink-0 font-mono text-xs tabular-nums text-fg">
          <span title={t("preview.position")} aria-label={t("preview.position")}>
            {timecode(positionMs)}
          </span>
          <span className="text-dim" title={t("preview.duration")} aria-label={t("preview.duration")}>
            {" / "}
            {timecode(durationMs)}
          </span>
        </span>

        {/* Scrub bar. In program mode this is the same playhead the timeline
            shows -- one position, two views of it. */}
        <Slider
          min={0}
          max={Math.max(1, durationMs)}
          value={Math.round(positionMs)}
          step={10}
          aria-label={t("preview.seek")}
          disabled={empty}
          onChange={(event) => seek(Number(event.target.value))}
          className="mx-2 min-w-[80px] flex-1"
        />

        <select
          value={speed}
          onChange={(event) => setSpeed(Number(event.target.value))}
          aria-label={t("preview.speed")}
          title={t("preview.speed")}
          className="h-control-sm shrink-0 rounded border border-line-strong bg-control px-1 font-mono text-2xs text-muted transition-colors hover:bg-control-hover focus:border-accent"
        >
          {SPEEDS.map((value) => (
            <option key={value} value={value}>
              {value}×
            </option>
          ))}
        </select>

        <Divider vertical />

        <IconButton
          label={muted || volume === 0 ? t("preview.unmute") : t("preview.mute")}
          size="sm"
          active={muted || volume === 0}
          onClick={() => setMuted(!muted)}
        >
          <Glyph name={muted || volume === 0 ? "mute" : "volume"} size={12} />
        </IconButton>
        <Slider
          min={0}
          max={1}
          step={0.05}
          value={muted ? 0 : volume}
          aria-label={t("preview.volume")}
          onChange={(event) => {
            const next = Number(event.target.value);
            setVolume(next);
            setMuted(next === 0);
          }}
          className="w-14"
        />

        <IconButton
          label={fullscreen ? t("preview.exitFullscreen") : t("preview.fullscreen")}
          size="sm"
          active={fullscreen}
          onClick={toggleFullscreen}
        >
          <Glyph name="fullscreen" size={12} />
        </IconButton>
      </div>
    </section>
  );
}
