"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type BeatsPayload, type MediaAsset } from "@/lib/api";
import { shortDuration, timecode } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { MAX_FADE_MS, MAX_GAIN, musicDuration, totalDuration } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  EmptyState,
  Field,
  Glyph,
  NumberInput,
  Row,
  SectionTitle,
  Slider,
  StatusDot,
} from "@/components/ui";

/**
 * The music bed, in the inspector.
 *
 * One track, matching what `EditPlan` can express. A second bed would need
 * overlap rules and a mix policy the plan has nowhere to put, so the editor
 * does not offer one rather than offering one the server would refuse.
 *
 * Levels are shown as percentages and stored as linear gain, which is the unit
 * the filter graph takes. The conversion happens at the control and nowhere
 * else -- a second place that scaled a level would be a second place an edit
 * could quietly get its character.
 */

/** Percent for the UI, linear gain for everything else. */
const asPercent = (gain: number) => Math.round(gain * 100);
const asGain = (percent: number) => percent / 100;

export function AudioPanel({
  projectId,
  media,
  mediaList,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
}) {
  const t = useT();

  const music = useEditorStore((s) => s.music);
  const setMusic = useEditorStore((s) => s.setMusic);
  const updateMusic = useEditorStore((s) => s.updateMusic);
  const clearMusic = useEditorStore((s) => s.clearMusic);
  const beatSync = useEditorStore((s) => s.beatSync);
  const setBeatSync = useEditorStore((s) => s.setBeatSync);
  const showBeats = useEditorStore((s) => s.showBeatMarkers);
  const setShowBeats = useEditorStore((s) => s.setShowBeatMarkers);
  const activeMediaId = useEditorStore((s) => s.activeMediaId);
  const clips = useEditorStore((s) => s.clips);
  const audioMode = useEditorStore((s) => s.audio);
  const sourceGain = useEditorStore((s) => s.sourceGain);
  const setOutput = useEditorStore((s) => s.setOutput);

  const track = music ? (media.get(music.mediaId) ?? null) : null;
  const trackMs = track?.duration_ms ?? null;

  /** The asset the browser is pointing at, if it could be a bed. */
  const candidate = activeMediaId ? media.get(activeMediaId) : undefined;
  const choosable =
    candidate && candidate.kind === "audio" && candidate.status === "ready"
      ? candidate
      : undefined;

  const timelineMs = totalDuration(clips);

  // Beats for the chosen track. Fetched here rather than pushed down from the
  // workstation: only this panel and the timeline lane want them, and the
  // query cache means the two share one request.
  const beats = useQuery({
    queryKey: ["analysis", music?.mediaId],
    queryFn: () => api.mediaAnalysis(projectId, music!.mediaId),
    enabled: Boolean(music?.mediaId),
    staleTime: 60_000,
  });

  const record = beats.data?.items.find((item) => item.analyzer === "beats");
  const grid = record?.status === "ok" ? (record.payload as unknown as BeatsPayload) : null;
  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });
  const minConfidence = capabilities.data?.beat_sync?.min_confidence ?? 0.35;
  const reliable = Boolean(grid && grid.confidence >= minConfidence && grid.bpm > 0);

  // ------------------------------------------------------------------ empty
  if (!music || !track) {
    const tracks = mediaList.filter(
      (asset) => asset.kind === "audio" && asset.status === "ready",
    );

    return (
      <div className="flex flex-col gap-panel-gap p-panel">
        <EmptyState
          icon="audio"
          title={t("audio.none.title")}
          action={
            choosable ? (
              <Button size="sm" tone="primary" onClick={() => setMusic(choosable)}>
                <Glyph name="audio" size={10} />
                {t("audio.choose")}
              </Button>
            ) : undefined
          }
        >
          {tracks.length > 0 ? t("audio.none.pick") : t("audio.none.body")}
        </EmptyState>

        {/* The project's audio, offered here rather than described. The panel
            is handed the whole media list already, and telling someone to go
            and select something in a browser that this workspace does not show
            is an instruction they cannot follow. */}
        {tracks.length > 0 && (
          <ul className="flex flex-col gap-1">
            {tracks.map((asset) => (
              <li key={asset.id}>
                <button
                  type="button"
                  onClick={() => setMusic(asset)}
                  title={asset.original_filename}
                  className="group flex w-full items-center gap-2.5 rounded-lg bg-elevated px-2.5 py-2 text-left transition-[background-color,transform] duration-fast hover:bg-hover active:scale-[0.99]"
                >
                  <span
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-success/15 text-success"
                    aria-hidden
                  >
                    <Glyph name="audio" size={13} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-fg">
                      {asset.original_filename}
                    </span>
                    <span className="mt-0.5 block font-mono text-2xs tabular-nums text-faint">
                      {shortDuration(asset.duration_ms)}
                      {asset.channels
                        ? ` · ${t("clip.audioValue", {
                            channels: asset.channels,
                            rate: asset.sample_rate ?? t("common.dash"),
                          })}`
                        : ""}
                    </span>
                  </span>
                  <span className="shrink-0 text-faint transition-colors duration-fast group-hover:text-accent-strong">
                    <Glyph name="chevron-right" size={12} />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  const cueMs = musicDuration(music);

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      {/* ---------------- the track ---------------- */}
      <section>
        <SectionTitle
          aside={
            <Button
              size="sm"
              tone="danger"
              title={t("audio.remove")}
              onClick={clearMusic}
              className="h-[18px] px-1.5"
            >
              <Glyph name="trash" size={10} />
            </Button>
          }
        >
          {t("audio.title")}
        </SectionTitle>

        <div className="mt-1">
          <Row
            label={t("audio.track")}
            value={track.original_filename}
            title={track.original_filename}
          />
          <Row label={t("audio.trackDuration")} value={timecode(trackMs)} />
          <Row label={t("audio.cueLength")} value={timecode(cueMs)} />
          <Row
            label={t("clip.audio")}
            value={
              track.channels
                ? t("clip.audioValue", {
                    channels: track.channels,
                    rate: track.sample_rate ?? t("common.dash"),
                  })
                : t("clip.audioNone")
            }
          />
        </div>

        {choosable && choosable.id !== music.mediaId && (
          <div className="mt-2">
            <Button size="sm" onClick={() => setMusic(choosable)}>
              <Glyph name="refresh" size={10} />
              {t("audio.replace")}
            </Button>
          </div>
        )}
      </section>

      {/* ---------------- level ---------------- */}
      <section>
        <SectionTitle>{t("audio.section.level")}</SectionTitle>

        <div className="mt-1.5 flex flex-col gap-2">
          <Field label={t("audio.volume")} hint={t("audio.volumeHint")}>
            <div className="flex items-center gap-2">
              <Slider
                min={0}
                max={MAX_GAIN * 100}
                step={5}
                value={asPercent(music.gain)}
                aria-label={t("audio.volume")}
                onChange={(event) =>
                  updateMusic({ gain: asGain(Number(event.target.value)) }, trackMs)
                }
                className="min-w-0 flex-1"
              />
              <span className="w-10 shrink-0 text-right font-mono text-2xs tabular-nums text-fg">
                {asPercent(music.gain)}%
              </span>
            </div>
          </Field>

          {/* The clips' own level, here rather than in Export, because ducking
              dialogue under a bed is one decision made while listening to both. */}
          <Field label={t("audio.sourceVolume")} hint={t("audio.sourceVolumeHint")}>
            <div className="flex items-center gap-2">
              <Slider
                min={0}
                max={MAX_GAIN * 100}
                step={5}
                value={asPercent(sourceGain)}
                aria-label={t("audio.sourceVolume")}
                disabled={audioMode === "none"}
                onChange={(event) =>
                  setOutput({ sourceGain: asGain(Number(event.target.value)) })
                }
                className="min-w-0 flex-1"
              />
              <span className="w-10 shrink-0 text-right font-mono text-2xs tabular-nums text-fg">
                {asPercent(sourceGain)}%
              </span>
            </div>
          </Field>
          {audioMode === "none" && (
            <p className="-mt-1 text-2xs leading-snug text-faint">{t("audio.sourceMuted")}</p>
          )}
        </div>

        <div className="mt-2 grid grid-cols-2 gap-2">
          <Field label={t("audio.fadeIn")}>
            <NumberInput
              min={0}
              max={MAX_FADE_MS}
              step={100}
              value={Math.round(music.fadeInMs)}
              aria-label={t("audio.fadeIn")}
              onChange={(event) =>
                updateMusic({ fadeInMs: Number(event.target.value) }, trackMs)
              }
            />
          </Field>
          <Field label={t("audio.fadeOut")}>
            <NumberInput
              min={0}
              max={MAX_FADE_MS}
              step={100}
              value={Math.round(music.fadeOutMs)}
              aria-label={t("audio.fadeOut")}
              onChange={(event) =>
                updateMusic({ fadeOutMs: Number(event.target.value) }, trackMs)
              }
            />
          </Field>
        </div>
      </section>

      {/* ---------------- timing ---------------- */}
      <section>
        <SectionTitle>{t("audio.section.timing")}</SectionTitle>
        <div className="mt-1.5 grid grid-cols-2 gap-2">
          <Field label={t("audio.in")} hint={t("audio.inHint")}>
            <NumberInput
              min={0}
              step={500}
              value={Math.round(music.inMs)}
              aria-label={t("audio.in")}
              onChange={(event) => updateMusic({ inMs: Number(event.target.value) }, trackMs)}
            />
          </Field>
          <Field label={t("audio.out")}>
            <NumberInput
              min={0}
              step={500}
              max={trackMs ?? undefined}
              value={Math.round(music.outMs)}
              aria-label={t("audio.out")}
              onChange={(event) => updateMusic({ outMs: Number(event.target.value) }, trackMs)}
            />
          </Field>
          <Field label={t("audio.start")} hint={t("audio.startHint")} className="col-span-2">
            <NumberInput
              min={0}
              step={500}
              max={Math.max(0, timelineMs - 1)}
              value={Math.round(music.startMs)}
              aria-label={t("audio.start")}
              onChange={(event) => updateMusic({ startMs: Number(event.target.value) }, trackMs)}
            />
          </Field>
        </div>
      </section>

      {/* ---------------- beats ---------------- */}
      <section>
        <SectionTitle
          aside={
            grid ? (
              <StatusDot tone={reliable ? "ok" : "warn"}>
                <span className="font-mono text-2xs text-faint">
                  {reliable ? t("audio.beatsReady") : t("audio.beatsUnreliable")}
                </span>
              </StatusDot>
            ) : undefined
          }
        >
          {t("audio.section.beats")}
        </SectionTitle>

        {beats.isLoading ? (
          <p className="py-2 text-2xs text-faint">{t("audio.analysing")}</p>
        ) : !grid ? (
          <p className="py-2 text-2xs leading-snug text-faint">{t("audio.beatsMissing")}</p>
        ) : (
          <div className="mt-1">
            <Row label={t("audio.bpm")} value={t("audio.bpmValue", { bpm: grid.bpm.toFixed(1) })} />
            <Row
              label={t("audio.confidence")}
              value={`${Math.round(grid.confidence * 100)}%`}
              tone={reliable ? "ok" : "warn"}
            />
            <Row label={t("audio.beatCount")} value={String(grid.beat_count)} />
          </div>
        )}

        <div className="mt-2 flex flex-col gap-1.5">
          <label className="flex items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={beatSync}
              disabled={!reliable}
              onChange={(event) => setBeatSync(event.target.checked)}
              className="h-3 w-3 accent-[rgb(var(--accent))]"
            />
            {t("audio.beatSync")}
          </label>
          <p className="text-2xs leading-snug text-faint">{t("audio.beatSyncHint")}</p>

          <label
            className="mt-1 flex items-center gap-2 text-xs text-muted"
            title={t("audio.markersHint")}
          >
            <input
              type="checkbox"
              checked={showBeats}
              disabled={!grid}
              onChange={(event) => setShowBeats(event.target.checked)}
              className="h-3 w-3 accent-[rgb(var(--accent))]"
            />
            {t("audio.markers")}
          </label>
        </div>
      </section>

      {/* What the track actually contains, at a glance. */}
      {track.status !== "ready" && (
        <Badge tone="warn">{t(`media.status.${track.status}`)}</Badge>
      )}
      <p className="font-mono text-2xs text-faint">
        {shortDuration(cueMs)} · {asPercent(music.gain)}%
      </p>
    </div>
  );
}
