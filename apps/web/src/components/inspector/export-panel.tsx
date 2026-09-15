"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  api,
  ApiError,
  type AspectRatio,
  type MediaAsset,
  type QualityPreset,
  type Render,
  type RenderStatus,
} from "@/lib/api";
import { bytes, fps as formatFps, seconds, timecode } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { draftProblems, toManualCuts, toMusicRequest } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  EmptyState,
  ErrorNote,
  Field,
  Glyph,
  Row,
  SectionTitle,
  Select,
} from "@/components/ui";

/**
 * Export.
 *
 * The whole panel sends intent and nothing else: a shape, a frame rate, a
 * quality *level*. There is no width field, no CRF field and no place to put an
 * FFmpeg argument, because the server chooses all three and the plan schema has
 * nowhere to carry them. What the user picks here maps onto presets the server
 * already declared through `/planner/capabilities`.
 *
 * Rendering is two steps and says so: the timeline is stored as a plan, then
 * that plan is rendered. A render is always of something the server has already
 * validated -- never of the browser's current scroll position.
 */

const QUALITY_NOTE: Record<QualityPreset, MessageKey> = {
  draft: "export.quality.draftNote",
  balanced: "export.quality.balancedNote",
  high: "export.quality.highNote",
};

const QUALITY_LABEL: Record<QualityPreset, MessageKey> = {
  draft: "export.quality.draft",
  balanced: "export.quality.balanced",
  high: "export.quality.high",
};

const RENDER_TONE: Record<RenderStatus, "neutral" | "warn" | "ok" | "danger"> = {
  pending: "neutral",
  rendering: "warn",
  ready: "ok",
  failed: "danger",
  cancelled: "neutral",
};

const RENDER_LABEL: Record<RenderStatus, MessageKey> = {
  pending: "status.render.pending",
  rendering: "status.render.rendering",
  ready: "status.render.ready",
  failed: "status.render.failed",
  cancelled: "status.render.cancelled",
};

export function ExportPanel({
  projectId,
  media,
  render,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  render: Render | null;
}) {
  const t = useT();
  const queryClient = useQueryClient();

  const clips = useEditorStore((s) => s.clips);
  const aspect = useEditorStore((s) => s.aspect);
  const fps = useEditorStore((s) => s.fps);
  const quality = useEditorStore((s) => s.quality);
  const audio = useEditorStore((s) => s.audio);
  const sourceGain = useEditorStore((s) => s.sourceGain);
  const musicBed = useEditorStore((s) => s.music);
  const setOutput = useEditorStore((s) => s.setOutput);
  const committedPlanId = useEditorStore((s) => s.committedPlanId);
  const sourcePlanId = useEditorStore((s) => s.sourcePlanId);
  const dirty = useEditorStore((s) => s.dirty);
  const markCommitted = useEditorStore((s) => s.markCommitted);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);

  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });

  const problems = draftProblems(clips, media);
  const timelineHasAudio = clips.some((clip) => Boolean(media.get(clip.mediaId)?.channels));
  const musicAsset = musicBed ? media.get(musicBed.mediaId) : undefined;

  /**
   * Store the timeline, then render it.
   *
   * A fresh plan is stored whenever the draft has been edited, so the render is
   * always of exactly what is on screen. When nothing has changed since the
   * last commit, the existing plan is reused rather than duplicated.
   */
  const renderNow = useMutation({
    mutationFn: async () => {
      let planId = committedPlanId;
      if (!planId || dirty) {
        const plan = await api.createManualEditPlan(projectId, {
          segments: toManualCuts(clips),
          aspect_ratio: aspect,
          fps,
          quality,
          audio,
          source_gain: sourceGain,
          // The bed travels with the timeline it was placed under. The server
          // validates it against the real media rows like everything else.
          music: toMusicRequest(musicBed),
          derived_from_edit_plan_id: sourcePlanId,
        });
        planId = plan.id;
        markCommitted(plan.id);
        void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
      }
      return api.createRender(projectId, planId);
    },
    onSuccess: () => {
      setError(null);
      setPreviewSource("render");
      void queryClient.invalidateQueries({ queryKey: ["renders", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
    onError: (caught: unknown) =>
      setError(
        caught instanceof ApiError
          ? { message: caught.message, hint: caught.hint }
          : { message: t("export.error"), hint: null },
      ),
  });

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      <section>
        <SectionTitle>{t("export.output")}</SectionTitle>

        <div className="mt-1.5 grid grid-cols-2 gap-2">
          <Field label={t("export.resolution")} hint={t("export.resolutionHint")}>
            <Select
              value={aspect}
              aria-label={t("export.shapeLabel")}
              onChange={(event) => setOutput({ aspect: event.target.value as AspectRatio })}
            >
              {(capabilities.data?.aspect_ratios ?? []).map((item) => (
                <option key={item.value} value={item.value}>
                  {item.width}×{item.height} · {item.value}
                </option>
              ))}
            </Select>
          </Field>

          <Field label={t("export.fps")}>
            <Select
              value={fps}
              aria-label={t("export.fpsLabel")}
              onChange={(event) => setOutput({ fps: Number(event.target.value) })}
            >
              {(capabilities.data?.fps_presets ?? [24, 30, 60]).map((value) => (
                <option key={value} value={value}>
                  {value} fps
                </option>
              ))}
            </Select>
          </Field>

          <Field label={t("export.quality")} hint={t(QUALITY_NOTE[quality])}>
            <Select
              value={quality}
              aria-label={t("export.qualityLabel")}
              onChange={(event) => setOutput({ quality: event.target.value as QualityPreset })}
            >
              {(capabilities.data?.quality_presets ?? ["draft", "balanced", "high"]).map(
                (value) => (
                  <option key={value} value={value}>
                    {t(QUALITY_LABEL[value as QualityPreset] ?? "export.quality.balanced")}
                  </option>
                ),
              )}
            </Select>
          </Field>

          <Field
            label={t("export.audio")}
            hint={timelineHasAudio ? t("export.audio.hasHint") : t("export.audio.noneHint")}
          >
            <Select
              value={audio}
              aria-label={t("export.audioLabel")}
              disabled={!timelineHasAudio}
              onChange={(event) =>
                setOutput({ audio: event.target.value as "none" | "source" })
              }
            >
              <option value="none">{t("export.audio.none")}</option>
              <option value="source">{t("export.audio.source")}</option>
            </Select>
          </Field>
        </div>

        <p className="mt-1.5 text-2xs leading-snug text-faint">{t(QUALITY_NOTE[quality])}</p>

        {/* What will actually be heard, stated where the render is started.
            "Silent" plus a music bed is a contradiction worth not shipping. */}
        {musicBed && (
          <div className="mt-1.5">
            <Row
              label={t("audio.title")}
              value={musicAsset?.original_filename ?? musicBed.mediaId.slice(0, 8)}
              title={musicAsset?.original_filename}
            />
          </div>
        )}
      </section>

      <section>
        <SectionTitle
          aside={
            <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
              {t.plural("timeline.clipCount", clips.length)}
            </span>
          }
        >
          {t("export.render")}
        </SectionTitle>

        <div className="mt-1">
          <Row
            label={t("export.timelineLength")}
            value={timecode(clips.reduce((sum, c) => sum + (c.outMs - c.inMs), 0))}
          />
          <Row
            label={t("export.storedPlan")}
            value={
              committedPlanId
                ? dirty
                  ? t("export.storedPlan.stale", { id: committedPlanId.slice(0, 8) })
                  : committedPlanId.slice(0, 8)
                : t("export.storedPlan.none")
            }
            tone={dirty ? "warn" : undefined}
          />
        </div>

        {problems.length > 0 && (
          <ul className="mt-2 flex flex-col gap-1">
            {problems.slice(0, 4).map((problem, index) => (
              <li
                key={index}
                className="flex items-start gap-1.5 text-2xs leading-snug text-warning"
              >
                <span className="mt-px shrink-0">
                  <Glyph name="warning" size={11} />
                </span>
                {problem.message}
              </li>
            ))}
          </ul>
        )}

        <Button
          tone="primary"
          className="mt-2 w-full"
          disabled={problems.length > 0 || renderNow.isPending}
          onClick={() => renderNow.mutate()}
        >
          {renderNow.isPending
            ? t("export.queuing")
            : dirty || !committedPlanId
              ? t("export.storeAndRender")
              : t("export.renderNow")}
        </Button>

        {error && (
          <div className="mt-2">
            <ErrorNote hint={error.hint}>{error.message}</ErrorNote>
          </div>
        )}
      </section>

      <section>
        <SectionTitle
          aside={
            render && (
              <Badge tone={RENDER_TONE[render.status]}>{t(RENDER_LABEL[render.status])}</Badge>
            )
          }
        >
          {t("export.lastOutput")}
        </SectionTitle>

        {!render ? (
          <EmptyState icon="download">{t("export.noRenders")}</EmptyState>
        ) : (
          <>
            <div className="mt-1">
              <Row
                label={t("export.resolution")}
                value={render.width ? `${render.width}×${render.height}` : t("common.dash")}
              />
              <Row label={t("export.fps")} value={formatFps(render.fps)} />
              <Row label={t("export.duration")} value={timecode(render.duration_ms)} />
              <Row label={t("export.size")} value={bytes(render.bytes_size)} />
              <Row
                label={t("export.encodeTime")}
                value={
                  render.metrics?.render_ms
                    ? seconds(Number(render.metrics.render_ms))
                    : t("common.dash")
                }
              />
            </div>

            {render.error?.message && (
              <div className="mt-2">
                <ErrorNote hint={render.error.hint ?? null}>{render.error.message}</ErrorNote>
              </div>
            )}

            {render.status === "ready" && render.playback_url && (
              <div className="mt-2 flex gap-1">
                <Button size="sm" onClick={() => setPreviewSource("render")}>
                  <Glyph name="play" size={10} />
                  {t("export.play")}
                </Button>
                <a
                  href={render.playback_url}
                  download
                  className="inline-flex h-control-sm items-center gap-1.5 rounded px-2.5 text-2xs font-medium text-fg shadow-raised transition-colors duration-fast bg-elevated hover:bg-hover"
                >
                  <Glyph name="download" size={10} />
                  {t("export.download")}
                </a>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
