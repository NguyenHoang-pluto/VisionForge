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
import { draftProblems, toManualCuts } from "@/lib/timeline";
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

const QUALITY_NOTE: Record<QualityPreset, string> = {
  draft: "Fastest encode. For checking the cut.",
  balanced: "The default. Good quality at a sane speed.",
  high: "Slowest encode, largest file.",
};

const RENDER_TONE: Record<RenderStatus, "neutral" | "warn" | "ok" | "danger"> = {
  pending: "neutral",
  rendering: "warn",
  ready: "ok",
  failed: "danger",
  cancelled: "neutral",
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
  const queryClient = useQueryClient();

  const clips = useEditorStore((s) => s.clips);
  const aspect = useEditorStore((s) => s.aspect);
  const fps = useEditorStore((s) => s.fps);
  const quality = useEditorStore((s) => s.quality);
  const audio = useEditorStore((s) => s.audio);
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
          : { message: "Could not start the render.", hint: null },
      ),
  });

  return (
    <div className="flex flex-col gap-4 p-2.5">
      <section>
        <SectionTitle>Output</SectionTitle>

        <div className="mt-1.5 grid grid-cols-2 gap-2">
          <Field label="Resolution" hint="Chosen by the server from the aspect ratio.">
            <Select
              value={aspect}
              aria-label="Output shape"
              onChange={(event) => setOutput({ aspect: event.target.value as AspectRatio })}
            >
              {(capabilities.data?.aspect_ratios ?? []).map((item) => (
                <option key={item.value} value={item.value}>
                  {item.width}×{item.height} · {item.value}
                </option>
              ))}
            </Select>
          </Field>

          <Field label="Frame rate">
            <Select
              value={fps}
              aria-label="Output frame rate"
              onChange={(event) => setOutput({ fps: Number(event.target.value) })}
            >
              {(capabilities.data?.fps_presets ?? [24, 30, 60]).map((value) => (
                <option key={value} value={value}>
                  {value} fps
                </option>
              ))}
            </Select>
          </Field>

          <Field label="Quality" hint={QUALITY_NOTE[quality]}>
            <Select
              value={quality}
              aria-label="Quality preset"
              onChange={(event) => setOutput({ quality: event.target.value as QualityPreset })}
            >
              {(capabilities.data?.quality_presets ?? ["draft", "balanced", "high"]).map(
                (value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ),
              )}
            </Select>
          </Field>

          <Field
            label="Audio"
            hint={
              timelineHasAudio
                ? "Source audio is concatenated with the clips."
                : "No clip on this timeline has an audio stream."
            }
          >
            <Select
              value={audio}
              aria-label="Audio mode"
              disabled={!timelineHasAudio}
              onChange={(event) =>
                setOutput({ audio: event.target.value as "none" | "source" })
              }
            >
              <option value="none">Silent</option>
              <option value="source">Source audio</option>
            </Select>
          </Field>
        </div>

        <p className="mt-1.5 text-2xs leading-snug text-dim">{QUALITY_NOTE[quality]}</p>
      </section>

      <section>
        <SectionTitle
          aside={
            <span className="font-mono text-2xs text-dim tabular-nums">
              {clips.length} clips
            </span>
          }
        >
          Render
        </SectionTitle>

        <div className="mt-1">
          <Row label="Timeline length" value={timecode(clips.reduce((sum, c) => sum + (c.outMs - c.inMs), 0))} />
          <Row
            label="Stored plan"
            value={committedPlanId ? `${committedPlanId.slice(0, 8)}${dirty ? " (stale)" : ""}` : "none"}
            tone={dirty ? "warn" : undefined}
          />
        </div>

        {problems.length > 0 && (
          <ul className="mt-2 flex flex-col gap-1">
            {problems.slice(0, 4).map((problem, index) => (
              <li key={index} className="text-2xs leading-snug text-warn">
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
            ? "Queuing…"
            : dirty || !committedPlanId
              ? "Store timeline and render"
              : "Render"}
        </Button>

        {error && (
          <div className="mt-2">
            <ErrorNote hint={error.hint}>{error.message}</ErrorNote>
          </div>
        )}
      </section>

      <section>
        <SectionTitle
          aside={render && <Badge tone={RENDER_TONE[render.status]}>{render.status}</Badge>}
        >
          Last output
        </SectionTitle>

        {!render ? (
          <EmptyState>No renders in this project yet.</EmptyState>
        ) : (
          <>
            <div className="mt-1">
              <Row
                label="Resolution"
                value={render.width ? `${render.width}×${render.height}` : "—"}
              />
              <Row label="Frame rate" value={formatFps(render.fps)} />
              <Row label="Duration" value={timecode(render.duration_ms)} />
              <Row label="Size" value={bytes(render.bytes_size)} />
              <Row
                label="Encode time"
                value={
                  render.metrics?.render_ms ? seconds(Number(render.metrics.render_ms)) : "—"
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
                  Play
                </Button>
                <a
                  href={render.playback_url}
                  download
                  className="inline-flex h-[22px] items-center gap-1 rounded border border-line-strong bg-control px-2 text-2xs text-fg transition-colors hover:bg-control-hover"
                >
                  <Glyph name="download" size={10} />
                  Download MP4
                </a>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
