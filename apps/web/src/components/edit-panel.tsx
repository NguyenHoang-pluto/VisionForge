"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  api,
  ApiError,
  type AspectRatio,
  type ClipOrder,
  type EditPlan,
  type MediaAsset,
  type PlanSegment,
  type Render,
  type RenderStatus,
} from "@/lib/api";

/**
 * Generate an edit, render it, play it back.
 *
 * Laid out as a workstation strip: settings, then the timeline the planner
 * produced, then the render. The timeline is a proportional bar rather than a
 * list, because the one thing worth seeing at a glance is how the duration is
 * distributed across clips.
 */

const ASPECT_RATIOS: { value: AspectRatio; label: string; geometry: string }[] = [
  { value: "16:9", label: "16:9", geometry: "1280×720" },
  { value: "9:16", label: "9:16", geometry: "720×1280" },
  { value: "1:1", label: "1:1", geometry: "720×720" },
];

const ORDERS: { value: ClipOrder; label: string }[] = [
  { value: "score_desc", label: "Strongest first" },
  { value: "sequence", label: "Upload order" },
];

const RENDER_STATUS_CLASS: Record<RenderStatus, string> = {
  pending: "text-slate-500",
  rendering: "text-amber-400",
  ready: "text-emerald-400",
  failed: "text-rose-400",
  cancelled: "text-slate-500",
};

/** Distinct hues so adjacent clips are separable in the timeline bar. */
const CLIP_COLOURS = [
  "bg-sky-700",
  "bg-teal-700",
  "bg-indigo-700",
  "bg-cyan-700",
  "bg-violet-700",
  "bg-emerald-700",
  "bg-blue-700",
  "bg-fuchsia-800",
];

function seconds(ms: number | null | undefined): string {
  return typeof ms === "number" ? `${(ms / 1000).toFixed(1)}s` : "—";
}

function megabytes(bytes: number | null | undefined): string {
  return typeof bytes === "number" ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "—";
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[9px] uppercase tracking-wider text-slate-600">
        {label}
      </span>
      {children}
    </label>
  );
}

const SELECT_CLASS =
  "rounded-sm border border-slate-800 bg-slate-950 px-2 py-1 text-[11px] text-slate-200 focus:border-sky-700 focus:outline-none";

/** The planner's output as a proportional strip, one block per clip. */
function TimelineBar({
  segments,
  totalMs,
  names,
}: {
  segments: PlanSegment[];
  totalMs: number;
  names: Map<string, string>;
}) {
  return (
    <div>
      <div className="flex h-7 w-full overflow-hidden rounded-sm border border-slate-800">
        {segments.map((segment, index) => (
          <div
            key={`${segment.media_id}-${segment.order}`}
            className={`flex items-center justify-center overflow-hidden border-r border-slate-950 last:border-r-0 ${
              CLIP_COLOURS[index % CLIP_COLOURS.length]
            }`}
            style={{ width: `${(segment.duration_ms / totalMs) * 100}%` }}
            title={`${names.get(segment.media_id) ?? segment.media_id.slice(0, 8)} · ${
              segment.source_in_ms
            }–${segment.source_out_ms} ms`}
          >
            <span className="truncate px-1 font-mono text-[9px] text-slate-200">
              {seconds(segment.duration_ms)}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-2 overflow-x-auto">
        <table className="w-full font-mono text-[10px] tabular-nums">
          <thead className="text-slate-600">
            <tr className="border-b border-slate-800">
              <th className="py-1 pr-2 text-left font-normal">#</th>
              <th className="py-1 pr-2 text-left font-normal">Source</th>
              <th className="py-1 pr-2 text-right font-normal">In</th>
              <th className="py-1 pr-2 text-right font-normal">Out</th>
              <th className="py-1 text-right font-normal">Dur</th>
            </tr>
          </thead>
          <tbody className="text-slate-400">
            {segments.map((segment) => (
              <tr
                key={`${segment.media_id}-${segment.order}`}
                className="border-b border-slate-800/50 last:border-b-0"
              >
                <td className="py-0.5 pr-2 text-slate-600">{segment.order}</td>
                <td className="max-w-[14rem] truncate py-0.5 pr-2 text-slate-300">
                  {names.get(segment.media_id) ?? segment.media_id.slice(0, 8)}
                </td>
                <td className="py-0.5 pr-2 text-right">{seconds(segment.source_in_ms)}</td>
                <td className="py-0.5 pr-2 text-right">{seconds(segment.source_out_ms)}</td>
                <td className="py-0.5 text-right">{seconds(segment.duration_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Why clips were dropped. An automatic edit that cannot explain itself is not reviewable. */
function RejectionList({
  plan,
  names,
}: {
  plan: EditPlan;
  names: Map<string, string>;
}) {
  const rejected = plan.selection?.rejected ?? [];
  if (rejected.length === 0) return null;

  return (
    <details className="mt-2">
      <summary className="cursor-pointer font-mono text-[10px] text-slate-600 marker:text-slate-700 hover:text-slate-400">
        {rejected.length} clip{rejected.length === 1 ? "" : "s"} not used
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5">
        {rejected.map((item) => (
          <li key={item.media_id} className="flex gap-2 font-mono text-[10px]">
            <span className="w-40 truncate text-slate-500">
              {names.get(item.media_id) ?? item.media_id.slice(0, 8)}
            </span>
            <span className="text-amber-500/80">{item.reason.replace(/_/g, " ")}</span>
            <span className="truncate text-slate-600">{item.detail}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

export function EditPanel({
  projectId,
  media,
}: {
  projectId: string;
  media: MediaAsset[];
}) {
  const queryClient = useQueryClient();

  const [targetSeconds, setTargetSeconds] = useState(25);
  const [maxClips, setMaxClips] = useState(5);
  const [aspect, setAspect] = useState<AspectRatio>("16:9");
  const [order, setOrder] = useState<ClipOrder>("score_desc");
  const [error, setError] = useState<string | null>(null);

  const names = new Map(media.map((asset) => [asset.id, asset.original_filename]));
  const readyCount = media.filter((asset) => asset.status === "ready").length;

  const plans = useQuery({
    queryKey: ["edit-plans", projectId],
    queryFn: () => api.listEditPlans(projectId),
  });

  const renders = useQuery({
    queryKey: ["renders", projectId],
    queryFn: () => api.listRenders(projectId),
    // Poll only while something is actually encoding.
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (render: Render) => render.status === "pending" || render.status === "rendering",
      )
        ? 2000
        : false,
  });

  const latestPlan = plans.data?.items[0] ?? null;
  const latestRender = renders.data?.items[0] ?? null;

  const generate = useMutation({
    mutationFn: () =>
      api.createEditPlan(projectId, {
        target_duration_ms: targetSeconds * 1000,
        max_clips: maxClips,
        min_clips: 1,
        aspect_ratio: aspect,
        order,
      }),
    onSuccess: () => {
      setError(null);
      void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
    },
    onError: (caught: unknown) => {
      setError(
        caught instanceof ApiError
          ? [caught.message, caught.hint].filter(Boolean).join(" — ")
          : "Could not generate an edit.",
      );
    },
  });

  const render = useMutation({
    mutationFn: (planId: string) => api.createRender(projectId, planId),
    onSuccess: () => {
      setError(null);
      void queryClient.invalidateQueries({ queryKey: ["renders", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
    onError: (caught: unknown) => {
      setError(caught instanceof ApiError ? caught.message : "Could not start the render.");
    },
  });

  return (
    <section className="border-t border-slate-800">
      <h2 className="border-b border-slate-800 px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-600">
        Automatic edit
      </h2>

      <div className="grid gap-4 p-3 lg:grid-cols-[minmax(0,1fr)_300px]">
        {/* ---------------- plan ---------------- */}
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Duration">
              <input
                id="target-seconds"
                type="number"
                min={2}
                max={120}
                value={targetSeconds}
                onChange={(event) => setTargetSeconds(Number(event.target.value))}
                className={`${SELECT_CLASS} w-16 tabular-nums`}
              />
            </Field>
            <Field label="Max clips">
              <input
                id="max-clips"
                type="number"
                min={1}
                max={20}
                value={maxClips}
                onChange={(event) => setMaxClips(Number(event.target.value))}
                className={`${SELECT_CLASS} w-16 tabular-nums`}
              />
            </Field>
            <Field label="Aspect">
              <select
                id="aspect-ratio"
                value={aspect}
                onChange={(event) => setAspect(event.target.value as AspectRatio)}
                className={SELECT_CLASS}
              >
                {ASPECT_RATIOS.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label} · {item.geometry}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Order">
              <select
                id="clip-order"
                value={order}
                onChange={(event) => setOrder(event.target.value as ClipOrder)}
                className={SELECT_CLASS}
              >
                {ORDERS.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </Field>

            <button
              type="button"
              onClick={() => generate.mutate()}
              disabled={readyCount === 0 || generate.isPending}
              className="rounded-sm border border-sky-700 bg-sky-900/40 px-2.5 py-1 text-[11px] text-sky-200 transition hover:bg-sky-900/70 focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {generate.isPending ? "Generating…" : "Generate edit"}
            </button>
          </div>

          {error && (
            <p className="border-l-2 border-rose-800 px-2 py-1 text-[10px] text-rose-400">
              {error}
            </p>
          )}

          {!latestPlan && !generate.isPending && (
            <p className="text-[11px] text-slate-600">
              {readyCount === 0
                ? "No analysed media yet. Upload clips and run analysis first."
                : "No edit generated. Clips are ranked on sharpness, exposure, contrast, resolution and duration; near-duplicates are dropped."}
            </p>
          )}

          {latestPlan && (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-slate-500 tabular-nums">
                <span>
                  {latestPlan.plan.planner}@{latestPlan.plan.planner_version} ·{" "}
                  {latestPlan.segment_count} clips · {seconds(latestPlan.total_duration_ms)}
                </span>
                <span>
                  {latestPlan.plan.output.width}×{latestPlan.plan.output.height} @{" "}
                  {latestPlan.plan.output.fps}fps
                </span>
              </div>

              <TimelineBar
                segments={latestPlan.plan.segments}
                totalMs={latestPlan.total_duration_ms}
                names={names}
              />
              <RejectionList plan={latestPlan} names={names} />

              <div>
                <button
                  type="button"
                  onClick={() => render.mutate(latestPlan.id)}
                  disabled={render.isPending}
                  className="rounded-sm border border-slate-700 px-2.5 py-1 text-[11px] text-slate-200 transition hover:border-slate-600 focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600 disabled:opacity-40"
                >
                  {render.isPending ? "Queuing…" : "Render"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ---------------- output ---------------- */}
        <div className="flex flex-col gap-2 border-slate-800 lg:border-l lg:pl-4">
          <h3 className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
            Output
          </h3>

          {!latestRender && (
            <p className="text-[11px] text-slate-600">No renders yet.</p>
          )}

          {latestRender && (
            <>
              {latestRender.status === "ready" && latestRender.playback_url ? (
                <video
                  key={latestRender.id}
                  src={latestRender.playback_url}
                  controls
                  className="w-full rounded-sm border border-slate-800 bg-black"
                />
              ) : (
                <div className="flex aspect-video items-center justify-center rounded-sm border border-slate-800 bg-slate-900">
                  <span
                    className={`font-mono text-[10px] uppercase ${RENDER_STATUS_CLASS[latestRender.status]}`}
                  >
                    {latestRender.status}
                  </span>
                </div>
              )}

              <dl className="flex flex-col divide-y divide-slate-800/60 font-mono text-[10px] tabular-nums">
                {[
                  ["Status", latestRender.status],
                  [
                    "Resolution",
                    latestRender.width ? `${latestRender.width}×${latestRender.height}` : "—",
                  ],
                  ["Frame rate", latestRender.fps ? `${latestRender.fps.toFixed(2)} fps` : "—"],
                  ["Duration", seconds(latestRender.duration_ms)],
                  ["Size", megabytes(latestRender.bytes_size)],
                  [
                    "Encode",
                    latestRender.metrics?.render_ms
                      ? seconds(Number(latestRender.metrics.render_ms))
                      : "—",
                  ],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-3 py-1">
                    <dt className="text-slate-600">{label}</dt>
                    <dd
                      className={
                        label === "Status"
                          ? RENDER_STATUS_CLASS[latestRender.status]
                          : "text-slate-300"
                      }
                    >
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>

              {latestRender.error?.message && (
                <p className="border-l-2 border-rose-800 px-2 py-1 text-[10px] leading-snug text-rose-400">
                  {latestRender.error.message}
                  {latestRender.error.hint && (
                    <span className="block text-rose-500/70">{latestRender.error.hint}</span>
                  )}
                </p>
              )}

              {latestRender.status === "ready" && latestRender.playback_url && (
                <a
                  href={latestRender.playback_url}
                  download
                  className="text-center font-mono text-[10px] text-slate-500 underline-offset-2 transition hover:text-slate-300 hover:underline"
                >
                  Download MP4
                </a>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
