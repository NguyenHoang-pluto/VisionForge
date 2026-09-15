"use client";

import type { Job, JobStep, MediaAsset, Render, RenderStatus } from "@/lib/api";
import { bytes, fps as formatFps, relativeTime, seconds, timecode } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { useEditorStore } from "@/stores/editor-store";
import { ExportPanel } from "@/components/inspector/export-panel";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Glyph,
  Panel,
  PanelHeader,
  Spinner,
} from "@/components/ui";

/**
 * The Export workspace.
 *
 * Settings on the left, what happened on the right. The settings are the same
 * `ExportPanel` the inspector shows -- one implementation of "what does the
 * server let me ask for" -- and everything beside it is the part a 300px column
 * had no room for: the stages of the render as it runs, and the history of
 * everything rendered from this project.
 */

/**
 * The render pipeline, named.
 *
 * These are the *real* step names the server creates job rows from
 * (`RENDER_VIDEO_STEPS` in the domain), not a story about what rendering
 * involves. If the backend gains or loses a step, this list goes stale in the
 * one way that matters: it stops matching, and the unmatched steps still
 * render, because the component walks the job's own steps rather than this map.
 */
const STEP_LABEL: Record<string, MessageKey> = {
  PREPARE: "export.stage.prepare",
  COMPILE: "export.stage.compile",
  RENDER: "export.stage.encode",
  PUBLISH: "export.stage.publish",
  FINALIZE: "export.stage.complete",
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

function StageRow({ step, last }: { step: JobStep; last: boolean }) {
  const t = useT();
  const label = STEP_LABEL[step.name];

  const done = step.status === "succeeded";
  const running = step.status === "running";
  const failed = step.status === "failed";

  return (
    <li className="flex gap-3">
      {/* The rail: a dot per stage joined by a line, so progress is a shape
          rather than five separate badges. */}
      <div className="flex flex-col items-center">
        <span
          aria-hidden
          className={`flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full transition-colors duration-base ${
            failed
              ? "bg-danger text-white"
              : done
                ? "bg-success/20 text-success"
                : running
                  ? "bg-accent text-accent-fg"
                  : "bg-hover text-faint"
          }`}
        >
          {failed ? (
            <Glyph name="close" size={10} />
          ) : done ? (
            <Glyph name="check" size={10} />
          ) : running ? (
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
          ) : (
            <span className="h-1 w-1 rounded-full bg-current" />
          )}
        </span>
        {!last && (
          <span
            aria-hidden
            className={`w-px flex-1 transition-colors duration-base ${
              done ? "bg-success/40" : "bg-subtle"
            }`}
          />
        )}
      </div>

      <div className="min-w-0 flex-1 pb-3">
        <p
          className={`text-xs ${running ? "font-medium text-fg" : done ? "text-muted" : "text-faint"}`}
        >
          {label ? t(label) : step.name}
        </p>
        {step.attempt > 1 && (
          <p className="mt-0.5 text-2xs text-warning">
            {t("export.stage.attempt", { attempt: step.attempt })}
          </p>
        )}
        {failed && typeof step.error?.message === "string" && (
          <p className="mt-0.5 text-2xs leading-snug text-danger">{step.error.message}</p>
        )}
      </div>
    </li>
  );
}

function RenderProgress({ job }: { job: Job }) {
  const t = useT();
  const steps = [...job.steps].sort((a, b) => a.seq - b.seq);
  const finished = ["succeeded", "failed", "cancelled"].includes(job.status);

  return (
    <Card className="p-panel">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-xs font-semibold text-fg">
          {!finished && <Spinner size={12} />}
          {t(finished ? "export.progress.done" : "export.progress.running")}
        </h3>
        <span className="font-mono text-2xs tabular-nums text-faint">
          {Math.round(job.progress * 100)}%
        </span>
      </div>

      <ol className="flex flex-col">
        {steps.map((step, index) => (
          <StageRow key={step.seq} step={step} last={index === steps.length - 1} />
        ))}
      </ol>
    </Card>
  );
}

function RenderRow({ render, onPlay }: { render: Render; onPlay: () => void }) {
  const t = useT();

  return (
    <Card className="flex items-center gap-3 p-3">
      <span
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-hover text-faint"
        aria-hidden
      >
        <Glyph name="film" size={15} />
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-mono text-xs text-fg">{render.id.slice(0, 8)}</span>
          <Badge tone={RENDER_TONE[render.status]}>{t(RENDER_LABEL[render.status])}</Badge>
        </div>
        <p className="mt-1 truncate font-mono text-2xs tabular-nums text-faint">
          {render.width ? `${render.width}×${render.height}` : t("common.dash")}
          {" · "}
          {formatFps(render.fps)} fps
          {" · "}
          {timecode(render.duration_ms, false)}
          {" · "}
          {bytes(render.bytes_size)}
          {render.metrics?.render_ms ? ` · ${seconds(Number(render.metrics.render_ms))}` : ""}
        </p>
      </div>

      <span className="shrink-0 text-2xs text-faint">
        {relativeTime(render.created_at, t.lang)}
      </span>

      {render.status === "ready" && (
        <div className="flex shrink-0 items-center gap-1">
          <Button size="sm" tone="quiet" onClick={onPlay}>
            <Glyph name="play" size={10} />
            {t("export.play")}
          </Button>
          {render.playback_url && (
            <a
              href={render.playback_url}
              download
              className="inline-flex h-control-sm shrink-0 items-center gap-1.5 rounded px-2.5 text-2xs font-medium text-muted transition-colors duration-fast hover:bg-hover hover:text-fg"
            >
              <Glyph name="download" size={10} />
              {t("export.download")}
            </a>
          )}
        </div>
      )}
    </Card>
  );
}

export function ExportWorkspace({
  projectId,
  media,
  renders,
  render,
  jobs,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  renders: Render[];
  render: Render | null;
  jobs: Job[];
}) {
  const t = useT();
  const setView = useEditorStore((s) => s.setView);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);

  /** The newest render job, whether or not it is still going. */
  const renderJob = jobs
    .filter((job) => job.type === "render_video")
    .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];

  function play() {
    setPreviewSource("render");
    setView("editor");
  }

  return (
    <div className="vf-view flex min-h-0 flex-1 gap-2 px-2 pb-1">
      <Panel className="w-[360px] shrink-0 overflow-hidden rounded-xl shadow-panel 2xl:w-[400px]">
        <PanelHeader title={t("export.title")} icon="export" />
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <ExportPanel projectId={projectId} media={media} render={render} />
        </div>
      </Panel>

      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <div className="w-full max-w-[880px] py-panel pl-panel pr-panel">
          {renderJob && <RenderProgress job={renderJob} />}

          <section className={renderJob ? "mt-panel-gap" : ""}>
            <h2 className="mb-3 text-2xs font-semibold uppercase tracking-[0.08em] text-faint">
              {t("export.history")}
            </h2>

            {renders.length === 0 ? (
              <Card className="py-2">
                <EmptyState icon="download" title={t("export.noRenders.title")}>
                  {t("export.noRenders")}
                </EmptyState>
              </Card>
            ) : (
              <ul className="flex flex-col gap-2">
                {renders.map((item) => (
                  <li key={item.id}>
                    <RenderRow render={item} onPlay={play} />
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
