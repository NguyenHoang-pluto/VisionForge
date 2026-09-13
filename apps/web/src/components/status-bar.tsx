"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, APP_VERSION, type Job, type JobStatus } from "@/lib/api";
import { useJobEvents, type JobEvent } from "@/lib/use-job-events";
import { Badge, Button, Glyph, IconButton, ProgressBar } from "@/components/ui";

/**
 * What VisionForge is doing right now.
 *
 * Collapsed to a single strip by default, because most of the time the answer
 * is "nothing" and a job list that is always open costs the timeline forty
 * pixels forever. Expanded, it is the full queue.
 *
 * Progress comes from the SSE stream the backend already exposes; the periodic
 * list fetch is only there to discover jobs this tab has not seen yet, and it
 * stops the moment nothing is running. An SSE frame always wins over the list,
 * because it is newer by construction.
 */

const TERMINAL: ReadonlySet<JobStatus> = new Set(["succeeded", "failed", "cancelled"]);

const STATUS_TONE: Record<JobStatus, "neutral" | "info" | "warn" | "ok" | "danger"> = {
  pending: "neutral",
  queued: "info",
  running: "warn",
  retry_wait: "warn",
  succeeded: "ok",
  failed: "danger",
  cancel_requested: "warn",
  cancelled: "neutral",
};

const BAR_TONE: Record<JobStatus, "accent" | "ok" | "warn" | "danger"> = {
  pending: "accent",
  queued: "accent",
  running: "accent",
  retry_wait: "warn",
  succeeded: "ok",
  failed: "danger",
  cancel_requested: "warn",
  cancelled: "accent",
};

const JOB_LABEL: Record<string, string> = {
  media_ingest: "ingest",
  media_analysis: "analysis",
  render_video: "render",
};

function retryHint(retryAt: string | null): string | null {
  if (!retryAt) return null;
  const remaining = Math.max(0, Math.round((Date.parse(retryAt) - Date.now()) / 1000));
  return `retrying in ${remaining}s`;
}

function JobRow({
  job,
  event,
  onCancel,
}: {
  job: Job;
  event: JobEvent | undefined;
  onCancel: () => void;
}) {
  // Where both exist, the SSE frame is fresher than the last list fetch.
  const status = event?.status ?? job.status;
  const progress = event?.progress ?? job.progress;
  const stepsTotal = event?.steps_total ?? job.steps.length;
  const stepsDone =
    event?.steps_done ??
    job.steps.filter((step) => step.status === "succeeded" || step.status === "skipped").length;
  const attempt = event?.attempt ?? job.attempts;
  const maxAttempts = event?.max_attempts ?? job.max_attempts;

  return (
    <li className="flex flex-col gap-0.5 px-2 py-1">
      <div className="flex items-center gap-2">
        <Badge tone={STATUS_TONE[status]}>{status.replace("_", " ")}</Badge>
        <span className="font-mono text-2xs text-muted">
          {JOB_LABEL[job.type] ?? job.type}
        </span>
        <span className="font-mono text-2xs text-dim">{job.id.slice(0, 8)}</span>
        {attempt > 1 && (
          <span className="font-mono text-2xs text-warn">
            attempt {attempt}/{maxAttempts}
          </span>
        )}

        <span className="ml-auto flex items-center gap-2">
          <span className="font-mono text-2xs text-dim tabular-nums">
            {event?.current_step ? `${event.current_step} · ` : ""}
            {stepsDone}/{stepsTotal} · {Math.round(progress * 100)}%
          </span>
          {!TERMINAL.has(status) && (
            <Button size="sm" tone="ghost" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </span>
      </div>

      <ProgressBar fraction={progress} tone={BAR_TONE[status]} label={`${job.type} progress`} />

      {status === "retry_wait" && (
        <p className="text-2xs text-warn">{retryHint(event?.retry_at ?? job.retry_at)}</p>
      )}
      {status === "failed" && (
        <p className="text-2xs leading-snug text-danger">
          {event?.error?.message ?? job.error?.message ?? "Job failed"}
        </p>
      )}
      {Boolean(job.result?.duplicate) && (
        <p className="text-2xs text-muted">
          Duplicate of an asset already in this project — not added again.
        </p>
      )}
    </li>
  );
}

export function StatusBar({
  projectId,
  jobs,
  onCancel,
}: {
  projectId: string | null;
  jobs: Job[];
  onCancel: (jobId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const active = jobs.filter((job) => !TERMINAL.has(job.status));
  const events = useJobEvents(active.map((job) => job.id));

  const readiness = useQuery({
    queryKey: ["health", "ready"],
    queryFn: api.readiness,
    refetchInterval: 30_000,
    retry: false,
  });

  const failed = jobs.filter((job) => job.status === "failed").length;
  const apiReachable = !readiness.isError;
  const infraOk =
    apiReachable && (readiness.data?.components ?? []).every((c) => c.status === "ok");

  // One overall progress figure while work is in flight: the strip should
  // answer "how far along" without being expanded.
  const overall =
    active.length > 0
      ? active.reduce(
          (sum, job) => sum + (events[job.id]?.progress ?? job.progress),
          0,
        ) / active.length
      : 0;

  return (
    <section className="shrink-0 border-t border-line bg-raised" aria-label="Status">
      {expanded && (
        <ul className="max-h-40 divide-y divide-line/50 overflow-y-auto border-b border-line">
          {jobs.length === 0 ? (
            <li className="px-2 py-3 text-center text-2xs text-dim">
              No jobs in this project yet.
            </li>
          ) : (
            jobs
              .slice(0, 20)
              .map((job) => (
                <JobRow
                  key={job.id}
                  job={job}
                  event={events[job.id]}
                  onCancel={() => onCancel(job.id)}
                />
              ))
          )}
        </ul>
      )}

      <div className="flex h-[24px] items-center gap-2 px-2">
        <IconButton
          label={expanded ? "Hide jobs" : "Show jobs"}
          active={expanded}
          onClick={() => setExpanded(!expanded)}
          className="h-[20px] w-[20px]"
        >
          <Glyph name={expanded ? "chevron-right" : "chevron-left"} size={11} />
        </IconButton>

        <span className="font-mono text-2xs text-muted tabular-nums">
          {active.length > 0
            ? `${active.length} job${active.length === 1 ? "" : "s"} running`
            : "idle"}
        </span>

        {active.length > 0 && (
          <span className="w-24">
            <ProgressBar fraction={overall} label="Overall job progress" />
          </span>
        )}

        {failed > 0 && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="font-mono text-2xs text-danger hover:underline"
          >
            {failed} failed
          </button>
        )}

        <span className="ml-auto flex items-center gap-2 font-mono text-2xs text-dim">
          <span
            title={
              apiReachable
                ? infraOk
                  ? "API and infrastructure healthy"
                  : "Some infrastructure components are degraded"
                : "Cannot reach the API"
            }
            className="flex items-center gap-1"
          >
            <span
              aria-hidden
              className={`h-1.5 w-1.5 rounded-full ${
                !apiReachable ? "bg-danger" : infraOk ? "bg-ok" : "bg-warn"
              }`}
            />
            {!apiReachable ? "api unreachable" : infraOk ? "healthy" : "degraded"}
          </span>
          {projectId && <span>project {projectId.slice(0, 8)}</span>}
          <span>v{APP_VERSION}</span>
        </span>
      </div>
    </section>
  );
}
