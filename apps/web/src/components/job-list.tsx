"use client";

import type { Job, JobStatus } from "@/lib/api";
import type { JobEvent } from "@/lib/use-job-events";

const STATUS_STYLES: Record<JobStatus, string> = {
  pending: "bg-slate-500/10 text-slate-400",
  queued: "bg-sky-500/10 text-sky-400",
  running: "bg-amber-500/10 text-amber-400",
  retry_wait: "bg-orange-500/10 text-orange-400",
  succeeded: "bg-emerald-500/10 text-emerald-400",
  failed: "bg-rose-500/10 text-rose-400",
  cancel_requested: "bg-orange-500/10 text-orange-400",
  cancelled: "bg-slate-500/10 text-slate-400",
};

const TERMINAL: ReadonlySet<JobStatus> = new Set([
  "succeeded",
  "failed",
  "cancelled",
]);

function retryHint(retryAt: string | null): string | null {
  if (!retryAt) return null;
  const seconds = Math.max(0, Math.round((Date.parse(retryAt) - Date.now()) / 1000));
  return `Retrying in ${seconds}s`;
}

function barColor(status: JobStatus): string {
  if (status === "failed") return "bg-rose-500";
  if (status === "succeeded") return "bg-emerald-500";
  if (status === "cancelled") return "bg-slate-600";
  return "bg-sky-500";
}

export function JobList({
  jobs,
  live,
  onCancel,
}: {
  jobs: Job[];
  live: Record<string, JobEvent>;
  onCancel: (jobId: string) => void;
}) {
  if (jobs.length === 0) return null;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
      <h2 className="text-sm font-semibold text-slate-200">Processing jobs</h2>

      <ul className="mt-4 flex flex-col divide-y divide-slate-800">
        {jobs.slice(0, 8).map((job) => {
          // When both exist, the SSE frame is fresher than the last list fetch.
          const event = live[job.id];
          const status = event?.status ?? job.status;
          const progress = event?.progress ?? job.progress;
          const stepsTotal = event?.steps_total ?? job.steps.length;
          const stepsDone =
            event?.steps_done ??
            job.steps.filter((s) => s.status === "succeeded" || s.status === "skipped")
              .length;
          const attempt = event?.attempt ?? job.attempts;
          const maxAttempts = event?.max_attempts ?? job.max_attempts;

          return (
            <li key={job.id} className="flex flex-col gap-2 py-3 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase ${STATUS_STYLES[status]}`}
                  >
                    {status.replace("_", " ")}
                  </span>
                  <span className="font-mono text-xs text-slate-500">
                    {job.id.slice(0, 8)}
                  </span>
                  {attempt > 1 && (
                    <span className="font-mono text-[10px] text-orange-400">
                      Attempt {attempt}/{maxAttempts}
                    </span>
                  )}
                </div>

                {!TERMINAL.has(status) && (
                  <button
                    type="button"
                    onClick={() => onCancel(job.id)}
                    className="rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-400 transition hover:border-rose-800 hover:text-rose-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
                  >
                    Cancel
                  </button>
                )}
              </div>

              <div
                className="h-1 overflow-hidden rounded bg-slate-800"
                role="progressbar"
                aria-valuenow={Math.round(progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className={`h-full transition-all ${barColor(status)}`}
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>

              <div className="flex flex-wrap justify-between gap-2 font-mono text-[10px] text-slate-500">
                <span>
                  {event?.current_step
                    ? `${event.current_step} · ${stepsDone}/${stepsTotal} steps`
                    : `${stepsDone}/${stepsTotal} steps`}
                </span>
                <span className="tabular-nums">{Math.round(progress * 100)}%</span>
              </div>

              {status === "retry_wait" && (
                <p className="text-[10px] text-orange-400">
                  {retryHint(event?.retry_at ?? job.retry_at)}
                </p>
              )}

              {status === "failed" && (
                <p className="text-[10px] leading-snug text-rose-400">
                  {event?.error?.message ?? job.error?.message ?? "Job failed"}
                </p>
              )}

              {Boolean(job.result?.duplicate) && (
                <p className="text-[10px] text-slate-400">
                  Duplicate of an asset already in this project — not added again.
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
