"use client";

import type { Job, JobStatus } from "@/lib/api";
import type { JobEvent } from "@/lib/use-job-events";

const STATUS_STYLES: Record<JobStatus, string> = {
  pending: "text-slate-500",
  queued: "text-sky-400",
  running: "text-amber-400",
  retry_wait: "text-orange-400",
  succeeded: "text-emerald-400",
  failed: "text-rose-400",
  cancel_requested: "text-orange-400",
  cancelled: "text-slate-500",
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
    <section className="border-t border-slate-800">
      <h2 className="border-b border-slate-800 px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-600">
        Jobs
      </h2>

      <ul className="flex max-h-56 flex-col divide-y divide-slate-800/60 overflow-y-auto">
        {jobs.slice(0, 12).map((job) => {
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
            <li key={job.id} className="flex flex-col gap-1 px-3 py-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`font-mono text-[10px] uppercase ${STATUS_STYLES[status]}`}>
                    {status.replace("_", " ")}
                  </span>
                  <span className="font-mono text-[10px] text-slate-600">
                    {job.id.slice(0, 8)}
                  </span>
                  <span className="font-mono text-[10px] text-slate-600">
                    {job.type.replace("media_", "")}
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
                    className="rounded-sm border border-slate-800 px-1.5 py-px text-[10px] text-slate-500 transition hover:border-rose-800 hover:text-rose-400 focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600"
                  >
                    Cancel
                  </button>
                )}
              </div>

              <div
                className="h-0.5 overflow-hidden bg-slate-800"
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
