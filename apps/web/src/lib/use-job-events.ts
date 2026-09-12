"use client";

import { useEffect, useState } from "react";

import { api, type JobStatus } from "@/lib/api";

/**
 * Live job progress over Server-Sent Events.
 *
 * The server sends the current state as the first frame, so a page refresh
 * mid-job is immediately correct rather than blank until the next event.
 * `EventSource` reconnects on its own, which is most of why SSE was chosen over
 * WebSockets for one-directional progress.
 */
export interface JobEvent {
  job_id: string;
  status: JobStatus;
  progress: number;
  steps_done: number;
  steps_total: number;
  current_step: string | null;
  attempt?: number;
  max_attempts?: number;
  retry_at: string | null;
  error: { code?: string; message?: string; hint?: string } | null;
}

const TERMINAL: ReadonlySet<JobStatus> = new Set([
  "succeeded",
  "failed",
  "cancelled",
]);

export function useJobEvents(jobIds: string[]): Record<string, JobEvent> {
  const [events, setEvents] = useState<Record<string, JobEvent>>({});
  // Join on a stable key so the effect does not re-run on every array identity.
  const key = jobIds.join(",");

  useEffect(() => {
    if (!key) return;

    const sources = key.split(",").map((jobId) => {
      const source = new EventSource(api.jobEventsUrl(jobId));

      source.addEventListener("state", (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as JobEvent;
          setEvents((previous) => ({ ...previous, [jobId]: payload }));
          if (TERMINAL.has(payload.status)) source.close();
        } catch {
          // A malformed frame must not take down the stream.
        }
      });

      // The browser retries automatically; closing here would defeat that.
      source.onerror = () => {
        if (source.readyState === EventSource.CLOSED) source.close();
      };

      return source;
    });

    return () => sources.forEach((source) => source.close());
  }, [key]);

  return events;
}
