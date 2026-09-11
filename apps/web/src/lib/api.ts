/**
 * Typed client for the VisionForge API.
 *
 * The base URL is injected at build time so the same bundle can point at a local
 * API, a staging API, or a production API without a code change.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const APP_VERSION = process.env.NEXT_PUBLIC_APP_VERSION ?? "0.1.0";

export type ComponentStatus = "ok" | "degraded" | "failed";

export interface ComponentHealth {
  name: string;
  status: ComponentStatus;
  latency_ms: number | null;
  detail: string | null;
}

export interface ReadinessResponse {
  status: "ok" | "not_ready";
  components: ComponentHealth[];
}

export interface VersionResponse {
  name: string;
  version: string;
  environment: string;
}

/** Thrown when the API is reachable but answers with an error status. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, timeoutMs = 4000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      signal: controller.signal,
      cache: "no-store",
    });

    // 503 from /health/ready carries a useful body, so parse before throwing.
    const body = (await response.json()) as T;
    if (!response.ok && response.status !== 503) {
      throw new ApiError(`Request to ${path} failed`, response.status);
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  readiness: () => request<ReadinessResponse>("/health/ready"),
  version: () => request<VersionResponse>("/version"),
};
