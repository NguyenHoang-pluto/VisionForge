/**
 * Typed client for the VisionForge API.
 *
 * Note what is absent: no upload endpoint. File bytes go from the browser
 * straight to object storage via a presigned PUT, never through the API.
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

export interface Project {
  id: string;
  title: string;
  description: string | null;
  created_at: string;
  media_count: number;
}

export type MediaStatus =
  | "pending_upload"
  | "uploaded"
  | "processing"
  | "ready"
  | "failed";

export interface MediaAsset {
  id: string;
  project_id: string;
  original_filename: string;
  kind: "image" | "video" | "audio";
  status: MediaStatus;
  bytes_size: number | null;
  mime_type: string | null;
  sha256: string | null;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  codec: string | null;
  error: { code?: string; message?: string; hint?: string } | null;
  created_at: string;
  has_thumbnail: boolean;
  has_proxy: boolean;
}

export interface MediaListResponse {
  items: MediaAsset[];
  total: number;
}

export type JobStatus =
  | "pending"
  | "queued"
  | "running"
  | "retry_wait"
  | "succeeded"
  | "failed"
  | "cancel_requested"
  | "cancelled";

export interface JobStep {
  seq: number;
  name: string;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped";
  attempt: number;
  error: Record<string, unknown> | null;
}

export interface Job {
  id: string;
  project_id: string;
  media_id: string | null;
  type: string;
  status: JobStatus;
  attempts: number;
  max_attempts: number;
  cancel_requested: boolean;
  progress: number;
  result: Record<string, unknown> | null;
  error: { code?: string; message?: string; hint?: string } | null;
  retry_at: string | null;
  created_at: string;
  steps: JobStep[];
}

export interface UploadTicket {
  media_id: string;
  object_key: string;
  upload_url: string;
  expires_at: string;
  max_bytes: number;
}

export interface CompleteUploadResponse {
  media_id: string;
  status: string;
  job_id: string | null;
}

export interface ApiErrorBody {
  error: { code: string; message: string; hint: string | null };
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly hint: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 30_000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
      headers: { "Content-Type": "application/json", ...init.headers },
    });

    const body: unknown = await response.json().catch(() => null);

    if (!response.ok && response.status !== 503) {
      const parsed = body as ApiErrorBody | null;
      throw new ApiError(
        parsed?.error?.message ?? `Request to ${path} failed`,
        response.status,
        parsed?.error?.code ?? "unknown",
        parsed?.error?.hint ?? null,
      );
    }
    return body as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  // --- health ---
  readiness: () => request<ReadinessResponse>("/health/ready"),
  version: () => request<VersionResponse>("/version"),

  // --- projects ---
  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (title: string) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),

  // --- media ---
  listMedia: (projectId: string) =>
    request<MediaListResponse>(`/api/projects/${projectId}/media`),

  createUploadUrl: (projectId: string, filename: string, sizeBytes: number) =>
    request<UploadTicket>(`/api/projects/${projectId}/media/upload-url`, {
      method: "POST",
      body: JSON.stringify({ filename, size_bytes: sizeBytes }),
    }),

  completeUpload: (projectId: string, mediaId: string) =>
    request<CompleteUploadResponse>(
      `/api/projects/${projectId}/media/${mediaId}/complete`,
      { method: "POST" },
    ),

  thumbnailUrl: (projectId: string, mediaId: string) =>
    request<{ url: string; expires_in_s: number }>(
      `/api/projects/${projectId}/media/${mediaId}/thumbnail`,
    ),

  // --- jobs ---
  listProjectJobs: (projectId: string) =>
    request<Job[]>(`/api/projects/${projectId}/jobs`),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),

  jobEventsUrl: (jobId: string) => `${API_BASE_URL}/api/jobs/${jobId}/events`,
};

/**
 * PUT the file straight to object storage.
 *
 * Deliberately uses the raw presigned URL rather than `request()`: this call
 * does not go to the VisionForge API at all.
 */
export async function uploadToStorage(
  url: string,
  file: File,
): Promise<void> {
  const response = await fetch(url, { method: "PUT", body: file });
  if (!response.ok) {
    throw new ApiError(
      `Upload failed (${response.status})`,
      response.status,
      "upload_failed",
    );
  }
}
