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
  container_format: string | null;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  codec: string | null;
  pix_fmt: string | null;
  bit_rate: number | null;
  sample_rate: number | null;
  /** Audio channel count from ffprobe. `null` means the file has no audio. */
  channels: number | null;
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

export type AnalyzerName = "quality" | "scenes" | "phash" | "clip" | "faces" | "beats";
export type AnalysisStatus = "ok" | "unsupported" | "failed";

/** Analyzer-specific. Deliberately loose: each analyzer reports a different shape. */
export interface AnalysisRecord {
  id: string;
  media_id: string;
  analyzer: AnalyzerName;
  analyzer_version: string;
  status: AnalysisStatus;
  payload: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
  has_embedding: boolean;
  created_at: string;
}

export interface AnalysisListResponse {
  items: AnalysisRecord[];
  total: number;
}

export interface AnalyzeResponse {
  queued: number;
  media_count: number;
  lanes: string[];
  jobs: Job[];
}

export interface SimilarityHit {
  media_id: string;
  distance: number;
  similarity: number;
}

export interface SimilarMediaResponse {
  query_media_id: string;
  results: SimilarityHit[];
}

export type AspectRatio = "16:9" | "9:16" | "1:1";
export type ClipOrder = "score_desc" | "sequence";
export type RenderStatus = "pending" | "rendering" | "ready" | "failed" | "cancelled";

export type PlannerMode = "automatic" | "rules" | "ai";
export type QualityPreset = "draft" | "balanced" | "high";
export type EditStyle =
  | "cinematic"
  | "fast_montage"
  | "sports_highlight"
  | "gaming"
  | "anime"
  | "nature"
  | "social"
  | "custom";

/** What the server can plan with. Never contains a credential. */
export interface PlannerCapabilities {
  ai_available: boolean;
  provider: string | null;
  model: string | null;
  /** True when the "provider" is the deterministic local stub, not a model. */
  is_stub: boolean;
  error: string | null;
  modes: PlannerMode[];
  styles: {
    value: EditStyle;
    label: string;
    description: string;
    default_duration_ms: number;
    default_aspect: AspectRatio;
    min_clip_ms: number;
    max_clip_ms: number;
  }[];
  aspect_ratios: { value: AspectRatio; width: number; height: number }[];
  fps_presets: number[];
  quality_presets: QualityPreset[];
  prompt_version: string;
  max_request_chars: number;
  /**
   * The bounds the server's plan validator enforces.
   *
   * The timeline clamps a drag against these while the pointer is moving, so it
   * needs them client-side. Taking them from here rather than hard-coding them
   * is what keeps the browser's idea of a legal trim identical to the one that
   * is actually enforced.
   */
  segment_bounds: {
    min_clip_ms: number;
    max_clip_ms: number;
    max_clips: number;
    min_total_ms: number;
    max_total_ms: number;
  };

  /**
   * The same declaration for audio.
   *
   * A volume slider and two fade handles need clamping while the pointer is
   * moving, for exactly the reason a trim handle does -- and a copy of the
   * numbers in the browser would drift out of agreement with the validator
   * that actually enforces them.
   */
  audio_bounds: {
    min_gain: number;
    max_gain: number;
    min_music_ms: number;
    max_music_ms: number;
    max_fade_ms: number;
  };

  /**
   * What beat detection reports, and when the planner will act on it.
   *
   * The confidence floor is here so the editor can explain a grid the server
   * has decided not to trust, rather than offering a toggle that silently does
   * nothing.
   */
  beat_sync: {
    analyzer: AnalyzerName;
    min_bpm: number;
    max_bpm: number;
    min_confidence: number;
  };
}

/**
 * A music bed, as the API takes it.
 *
 * Note what is absent, and note that it is the same list absent from every
 * other request type here: no path, no storage key, no codec, no filter. A cue
 * is a media id the project already owns plus six numbers.
 */
export interface MusicRequest {
  media_id: string;
  source_in_ms: number;
  source_out_ms: number;
  timeline_start_ms: number;
  /** Linear; 1.0 is unity. Shown to the user as a percentage. */
  volume: number;
  fade_in_ms: number;
  fade_out_ms: number;
}

/** The cue as it comes back on a stored plan. Carries its derived duration. */
export interface PlanMusic extends MusicRequest {
  duration_ms: number;
  gain: number;
}

/** Tempo and beat positions, from the `beats` analyzer. */
export interface BeatsPayload {
  bpm: number;
  confidence: number;
  beat_count: number;
  beats_ms: number[];
  source_duration_ms: number | null;
}

/** Which planner ran, and why. Present on every plan. */
export interface ModeRecord {
  mode: PlannerMode;
  reason: string;
  inferred_style: EditStyle | null;
}

/** Provenance for a plan a model was involved in. Never contains a key. */
export interface LlmRecord {
  provider: string | null;
  model: string | null;
  prompt_version: string;
  status: string;
  attempts: number;
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  request_id: string | null;
  fallback_reason: string | null;
  fallback_detail: string | null;
  violations: { code: string; message: string }[];
}

export interface PlanSegment {
  media_id: string;
  order: number;
  source_in_ms: number;
  source_out_ms: number;
  duration_ms: number;
  transition_in: string;
}

export interface PlanDocument {
  project_id: string;
  planner: string;
  planner_version: string;
  output: {
    aspect_ratio: AspectRatio;
    width: number;
    height: number;
    fps: number;
    fit: string;
    audio: string;
    quality: QualityPreset;
    source_gain: number;
  };
  music: PlanMusic | null;
  segments: PlanSegment[];
  total_duration_ms: number;
  metadata: Record<string, unknown>;
}

export interface SelectionRecord {
  selected: { media_id: string; score: number; components: Record<string, number> }[];
  rejected: { media_id: string; reason: string; detail: string | null }[];
  duplicate_groups: string[][];
}

export interface EditPlan {
  id: string;
  project_id: string;
  planner: string;
  planner_version: string;
  segment_count: number;
  total_duration_ms: number;
  created_at: string;
  plan: PlanDocument;
  selection: SelectionRecord;
  mode: ModeRecord | null;
  llm: LlmRecord | null;
}

export interface Render {
  id: string;
  project_id: string;
  edit_plan_id: string;
  job_id: string | null;
  status: RenderStatus;
  bytes_size: number | null;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  spec: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  error: { code?: string; message?: string; hint?: string } | null;
  created_at: string;
  playback_url: string | null;
  playback_expires_in_s: number | null;
}

/** One clip on a hand-cut timeline. Position in the array is the edit order. */
export interface ManualCut {
  media_id: string;
  source_in_ms: number;
  source_out_ms: number;
}

/**
 * A timeline the user assembled, submitted for validation.
 *
 * Carries intent only. There is no width, height, CRF or path here for the
 * same reason there is none on `PlanOptions` -- the server owns geometry and
 * encoder settings, and the editor never learns them.
 */
export interface ManualPlanOptions {
  segments: ManualCut[];
  aspect_ratio?: AspectRatio;
  fps?: number;
  quality?: QualityPreset;
  audio?: "none" | "source";
  /** Gain on the clips' own audio, independent of the bed. */
  source_gain?: number;
  /** The bed the editor placed, if any. */
  music?: MusicRequest | null;
  derived_from_edit_plan_id?: string | null;
}

export interface PlanOptions {
  mode?: PlannerMode;
  style?: EditStyle | null;
  /** The user's own words. Bounded server-side; never stored, only digested. */
  request_text?: string | null;
  target_duration_ms?: number;
  max_clips?: number;
  min_clips?: number;
  aspect_ratio?: AspectRatio;
  fps?: number;
  order?: ClipOrder;
  quality?: QualityPreset;
  source_gain?: number;
  music?: MusicRequest | null;
  /** Let the track's detected beats decide clip length. */
  beat_sync?: boolean;
}

/** One planning call, successful or not. */
export interface LlmRun {
  id: string;
  edit_plan_id: string | null;
  provider: string;
  model: string;
  prompt_version: string;
  status: string;
  attempts: number;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  fallback_reason: string | null;
  fallback_detail: string | null;
  created_at: string;
}

export interface SignedUrl {
  url: string;
  expires_in_s: number;
}

/** What a plan listing returns: provenance without the document. */
export type EditPlanSummary = Omit<EditPlan, "plan" | "selection">;

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
    request<SignedUrl>(`/api/projects/${projectId}/media/${mediaId}/thumbnail`),

  /**
   * The 720p proxy, for scrubbing.
   *
   * Always the proxy, never the original: a 4K source would stall the preview
   * and the network on every seek, and the proxy exists precisely so that the
   * editor never touches the master file.
   */
  proxyUrl: (projectId: string, mediaId: string) =>
    request<SignedUrl>(`/api/projects/${projectId}/media/${mediaId}/proxy`),

  // --- jobs ---
  listProjectJobs: (projectId: string) =>
    request<Job[]>(`/api/projects/${projectId}/jobs`),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),

  jobEventsUrl: (jobId: string) => `${API_BASE_URL}/api/jobs/${jobId}/events`,

  // --- analysis ---
  requestAnalysis: (projectId: string, mediaId?: string, lanes?: string[]) =>
    request<AnalyzeResponse>(`/api/projects/${projectId}/analysis`, {
      method: "POST",
      body: JSON.stringify({ media_id: mediaId ?? null, lanes: lanes ?? null }),
    }),

  mediaAnalysis: (projectId: string, mediaId: string) =>
    request<AnalysisListResponse>(
      `/api/projects/${projectId}/media/${mediaId}/analysis`,
    ),

  projectAnalysis: (projectId: string) =>
    request<AnalysisListResponse>(`/api/projects/${projectId}/analysis`),

  similarMedia: (projectId: string, mediaId: string) =>
    request<SimilarMediaResponse>(
      `/api/projects/${projectId}/media/${mediaId}/similar`,
    ),

  // --- edit plans and renders ---
  createEditPlan: (projectId: string, options: PlanOptions = {}) =>
    request<EditPlan>(`/api/projects/${projectId}/edit-plan`, {
      method: "POST",
      body: JSON.stringify(options),
    }),

  /**
   * Plan summaries. Note the type: the listing carries no `plan` document, by
   * design -- twenty plans would mean twenty embedded segment arrays. Use
   * `getEditPlan` for the one being looked at.
   */
  listEditPlans: (projectId: string) =>
    request<{ items: EditPlanSummary[]; total: number }>(
      `/api/projects/${projectId}/edit-plan`,
    ),

  getEditPlan: (projectId: string, editPlanId: string) =>
    request<EditPlan>(`/api/projects/${projectId}/edit-plan/${editPlanId}`),

  /** Store a timeline the user cut by hand. The server validates it. */
  createManualEditPlan: (projectId: string, options: ManualPlanOptions) =>
    request<EditPlan>(`/api/projects/${projectId}/edit-plan/manual`, {
      method: "POST",
      body: JSON.stringify(options),
    }),

  createRender: (projectId: string, editPlanId: string) =>
    request<Render>(`/api/projects/${projectId}/render`, {
      method: "POST",
      body: JSON.stringify({ edit_plan_id: editPlanId }),
    }),

  /**
   * Render summaries. Like the plan listing, deliberately thinner than the
   * detail: `playback_url` is presigned per call and is only issued by
   * `getRender`.
   */
  listRenders: (projectId: string) =>
    request<{ items: Render[]; total: number }>(
      `/api/projects/${projectId}/renders`,
    ),

  getRender: (projectId: string, renderId: string) =>
    request<Render>(`/api/projects/${projectId}/renders/${renderId}`),

  // --- planner ---
  plannerCapabilities: () => request<PlannerCapabilities>("/api/planner/capabilities"),

  listLlmRuns: (projectId: string) =>
    request<{ items: LlmRun[]; total: number }>(`/api/projects/${projectId}/llm-runs`),
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
