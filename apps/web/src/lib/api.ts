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
  /**
   * The Phase 9 vocabulary, declared by the server.
   *
   * The same argument as `segment_bounds`: the editor has to grey out a
   * transition the clips are too short for while the pointer is moving, and a
   * slider whose range disagrees with the validator is a slider that produces
   * a 422. These are read once and applied over the module's defaults.
   */
  transitions: {
    value: TransitionKind;
    consumes_time: boolean;
    needs_previous: boolean;
    min_ms: number;
    max_ms: number;
    max_share: number;
  }[];
  effects: {
    kind: EffectKind;
    minimum: number;
    maximum: number;
    neutral: number;
    whole_segment_only: boolean;
  }[];
  /** What each preset actually looks like, so the panel previews it honestly
   *  rather than guessing at the font the renderer will use. */
  subtitle_styles: { id: SubtitleStyle; label: string; font: string; size: number; bold: boolean }[];
  subtitle_positions: SubtitlePosition[];
  subtitle_bounds: {
    min_cue_ms: number;
    max_cue_ms: number;
    max_chars: number;
    max_cues: number;
    max_effects_per_clip: number;
  };

  beat_sync: {
    analyzer: AnalyzerName;
    min_bpm: number;
    max_bpm: number;
    min_confidence: number;
  };

  /**
   * The closed vocabulary a co-edit may use (Phase 10).
   *
   * Declared for the same reason every other bound is: the panel should offer
   * what the parser accepts, and an operation removed on the server should stop
   * being offered rather than become a 422.
   */
  operations: { kind: OperationKind }[];
  coedit_prompt_version: string;
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
  /** Present from Phase 9. Absent on plans stored before it. */
  transition_ms?: number;
  effects?: EffectRequest[];
}

/** Subtitles as the plan records them: ids, cues, and no styling. */
export interface PlanSubtitles {
  cues: { start_ms: number; end_ms: number; text: string }[];
  style: SubtitleStyle;
  position: SubtitlePosition;
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
  subtitles?: PlanSubtitles | null;
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

/**
 * How a clip enters. A closed set, validated server-side (Phase 9).
 *
 * Only `crossfade` changes the length of the edit -- it overlaps the previous
 * clip -- which is why the editor computes durations from the same rule the
 * server does rather than summing clip lengths.
 */
export type TransitionKind = "cut" | "crossfade" | "fade_in" | "fade_to_black";

export const TRANSITIONS: TransitionKind[] = ["cut", "crossfade", "fade_in", "fade_to_black"];

/** Transitions that overlap the previous clip and shorten the programme. */
export const OVERLAPPING: TransitionKind[] = ["crossfade"];

export type EffectKind =
  | "zoom_in"
  | "zoom_out"
  | "slow_motion"
  | "speed_up"
  | "brightness"
  | "contrast"
  | "saturation";

/**
 * One effect: a kind and a number, with the number's meaning set by the kind.
 *
 * There is no parameter object here for the same reason there is none on the
 * server: a free-form bag on a renderer instruction is a hole in the shape of
 * an arbitrary filter argument.
 */
export interface EffectRequest {
  kind: EffectKind;
  amount: number;
  /** Offsets inside the clip. Only the colour effects accept a window. */
  start_ms?: number | null;
  end_ms?: number | null;
}

/** The server's bounds per kind, mirrored so a slider cannot offer an invalid
 *  value. The server validates regardless; this is about not showing the user a
 *  control whose range would be refused. */
export const EFFECT_BOUNDS: Record<
  EffectKind,
  { min: number; max: number; neutral: number; step: number; wholeClipOnly: boolean }
> = {
  zoom_in: { min: 0, max: 0.3, neutral: 0, step: 0.01, wholeClipOnly: true },
  zoom_out: { min: 0, max: 0.3, neutral: 0, step: 0.01, wholeClipOnly: true },
  slow_motion: { min: 0.25, max: 1, neutral: 1, step: 0.05, wholeClipOnly: true },
  speed_up: { min: 1, max: 4, neutral: 1, step: 0.05, wholeClipOnly: true },
  brightness: { min: -1, max: 1, neutral: 0, step: 0.05, wholeClipOnly: false },
  contrast: { min: 0.5, max: 1.5, neutral: 1, step: 0.05, wholeClipOnly: false },
  saturation: { min: 0, max: 2, neutral: 1, step: 0.05, wholeClipOnly: false },
};

export const TRANSITION_BOUNDS = { min: 80, max: 4000 } as const;

export type SubtitleStyle = "clean" | "bold" | "minimal" | "cinematic" | "social";
export type SubtitlePosition = "bottom" | "center" | "top" | "bottom_left" | "bottom_right";

export const SUBTITLE_STYLES: SubtitleStyle[] = [
  "clean",
  "bold",
  "minimal",
  "cinematic",
  "social",
];
export const SUBTITLE_POSITIONS: SubtitlePosition[] = [
  "bottom",
  "center",
  "top",
  "bottom_left",
  "bottom_right",
];

/** Cue bounds, mirrored from the server so the editor can refuse locally too. */
export const CUE_BOUNDS = { minMs: 400, maxMs: 10_000, maxChars: 120, maxCues: 300 } as const;

export const MIN_CUE_MS = CUE_BOUNDS.minMs;

/** A transition may eat at most half of the shorter side it joins. Mirrors
 *  `MAX_TRANSITION_SHARE` in the domain, for the same reason the other bounds
 *  are mirrored: to disable a control rather than to offer a 422. */
export const MAX_TRANSITION_SHARE = 0.5;

export interface SubtitleCueRequest {
  start_ms: number;
  end_ms: number;
  text: string;
}

/**
 * Subtitles as the API takes them.
 *
 * `style` and `position` are ids. There is no font, size, colour or coordinate
 * here: the server's preset table decides all of them, which is what keeps a
 * font path out of the render.
 */
export interface SubtitleTrackRequest {
  cues: SubtitleCueRequest[];
  style: SubtitleStyle;
  position: SubtitlePosition;
}

/**
 * What the AI subtitle route answers.
 *
 * Two states and no third: `ok` with a track, or not-ok with a `failure` naming
 * which of the known failures happened. There is no partial result and nothing
 * is filled in when the model could not answer -- an empty panel that says why
 * beats subtitles nobody wrote.
 */
export type SubtitleFailure =
  | "provider_disabled"
  | "provider_unavailable"
  | "provider_error"
  | "unreadable"
  | "no_usable_cues"
  | "timeline_too_short";

export interface SubtitleSuggestion {
  ok: boolean;
  subtitles: SubtitleTrackRequest | null;
  failure: SubtitleFailure | null;
  detail: string;
  provider: string;
  model: string;
  prompt_version: string;
  latency_ms: number;
}

/** One clip on a hand-cut timeline. Position in the array is the edit order. */
export interface ManualCut {
  media_id: string;
  source_in_ms: number;
  source_out_ms: number;
  transition_in?: TransitionKind;
  transition_ms?: number;
  effects?: EffectRequest[];
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
  /** Subtitles the editor wrote (Phase 9). */
  subtitles?: SubtitleTrackRequest | null;
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
  /**
   * How much of the project's reference video to apply (Phase 8).
   *
   * There is deliberately no field naming the reference itself: which clip that
   * is belongs to the project and is resolved server-side, so a request cannot
   * point at media it does not own.
   */
  style_strength?: StyleStrength;
}

/** The five stops on the style dial. A closed set, validated server-side. */
export type StyleStrength = "0" | "25" | "50" | "75" | "100";

export const STYLE_STRENGTHS: StyleStrength[] = ["0", "25", "50", "75", "100"];

/**
 * A measured value and how far it should be trusted.
 *
 * The two travel together rather than being folded into one number, because a
 * confidently-measured zero and an unmeasured field are different facts and the
 * UI has to be able to show the difference.
 */
export interface Measurement {
  value: number;
  confidence: number;
}

/**
 * What a reference video measured as.
 *
 * Every field is nullable, and every null means "not measured" -- never zero,
 * never an average. Nothing here is a path, a key or a filename: the profile is
 * numbers and bounded enums by construction.
 */
export interface ReferenceProfile {
  media_id: string;
  version: string;
  duration_ms: number;
  confidence: number;
  usable: boolean;

  pacing: "slow" | "measured" | "brisk" | "rapid" | null;
  scene_count: number | null;
  shot_ms: Measurement | null;
  shot_ms_p25: number | null;
  shot_ms_p75: number | null;
  cut_rate: Measurement | null;

  luminance: Measurement | null;
  contrast: Measurement | null;
  saturation: Measurement | null;
  motion: Measurement | null;

  bpm: number | null;
  beat_confidence: number | null;
  beat_sync: Measurement | null;
  /** Advisory: the server never switches beat sync on by itself. */
  suggests_beat_sync: boolean;
}

export interface ReferenceState {
  media_id: string | null;
  /** Analyzers this reference still owes, so the UI can say what to run. */
  pending_analyzers: string[];
  profile: ReferenceProfile | null;
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

/**
 * The operations a co-edit may ask for. A closed set, mirrored from the server.
 *
 * Note what a client cannot construct: there is no operation here that names a
 * file, a path, a filter or an encoder setting, because there is none on the
 * server either. The mirror is for labelling and for disabling controls -- the
 * server validates regardless.
 */
export type OperationKind =
  | "REMOVE_SEGMENT"
  | "REORDER_SEGMENT"
  | "TRIM_SEGMENT"
  | "CHANGE_DURATION"
  | "CHANGE_STYLE_STRENGTH"
  | "CHANGE_TRANSITION"
  | "ADD_EFFECT"
  | "REMOVE_EFFECT"
  | "MODIFY_EFFECT"
  | "ADD_SUBTITLE"
  | "MODIFY_SUBTITLE"
  | "REMOVE_SUBTITLE"
  | "CHANGE_MUSIC_VOLUME"
  | "CHANGE_MUSIC_FADE"
  | "CHANGE_BEAT_SYNC"
  | "CHANGE_OUTPUT_PRESET";

/**
 * One operation, as it travels.
 *
 * Deliberately loose on this side: the fields differ per kind and the server
 * owns the schema. The editor never builds one of these by hand -- it sends a
 * sentence and hands back whatever the preview returned -- so a precise union
 * here would be a second copy of the vocabulary to keep in step for no gain.
 */
export interface EditOperation {
  kind: OperationKind;
  [field: string]: unknown;
}

/** One line of the before/after the panel draws. Already formatted by the server. */
export interface DiffEntry {
  field: string;
  label: string;
  before: string;
  after: string;
  /** 1-based, when the change is about one clip. */
  clip: number | null;
}

export interface PlanDiff {
  entries: DiffEntry[];
  /** What the operations said they would do, in their own words. */
  applied: string[];
  /** Adjustments the server made on its own to keep the plan renderable. */
  adjustments: string[];
}

/** Why a change produced nothing. Named, so the panel can explain rather than shrug. */
export type CoEditFailure =
  | "provider_disabled"
  | "provider_unavailable"
  | "provider_error"
  | "unreadable"
  | "no_usable_operations"
  | "not_understood"
  | "empty_request";

/** Which resolver read the request. `rules` never touches a model. */
export type CoEditSource = "rules" | "llm" | "client";

/**
 * What a change would do. Nothing has been written when this comes back.
 *
 * `needs_confirmation` is false when the server's own rules resolved the
 * request, which is the signal the panel uses to apply a simple change
 * immediately instead of showing a diff nobody needs to read.
 */
export interface CoEditPreview {
  ok: boolean;
  base_version_id: string | null;
  base_version: number | null;
  operations: EditOperation[];
  rationale: string;
  diff: PlanDiff;
  failure: CoEditFailure | null;
  detail: string;
  source: CoEditSource;
  provider: string;
  model: string;
  latency_ms: number;
  violations: { code: string; message: string; index: number | null }[];
  needs_confirmation: boolean;
}

/**
 * One step in the edit history.
 *
 * No request text, only a digest: the server stores a fingerprint of what was
 * asked and never the words, so there is nothing else to send.
 */
export interface EditVersion {
  id: string;
  version: number;
  parent_id: string | null;
  edit_plan_id: string;
  origin: "generated" | "manual" | "co_edit" | "undo" | "redo";
  is_current: boolean;
  applied: string[];
  operation_count: number;
  source: string;
  provider: string | null;
  model: string | null;
  latency_ms: number | null;
  request_digest: string | null;
  total_duration_ms: number;
  segment_count: number;
  created_at: string;
  render_id: string | null;
  render_status: RenderStatus | null;
}

export interface EditVersionList {
  items: EditVersion[];
  total: number;
  current_version_id: string | null;
  /** Decided by the server that owns the history, not guessed from the list. */
  can_undo: boolean;
  can_redo: boolean;
}

/** A committed change: the new version, its plan, and what it did. */
export interface CoEditResult {
  version: EditVersion;
  plan: EditPlan;
  diff: PlanDiff;
  rationale: string;
  source: CoEditSource;
  provider: string;
  model: string;
  latency_ms: number;
}

/**
 * A change request.
 *
 * Either words or operations from a preview the user approved. There is no
 * field here for a plan, a segment's media id or an output setting: a change
 * names what to change about the edit the server already holds.
 */
export interface CoEditRequest {
  request_text?: string | null;
  operations?: EditOperation[] | null;
  base_version_id?: string | null;
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

  /**
   * Ask a model to draft subtitles for a stored plan.
   *
   * Against a stored plan, so the model is shown the timing that will actually
   * be rendered. Read-only: the cues come back to the editor and reach the
   * database only if the user submits a plan containing them.
   *
   * Never throws for "the model could not": a failure arrives as `ok: false`
   * with a named reason, because an empty panel with an explanation is the
   * honest outcome and invented subtitles are not.
   */
  suggestSubtitles: (
    projectId: string,
    editPlanId: string,
    body: { request_text?: string | null; style?: SubtitleStyle; position?: SubtitlePosition },
  ) =>
    request<SubtitleSuggestion>(
      `/api/projects/${projectId}/edit-plan/${editPlanId}/subtitles/suggest`,
      { method: "POST", body: JSON.stringify(body) },
    ),

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

  // ------------------------------------------------------------ reference
  /** The project's style reference and what it measures as. */
  reference: (projectId: string) =>
    request<ReferenceState>(`/api/projects/${projectId}/reference`),

  /** Nominate a clip in this project. The server checks it is one. */
  setReference: (projectId: string, mediaId: string) =>
    request<ReferenceState>(`/api/projects/${projectId}/reference`, {
      method: "PUT",
      body: JSON.stringify({ media_id: mediaId }),
    }),

  /** Stop styling after anything. The clip itself is untouched. */
  clearReference: (projectId: string) =>
    request<ReferenceState>(`/api/projects/${projectId}/reference`, {
      method: "DELETE",
    }),

  listLlmRuns: (projectId: string) =>
    request<{ items: LlmRun[]; total: number }>(`/api/projects/${projectId}/llm-runs`),

  // ----------------------------------------------------------- co-editor
  /**
   * What a change would do. Writes nothing.
   *
   * The whole pipeline runs server-side and everything but the diff is thrown
   * away, so what this returns cannot disagree with what applying it does.
   */
  previewCoEdit: (projectId: string, body: CoEditRequest) =>
    request<CoEditPreview>(`/api/projects/${projectId}/edit-plan/co-edit/preview`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /**
   * Apply a change and store it as a new version.
   *
   * Does not render. A plan change must not queue an encode -- the render is a
   * separate, explicit request through the route that already exists.
   */
  applyCoEdit: (projectId: string, body: CoEditRequest) =>
    request<CoEditResult>(`/api/projects/${projectId}/edit-plan/co-edit`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listEditVersions: (projectId: string) =>
    request<EditVersionList>(`/api/projects/${projectId}/edit-plan/versions`),

  /**
   * Step back one version. The server is authoritative: no plan is written and
   * the restored version is the one that was stored, not a recomputation.
   */
  undoEditVersion: (projectId: string) =>
    request<EditVersion>(`/api/projects/${projectId}/edit-plan/versions/undo`, {
      method: "POST",
    }),

  redoEditVersion: (projectId: string) =>
    request<EditVersion>(`/api/projects/${projectId}/edit-plan/versions/redo`, {
      method: "POST",
    }),

  restoreEditVersion: (projectId: string, versionId: string) =>
    request<EditVersion>(
      `/api/projects/${projectId}/edit-plan/versions/${versionId}/restore`,
      { method: "POST" },
    ),
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
