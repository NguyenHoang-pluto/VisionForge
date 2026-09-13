"use client";

import { useQuery } from "@tanstack/react-query";

import {
  api,
  type AnalysisRecord,
  type AnalysisStatus,
  type AnalyzerName,
  type MediaAsset,
} from "@/lib/api";
import { elapsed, int, num } from "@/lib/format";
import { Badge, EmptyState, Meter, Row, SectionTitle } from "@/components/ui";

/**
 * Analysis, in the inspector.
 *
 * A technical readout rather than a dashboard: monospace values in a fixed
 * column, hairline separators, no cards. A bar appears only where the value has
 * a meaningful range to be read against -- sharpness against the blur
 * threshold, luminance against mid-grey. A bar with an invented maximum implies
 * a comparison that does not exist, which is worse than a bare number.
 */

const STATUS: Record<AnalysisStatus, { label: string; tone: "ok" | "neutral" | "danger" }> = {
  ok: { label: "ok", tone: "ok" },
  unsupported: { label: "n/a", tone: "neutral" },
  failed: { label: "fail", tone: "danger" },
};

const ANALYZER_LABEL: Record<AnalyzerName, string> = {
  quality: "Quality",
  scenes: "Scenes",
  phash: "Perceptual hash",
  clip: "CLIP embedding",
  faces: "Faces",
};

/** The Laplacian variance below which the analyzer calls a frame blurry. */
const BLUR_REFERENCE = 200;

function Section({
  record,
  children,
}: {
  record: AnalysisRecord;
  children?: React.ReactNode;
}) {
  const status = STATUS[record.status];
  return (
    <section>
      <SectionTitle
        aside={
          <span className="flex items-center gap-1.5 font-mono text-2xs text-dim">
            v{record.analyzer_version}
            <Badge tone={status.tone}>{status.label}</Badge>
          </span>
        }
      >
        {ANALYZER_LABEL[record.analyzer] ?? record.analyzer}
      </SectionTitle>

      {record.status === "unsupported" ? (
        <p className="py-1 text-xs text-dim">
          {String(record.payload.reason ?? "Not applicable to this media type.")}
        </p>
      ) : (
        <div className="mt-1">{children}</div>
      )}
    </section>
  );
}

function Quality({ p }: { p: Record<string, unknown> }) {
  const blur = typeof p.blur_score === "number" ? p.blur_score : null;
  const luma = typeof p.mean_luminance === "number" ? p.mean_luminance : null;
  const badFrames = Number(p.badly_exposed_frames ?? 0);

  return (
    <>
      <Row
        label="Sharpness"
        value={num(p.blur_score)}
        tone={p.is_blurry ? "warn" : undefined}
        title="Laplacian variance. Lower is blurrier."
      />
      {blur !== null && (
        <div className="pb-1 pt-0.5">
          <Meter
            value={Math.min(blur, BLUR_REFERENCE * 2)}
            max={BLUR_REFERENCE * 2}
            tone={p.is_blurry ? "warn" : "ok"}
          />
        </div>
      )}

      <Row
        label="Contrast"
        value={num(p.contrast)}
        tone={p.is_low_contrast ? "warn" : undefined}
        title="Standard deviation of luma."
      />
      <Row label="Brightness" value={num(p.mean_luminance, 1)} title="Mean luma, 0–255." />
      {luma !== null && (
        <div className="pb-1 pt-0.5">
          <Meter value={luma} max={255} tone={luma < 40 || luma > 215 ? "warn" : "ok"} />
        </div>
      )}

      <Row
        label="Exposure faults"
        value={`${int(p.badly_exposed_frames)} / ${int(p.frame_count)}`}
        tone={badFrames > 0 ? "warn" : undefined}
        title="Frames clipped to black or white."
      />
      <Row
        label="Sampled at"
        value={`${int(p.analysis_width)}×${int(p.analysis_height)}`}
      />
    </>
  );
}

function Scenes({ p }: { p: Record<string, unknown> }) {
  const scenes = Array.isArray(p.scenes) ? (p.scenes as Record<string, number>[]) : [];
  const total = scenes.length > 0 ? scenes[scenes.length - 1].end_ms : 0;

  return (
    <>
      <Row label="Scenes" value={int(p.scene_count)} />
      <Row label="Mean length" value={elapsed(p.mean_scene_ms)} />
      <Row
        label="Shortest / longest"
        value={`${elapsed(p.shortest_scene_ms)} / ${elapsed(p.longest_scene_ms)}`}
      />
      <Row label="Detector" value={`${String(p.detector)} @ ${num(p.threshold, 0)}`} />

      {/* Boundaries as a strip: where the cuts fall matters more than when. */}
      {scenes.length > 0 && total > 0 && (
        <div
          className="mt-1.5 flex h-4 w-full overflow-hidden rounded-sm border border-line"
          title={`${scenes.length} scenes`}
        >
          {scenes.map((scene, index) => (
            <div
              key={index}
              className={`border-r border-panel last:border-r-0 ${
                index % 2 ? "bg-accent/35" : "bg-accent/60"
              }`}
              style={{ width: `${((scene.end_ms - scene.start_ms) / total) * 100}%` }}
              title={`Scene ${index + 1}: ${elapsed(scene.start_ms)} → ${elapsed(scene.end_ms)}`}
            />
          ))}
        </div>
      )}
    </>
  );
}

function Hashes({ p }: { p: Record<string, unknown> }) {
  return (
    <>
      <Row label="pHash" value={String(p.phash ?? "—")} title="DCT hash, 64-bit" />
      <Row label="aHash" value={String(p.ahash ?? "—")} title="Mean hash, 64-bit" />
      <Row label="Duplicate within" value={`≤ ${int(p.duplicate_max_distance)} bits`} />
    </>
  );
}

function Embedding({
  record,
  p,
}: {
  record: AnalysisRecord;
  p: Record<string, unknown>;
}) {
  const metrics = record.metrics ?? {};
  return (
    <>
      <Row label="Model" value={`${String(p.model)} / ${String(p.pretrained)}`} />
      <Row label="Dimensions" value={int(p.dim)} />
      <Row label="Vector stored" value={record.has_embedding ? "yes" : "no"} />
      <Row label="Frames encoded" value={int(p.frame_count)} />
      <Row label="Device" value={String(metrics.device ?? "—")} />
      <Row
        label="VRAM peak"
        value={metrics.vram_peak_mb ? `${num(metrics.vram_peak_mb, 1)} MiB` : "—"}
      />
      <Row label="Inference" value={elapsed(metrics.duration_ms)} />
    </>
  );
}

function Faces({ p }: { p: Record<string, unknown> }) {
  return (
    <>
      <Row label="Faces detected" value={int(p.face_count)} />
      <Row
        label="Frames with faces"
        value={`${int(p.frames_with_faces)} / ${int(p.frame_count)}`}
      />
      <Row label="Detector" value={String(p.model ?? "—")} />
      <Row label="Device" value={String(p.device ?? "—")} />
      <Row label="Confidence floor" value={num(p.score_threshold)} />
      {/* Phase 3 decided identity is never computed. Saying so is the point. */}
      <Row label="Identity stored" value={p.identity_stored ? "yes" : "no"} />
    </>
  );
}

const ORDER: AnalyzerName[] = ["quality", "scenes", "phash", "clip", "faces"];

export function AnalysisPanel({
  projectId,
  asset,
  onSelectMedia,
}: {
  projectId: string;
  asset: MediaAsset;
  onSelectMedia: (mediaId: string) => void;
}) {
  const analysis = useQuery({
    queryKey: ["analysis", asset.id],
    queryFn: () => api.mediaAnalysis(projectId, asset.id),
    // Poll only while there is nothing yet -- once results exist they do not
    // change unless analysis is re-run, which invalidates this key anyway.
    refetchInterval: (query) => (query.state.data?.items.length ? false : 4000),
    staleTime: 30_000,
  });

  const similar = useQuery({
    queryKey: ["similar", asset.id],
    queryFn: () => api.similarMedia(projectId, asset.id),
    enabled: analysis.data?.items.some((item) => item.has_embedding) ?? false,
    staleTime: 60_000,
    retry: false,
  });

  const byAnalyzer = new Map((analysis.data?.items ?? []).map((item) => [item.analyzer, item]));

  if (analysis.isLoading) {
    return <EmptyState>Loading analysis…</EmptyState>;
  }

  if (byAnalyzer.size === 0) {
    return (
      <EmptyState>
        No analysis for this clip. Run Analyse from the top bar to compute quality signals,
        scene boundaries, perceptual hashes, embeddings and face counts.
      </EmptyState>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-2.5">
      {ORDER.map((name) => {
        const record = byAnalyzer.get(name);
        if (!record) return null;
        return (
          <Section key={name} record={record}>
            {name === "quality" && <Quality p={record.payload} />}
            {name === "scenes" && <Scenes p={record.payload} />}
            {name === "phash" && <Hashes p={record.payload} />}
            {name === "clip" && <Embedding record={record} p={record.payload} />}
            {name === "faces" && <Faces p={record.payload} />}
          </Section>
        );
      })}

      {similar.data && similar.data.results.length > 0 && (
        <section>
          <SectionTitle>Visually similar</SectionTitle>
          <ul className="mt-1">
            {similar.data.results.slice(0, 6).map((hit) => (
              <li key={hit.media_id}>
                <button
                  type="button"
                  onClick={() => onSelectMedia(hit.media_id)}
                  className="flex w-full items-baseline justify-between gap-3 border-b border-line/60 py-[3px] text-left transition-colors last:border-b-0 hover:text-accent-strong"
                >
                  <span className="truncate text-xs text-muted">
                    {hit.media_id.slice(0, 8)}
                  </span>
                  <span className="font-mono text-xs text-fg tabular-nums">
                    {(hit.similarity * 100).toFixed(1)}%
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
