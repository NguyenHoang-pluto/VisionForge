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
import { useT, type MessageKey } from "@/lib/i18n";
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

const STATUS: Record<AnalysisStatus, { label: MessageKey; tone: "ok" | "neutral" | "danger" }> = {
  ok: { label: "analysis.status.ok", tone: "ok" },
  unsupported: { label: "analysis.status.unsupported", tone: "neutral" },
  failed: { label: "analysis.status.failed", tone: "danger" },
};

const ANALYZER_LABEL: Record<AnalyzerName, MessageKey> = {
  quality: "analysis.analyzer.quality",
  scenes: "analysis.analyzer.scenes",
  phash: "analysis.analyzer.phash",
  clip: "analysis.analyzer.clip",
  faces: "analysis.analyzer.faces",
  beats: "analysis.analyzer.beats",
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
  const t = useT();
  const status = STATUS[record.status];
  return (
    <section>
      <SectionTitle
        aside={
          <span className="flex shrink-0 items-center gap-1.5 font-mono text-2xs text-dim">
            v{record.analyzer_version}
            <Badge tone={status.tone}>{t(status.label)}</Badge>
          </span>
        }
      >
        {ANALYZER_LABEL[record.analyzer] ? t(ANALYZER_LABEL[record.analyzer]) : record.analyzer}
      </SectionTitle>

      {record.status === "unsupported" ? (
        <p className="py-1 text-xs text-dim">
          {String(record.payload.reason ?? t("analysis.notApplicable"))}
        </p>
      ) : (
        <div className="mt-1">{children}</div>
      )}
    </section>
  );
}

function Quality({ p }: { p: Record<string, unknown> }) {
  const t = useT();
  const blur = typeof p.blur_score === "number" ? p.blur_score : null;
  const luma = typeof p.mean_luminance === "number" ? p.mean_luminance : null;
  const badFrames = Number(p.badly_exposed_frames ?? 0);

  return (
    <>
      <Row
        label={t("analysis.quality.sharpness")}
        value={num(p.blur_score)}
        tone={p.is_blurry ? "warn" : undefined}
        title={t("analysis.quality.sharpnessHint")}
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
        label={t("analysis.quality.contrast")}
        value={num(p.contrast)}
        tone={p.is_low_contrast ? "warn" : undefined}
        title={t("analysis.quality.contrastHint")}
      />
      <Row
        label={t("analysis.quality.brightness")}
        value={num(p.mean_luminance, 1)}
        title={t("analysis.quality.brightnessHint")}
      />
      {luma !== null && (
        <div className="pb-1 pt-0.5">
          <Meter value={luma} max={255} tone={luma < 40 || luma > 215 ? "warn" : "ok"} />
        </div>
      )}

      <Row
        label={t("analysis.quality.exposure")}
        value={`${int(p.badly_exposed_frames)} / ${int(p.frame_count)}`}
        tone={badFrames > 0 ? "warn" : undefined}
        title={t("analysis.quality.exposureHint")}
      />
      <Row
        label={t("analysis.quality.sampledAt")}
        value={`${int(p.analysis_width)}×${int(p.analysis_height)}`}
      />
    </>
  );
}

function Scenes({ p }: { p: Record<string, unknown> }) {
  const t = useT();
  const scenes = Array.isArray(p.scenes) ? (p.scenes as Record<string, number>[]) : [];
  const total = scenes.length > 0 ? scenes[scenes.length - 1].end_ms : 0;

  return (
    <>
      <Row label={t("analysis.scenes.count")} value={int(p.scene_count)} />
      <Row label={t("analysis.scenes.mean")} value={elapsed(p.mean_scene_ms)} />
      <Row
        label={t("analysis.scenes.range")}
        value={`${elapsed(p.shortest_scene_ms)} / ${elapsed(p.longest_scene_ms)}`}
      />
      <Row
        label={t("analysis.scenes.detector")}
        value={`${String(p.detector)} @ ${num(p.threshold, 0)}`}
      />

      {/* Boundaries as a strip: where the cuts fall matters more than when. */}
      {scenes.length > 0 && total > 0 && (
        <div
          className="mt-1.5 flex h-4 w-full overflow-hidden rounded-sm border border-line"
          title={t("analysis.scenes.strip", { count: scenes.length })}
        >
          {scenes.map((scene, index) => (
            <div
              key={index}
              className={`border-r border-panel last:border-r-0 ${
                index % 2 ? "bg-accent/35" : "bg-accent/60"
              }`}
              style={{ width: `${((scene.end_ms - scene.start_ms) / total) * 100}%` }}
              title={t("analysis.scenes.item", {
                index: index + 1,
                from: elapsed(scene.start_ms),
                to: elapsed(scene.end_ms),
              })}
            />
          ))}
        </div>
      )}
    </>
  );
}

function Hashes({ p }: { p: Record<string, unknown> }) {
  const t = useT();
  return (
    <>
      <Row
        label={t("analysis.hash.phash")}
        value={String(p.phash ?? "—")}
        title={t("analysis.hash.phashHint")}
      />
      <Row
        label={t("analysis.hash.ahash")}
        value={String(p.ahash ?? "—")}
        title={t("analysis.hash.ahashHint")}
      />
      <Row
        label={t("analysis.hash.duplicate")}
        value={t("analysis.hash.bits", { bits: int(p.duplicate_max_distance) })}
      />
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
  const t = useT();
  const metrics = record.metrics ?? {};
  return (
    <>
      <Row
        label={t("analysis.embed.model")}
        value={`${String(p.model)} / ${String(p.pretrained)}`}
      />
      <Row label={t("analysis.embed.dim")} value={int(p.dim)} />
      <Row
        label={t("analysis.embed.stored")}
        value={record.has_embedding ? t("common.yes") : t("common.no")}
      />
      <Row label={t("analysis.embed.frames")} value={int(p.frame_count)} />
      <Row label={t("analysis.embed.device")} value={String(metrics.device ?? "—")} />
      <Row
        label={t("analysis.embed.vram")}
        value={metrics.vram_peak_mb ? `${num(metrics.vram_peak_mb, 1)} MiB` : t("common.dash")}
      />
      <Row label={t("analysis.embed.inference")} value={elapsed(metrics.duration_ms)} />
    </>
  );
}

function Beats({ p }: { p: Record<string, unknown> }) {
    const t = useT();
    const bpm = typeof p.bpm === "number" ? p.bpm : null;
    const confidence = typeof p.confidence === "number" ? p.confidence : 0;
    return (
      <>
        <Row
          label={t("audio.bpm")}
          value={bpm ? t("audio.bpmValue", { bpm: bpm.toFixed(1) }) : t("common.dash")}
        />
        <Row label={t("audio.confidence")} value={`${Math.round(confidence * 100)}%`} />
        {/* A bar, because confidence has a meaningful range: the planner's
            threshold sits inside it, so "is this enough" is a comparison the
            reader can actually make. */}
        <div className="pb-1 pt-0.5">
          <Meter value={confidence} max={1} tone={confidence >= 0.35 ? "ok" : "warn"} />
        </div>
        <Row label={t("audio.beatCount")} value={int(p.beat_count)} />
        <Row label={t("analysis.scenes.detector")} value={String(p.method ?? "—")} />
      </>
    );
}


function Faces({ p }: { p: Record<string, unknown> }) {
  const t = useT();
  return (
    <>
      <Row label={t("analysis.faces.count")} value={int(p.face_count)} />
      <Row
        label={t("analysis.faces.frames")}
        value={`${int(p.frames_with_faces)} / ${int(p.frame_count)}`}
      />
      <Row label={t("analysis.faces.detector")} value={String(p.model ?? "—")} />
      <Row label={t("analysis.faces.device")} value={String(p.device ?? "—")} />
      <Row label={t("analysis.faces.threshold")} value={num(p.score_threshold)} />
      {/* Phase 3 decided identity is never computed. Saying so is the point. */}
      <Row
        label={t("analysis.faces.identity")}
        value={p.identity_stored ? t("common.yes") : t("common.no")}
      />
    </>
  );
}

const ORDER: AnalyzerName[] = ["quality", "scenes", "phash", "clip", "faces", "beats"];

export function AnalysisPanel({
  projectId,
  asset,
  onSelectMedia,
}: {
  projectId: string;
  asset: MediaAsset;
  onSelectMedia: (mediaId: string) => void;
}) {
  const t = useT();
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
    return <EmptyState icon="analyse">{t("analysis.loading")}</EmptyState>;
  }

  if (byAnalyzer.size === 0) {
    return <EmptyState icon="analyse">{t("analysis.empty")}</EmptyState>;
  }

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
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
            {name === "beats" && <Beats p={record.payload} />}
          </Section>
        );
      })}

      {similar.data && similar.data.results.length > 0 && (
        <section>
          <SectionTitle>{t("analysis.similar")}</SectionTitle>
          <ul className="mt-1">
            {similar.data.results.slice(0, 6).map((hit) => (
              <li key={hit.media_id}>
                <button
                  type="button"
                  onClick={() => onSelectMedia(hit.media_id)}
                  className="flex min-h-row w-full items-baseline justify-between gap-3 border-b border-line/60 py-[3px] text-left transition-colors last:border-b-0 hover:text-accent-strong"
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
