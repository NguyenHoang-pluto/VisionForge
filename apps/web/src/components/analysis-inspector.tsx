"use client";

import { useQuery } from "@tanstack/react-query";

import {
  api,
  type AnalysisRecord,
  type AnalysisStatus,
  type MediaAsset,
} from "@/lib/api";

/**
 * Per-asset analysis inspector.
 *
 * Laid out as a technical readout, not a marketing panel: monospace values, a
 * fixed label column, hairline separators, no cards. The point is that a number
 * can be read off quickly and compared against the one above it.
 */

const STATUS_LABEL: Record<AnalysisStatus, string> = {
  ok: "OK",
  unsupported: "N/A",
  failed: "FAIL",
};

const STATUS_CLASS: Record<AnalysisStatus, string> = {
  ok: "text-emerald-400",
  unsupported: "text-slate-500",
  failed: "text-rose-400",
};

const ANALYZER_LABEL: Record<string, string> = {
  quality: "Quality",
  scenes: "Scenes",
  phash: "Perceptual hash",
  clip: "CLIP embedding",
  faces: "Face detection",
};

function num(value: unknown, digits = 2): string {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function int(value: unknown): string {
  return typeof value === "number" ? String(Math.round(value)) : "—";
}

function ms(value: unknown): string {
  if (typeof value !== "number") return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

/** One label/value line. The unit of this whole panel. */
function Row({
  label,
  value,
  flag,
}: {
  label: string;
  value: string;
  flag?: "warn" | "ok";
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-slate-800/60 py-1 last:border-b-0">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span
        className={`font-mono text-[11px] tabular-nums ${
          flag === "warn"
            ? "text-amber-400"
            : flag === "ok"
              ? "text-emerald-400"
              : "text-slate-300"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function Section({
  record,
  children,
}: {
  record: AnalysisRecord;
  children?: React.ReactNode;
}) {
  return (
    <section>
      <header className="flex items-baseline justify-between gap-3 border-b border-slate-700 pb-1">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
          {ANALYZER_LABEL[record.analyzer] ?? record.analyzer}
        </h4>
        <span className="flex items-baseline gap-2 font-mono text-[10px]">
          <span className="text-slate-600">v{record.analyzer_version}</span>
          <span className={STATUS_CLASS[record.status]}>
            {STATUS_LABEL[record.status]}
          </span>
        </span>
      </header>
      {record.status === "unsupported" ? (
        <p className="py-1.5 text-[11px] text-slate-600">
          {String(record.payload.reason ?? "Not applicable to this media type.")}
        </p>
      ) : (
        <div className="mt-1">{children}</div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- per-analyzer
function QualityBody({ p }: { p: Record<string, unknown> }) {
  return (
    <>
      <Row
        label="Sharpness (Laplacian var)"
        value={num(p.blur_score)}
        flag={p.is_blurry ? "warn" : undefined}
      />
      <Row
        label="Contrast (luma σ)"
        value={num(p.contrast)}
        flag={p.is_low_contrast ? "warn" : undefined}
      />
      <Row label="Mean luminance" value={num(p.mean_luminance, 1)} />
      <Row label="Frames sampled" value={int(p.frame_count)} />
      <Row
        label="Badly exposed frames"
        value={int(p.badly_exposed_frames)}
        flag={Number(p.badly_exposed_frames) > 0 ? "warn" : undefined}
      />
      <Row
        label="Analysis resolution"
        value={`${int(p.analysis_width)}×${int(p.analysis_height)}`}
      />
    </>
  );
}

function ScenesBody({ p }: { p: Record<string, unknown> }) {
  const scenes = Array.isArray(p.scenes) ? p.scenes : [];
  return (
    <>
      <Row label="Scene count" value={int(p.scene_count)} />
      <Row label="Mean duration" value={ms(p.mean_scene_ms)} />
      <Row label="Shortest / longest" value={`${ms(p.shortest_scene_ms)} / ${ms(p.longest_scene_ms)}`} />
      <Row label="Detector" value={`${String(p.detector)} @ ${num(p.threshold, 0)}`} />
      {scenes.length > 0 && (
        <div className="mt-2 max-h-32 overflow-y-auto border border-slate-800">
          <table className="w-full font-mono text-[10px] tabular-nums">
            <thead className="sticky top-0 bg-slate-900 text-slate-500">
              <tr>
                <th className="px-2 py-1 text-left font-normal">#</th>
                <th className="px-2 py-1 text-right font-normal">In</th>
                <th className="px-2 py-1 text-right font-normal">Out</th>
                <th className="px-2 py-1 text-right font-normal">Dur</th>
              </tr>
            </thead>
            <tbody className="text-slate-400">
              {scenes.map((raw, index) => {
                const scene = raw as Record<string, number>;
                return (
                  <tr key={index} className="border-t border-slate-800/60">
                    <td className="px-2 py-0.5 text-slate-600">{scene.scene_id}</td>
                    <td className="px-2 py-0.5 text-right">{ms(scene.start_ms)}</td>
                    <td className="px-2 py-0.5 text-right">{ms(scene.end_ms)}</td>
                    <td className="px-2 py-0.5 text-right">{ms(scene.duration_ms)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function HashBody({ p }: { p: Record<string, unknown> }) {
  return (
    <>
      <Row label="pHash (DCT, 64-bit)" value={String(p.phash ?? "—")} />
      <Row label="aHash (mean, 64-bit)" value={String(p.ahash ?? "—")} />
      <Row label="Duplicate threshold" value={`≤ ${int(p.duplicate_max_distance)} bits`} />
    </>
  );
}

function ClipBody({
  record,
  p,
}: {
  record: AnalysisRecord;
  p: Record<string, unknown>;
}) {
  const m = record.metrics ?? {};
  return (
    <>
      <Row label="Model" value={`${String(p.model)} / ${String(p.pretrained)}`} />
      <Row label="Dimensions" value={int(p.dim)} />
      <Row label="Vector stored" value={record.has_embedding ? "yes" : "no"} />
      <Row label="Frames encoded" value={int(p.frame_count)} />
      <Row label="Device" value={String(m.device ?? "—")} />
      <Row label="VRAM peak" value={m.vram_peak_mb ? `${num(m.vram_peak_mb, 1)} MiB` : "—"} />
      <Row label="Inference" value={ms(m.duration_ms)} />
    </>
  );
}

function FacesBody({ p }: { p: Record<string, unknown> }) {
  return (
    <>
      <Row label="Faces detected" value={int(p.face_count)} />
      <Row label="Frames with faces" value={`${int(p.frames_with_faces)} / ${int(p.frame_count)}`} />
      <Row label="Detector" value={String(p.model ?? "—")} />
      <Row label="Device" value={String(p.device ?? "—")} />
      <Row label="Confidence floor" value={num(p.score_threshold)} />
      <Row label="Identity stored" value={p.identity_stored ? "yes" : "no"} />
    </>
  );
}

// -------------------------------------------------------------------- panel
export function AnalysisInspector({
  projectId,
  asset,
  onClose,
}: {
  projectId: string;
  asset: MediaAsset;
  onClose: () => void;
}) {
  const analysis = useQuery({
    queryKey: ["analysis", asset.id],
    queryFn: () => api.mediaAnalysis(projectId, asset.id),
    refetchInterval: (query) => (query.state.data?.items.length ? false : 4000),
  });

  const similar = useQuery({
    queryKey: ["similar", asset.id],
    queryFn: () => api.similarMedia(projectId, asset.id),
    enabled: analysis.data?.items.some((i) => i.has_embedding) ?? false,
    retry: false,
  });

  const byAnalyzer = new Map(
    (analysis.data?.items ?? []).map((item) => [item.analyzer, item]),
  );
  const order: AnalysisRecord["analyzer"][] = [
    "quality",
    "scenes",
    "phash",
    "clip",
    "faces",
  ];

  return (
    <aside className="flex h-full flex-col border-l border-slate-800 bg-slate-950">
      <header className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium text-slate-200" title={asset.original_filename}>
            {asset.original_filename}
          </p>
          <p className="font-mono text-[10px] text-slate-600">
            {asset.kind}
            {asset.width ? ` · ${asset.width}×${asset.height}` : ""}
            {asset.codec ? ` · ${asset.codec}` : ""}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close inspector"
          className="shrink-0 px-1.5 py-0.5 font-mono text-xs text-slate-500 transition hover:text-slate-200 focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600"
        >
          ✕
        </button>
      </header>

      <div className="flex flex-col gap-4 overflow-y-auto p-4">
        {analysis.isLoading && (
          <p className="text-[11px] text-slate-600">Loading analysis…</p>
        )}

        {!analysis.isLoading && byAnalyzer.size === 0 && (
          <p className="text-[11px] leading-relaxed text-slate-500">
            No analysis recorded. Run analysis from the toolbar to compute quality
            signals, scene boundaries, perceptual hashes, embeddings and face
            counts.
          </p>
        )}

        {order.map((name) => {
          const record = byAnalyzer.get(name);
          if (!record) return null;
          return (
            <Section key={name} record={record}>
              {name === "quality" && <QualityBody p={record.payload} />}
              {name === "scenes" && <ScenesBody p={record.payload} />}
              {name === "phash" && <HashBody p={record.payload} />}
              {name === "clip" && <ClipBody record={record} p={record.payload} />}
              {name === "faces" && <FacesBody p={record.payload} />}
            </Section>
          );
        })}

        {similar.data && similar.data.results.length > 0 && (
          <section>
            <header className="border-b border-slate-700 pb-1">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
                Visually similar
              </h4>
            </header>
            <div className="mt-1">
              {similar.data.results.slice(0, 5).map((hit) => (
                <Row
                  key={hit.media_id}
                  label={hit.media_id.slice(0, 8)}
                  value={`${(hit.similarity * 100).toFixed(1)}%`}
                />
              ))}
            </div>
          </section>
        )}
      </div>
    </aside>
  );
}
