"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type MediaAsset, type MediaStatus } from "@/lib/api";

const STATUS_STYLES: Record<MediaStatus, string> = {
  pending_upload: "text-slate-500",
  uploaded: "text-sky-400",
  processing: "text-amber-400",
  ready: "text-emerald-400",
  failed: "text-rose-400",
};

const STATUS_LABELS: Record<MediaStatus, string> = {
  pending_upload: "PENDING",
  uploaded: "QUEUED",
  processing: "PROCESSING",
  ready: "READY",
  failed: "FAILED",
};

const KIND_GLYPH: Record<MediaAsset["kind"], string> = {
  video: "▶",
  audio: "♪",
  image: "▣",
};

function formatDuration(ms: number | null): string | null {
  if (!ms) return null;
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function formatBytes(bytes: number | null): string | null {
  if (!bytes) return null;
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function MediaCard({
  asset,
  projectId,
  selected,
  onSelect,
  analyzed,
}: {
  asset: MediaAsset;
  projectId: string;
  selected: boolean;
  onSelect: () => void;
  analyzed: boolean;
}) {
  // Thumbnail URLs are presigned and short-lived, so they are fetched per asset
  // rather than embedded in the listing response.
  const thumbnail = useQuery({
    queryKey: ["thumbnail", asset.id],
    queryFn: () => api.thumbnailUrl(projectId, asset.id),
    enabled: asset.has_thumbnail,
    staleTime: 4 * 60 * 1000, // just under the 5-minute signature TTL
    retry: false,
  });

  const meta = [
    asset.width && asset.height ? `${asset.width}x${asset.height}` : null,
    formatDuration(asset.duration_ms),
    formatBytes(asset.bytes_size),
  ].filter(Boolean);

  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className={`w-full overflow-hidden rounded-sm border text-left transition focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600 ${
          selected
            ? "border-sky-600 bg-slate-900"
            : "border-slate-800 bg-slate-950 hover:border-slate-700"
        }`}
      >
      <div className="relative flex aspect-video items-center justify-center bg-slate-900">
        {thumbnail.data?.url ? (
          /* eslint-disable-next-line @next/next/no-img-element --
             presigned URL on a dynamic host; next/image would need a
             remotePatterns entry per environment. */
          <img
            src={thumbnail.data.url}
            alt={asset.original_filename}
            className="h-full w-full object-cover"
          />
        ) : (
          <span className="text-2xl text-slate-700" aria-hidden>
            {KIND_GLYPH[asset.kind]}
          </span>
        )}

        <div className="absolute bottom-1 right-1 flex gap-1">
          {asset.has_proxy && (
            <span className="bg-slate-950/85 px-1 py-px font-mono text-[9px] text-slate-400">
              720p
            </span>
          )}
          {analyzed && (
            <span
              className="bg-slate-950/85 px-1 py-px font-mono text-[9px] text-emerald-400"
              title="Analysis recorded"
            >
              ANL
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-1 border-t border-slate-800 px-2 py-1.5">
        <p
          className="truncate text-[11px] text-slate-200"
          title={asset.original_filename}
        >
          {asset.original_filename}
        </p>

        <div className="flex items-center justify-between gap-2 font-mono text-[9px]">
          <span className="text-slate-600 uppercase">{asset.kind}</span>
          <span className={STATUS_STYLES[asset.status]}>
            {STATUS_LABELS[asset.status]}
          </span>
        </div>

        {meta.length > 0 && (
          <p className="truncate font-mono text-[9px] text-slate-600 tabular-nums">
            {meta.join(" · ")}
          </p>
        )}

        {asset.status === "failed" && asset.error?.message && (
          <p className="text-[9px] leading-snug text-rose-400">
            {asset.error.message}
          </p>
        )}
      </div>
      </button>
    </li>
  );
}
