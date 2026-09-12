"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type MediaAsset, type MediaStatus } from "@/lib/api";

const STATUS_STYLES: Record<MediaStatus, string> = {
  pending_upload: "bg-slate-500/10 text-slate-400",
  uploaded: "bg-sky-500/10 text-sky-400",
  processing: "bg-amber-500/10 text-amber-400",
  ready: "bg-emerald-500/10 text-emerald-400",
  failed: "bg-rose-500/10 text-rose-400",
};

const STATUS_LABELS: Record<MediaStatus, string> = {
  pending_upload: "Waiting for upload",
  uploaded: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
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
}: {
  asset: MediaAsset;
  projectId: string;
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
    <li className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
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

        {asset.has_proxy && (
          <span className="absolute bottom-1.5 right-1.5 rounded bg-slate-950/80 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
            720p
          </span>
        )}
      </div>

      <div className="flex flex-col gap-1.5 p-3">
        <p
          className="truncate text-xs font-medium text-slate-200"
          title={asset.original_filename}
        >
          {asset.original_filename}
        </p>

        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase ${STATUS_STYLES[asset.status]}`}
          >
            {STATUS_LABELS[asset.status]}
          </span>
          <span className="font-mono text-[10px] text-slate-500">{asset.kind}</span>
        </div>

        {meta.length > 0 && (
          <p className="font-mono text-[10px] text-slate-500 tabular-nums">
            {meta.join(" · ")}
          </p>
        )}

        {asset.status === "failed" && asset.error?.message && (
          <p className="text-[10px] leading-snug text-rose-400">
            {asset.error.message}
          </p>
        )}
      </div>
    </li>
  );
}
