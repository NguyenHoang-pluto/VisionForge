"use client";

import { useQueries, useQuery } from "@tanstack/react-query";

import { api, type MediaAsset } from "@/lib/api";

/**
 * Presigned media URLs, cached just under their own expiry.
 *
 * Every thumbnail and proxy URL is signed and short-lived, so it cannot travel
 * in the media listing and has to be fetched per asset. Routing them through
 * the query cache is what stops a grid of two hundred tiles from re-signing on
 * every scroll, every re-render and every panel toggle.
 *
 * `staleTime` sits below the signature's five-minute TTL: refetching slightly
 * early costs one request, whereas refetching late hands the `<img>` a URL the
 * storage service has already stopped honouring.
 */
const SIGNATURE_TTL_MS = 5 * 60 * 1000;
const STALE_MS = 4 * 60 * 1000;

export function useThumbnailUrl(
  projectId: string | null,
  asset: Pick<MediaAsset, "id" | "has_thumbnail"> | null,
) {
  return useQuery({
    queryKey: ["thumbnail", asset?.id],
    queryFn: () => api.thumbnailUrl(projectId!, asset!.id),
    enabled: Boolean(projectId && asset?.has_thumbnail),
    staleTime: STALE_MS,
    gcTime: SIGNATURE_TTL_MS,
    retry: false,
  });
}

/** The 720p proxy. Absent until ingest has produced one. */
export function useProxyUrl(
  projectId: string | null,
  asset: Pick<MediaAsset, "id" | "has_proxy"> | null,
) {
  return useQuery({
    queryKey: ["proxy", asset?.id],
    queryFn: () => api.proxyUrl(projectId!, asset!.id),
    enabled: Boolean(projectId && asset?.has_proxy),
    staleTime: STALE_MS,
    gcTime: SIGNATURE_TTL_MS,
    retry: false,
  });
}

/**
 * Proxy URLs for several assets at once.
 *
 * Program playback steps between clips, and re-fetching a URL at every cut
 * would put a network round trip in the middle of the edit. Fetching the whole
 * timeline's sources up front means a cut is an `src` swap.
 */
export function useProxyUrls(
  projectId: string | null,
  mediaIds: readonly string[],
): Map<string, string> {
  const unique = Array.from(new Set(mediaIds));

  const results = useQueries({
    queries: unique.map((mediaId) => ({
      queryKey: ["proxy", mediaId],
      queryFn: () => api.proxyUrl(projectId!, mediaId),
      enabled: Boolean(projectId),
      staleTime: STALE_MS,
      gcTime: SIGNATURE_TTL_MS,
      retry: false,
    })),
  });

  const urls = new Map<string, string>();
  unique.forEach((mediaId, index) => {
    const url = results[index]?.data?.url;
    if (url) urls.set(mediaId, url);
  });
  return urls;
}
