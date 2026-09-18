"use client";

import { useQueries } from "@tanstack/react-query";

import { api, type MediaAsset, type Project } from "@/lib/api";

/**
 * What a project card can honestly say.
 *
 * The projects listing carries an id, a title, a description, a creation time
 * and a media count -- and nothing else. It has no thumbnail, no duration, no
 * resolution and no "last edited". Those are the things a home screen wants,
 * and there were two ways to get them: add them to the API, or derive them from
 * data the client can already ask for.
 *
 * This is the second. Phase 8 is not open and the backend is not in scope, so
 * every figure on a card is computed here from that project's own media
 * listing, and anything that cannot be computed is simply not shown. Nothing is
 * invented: a project with no ready footage shows no duration, not a zero.
 *
 * The cost is one media request per card, which is the reason `RECENT_LIMIT`
 * exists. It is not wasted work -- the request is keyed `["media", projectId]`,
 * which is exactly the key the workstation uses, so opening a project from the
 * home screen finds its library already in the cache.
 */

/** How many projects the home screen fetches media for. */
export const RECENT_LIMIT = 8;

export interface ProjectSummary {
  project: Project;
  /** Ready assets, by kind. */
  videos: number;
  images: number;
  audio: number;
  /** Total duration of the ready video assets, or null if none have one. */
  footageMs: number | null;
  /** The commonest video geometry in the project, if any video has one. */
  width: number | null;
  height: number | null;
  /** The asset whose thumbnail represents the project. */
  cover: MediaAsset | null;
  /**
   * The most recent timestamp the client can observe: the newest media, or the
   * project's own creation. Deliberately *not* called "last edited" -- nothing
   * in the API records when the timeline was last touched, and labelling a
   * media upload as an edit would be a small, confident lie.
   */
  activeAt: string;
  /** False while this project's media listing is still in flight. */
  loaded: boolean;
}

/** Largest dimension wins ties, so a project reports its best available format. */
function dominantGeometry(videos: MediaAsset[]): { width: number | null; height: number | null } {
  const counts = new Map<string, { width: number; height: number; n: number }>();
  for (const asset of videos) {
    if (!asset.width || !asset.height) continue;
    const key = `${asset.width}x${asset.height}`;
    const entry = counts.get(key) ?? { width: asset.width, height: asset.height, n: 0 };
    entry.n += 1;
    counts.set(key, entry);
  }
  if (counts.size === 0) return { width: null, height: null };

  const best = [...counts.values()].sort(
    (a, b) => b.n - a.n || b.width * b.height - a.width * a.height,
  )[0];
  return { width: best.width, height: best.height };
}

/**
 * The cover frame.
 *
 * A video if there is one, because a project of footage should look like
 * footage; an image otherwise. Audio never covers a project -- it has no
 * thumbnail, and a card showing a placeholder where the picture goes is worse
 * than a card that admits it has none.
 */
function pickCover(assets: MediaAsset[]): MediaAsset | null {
  const usable = assets.filter((asset) => asset.status === "ready" && asset.has_thumbnail);
  return (
    usable.find((asset) => asset.kind === "video") ??
    usable.find((asset) => asset.kind === "image") ??
    null
  );
}

export function useProjectSummaries(projects: Project[]): ProjectSummary[] {
  const recent = projects.slice(0, RECENT_LIMIT);

  const results = useQueries({
    queries: recent.map((project) => ({
      // The workstation's own key: the cache is shared, not duplicated.
      queryKey: ["media", project.id],
      queryFn: () => api.listMedia(project.id),
      staleTime: 60_000,
    })),
  });

  return recent.map((project, index) => {
    const result = results[index];
    const assets = result?.data?.items ?? [];
    const ready = assets.filter((asset) => asset.status === "ready");
    const videos = ready.filter((asset) => asset.kind === "video");

    const durations = videos
      .map((asset) => asset.duration_ms)
      .filter((ms): ms is number => typeof ms === "number" && ms > 0);

    const newest = assets
      .map((asset) => asset.created_at)
      .concat(project.created_at)
      .sort()
      .at(-1);

    return {
      project,
      videos: videos.length,
      images: ready.filter((asset) => asset.kind === "image").length,
      audio: ready.filter((asset) => asset.kind === "audio").length,
      footageMs: durations.length > 0 ? durations.reduce((a, b) => a + b, 0) : null,
      ...dominantGeometry(videos),
      cover: pickCover(assets),
      activeAt: newest ?? project.created_at,
      loaded: !result?.isLoading,
    };
  });
}

/**
 * A geometry as an editor names it: `16:9`, `9:16`, `4:3`, or the ratio itself.
 *
 * Only the ratios the product actually offers get a name. Anything else prints
 * as a reduced fraction rather than being forced into the nearest preset, which
 * would tell the user their 2.39:1 footage is 16:9.
 */
export function aspectLabel(width: number | null, height: number | null): string | null {
  if (!width || !height) return null;
  const ratio = width / height;
  const named: [number, string][] = [
    [16 / 9, "16:9"],
    [9 / 16, "9:16"],
    [1, "1:1"],
    [4 / 5, "4:5"],
    [4 / 3, "4:3"],
    [21 / 9, "21:9"],
  ];
  for (const [value, label] of named) {
    if (Math.abs(ratio - value) < 0.02) return label;
  }
  const divisor = gcd(width, height);
  return `${Math.round(width / divisor)}:${Math.round(height / divisor)}`;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/** `1080p`, `720p`, `4K` — the way footage is actually described. */
export function formatLabel(width: number | null, height: number | null): string | null {
  if (!width || !height) return null;
  const shortest = Math.min(width, height);
  if (shortest >= 2000) return "4K";
  if (shortest >= 1400) return "1440p";
  if (shortest >= 1000) return "1080p";
  if (shortest >= 700) return "720p";
  if (shortest >= 460) return "480p";
  return `${shortest}p`;
}
