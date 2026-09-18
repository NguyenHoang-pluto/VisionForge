"use client";

import type { MediaAsset } from "@/lib/api";
import { MediaBrowser } from "@/components/media-browser";

/**
 * The Assets workspace.
 *
 * The media browser with the whole viewport instead of a 260px column. Same
 * component, same state, same selection -- what changes is how many tiles fit
 * on a row, which is the entire difference between "find the clip I am about to
 * trim" and "go through two hundred files and decide which twelve are any
 * good".
 *
 * Deliberately not a second implementation. A library view that duplicated the
 * browser would be a second place to fix an upload bug.
 */
export function AssetsWorkspace({
  projectId,
  media,
  loading,
  analyzedIds,
  onUploaded,
}: {
  projectId: string;
  media: MediaAsset[];
  loading: boolean;
  analyzedIds: Set<string>;
  onUploaded: () => void;
}) {
  return (
    <div className="vf-view flex min-h-0 flex-1 px-2 pb-1">
      <MediaBrowser
        projectId={projectId}
        media={media}
        loading={loading}
        analyzedIds={analyzedIds}
        onUploaded={onUploaded}
        wide
      />
    </div>
  );
}
