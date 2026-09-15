"use client";

import type { MediaAsset } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { AudioPanel } from "@/components/inspector/audio-panel";
import { Timeline } from "@/components/timeline";
import { Panel, PanelHeader } from "@/components/ui";

/**
 * The Audio workspace.
 *
 * The music bed given a room of its own. The controls are the same inspector
 * panel -- one implementation, one set of bounds, one place a fade is clamped
 * -- but beside a full-width timeline rather than a 300px column, because the
 * question you are answering here ("does this cut land on the beat") is asked
 * of the whole programme at once and cannot be answered through a keyhole.
 *
 * Phase 7 is where beats, the bed and the mix arrived; this is where they
 * became something you can look at.
 */
export function AudioWorkspace({
  projectId,
  media,
  mediaList,
  invalidClipIds,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
  invalidClipIds: Set<string>;
}) {
  const t = useT();

  return (
    <div className="vf-view flex min-h-0 flex-1 gap-2 px-2 pb-1">
      <Panel className="w-[340px] shrink-0 overflow-hidden rounded-xl shadow-panel 2xl:w-[380px]">
        <PanelHeader title={t("audio.title")} icon="audio" />
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <AudioPanel projectId={projectId} media={media} mediaList={mediaList} />
        </div>
      </Panel>

      <div className="flex min-w-0 flex-1 flex-col">
        <Timeline projectId={projectId} media={media} invalidClipIds={invalidClipIds} />
      </div>
    </div>
  );
}
