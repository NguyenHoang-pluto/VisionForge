"use client";

import type { MediaAsset, Render } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useEditorStore } from "@/stores/editor-store";
import { AudioPanel } from "@/components/inspector/audio-panel";
import { Preview } from "@/components/preview";
import { Timeline } from "@/components/timeline";
import { Panel, PanelHeader } from "@/components/ui";

/**
 * The Audio workspace.
 *
 * The music bed given a room of its own: the controls on the left, the picture
 * and the timeline on the right. The controls are the same inspector panel --
 * one implementation, one set of bounds, one place a fade is clamped -- but
 * beside a full-width timeline rather than a 300px column, because the question
 * you are answering here ("does this cut land on the beat") is asked of the
 * whole programme at once and cannot be answered through a keyhole.
 *
 * The preview is here rather than omitted because scoring is listening. Setting
 * a level or a fade without being able to play the picture under it is setting
 * a number, not mixing.
 *
 * Phase 7 is where beats, the bed and the mix arrived; this is where they
 * became something you can look at.
 */
export function AudioWorkspace({
  projectId,
  media,
  mediaList,
  render,
  invalidClipIds,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
  render: Render | null;
  invalidClipIds: Set<string>;
}) {
  const t = useT();
  const timelineHeight = useEditorStore((s) => s.timelineHeight);

  return (
    <div className="vf-view flex min-h-0 flex-1 gap-2 px-2 pb-1">
      <Panel className="w-[340px] shrink-0 overflow-hidden rounded-xl shadow-panel 2xl:w-[380px]">
        <PanelHeader title={t("audio.title")} icon="audio" />
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <AudioPanel projectId={projectId} media={media} mediaList={mediaList} />
        </div>
      </Panel>

      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <Preview projectId={projectId} media={media} render={render} />
        <div className="flex shrink-0 flex-col" style={{ height: timelineHeight }}>
          <Timeline projectId={projectId} media={media} invalidClipIds={invalidClipIds} />
        </div>
      </div>
    </div>
  );
}
