"use client";

import type { EditPlan, MediaAsset, Render } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { useEditorStore, type InspectorTab } from "@/stores/editor-store";
import { EmptyState, Panel, Tabs } from "@/components/ui";
import { AiEditPanel } from "@/components/inspector/ai-edit-panel";
import { AnalysisPanel } from "@/components/inspector/analysis-panel";
import { AudioPanel } from "@/components/inspector/audio-panel";
import { ClipProperties } from "@/components/inspector/clip-properties";
import { ExportPanel } from "@/components/inspector/export-panel";

/**
 * The inspector.
 *
 * Five tabs over one selection. AI lives here, as one tab among five, which is
 * the honest weighting: it is a way of producing a first cut, not the point of
 * the application. Audio joined them in Phase 7 rather than becoming a mode of
 * its own, for the same reason -- a music bed is a property of the edit, not a
 * separate activity.
 */
const TABS: { value: InspectorTab; label: MessageKey }[] = [
  { value: "clip", label: "inspector.tab.clip" },
  { value: "analysis", label: "inspector.tab.analysis" },
  { value: "ai", label: "inspector.tab.ai" },
  { value: "audio", label: "inspector.tab.audio" },
  { value: "export", label: "inspector.tab.export" },
];

export function Inspector({
  projectId,
  media,
  mediaList,
  render,
  onPlanned,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
  render: Render | null;
  onPlanned: (plan: EditPlan) => void;
}) {
  const t = useT();
  const tab = useEditorStore((s) => s.inspectorTab);
  const setTab = useEditorStore((s) => s.setInspectorTab);
  const activeMediaId = useEditorStore((s) => s.activeMediaId);
  const selectedClipId = useEditorStore((s) => s.selectedClipId);
  const clips = useEditorStore((s) => s.clips);
  const selectMedia = useEditorStore((s) => s.selectMedia);

  // Analysis follows the timeline selection when there is one, so clicking a
  // clip and switching to Analysis shows that clip's source rather than
  // whatever was last touched in the browser.
  const selectedClip = clips.find((clip) => clip.id === selectedClipId);
  const analysisAsset = selectedClip
    ? media.get(selectedClip.mediaId)
    : activeMediaId
      ? media.get(activeMediaId)
      : undefined;

  const readyCount = mediaList.filter((asset) => asset.status === "ready").length;

  return (
    <Panel className="h-full border-l border-subtle">
      <Tabs
        value={tab}
        onChange={setTab}
        tabs={TABS.map((item) => ({ value: item.value, label: t(item.label) }))}
        label={t("inspector.title")}
      />

      <div
        role="tabpanel"
        aria-label={t(TABS.find((item) => item.value === tab)?.label ?? "inspector.title")}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      >
        {tab === "clip" && <ClipProperties media={media} />}

        {tab === "analysis" &&
          (analysisAsset ? (
            <AnalysisPanel
              projectId={projectId}
              asset={analysisAsset}
              onSelectMedia={(mediaId) => selectMedia(mediaId)}
            />
          ) : (
            <EmptyState icon="analyse">{t("analysis.selectSomething")}</EmptyState>
          ))}

        {tab === "ai" && (
          <AiEditPanel projectId={projectId} readyCount={readyCount} onPlanned={onPlanned} />
        )}

        {tab === "audio" && (
          <AudioPanel projectId={projectId} media={media} mediaList={mediaList} />
        )}

        {tab === "export" && (
          <ExportPanel projectId={projectId} media={media} render={render} />
        )}
      </div>
    </Panel>
  );
}
