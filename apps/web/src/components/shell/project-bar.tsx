"use client";

import type { MediaAsset, Project } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { useEditorStore, type AppView } from "@/stores/editor-store";
import { Badge, Button, Glyph, IconButton, Menu, ToolGroup } from "@/components/ui";

/**
 * The project bar.
 *
 * What used to be the application bar, minus everything the navigation rail
 * took over. The old version carried the brand, a project picker, four verbs,
 * two panel toggles, a language select, a theme toggle, a help button and a
 * settings button -- fifteen controls in one strip, which is how a toolbar
 * turns into an admin panel.
 *
 * What is left is what belongs to *this project, right now*: which project is
 * open, where in it you are, the three verbs that act on the whole edit, and
 * the state the timeline is in. Global chrome lives on the rail.
 *
 * What is still deliberately absent: anything acting on a selection, which
 * belongs to the panel that owns that selection.
 */

const VIEW_LABEL: Record<Exclude<AppView, "home" | "settings">, MessageKey> = {
  editor: "nav.editor",
  assets: "nav.assets",
  templates: "nav.templates",
  audio: "nav.audio",
  export: "nav.exports",
};

export function ProjectBar({
  projects,
  project,
  media,
  analyzing,
  onOpenProject,
  onCreateProject,
  onAnalyze,
  onGoToPanel,
}: {
  projects: Project[];
  project: Project | null;
  media: MediaAsset[];
  analyzing: boolean;
  onOpenProject: (projectId: string) => void;
  onCreateProject: () => void;
  onAnalyze: (scope: "all" | "selection") => void;
  onGoToPanel: (tab: "ai" | "export") => void;
}) {
  const t = useT();

  const view = useEditorStore((s) => s.view);
  const selectedIds = useEditorStore((s) => s.selectedMediaIds);
  const browserOpen = useEditorStore((s) => s.browserOpen);
  const inspectorOpen = useEditorStore((s) => s.inspectorOpen);
  const toggleBrowser = useEditorStore((s) => s.toggleBrowser);
  const toggleInspector = useEditorStore((s) => s.toggleInspector);
  const dirty = useEditorStore((s) => s.dirty);
  const clipCount = useEditorStore((s) => s.clips.length);

  const readyCount = media.filter((asset) => asset.status === "ready").length;
  const selectedReady = media.filter(
    (asset) => selectedIds.includes(asset.id) && asset.status === "ready",
  ).length;

  const canAnalyseAll = readyCount > 0 && !analyzing;
  const canAnalyseSelection = selectedReady > 0 && !analyzing;

  const viewLabel = view in VIEW_LABEL ? VIEW_LABEL[view as keyof typeof VIEW_LABEL] : null;

  return (
    <header className="flex h-strip shrink-0 items-center gap-2.5 px-3">
      {/* ---- which project, as a breadcrumb you can steer with ---- */}
      <Menu
        label={t("top.project")}
        className="min-w-0"
        items={[
          ...projects.slice(0, 12).map((item) => ({
            key: item.id,
            label: item.title,
            icon: "folder" as const,
            onSelect: () => onOpenProject(item.id),
            disabled: item.id === project?.id,
          })),
          {
            key: "new",
            label: t("top.project.new"),
            icon: "plus" as const,
            onSelect: onCreateProject,
          },
        ]}
      >
        <span className="min-w-0 max-w-[220px] truncate text-sm font-semibold tracking-tight text-fg">
          {project?.title ?? t("top.project.select")}
        </span>
        <span className="text-faint">
          <Glyph name="chevron-down" size={11} />
        </span>
      </Menu>

      {viewLabel && (
        <>
          <span className="shrink-0 text-faint" aria-hidden>
            <Glyph name="chevron-right" size={11} />
          </span>
          <span className="shrink-0 text-xs text-muted">{t(viewLabel)}</span>
        </>
      )}

      {/* ---- what you do to the whole edit ---- */}
      <ToolGroup label={t("top.group.edit")} className="ml-3">
        <Button
          disabled={!canAnalyseAll}
          title={
            readyCount === 0
              ? t("top.analyse.noMedia")
              : t("top.analyse.allHint", { count: readyCount })
          }
          onClick={() => onAnalyze("all")}
        >
          <Glyph name="analyse" size={12} />
          {analyzing ? t("top.analyse.busy") : t("top.analyse")}
          <span className="font-mono tabular-nums text-faint">{readyCount}</span>
        </Button>
        <Button
          disabled={!canAnalyseSelection}
          title={
            selectedReady === 0
              ? t("top.analyse.noSelection")
              : t("top.analyse.selectionHint", { count: selectedReady })
          }
          onClick={() => onAnalyze("selection")}
        >
          {t("top.analyse.selection")}
          {selectedReady > 0 && (
            <span className="font-mono tabular-nums text-faint">{selectedReady}</span>
          )}
        </Button>
      </ToolGroup>

      <ToolGroup>
        <Button title={t("top.generate.hint")} onClick={() => onGoToPanel("ai")}>
          <Glyph name="wand" size={12} />
          {t("top.generate")}
        </Button>
        <Button
          tone="primary"
          disabled={clipCount === 0}
          title={t("top.render.hint")}
          onClick={() => onGoToPanel("export")}
        >
          <Glyph name="export" size={12} />
          {t("top.render")}
        </Button>
      </ToolGroup>

      {/* ---- what state it is in ----
          A readout, not a control. Sits beside the verbs because "17 clips,
          unsaved" is the answer to the question the verbs raise. */}
      {clipCount > 0 && (
        <div className="ml-1 flex shrink-0 items-center gap-2">
          <span className="font-mono text-2xs tabular-nums text-faint">
            {t.plural("timeline.clipCount", clipCount)}
          </span>
          {dirty && (
            <Badge tone="warn" title={t("timeline.editedHint")}>
              {t("timeline.edited")}
            </Badge>
          )}
        </div>
      )}

      {/* ---- how you look at it ----
          Only meaningful in the editor: the other workspaces have no side
          panels to toggle, and a control that does nothing where it is shown is
          worse than one that is not shown. */}
      {view === "editor" && (
        <ToolGroup label={t("top.group.workspace")} className="ml-auto">
          <IconButton
            label={browserOpen ? t("top.view.browser.hide") : t("top.view.browser.show")}
            active={browserOpen}
            onClick={toggleBrowser}
          >
            <Glyph name="panel-left" />
          </IconButton>
          <IconButton
            label={inspectorOpen ? t("top.view.inspector.hide") : t("top.view.inspector.show")}
            active={inspectorOpen}
            onClick={toggleInspector}
          >
            <Glyph name="panel-right" />
          </IconButton>
        </ToolGroup>
      )}
    </header>
  );
}
