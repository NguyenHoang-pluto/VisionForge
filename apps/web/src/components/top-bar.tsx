"use client";

import { useState } from "react";

import { useT } from "@/lib/i18n";
import { LANGUAGES } from "@/lib/i18n";
import type { MediaAsset, Project } from "@/lib/api";
import { usePreferences, type Language } from "@/stores/preferences-store";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  BrandMark,
  Button,
  Dialog,
  Glyph,
  IconButton,
  Menu,
  TextInput,
  ToolGroup,
} from "@/components/ui";

/**
 * The application bar.
 *
 * Grouped left to right in the order the work happens — what you are editing,
 * then what you do to it, then how you look at it — with the global chrome
 * pushed to the right where a desktop application keeps it. The separators are
 * drawn by `ToolGroup`, so the grouping is structural rather than a matter of
 * remembering to put a divider between the right two buttons.
 *
 * What is *not* here matters as much. Anything acting on a selection lives in
 * the panel that owns that selection; this bar holds only what is global. The
 * two navigation buttons (Generate, Render) are labelled as what they do —
 * open a panel — rather than implying the bar can plan or encode by itself.
 */
export function TopBar({
  projects,
  projectId,
  media,
  analyzing,
  onOpenProject,
  onCreateProject,
  onAnalyze,
  onShowShortcuts,
  onShowPreferences,
  onGoToPanel,
}: {
  projects: Project[];
  projectId: string | null;
  media: MediaAsset[];
  analyzing: boolean;
  onOpenProject: (projectId: string) => void;
  onCreateProject: (title: string) => void;
  onAnalyze: (scope: "all" | "selection") => void;
  onShowShortcuts: () => void;
  onShowPreferences: () => void;
  onGoToPanel: (tab: "ai" | "export") => void;
}) {
  const t = useT();
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState("");

  const selectedIds = useEditorStore((s) => s.selectedMediaIds);
  const browserOpen = useEditorStore((s) => s.browserOpen);
  const inspectorOpen = useEditorStore((s) => s.inspectorOpen);
  const toggleBrowser = useEditorStore((s) => s.toggleBrowser);
  const toggleInspector = useEditorStore((s) => s.toggleInspector);
  const dirty = useEditorStore((s) => s.dirty);
  const clipCount = useEditorStore((s) => s.clips.length);

  const theme = usePreferences((s) => s.theme);
  const toggleTheme = usePreferences((s) => s.toggleTheme);
  const language = usePreferences((s) => s.language);
  const setLanguage = usePreferences((s) => s.setLanguage);

  const readyCount = media.filter((asset) => asset.status === "ready").length;
  const selectedReady = media.filter(
    (asset) => selectedIds.includes(asset.id) && asset.status === "ready",
  ).length;

  const canAnalyseAll = Boolean(projectId) && readyCount > 0 && !analyzing;
  const canAnalyseSelection = selectedReady > 0 && !analyzing;

  return (
    <header className="flex h-strip shrink-0 items-center gap-2 border-b border-subtle bg-elevated px-2">
      {/* ---- brand, doubling as the application menu ---- */}
      <Menu
        label={t("app.menu")}
        items={[
          { key: "new", label: t("top.menu.newProject"), onSelect: () => setCreating(true) },
          { key: "prefs", label: t("top.menu.preferences"), onSelect: onShowPreferences },
          { key: "keys", label: t("top.menu.shortcuts"), onSelect: onShowShortcuts },
        ]}
      >
        <BrandMark size={15} />
        <span className="select-none whitespace-nowrap text-sm font-semibold tracking-tight text-fg">
          {t("app.name")}
        </span>
        <span className="text-faint">
          <Glyph name="chevron-down" size={10} />
        </span>
      </Menu>

      {/* ---- what is open ---- */}
      <ToolGroup label={t("top.group.project")} className="ml-1 min-w-0">
        <label htmlFor="project-picker" className="sr-only">
          {t("top.project")}
        </label>
        <select
          id="project-picker"
          value={projectId ?? ""}
          onChange={(event) => event.target.value && onOpenProject(event.target.value)}
          className="h-control min-w-0 max-w-[200px] rounded border border-strong bg-elevated px-1.5 text-xs text-fg transition-colors hover:bg-hover focus:border-accent"
        >
          <option value="" disabled>
            {t("top.project.select")}
          </option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.title} ({project.media_count})
            </option>
          ))}
        </select>

        <IconButton label={t("top.project.new")} onClick={() => setCreating(true)}>
          <Glyph name="plus" />
        </IconButton>
      </ToolGroup>

      {/* ---- what you do to it ---- */}
      <ToolGroup label={t("top.group.edit")}>
        <Button
          size="sm"
          disabled={!canAnalyseAll}
          title={
            readyCount === 0
              ? t("top.analyse.noMedia")
              : t("top.analyse.allHint", { count: readyCount })
          }
          onClick={() => onAnalyze("all")}
        >
          <Glyph name="analyse" size={11} />
          {analyzing ? t("top.analyse.busy") : t("top.analyse")}
          <span className="font-mono tabular-nums text-faint">{readyCount}</span>
        </Button>
        <Button
          size="sm"
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

        <Button
          size="sm"
          disabled={!projectId}
          title={t("top.generate.hint")}
          onClick={() => onGoToPanel("ai")}
        >
          <Glyph name="spark" size={11} />
          {t("top.generate")}
        </Button>
        <Button
          size="sm"
          tone="primary"
          disabled={!projectId || clipCount === 0}
          title={t("top.render.hint")}
          onClick={() => onGoToPanel("export")}
        >
          {t("top.render")}
        </Button>
      </ToolGroup>

      {/* ---- what state it is in ----
          A readout, not a control. Sits beside the verbs because "17 clips,
          unsaved" is the answer to the question the verbs raise. */}
      {clipCount > 0 && (
        <ToolGroup className="font-mono text-2xs tabular-nums text-faint">
          <span>{t.plural("timeline.clipCount", clipCount)}</span>
          {dirty && (
            <Badge tone="warn" title={t("timeline.editedHint")}>
              {t("timeline.edited")}
            </Badge>
          )}
        </ToolGroup>
      )}

      {/* ---- how you look at it ---- */}
      <div className="ml-auto flex items-center gap-2">
        <ToolGroup label={t("top.group.workspace")}>
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

        <ToolGroup>
          {/* Language is one click from anywhere rather than buried in
              preferences: it is the setting most likely to be wrong on first
              run, and the one whose wrongness makes everything else harder. */}
          <label htmlFor="language-picker" className="sr-only">
            {t("top.language")}
          </label>
          <select
            id="language-picker"
            value={language}
            title={t("top.language")}
            onChange={(event) => setLanguage(event.target.value as Language)}
            className="h-control rounded border border-transparent bg-transparent px-1 text-2xs uppercase tracking-wider text-muted transition-colors hover:bg-elevated hover:text-fg focus:border-accent"
          >
            {LANGUAGES.map((item) => (
              <option key={item.value} value={item.value}>
                {item.value.toUpperCase()}
              </option>
            ))}
          </select>

          <IconButton
            label={theme === "dark" ? t("top.theme.toLight") : t("top.theme.toDark")}
            onClick={toggleTheme}
          >
            <Glyph name={theme === "dark" ? "sun" : "moon"} />
          </IconButton>
          <IconButton label={t("top.shortcuts")} onClick={onShowShortcuts}>
            <span className="font-mono text-xs leading-none">?</span>
          </IconButton>
          <IconButton label={t("top.settings")} onClick={onShowPreferences}>
            <Glyph name="settings" />
          </IconButton>
        </ToolGroup>
      </div>

      <NewProjectDialog
        open={creating}
        title={title}
        onTitle={setTitle}
        onClose={() => setCreating(false)}
        onCreate={(value) => {
          onCreateProject(value);
          setTitle("");
          setCreating(false);
        }}
      />
    </header>
  );
}

export function NewProjectDialog({
  open,
  title,
  onTitle,
  onClose,
  onCreate,
}: {
  open: boolean;
  title: string;
  onTitle: (title: string) => void;
  onClose: () => void;
  onCreate: (title: string) => void;
}) {
  const t = useT();
  return (
    <Dialog open={open} onClose={onClose} title={t("top.project.new")}>
      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (title.trim()) onCreate(title.trim());
        }}
      >
        <label htmlFor="new-project-title" className="text-xs text-muted">
          {t("top.project.newTitle")}
        </label>
        <TextInput
          id="new-project-title"
          autoFocus
          value={title}
          maxLength={200}
          placeholder={t("top.project.newPlaceholder")}
          onChange={(event) => onTitle(event.target.value)}
        />
        <div className="mt-1 flex justify-end gap-1">
          <Button onClick={onClose}>{t("common.cancel")}</Button>
          <Button type="submit" tone="primary" disabled={!title.trim()}>
            {t("top.project.create")}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
