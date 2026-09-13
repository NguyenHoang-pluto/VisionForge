"use client";

import { useState } from "react";

import type { MediaAsset, Project } from "@/lib/api";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  Dialog,
  Divider,
  Glyph,
  IconButton,
  TextInput,
} from "@/components/ui";

/**
 * The application bar.
 *
 * Verbs, left to right, in the order the work happens: pick a project, import,
 * analyse, then the panel toggles. Everything that acts on a *selection* lives
 * in the panel that owns the selection; this bar only holds what is global.
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
}: {
  projects: Project[];
  projectId: string | null;
  media: MediaAsset[];
  analyzing: boolean;
  onOpenProject: (projectId: string) => void;
  onCreateProject: (title: string) => void;
  onAnalyze: (scope: "all" | "selection") => void;
  onShowShortcuts: () => void;
}) {
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState("");

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

  return (
    <header className="flex h-[34px] shrink-0 items-center gap-2 border-b border-line bg-raised px-2">
      <span className="select-none whitespace-nowrap text-sm font-semibold tracking-tight text-fg">
        VisionForge
      </span>

      <Divider vertical />

      {/* ---- project ---- */}
      <label htmlFor="project-picker" className="sr-only">
        Project
      </label>
      <select
        id="project-picker"
        value={projectId ?? ""}
        onChange={(event) => event.target.value && onOpenProject(event.target.value)}
        className="h-[24px] max-w-[190px] rounded border border-line-strong bg-control px-1.5 text-xs text-fg focus:border-accent"
      >
        <option value="" disabled>
          Select a project…
        </option>
        {projects.map((project) => (
          <option key={project.id} value={project.id}>
            {project.title} ({project.media_count})
          </option>
        ))}
      </select>

      <IconButton label="New project" onClick={() => setCreating(true)}>
        <Glyph name="plus" />
      </IconButton>

      <Divider vertical />

      {/* ---- analysis ---- */}
      <Button
        size="sm"
        disabled={!projectId || readyCount === 0 || analyzing}
        title="Analyse every ready asset in this project"
        onClick={() => onAnalyze("all")}
      >
        Analyse all
        <span className="font-mono text-dim tabular-nums">{readyCount}</span>
      </Button>
      <Button
        size="sm"
        disabled={selectedReady === 0 || analyzing}
        title="Analyse the selected assets"
        onClick={() => onAnalyze("selection")}
      >
        Analyse selection
        {selectedReady > 0 && (
          <span className="font-mono text-dim tabular-nums">{selectedReady}</span>
        )}
      </Button>

      <Divider vertical />

      <span className="flex items-center gap-1.5 font-mono text-2xs text-dim tabular-nums">
        {clipCount > 0 && (
          <>
            <span>
              {clipCount} clip{clipCount === 1 ? "" : "s"}
            </span>
            {dirty && <Badge tone="warn">unsaved</Badge>}
          </>
        )}
      </span>

      {/* ---- panels ---- */}
      <span className="ml-auto flex items-center gap-1">
        <Button size="sm" tone="ghost" onClick={onShowShortcuts}>
          Shortcuts
        </Button>
        <IconButton
          label={browserOpen ? "Hide media browser" : "Show media browser"}
          active={browserOpen}
          onClick={toggleBrowser}
        >
          <Glyph name="grid" />
        </IconButton>
        <IconButton
          label={inspectorOpen ? "Hide inspector" : "Show inspector"}
          active={inspectorOpen}
          onClick={toggleInspector}
        >
          <Glyph name="list" />
        </IconButton>
      </span>

      <Dialog open={creating} onClose={() => setCreating(false)} title="New project">
        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (!title.trim()) return;
            onCreateProject(title.trim());
            setTitle("");
            setCreating(false);
          }}
        >
          <label htmlFor="new-project-title" className="text-xs text-muted">
            Title
          </label>
          <TextInput
            id="new-project-title"
            autoFocus
            value={title}
            maxLength={200}
            placeholder="Football highlights"
            onChange={(event) => setTitle(event.target.value)}
          />
          <div className="mt-1 flex justify-end gap-1">
            <Button onClick={() => setCreating(false)}>Cancel</Button>
            <Button type="submit" tone="primary" disabled={!title.trim()}>
              Create
            </Button>
          </div>
        </form>
      </Dialog>
    </header>
  );
}
