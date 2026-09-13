"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type EditPlan, type Job, type MediaAsset } from "@/lib/api";
import { applyBounds, draftProblems } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import { Button, Dialog, EmptyState, useShortcuts } from "@/components/ui";
import { Inspector } from "@/components/inspector/inspector";
import { MediaBrowser } from "@/components/media-browser";
import { PlanReview } from "@/components/plan-review";
import { Preview } from "@/components/preview";
import { StatusBar } from "@/components/status-bar";
import { Timeline } from "@/components/timeline";
import { TopBar } from "@/components/top-bar";

/**
 * The editing workstation.
 *
 * Owns the layout and the project-scoped queries; every panel below is given
 * what it needs rather than fetching for itself, so the media list is fetched
 * once per project and not once per panel that happens to want a filename.
 *
 *   ┌────────────────────── top bar ──────────────────────┐
 *   │ browser │            preview           │ inspector  │
 *   │         ├─────────────────────────────┤             │
 *   │         │            timeline          │            │
 *   └───────────────────── status bar ─────────────────────┘
 */

const POLL_WHILE_WORKING_MS = 3000;
const BROWSER_WIDTH = 250;
const INSPECTOR_WIDTH = 300;

/** Below this, the side panels stop being useful and start being in the way. */
const NARROW_BREAKPOINT = 1180;

function activeJob(job: Job): boolean {
  return !["succeeded", "failed", "cancelled"].includes(job.status);
}

export function Workstation() {
  const queryClient = useQueryClient();
  const layoutRef = useRef<HTMLDivElement>(null);
  const resizeRef = useRef<{ startY: number; startHeight: number } | null>(null);

  const projectId = useEditorStore((s) => s.projectId);
  const openProject = useEditorStore((s) => s.openProject);
  const browserOpen = useEditorStore((s) => s.browserOpen);
  const inspectorOpen = useEditorStore((s) => s.inspectorOpen);
  const toggleBrowser = useEditorStore((s) => s.toggleBrowser);
  const toggleInspector = useEditorStore((s) => s.toggleInspector);
  const timelineHeight = useEditorStore((s) => s.timelineHeight);
  const setTimelineHeight = useEditorStore((s) => s.setTimelineHeight);
  const clips = useEditorStore((s) => s.clips);

  const [reviewPlanId, setReviewPlanId] = useState<string | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  // ------------------------------------------------------------------ queries
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });

  /**
   * What the server can do, fetched once here rather than per panel.
   *
   * Held at the top of the tree deliberately: the segment bounds it carries are
   * adopted as the editor's clamps, and resolving this query re-renders
   * everything below, so no panel can be left drawing against the defaults
   * after the real values have arrived.
   */
  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });

  const bounds = capabilities.data?.segment_bounds;
  useEffect(() => {
    if (bounds) applyBounds(bounds);
  }, [bounds]);

  const media = useQuery({
    queryKey: ["media", projectId],
    queryFn: () => api.listMedia(projectId!),
    enabled: Boolean(projectId),
    // Poll only while something is still being ingested. Ready media does not
    // change on its own, and a library of two hundred assets is not free.
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (asset: MediaAsset) => asset.status === "processing" || asset.status === "uploaded",
      )
        ? POLL_WHILE_WORKING_MS
        : false,
  });

  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.listProjectJobs(projectId!),
    enabled: Boolean(projectId),
    // Discovers jobs this tab has not seen; live progress arrives over SSE.
    refetchInterval: (query) =>
      query.state.data?.some(activeJob) ? POLL_WHILE_WORKING_MS : false,
  });

  const analysis = useQuery({
    queryKey: ["project-analysis", projectId],
    queryFn: () => api.projectAnalysis(projectId!),
    enabled: Boolean(projectId),
    staleTime: 30_000,
  });

  const renders = useQuery({
    queryKey: ["renders", projectId],
    queryFn: () => api.listRenders(projectId!),
    enabled: Boolean(projectId),
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (render) => render.status === "pending" || render.status === "rendering",
      )
        ? 2000
        : false,
  });

  const latestRenderId = renders.data?.items[0]?.id ?? null;

  /**
   * The latest render, fetched by id.
   *
   * The listing deliberately carries no playback URL -- it is presigned and
   * short-lived, so issuing one per row would sign twenty URLs to use one. The
   * detail route issues exactly the one the preview needs.
   */
  const render = useQuery({
    queryKey: ["render", latestRenderId],
    queryFn: () => api.getRender(projectId!, latestRenderId!),
    enabled: Boolean(projectId && latestRenderId),
    refetchInterval: (query) =>
      query.state.data?.status === "pending" || query.state.data?.status === "rendering"
        ? 2000
        : false,
    staleTime: 60_000,
  });

  const items = useMemo(() => media.data?.items ?? [], [media.data]);
  const byId = useMemo(() => new Map(items.map((asset) => [asset.id, asset])), [items]);
  const analyzedIds = useMemo(
    () =>
      new Set(
        (analysis.data?.items ?? [])
          .filter((record) => record.status === "ok")
          .map((record) => record.media_id),
      ),
    [analysis.data],
  );

  const invalidClipIds = useMemo(
    () =>
      new Set(
        draftProblems(clips, byId)
          .map((problem) => problem.clipId)
          .filter((id): id is string => id !== null),
      ),
    [clips, byId],
  );

  // ---------------------------------------------------------------- mutations
  const invalidateProject = useCallback(() => {
    for (const key of ["media", "jobs", "project-analysis", "renders"]) {
      void queryClient.invalidateQueries({ queryKey: [key, projectId] });
    }
  }, [queryClient, projectId]);

  const createProject = useMutation({
    mutationFn: (title: string) => api.createProject(title),
    onSuccess: (project) => {
      openProject(project.id);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const analyze = useMutation({
    mutationFn: (mediaIds: string[]) =>
      mediaIds.length === 0
        ? api.requestAnalysis(projectId!)
        : Promise.all(mediaIds.map((mediaId) => api.requestAnalysis(projectId!, mediaId))).then(
            (responses) => responses[0],
          ),
    onSuccess: invalidateProject,
  });

  const cancelJob = useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: invalidateProject,
  });

  // Open the first project automatically: an editor that opens to a chooser
  // when there is exactly one thing to choose is asking a pointless question.
  useEffect(() => {
    if (!projectId && projects.data && projects.data.length > 0) {
      openProject(projects.data[0].id);
    }
  }, [projectId, projects.data, openProject]);

  // ------------------------------------------------------------ layout sizing
  /** Drag the divider between the preview and the timeline. */
  useEffect(() => {
    function onMove(event: PointerEvent) {
      const state = resizeRef.current;
      if (!state) return;
      setTimelineHeight(state.startHeight - (event.clientY - state.startY));
    }
    function onUp() {
      resizeRef.current = null;
      document.body.classList.remove("vf-dragging");
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [setTimelineHeight]);

  /**
   * Collapse the side panels on a narrow desktop.
   *
   * Desktop-first, as an editor should be, but a 1280-wide laptop is a real
   * desktop and three panels plus a preview at that width leaves the preview
   * unusable. The browser goes first because it is the one whose work (picking
   * clips) happens before the work the other panels support.
   */
  useEffect(() => {
    function apply() {
      const narrow = window.innerWidth < NARROW_BREAKPOINT;
      const state = useEditorStore.getState();
      if (narrow && state.browserOpen && state.inspectorOpen) state.toggleBrowser();
    }
    apply();
    window.addEventListener("resize", apply);
    return () => window.removeEventListener("resize", apply);
  }, []);

  // -------------------------------------------------------------- shortcuts
  useShortcuts({
    " ": () => useEditorStore.getState().togglePlaying(),
    s: () => useEditorStore.getState().splitAtPlayhead(),
    Delete: () => {
      const state = useEditorStore.getState();
      if (state.selectedClipId) state.removeClip(state.selectedClipId);
    },
    Backspace: () => {
      const state = useEditorStore.getState();
      if (state.selectedClipId) state.removeClip(state.selectedClipId);
    },
    ArrowLeft: () => useEditorStore.getState().nudgePlayhead(-100),
    ArrowRight: () => useEditorStore.getState().nudgePlayhead(100),
    "shift+ArrowLeft": () => useEditorStore.getState().nudgePlayhead(-1000),
    "shift+ArrowRight": () => useEditorStore.getState().nudgePlayhead(1000),
    Home: () => useEditorStore.getState().setPlayhead(0),
    "+": () => useEditorStore.getState().zoom(1.4),
    "=": () => useEditorStore.getState().zoom(1.4),
    "-": () => useEditorStore.getState().zoom(1 / 1.4),
    b: toggleBrowser,
    i: toggleInspector,
    "?": () => setShortcutsOpen(true),
  });

  // ------------------------------------------------------------------ render
  if (!projectId) {
    return (
      <div className="flex h-screen flex-col">
        <TopBar
          projects={projects.data ?? []}
          projectId={null}
          media={[]}
          analyzing={false}
          onOpenProject={openProject}
          onCreateProject={(title) => createProject.mutate(title)}
          onAnalyze={() => undefined}
          onShowShortcuts={() => setShortcutsOpen(true)}
        />
        <div className="flex flex-1 items-center justify-center">
          <EmptyState>
            {projects.isLoading
              ? "Loading projects…"
              : projects.isError
                ? "Cannot reach the API. Start it with .\\scripts\\vf.ps1 api"
                : "No projects yet. Create one from the top bar to begin."}
          </EmptyState>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar
        projects={projects.data ?? []}
        projectId={projectId}
        media={items}
        analyzing={analyze.isPending}
        onOpenProject={openProject}
        onCreateProject={(title) => createProject.mutate(title)}
        onAnalyze={(scope) =>
          analyze.mutate(
            scope === "all" ? [] : useEditorStore.getState().selectedMediaIds,
          )
        }
        onShowShortcuts={() => setShortcutsOpen(true)}
      />

      <div ref={layoutRef} className="flex min-h-0 flex-1">
        {browserOpen && (
          <div className="min-h-0 shrink-0" style={{ width: BROWSER_WIDTH }}>
            <MediaBrowser
              projectId={projectId}
              media={items}
              loading={media.isLoading}
              analyzedIds={analyzedIds}
              onUploaded={invalidateProject}
            />
          </div>
        )}

        {/* ---- centre column: preview over timeline ---- */}
        <div className="flex min-w-0 flex-1 flex-col">
          <Preview projectId={projectId} media={byId} render={render.data ?? null} />

          <div
            role="separator"
            aria-label="Resize timeline"
            aria-orientation="horizontal"
            tabIndex={0}
            onPointerDown={(event) => {
              resizeRef.current = { startY: event.clientY, startHeight: timelineHeight };
              document.body.classList.add("vf-dragging");
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowUp") setTimelineHeight(timelineHeight + 20);
              if (event.key === "ArrowDown") setTimelineHeight(timelineHeight - 20);
            }}
            className="h-[3px] shrink-0 cursor-row-resize bg-line transition-colors hover:bg-accent"
          />

          <div className="flex shrink-0 flex-col" style={{ height: timelineHeight }}>
            <Timeline projectId={projectId} media={byId} invalidClipIds={invalidClipIds} />
          </div>
        </div>

        {inspectorOpen && (
          <div className="min-h-0 shrink-0" style={{ width: INSPECTOR_WIDTH }}>
            <Inspector
              projectId={projectId}
              media={byId}
              mediaList={items}
              render={render.data ?? null}
              onPlanned={(plan: EditPlan) => {
                setReviewPlanId(plan.id);
                setReviewOpen(true);
              }}
            />
          </div>
        )}
      </div>

      <StatusBar
        projectId={projectId}
        jobs={jobs.data ?? []}
        onCancel={(jobId) => cancelJob.mutate(jobId)}
      />

      <PlanReview
        projectId={projectId}
        planId={reviewPlanId}
        media={byId}
        open={reviewOpen}
        onClose={() => setReviewOpen(false)}
      />

      <ShortcutHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  );
}

const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["S", "Split the clip under the playhead"],
  ["Del / Backspace", "Delete the selected clip"],
  ["← →", "Nudge the playhead 100 ms"],
  ["Shift + ← →", "Nudge the playhead 1 s"],
  ["Home", "Playhead to start"],
  ["+ / -", "Zoom the timeline"],
  ["Ctrl + wheel", "Zoom the timeline at the pointer"],
  ["B", "Toggle the media browser"],
  ["I", "Toggle the inspector"],
  ["?", "This list"],
];

function ShortcutHelp({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard">
      <dl className="flex flex-col">
        {SHORTCUTS.map(([keys, description]) => (
          <div
            key={keys}
            className="flex items-baseline justify-between gap-4 border-b border-line/60 py-1 last:border-b-0"
          >
            <dt className="font-mono text-2xs text-fg">{keys}</dt>
            <dd className="text-xs text-muted">{description}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-2xs leading-snug text-dim">
        Single-key shortcuts are ignored while a text field has focus.
      </p>
      <div className="mt-2 flex justify-end">
        <Button onClick={onClose}>Close</Button>
      </div>
    </Dialog>
  );
}
