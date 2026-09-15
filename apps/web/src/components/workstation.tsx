"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, APP_VERSION, type EditPlan, type Job, type MediaAsset } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { applyAudioBounds, applyBounds, draftProblems } from "@/lib/timeline";
import { useEditorStore, type InspectorTab } from "@/stores/editor-store";
import { Button, Dialog, KeyCap, useShortcuts } from "@/components/ui";
import { Inspector } from "@/components/inspector/inspector";
import { MediaBrowser } from "@/components/media-browser";
import { PlanReview } from "@/components/plan-review";
import { PreferencesDialog } from "@/components/preferences-dialog";
import { Preview } from "@/components/preview";
import { StatusBar } from "@/components/status-bar";
import { Timeline } from "@/components/timeline";
import { NewProjectDialog, TopBar } from "@/components/top-bar";
import { Welcome } from "@/components/welcome";

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

/**
 * Panel widths, in the three sizes that matter.
 *
 * A fixed 250px browser is a third of the screen at 1280 and a strip at 1920.
 * These are the widths at which the browser shows two tiles per row and the
 * inspector's two-column forms stay two columns, measured at each breakpoint
 * rather than picked once and lived with.
 */
const BROWSER_WIDTH = { base: 232, md: 256, lg: 288 };
const INSPECTOR_WIDTH = { base: 280, md: 300, lg: 328 };

/** Below this, the side panels stop being useful and start being in the way. */
const NARROW_BREAKPOINT = 1180;
/** And below this, one side panel is all that fits beside a usable preview. */
const VERY_NARROW_BREAKPOINT = 1024;

function activeJob(job: Job): boolean {
  return !["succeeded", "failed", "cancelled"].includes(job.status);
}

/** Which of the three width steps the viewport is in. */
function widthStep(width: number): "base" | "md" | "lg" {
  if (width >= 1800) return "lg";
  if (width >= 1400) return "md";
  return "base";
}

export function Workstation() {
  const t = useT();
  const queryClient = useQueryClient();
  const resizeRef = useRef<{ startY: number; startHeight: number } | null>(null);

  const projectId = useEditorStore((s) => s.projectId);
  const openProject = useEditorStore((s) => s.openProject);
  const browserOpen = useEditorStore((s) => s.browserOpen);
  const inspectorOpen = useEditorStore((s) => s.inspectorOpen);
  const toggleBrowser = useEditorStore((s) => s.toggleBrowser);
  const toggleInspector = useEditorStore((s) => s.toggleInspector);
  const setInspectorTab = useEditorStore((s) => s.setInspectorTab);
  const timelineHeight = useEditorStore((s) => s.timelineHeight);
  const setTimelineHeight = useEditorStore((s) => s.setTimelineHeight);
  const clips = useEditorStore((s) => s.clips);

  const [reviewPlanId, setReviewPlanId] = useState<string | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [step, setStep] = useState<"base" | "md" | "lg">("base");

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

  const audioBounds = capabilities.data?.audio_bounds;
  useEffect(() => {
    if (audioBounds) applyAudioBounds(audioBounds);
  }, [audioBounds]);

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

  /** Show the inspector tab a top-bar verb refers to, opening the panel if shut. */
  const goToPanel = useCallback(
    (tab: InspectorTab) => {
      setInspectorTab(tab);
      if (!useEditorStore.getState().inspectorOpen) toggleInspector();
    },
    [setInspectorTab, toggleInspector],
  );

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
   * Adapt to the width of the desktop.
   *
   * Desktop-first, as an editor should be, but 1280 and 1920 are both real
   * desktops and one set of panel widths cannot serve them. Three steps, and
   * below 1180 the browser is collapsed rather than squeezed -- it goes first
   * because its work (picking clips) happens before the work the other panels
   * support. Below 1024 the inspector follows it.
   *
   * Collapsing is done once per crossing rather than on every resize event, so
   * a user who reopens a panel at a narrow width keeps it open.
   */
  useEffect(() => {
    let previous: number | null = null;

    function apply() {
      const width = window.innerWidth;
      setStep(widthStep(width));

      const band = width < VERY_NARROW_BREAKPOINT ? 0 : width < NARROW_BREAKPOINT ? 1 : 2;
      if (band === previous) return;
      const shrinking = previous === null || band < previous;
      previous = band;
      // Widening never closes a panel, and re-entering a band the user has
      // already overruled by reopening a panel does not close it again.
      if (!shrinking) return;

      const state = useEditorStore.getState();
      if (band <= 1 && state.browserOpen && state.inspectorOpen) state.toggleBrowser();
      if (band === 0 && state.inspectorOpen) state.toggleInspector();
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
  const chrome = (
    <>
      <PreferencesDialog open={preferencesOpen} onClose={() => setPreferencesOpen(false)} />
      <ShortcutHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </>
  );

  if (!projectId) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <TopBar
          projects={projects.data ?? []}
          projectId={null}
          media={[]}
          analyzing={false}
          onOpenProject={openProject}
          onCreateProject={(title) => createProject.mutate(title)}
          onAnalyze={() => undefined}
          onShowShortcuts={() => setShortcutsOpen(true)}
          onShowPreferences={() => setPreferencesOpen(true)}
          onGoToPanel={goToPanel}
        />

        <Welcome
          projects={projects.data ?? []}
          loading={projects.isLoading}
          unreachable={projects.isError}
          onOpen={openProject}
          onCreate={() => setCreatingProject(true)}
          onRetry={() => void projects.refetch()}
        />

        <NewProjectDialog
          open={creatingProject}
          title={newTitle}
          onTitle={setNewTitle}
          onClose={() => setCreatingProject(false)}
          onCreate={(value) => {
            createProject.mutate(value);
            setNewTitle("");
            setCreatingProject(false);
          }}
        />
        {chrome}
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
        onShowPreferences={() => setPreferencesOpen(true)}
        onGoToPanel={goToPanel}
      />

      <div className="flex min-h-0 flex-1">
        {browserOpen && (
          <div className="min-h-0 shrink-0" style={{ width: BROWSER_WIDTH[step] }}>
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
            aria-label={t("timeline.resize")}
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
            className="group relative h-[4px] shrink-0 cursor-row-resize bg-line transition-colors hover:bg-accent"
          >
            {/* A grip, so the divider looks draggable before it is dragged. */}
            <span
              aria-hidden
              className="absolute left-1/2 top-1/2 h-[2px] w-8 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-line-strong transition-colors group-hover:bg-accent-strong"
            />
          </div>

          <div className="flex shrink-0 flex-col" style={{ height: timelineHeight }}>
            <Timeline projectId={projectId} media={byId} invalidClipIds={invalidClipIds} />
          </div>
        </div>

        {inspectorOpen && (
          <div className="min-h-0 shrink-0" style={{ width: INSPECTOR_WIDTH[step] }}>
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
      {chrome}
    </div>
  );
}

/**
 * The shortcut list.
 *
 * Grouped by what the keys are for rather than listed alphabetically: someone
 * opening this is looking for "how do I cut", not for "what does S do".
 */
const SHORTCUT_GROUPS: {
  title: MessageKey;
  rows: { keys: string[]; label: MessageKey }[];
}[] = [
  {
    title: "shortcuts.group.playback",
    rows: [
      { keys: ["Space"], label: "shortcuts.playPause" },
      { keys: ["←", "→"], label: "shortcuts.nudge" },
      { keys: ["Shift", "←", "→"], label: "shortcuts.nudgeBig" },
      { keys: ["Home"], label: "shortcuts.home" },
    ],
  },
  {
    title: "shortcuts.group.editing",
    rows: [
      { keys: ["S"], label: "shortcuts.split" },
      { keys: ["Del"], label: "shortcuts.delete" },
    ],
  },
  {
    title: "shortcuts.group.view",
    rows: [
      { keys: ["+", "−"], label: "shortcuts.zoom" },
      { keys: ["Ctrl", "wheel"], label: "shortcuts.zoomPointer" },
      { keys: ["B"], label: "shortcuts.browser" },
      { keys: ["I"], label: "shortcuts.inspector" },
      { keys: ["?"], label: "shortcuts.help" },
    ],
  },
];

function ShortcutHelp({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useT();
  return (
    <Dialog open={open} onClose={onClose} title={t("shortcuts.title")}>
      <div className="flex flex-col gap-panel-gap">
        {SHORTCUT_GROUPS.map((group) => (
          <section key={group.title}>
            <h3 className="border-b border-line-strong pb-1 text-xs font-semibold uppercase tracking-wider text-muted">
              {t(group.title)}
            </h3>
            <dl className="mt-1 flex flex-col">
              {group.rows.map((row) => (
                <div
                  key={row.label}
                  className="flex min-h-row items-center justify-between gap-4 border-b border-line/60 py-1 last:border-b-0"
                >
                  <dt className="flex shrink-0 items-center gap-1">
                    {row.keys.map((key) => (
                      <KeyCap key={key}>{key}</KeyCap>
                    ))}
                  </dt>
                  <dd className="truncate text-xs text-muted">{t(row.label)}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>

      <p className="mt-3 text-2xs leading-snug text-dim">{t("shortcuts.note")}</p>
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="font-mono text-2xs text-dim">
          {t("app.version", { version: APP_VERSION })}
        </span>
        <Button onClick={onClose}>{t("common.close")}</Button>
      </div>
    </Dialog>
  );
}
