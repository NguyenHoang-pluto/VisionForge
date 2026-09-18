"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, APP_VERSION, type EditPlan, type Job, type MediaAsset } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { applyAudioBounds, applyBounds, draftProblems } from "@/lib/timeline";
import { useEditorStore, type InspectorTab } from "@/stores/editor-store";
import { Button, Dialog, KeyCap, useShortcuts } from "@/components/ui";
import { ProjectHome } from "@/components/home/project-home";
import { Inspector } from "@/components/inspector/inspector";
import { MediaBrowser } from "@/components/media-browser";
import { PlanReview } from "@/components/plan-review";
import { PreferencesDialog } from "@/components/preferences-dialog";
import { Preview } from "@/components/preview";
import { NavRail } from "@/components/shell/nav-rail";
import { ProjectBar } from "@/components/shell/project-bar";
import { StatusBar } from "@/components/status-bar";
import { Timeline } from "@/components/timeline";
import { NewProjectDialog } from "@/components/shell/new-project-dialog";
import { AudioWorkspace } from "@/components/views/audio-workspace";
import { AssetsWorkspace } from "@/components/views/assets-workspace";
import { ExportWorkspace } from "@/components/views/export-workspace";
import { TemplatesWorkspace } from "@/components/views/templates-workspace";

/**
 * The application shell.
 *
 * Owns the navigation, the layout and the project-scoped queries; every panel
 * below is given what it needs rather than fetching for itself, so the media
 * list is fetched once per project and not once per panel that happens to want
 * a filename.
 *
 *   ┌────┬────────────────── project bar ──────────────────┐
 *   │    ├─────────┬───────────────────────┬───────────────┤
 *   │ n  │ browser │        preview        │   inspector   │
 *   │ a  │         ├───────────────────────┤               │
 *   │ v  │         │       timeline        │               │
 *   │    ├─────────────────── status bar ──────────────────┤
 *
 * The panels are inset cards on the application ground rather than regions
 * divided by hairlines. That one change is most of why the workspace stopped
 * reading as a grid of boxes: the gaps do the separating, so the borders could
 * go, and a panel now has a shape instead of an outline.
 */

const POLL_WHILE_WORKING_MS = 3000;

/**
 * Panel widths, in the three sizes that matter.
 *
 * A fixed 250px browser is a third of the screen at 1280 and a strip at 1920.
 * These are the widths at which the browser shows two tiles per row and the
 * inspector's two-column forms stay two columns, measured at each breakpoint
 * rather than picked once and lived with. They are a little wider than the
 * previous revision's because the type inside them is a step larger.
 */
const BROWSER_WIDTH = { base: 244, md: 272, lg: 304 };
const INSPECTOR_WIDTH = { base: 308, md: 342, lg: 376 };

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
  const view = useEditorStore((s) => s.view);
  const setView = useEditorStore((s) => s.setView);
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

  const project = projects.data?.find((item) => item.id === projectId) ?? null;

  // ---------------------------------------------------------------- mutations
  const invalidateProject = useCallback(() => {
    for (const key of ["media", "jobs", "project-analysis", "renders"]) {
      void queryClient.invalidateQueries({ queryKey: [key, projectId] });
    }
  }, [queryClient, projectId]);

  const createProject = useMutation({
    mutationFn: (title: string) => api.createProject(title),
    onSuccess: (created) => {
      openProject(created.id);
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

  /** Show the inspector tab a verb refers to, opening the panel if shut. */
  const goToPanel = useCallback(
    (tab: InspectorTab) => {
      setInspectorTab(tab);
      const state = useEditorStore.getState();
      if (state.view !== "editor") state.setView("editor");
      if (!state.inspectorOpen) state.toggleInspector();
    },
    [setInspectorTab],
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
   * support, and because the Assets workspace gives it the whole viewport when
   * that is the job at hand. Below 1024 the inspector follows it.
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
    // The workspaces, on the digits. One press from anywhere to anywhere, which
    // is the point of having named places at all.
    "1": () => setView("home"),
    "2": () => projectId && setView("editor"),
    "3": () => projectId && setView("assets"),
    "4": () => projectId && setView("audio"),
    "5": () => projectId && setView("export"),
    "?": () => setShortcutsOpen(true),
  });

  // ------------------------------------------------------------------ chrome
  const chrome = (
    <>
      <PreferencesDialog open={preferencesOpen} onClose={() => setPreferencesOpen(false)} />
      <ShortcutHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
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
      {projectId && (
        <PlanReview
          projectId={projectId}
          planId={reviewPlanId}
          media={byId}
          open={reviewOpen}
          onClose={() => setReviewOpen(false)}
        />
      )}
    </>
  );

  // Home is the only workspace that works without a project, so it is also the
  // fallback: a project that has not loaded cannot be edited, exported or
  // listened to, and pretending otherwise produces four empty panels.
  const current = projectId ? view : "home";

  const onPlanned = (plan: EditPlan) => {
    setReviewPlanId(plan.id);
    setReviewOpen(true);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-ground">
      <NavRail
        onShowShortcuts={() => setShortcutsOpen(true)}
        onShowPreferences={() => setPreferencesOpen(true)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {current === "home" ? (
          <ProjectHome
            projects={projects.data ?? []}
            loading={projects.isLoading}
            unreachable={projects.isError}
            onOpen={openProject}
            onCreate={() => setCreatingProject(true)}
            onRetry={() => void projects.refetch()}
          />
        ) : (
          <>
            <ProjectBar
              projects={projects.data ?? []}
              project={project}
              media={items}
              analyzing={analyze.isPending}
              onOpenProject={openProject}
              onCreateProject={() => setCreatingProject(true)}
              onAnalyze={(scope) =>
                analyze.mutate(scope === "all" ? [] : useEditorStore.getState().selectedMediaIds)
              }
              onGoToPanel={goToPanel}
            />

            {current === "editor" && (
              <div className="vf-view flex min-h-0 flex-1 gap-2 px-2 pb-1">
                {browserOpen && (
                  <div className="min-h-0 shrink-0" style={{ width: BROWSER_WIDTH[step] }}>
                    <MediaBrowser
                      projectId={projectId!}
                      media={items}
                      loading={media.isLoading}
                      analyzedIds={analyzedIds}
                      onUploaded={invalidateProject}
                    />
                  </div>
                )}

                {/* ---- centre column: preview over timeline ---- */}
                <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                  <Preview projectId={projectId!} media={byId} render={render.data ?? null} />

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
                    className="group relative -my-1 flex h-3 shrink-0 cursor-row-resize items-center justify-center"
                  >
                    {/* A grip, so the divider looks draggable before it is
                        dragged. It brightens rather than appearing, so the
                        divider is never invisible and never loud. */}
                    <span
                      aria-hidden
                      className="h-[3px] w-10 rounded-full bg-strong transition-[background-color,width] duration-base group-hover:w-16 group-hover:bg-accent"
                    />
                  </div>

                  <div className="flex shrink-0 flex-col" style={{ height: timelineHeight }}>
                    <Timeline
                      projectId={projectId!}
                      media={byId}
                      invalidClipIds={invalidClipIds}
                    />
                  </div>
                </div>

                {inspectorOpen && (
                  <div className="min-h-0 shrink-0" style={{ width: INSPECTOR_WIDTH[step] }}>
                    <Inspector
                      projectId={projectId!}
                      media={byId}
                      mediaList={items}
                      render={render.data ?? null}
                      onPlanned={onPlanned}
                      onAnalyze={(mediaId) => analyze.mutate([mediaId])}
                    />
                  </div>
                )}
              </div>
            )}

            {current === "assets" && (
              <AssetsWorkspace
                projectId={projectId!}
                media={items}
                loading={media.isLoading}
                analyzedIds={analyzedIds}
                onUploaded={invalidateProject}
              />
            )}

            {current === "templates" && (
              <TemplatesWorkspace
                projectId={projectId!}
                media={items}
                loading={media.isLoading}
                analyzedIds={analyzedIds}
                onUploaded={invalidateProject}
                onAnalyzeAll={() => analyze.mutate([])}
                onPlanned={onPlanned}
              />
            )}

            {current === "audio" && (
              <AudioWorkspace
                projectId={projectId!}
                media={byId}
                mediaList={items}
                render={render.data ?? null}
                invalidClipIds={invalidClipIds}
              />
            )}

            {current === "export" && (
              <ExportWorkspace
                projectId={projectId!}
                media={byId}
                renders={renders.data?.items ?? []}
                render={render.data ?? null}
                jobs={jobs.data ?? []}
              />
            )}

            <StatusBar
              projectId={projectId!}
              jobs={jobs.data ?? []}
              onCancel={(jobId) => cancelJob.mutate(jobId)}
            />
          </>
        )}
      </div>

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
    title: "shortcuts.group.workspaces",
    rows: [
      { keys: ["1"], label: "nav.home" },
      { keys: ["2"], label: "nav.editor" },
      { keys: ["3"], label: "nav.assets" },
      { keys: ["4"], label: "nav.audio" },
      { keys: ["5"], label: "nav.exports" },
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
    <Dialog open={open} onClose={onClose} title={t("shortcuts.title")} size="lg">
      <div className="grid grid-cols-2 gap-x-panel-gap gap-y-5">
        {SHORTCUT_GROUPS.map((group) => (
          <section key={group.title}>
            <h3 className="text-2xs font-semibold uppercase tracking-[0.08em] text-faint">
              {t(group.title)}
            </h3>
            <dl className="mt-2 flex flex-col gap-1">
              {group.rows.map((row) => (
                <div
                  key={row.label}
                  className="flex min-h-row items-center justify-between gap-4"
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

      <p className="mt-5 text-2xs leading-snug text-faint">{t("shortcuts.note")}</p>
      <div className="mt-3 flex items-center justify-between gap-2">
        <span className="font-mono text-2xs text-faint">
          {t("app.version", { version: APP_VERSION })}
        </span>
        <Button onClick={onClose}>{t("common.close")}</Button>
      </div>
    </Dialog>
  );
}
