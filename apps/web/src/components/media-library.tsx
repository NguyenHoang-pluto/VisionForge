"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import {
  api,
  ApiError,
  uploadToStorage,
  type Job,
  type MediaAsset,
} from "@/lib/api";
import { useJobEvents } from "@/lib/use-job-events";
import { AnalysisInspector } from "@/components/analysis-inspector";
import { EditPanel } from "@/components/edit-panel";
import { MediaCard } from "@/components/media-card";
import { JobList } from "@/components/job-list";

const POLL_WHILE_WORKING_MS = 3000;

interface UploadFailure {
  filename: string;
  message: string;
  hint: string | null;
}

/** Toolbar button. One visual weight, used consistently. */
function ToolButton({
  children,
  onClick,
  disabled,
  primary,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-sm border px-2.5 py-1 text-[11px] transition focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600 disabled:cursor-not-allowed disabled:opacity-40 ${
        primary
          ? "border-sky-700 bg-sky-900/40 text-sky-200 hover:bg-sky-900/70"
          : "border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100"
      }`}
    >
      {children}
    </button>
  );
}

export function MediaLibrary() {
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const [projectId, setProjectId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(
    null,
  );
  const [failures, setFailures] = useState<UploadFailure[]>([]);

  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });

  const media = useQuery({
    queryKey: ["media", projectId],
    queryFn: () => api.listMedia(projectId!),
    enabled: Boolean(projectId),
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (m: MediaAsset) => m.status === "processing" || m.status === "uploaded",
      )
        ? POLL_WHILE_WORKING_MS
        : false,
  });

  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.listProjectJobs(projectId!),
    enabled: Boolean(projectId),
    refetchInterval: (query) =>
      query.state.data?.some(
        (j: Job) => !["succeeded", "failed", "cancelled"].includes(j.status),
      )
        ? POLL_WHILE_WORKING_MS
        : false,
  });

  /** Which assets already have analysis, for the grid badge. */
  const analysis = useQuery({
    queryKey: ["project-analysis", projectId],
    queryFn: () => api.projectAnalysis(projectId!),
    enabled: Boolean(projectId),
  });
  const analyzedIds = new Set(
    (analysis.data?.items ?? []).filter((a) => a.status === "ok").map((a) => a.media_id),
  );

  const activeJobIds = (jobs.data ?? [])
    .filter((job: Job) => !["succeeded", "failed", "cancelled"].includes(job.status))
    .map((job: Job) => job.id);
  const liveEvents = useJobEvents(activeJobIds);

  const createProject = useMutation({
    mutationFn: (title: string) => api.createProject(title),
    onSuccess: (project) => {
      setNewTitle("");
      setProjectId(project.id);
      setSelectedId(null);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const cancelJob = useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: () => invalidateProject(),
  });

  const analyze = useMutation({
    mutationFn: ({ mediaId, lanes }: { mediaId?: string; lanes?: string[] }) =>
      api.requestAnalysis(projectId!, mediaId, lanes),
    onSuccess: () => invalidateProject(),
  });

  function invalidateProject() {
    for (const key of ["media", "jobs", "project-analysis"]) {
      void queryClient.invalidateQueries({ queryKey: [key, projectId] });
    }
    if (selectedId) {
      void queryClient.invalidateQueries({ queryKey: ["analysis", selectedId] });
    }
  }

  /**
   * Upload each file: presign, PUT straight to storage, then tell the API it
   * landed. One at a time — this machine has limited memory and bandwidth, and a
   * stampede of parallel PUTs makes the progress count meaningless.
   */
  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length || !projectId) return;

      const list = Array.from(files);
      setFailures([]);
      setUploading({ done: 0, total: list.length });

      const problems: UploadFailure[] = [];
      for (const [index, file] of list.entries()) {
        try {
          const ticket = await api.createUploadUrl(projectId, file.name, file.size);
          await uploadToStorage(ticket.upload_url, file);
          await api.completeUpload(projectId, ticket.media_id);
        } catch (error) {
          problems.push({
            filename: file.name,
            message: error instanceof ApiError ? error.message : "Upload failed",
            hint: error instanceof ApiError ? error.hint : null,
          });
        }
        setUploading({ done: index + 1, total: list.length });
      }

      setFailures(problems);
      setUploading(null);
      invalidateProject();
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectId, queryClient],
  );

  const items = media.data?.items ?? [];
  const selected = items.find((m) => m.id === selectedId) ?? null;
  const readyCount = items.filter((m) => m.status === "ready").length;

  return (
    <div className="flex flex-col border border-slate-800 bg-slate-950">
      {/* ---------------- project bar ---------------- */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
          Project
        </span>

        {projects.data?.map((project) => (
          <button
            key={project.id}
            type="button"
            onClick={() => {
              setProjectId(project.id);
              setSelectedId(null);
            }}
            className={`rounded-sm border px-2 py-0.5 text-[11px] transition focus:outline-none focus-visible:ring-1 focus-visible:ring-sky-600 ${
              project.id === projectId
                ? "border-sky-700 bg-sky-900/40 text-sky-200"
                : "border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
            }`}
          >
            {project.title}
            <span className="ml-1.5 font-mono text-[9px] text-slate-600 tabular-nums">
              {project.media_count}
            </span>
          </button>
        ))}

        <form
          className="ml-auto flex gap-1.5"
          onSubmit={(event) => {
            event.preventDefault();
            if (newTitle.trim()) createProject.mutate(newTitle.trim());
          }}
        >
          <input
            id="project-title"
            value={newTitle}
            onChange={(event) => setNewTitle(event.target.value)}
            placeholder="New project"
            className="w-36 rounded-sm border border-slate-800 bg-slate-950 px-2 py-0.5 text-[11px] text-slate-200 placeholder:text-slate-700 focus:border-sky-700 focus:outline-none"
          />
          <ToolButton
            onClick={() => newTitle.trim() && createProject.mutate(newTitle.trim())}
            disabled={!newTitle.trim() || createProject.isPending}
          >
            Create
          </ToolButton>
        </form>
      </div>

      {!projectId ? (
        <p className="px-3 py-8 text-center text-[11px] text-slate-600">
          Select or create a project to begin.
        </p>
      ) : (
        <>
          {/* ---------------- toolbar ---------------- */}
          <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
            <ToolButton
              onClick={() => fileInput.current?.click()}
              disabled={Boolean(uploading)}
            >
              Add files
            </ToolButton>
            <ToolButton
              onClick={() => folderInput.current?.click()}
              disabled={Boolean(uploading)}
            >
              Add folder
            </ToolButton>

            <span className="mx-1 h-4 w-px bg-slate-800" aria-hidden />

            <ToolButton
              primary
              onClick={() => analyze.mutate({})}
              disabled={readyCount === 0 || analyze.isPending}
            >
              Analyze all ({readyCount})
            </ToolButton>
            <ToolButton
              onClick={() => selected && analyze.mutate({ mediaId: selected.id })}
              disabled={!selected || selected.status !== "ready" || analyze.isPending}
            >
              Analyze selection
            </ToolButton>
            <ToolButton
              onClick={() =>
                selected && analyze.mutate({ mediaId: selected.id, lanes: ["cpu"] })
              }
              disabled={!selected || selected.status !== "ready" || analyze.isPending}
            >
              CPU only
            </ToolButton>

            <span className="ml-auto font-mono text-[10px] text-slate-600 tabular-nums">
              {items.length} assets · {analyzedIds.size} analyzed
            </span>

            <input
              ref={fileInput}
              id="file-input"
              type="file"
              multiple
              hidden
              accept="image/*,video/*,audio/*"
              onChange={(event) => void handleFiles(event.target.files)}
            />
            <input
              ref={folderInput}
              id="folder-input"
              type="file"
              multiple
              hidden
              {...{ webkitdirectory: "", directory: "" }}
              onChange={(event) => void handleFiles(event.target.files)}
            />
          </div>

          {uploading && (
            <div className="border-b border-slate-800 px-3 py-1.5">
              <div className="flex justify-between font-mono text-[10px] text-slate-500 tabular-nums">
                <span>Uploading</span>
                <span>
                  {uploading.done} / {uploading.total}
                </span>
              </div>
              <div className="mt-1 h-0.5 bg-slate-800">
                <div
                  className="h-full bg-sky-500 transition-all"
                  style={{ width: `${(uploading.done / uploading.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          {failures.length > 0 && (
            <ul className="border-b border-slate-800">
              {failures.map((failure) => (
                <li
                  key={failure.filename}
                  className="border-l-2 border-rose-800 px-3 py-1 text-[10px]"
                >
                  <span className="font-mono text-rose-300">{failure.filename}</span>
                  <span className="text-rose-400"> — {failure.message}</span>
                  {failure.hint && (
                    <span className="block text-rose-500/70">{failure.hint}</span>
                  )}
                </li>
              ))}
            </ul>
          )}

          {/* ---------------- grid + inspector ---------------- */}
          <div className="grid lg:grid-cols-[1fr_320px]">
            <div className="min-h-[240px] p-3">
              {items.length > 0 ? (
                <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
                  {items.map((asset) => (
                    <MediaCard
                      key={asset.id}
                      asset={asset}
                      projectId={projectId}
                      selected={asset.id === selectedId}
                      analyzed={analyzedIds.has(asset.id)}
                      onSelect={() =>
                        setSelectedId(asset.id === selectedId ? null : asset.id)
                      }
                    />
                  ))}
                </ul>
              ) : (
                <p className="py-8 text-center text-[11px] text-slate-600">
                  {media.isLoading ? "Loading…" : "No media. Add files to begin."}
                </p>
              )}
            </div>

            {selected && (
              <AnalysisInspector
                projectId={projectId}
                asset={selected}
                onClose={() => setSelectedId(null)}
              />
            )}
          </div>

          <EditPanel projectId={projectId} media={items} />

          <JobList
            jobs={jobs.data ?? []}
            live={liveEvents}
            onCancel={(jobId) => cancelJob.mutate(jobId)}
          />
        </>
      )}
    </div>
  );
}
