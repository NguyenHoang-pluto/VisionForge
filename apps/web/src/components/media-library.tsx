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
import { MediaCard } from "@/components/media-card";
import { JobList } from "@/components/job-list";

const POLL_WHILE_PROCESSING_MS = 3000;

interface UploadFailure {
  filename: string;
  message: string;
  hint: string | null;
}

export function MediaLibrary() {
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const [projectId, setProjectId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(
    null,
  );
  const [failures, setFailures] = useState<UploadFailure[]>([]);

  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  const media = useQuery({
    queryKey: ["media", projectId],
    queryFn: () => api.listMedia(projectId!),
    enabled: Boolean(projectId),
    // Poll only while something is still being processed.
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (m: MediaAsset) => m.status === "processing" || m.status === "uploaded",
      )
        ? POLL_WHILE_PROCESSING_MS
        : false,
  });

  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.listProjectJobs(projectId!),
    enabled: Boolean(projectId),
  });

  const activeJobIds = (jobs.data ?? [])
    .filter((job: Job) => !["succeeded", "failed", "cancelled"].includes(job.status))
    .map((job: Job) => job.id);

  const liveEvents = useJobEvents(activeJobIds);

  const createProject = useMutation({
    mutationFn: (title: string) => api.createProject(title),
    onSuccess: (project) => {
      setNewTitle("");
      setProjectId(project.id);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  const cancelJob = useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["media", projectId] });
    },
  });

  /**
   * Upload each file: ask the API for a presigned URL, PUT the bytes straight to
   * storage, then tell the API it landed. Files are sent one at a time — this
   * machine has limited memory and bandwidth, and a stampede of parallel PUTs
   * makes progress reporting meaningless.
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
            message:
              error instanceof ApiError ? error.message : "Upload failed",
            hint: error instanceof ApiError ? error.hint : null,
          });
        }
        setUploading({ done: index + 1, total: list.length });
      }

      setFailures(problems);
      setUploading(null);
      void queryClient.invalidateQueries({ queryKey: ["media", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
    [projectId, queryClient],
  );

  return (
    <div className="flex flex-col gap-8">
      {/* ---- projects ---- */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
        <h2 className="text-sm font-semibold text-slate-200">Projects</h2>

        <form
          className="mt-4 flex flex-wrap gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (newTitle.trim()) createProject.mutate(newTitle.trim());
          }}
        >
          <input
            id="project-title"
            value={newTitle}
            onChange={(event) => setNewTitle(event.target.value)}
            placeholder="New project name"
            className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-600 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!newTitle.trim() || createProject.isPending}
            className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {createProject.isPending ? "Creating…" : "Create project"}
          </button>
        </form>

        {projects.data && projects.data.length > 0 && (
          <ul className="mt-4 flex flex-wrap gap-2">
            {projects.data.map((project) => (
              <li key={project.id}>
                <button
                  type="button"
                  onClick={() => setProjectId(project.id)}
                  className={`rounded border px-3 py-1.5 text-xs transition ${
                    project.id === projectId
                      ? "border-sky-600 bg-sky-950/60 text-sky-300"
                      : "border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-200"
                  }`}
                >
                  {project.title}
                  <span className="ml-2 font-mono text-slate-500 tabular-nums">
                    {project.media_count}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {projects.data?.length === 0 && (
          <p className="mt-4 text-xs text-slate-500">
            No projects yet. Create one to start uploading media.
          </p>
        )}
      </section>

      {projectId && (
        <>
          {/* ---- upload ---- */}
          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <h2 className="text-sm font-semibold text-slate-200">Upload media</h2>
            <p className="mt-1 text-xs text-slate-500">
              Files are uploaded directly to object storage with a presigned URL —
              they never pass through the API.
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                disabled={Boolean(uploading)}
                className="rounded border border-slate-700 px-4 py-2 text-sm text-slate-200 transition hover:border-slate-600 disabled:opacity-40"
              >
                Select files
              </button>
              <button
                type="button"
                onClick={() => folderInput.current?.click()}
                disabled={Boolean(uploading)}
                className="rounded border border-slate-700 px-4 py-2 text-sm text-slate-200 transition hover:border-slate-600 disabled:opacity-40"
              >
                Select folder
              </button>

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
                // Non-standard but supported in Chromium and Safari.
                {...{ webkitdirectory: "", directory: "" }}
                onChange={(event) => void handleFiles(event.target.files)}
              />
            </div>

            {uploading && (
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Uploading</span>
                  <span className="font-mono tabular-nums">
                    {uploading.done} / {uploading.total}
                  </span>
                </div>
                <div className="mt-1.5 h-1 overflow-hidden rounded bg-slate-800">
                  <div
                    className="h-full bg-sky-500 transition-all"
                    style={{
                      width: `${(uploading.done / uploading.total) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )}

            {failures.length > 0 && (
              <ul className="mt-4 flex flex-col gap-2">
                {failures.map((failure) => (
                  <li
                    key={failure.filename}
                    className="rounded border border-rose-900/60 bg-rose-950/30 px-3 py-2 text-xs"
                  >
                    <span className="font-medium text-rose-300">
                      {failure.filename}
                    </span>
                    <span className="text-rose-400"> — {failure.message}</span>
                    {failure.hint && (
                      <span className="block text-rose-500/80">{failure.hint}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* ---- jobs ---- */}
          <JobList
            jobs={jobs.data ?? []}
            live={liveEvents}
            onCancel={(jobId) => cancelJob.mutate(jobId)}
          />

          {/* ---- media grid ---- */}
          <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
            <div className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold text-slate-200">Media library</h2>
              <span className="font-mono text-xs text-slate-500 tabular-nums">
                {media.data?.total ?? 0} assets
              </span>
            </div>

            {media.data?.items.length ? (
              <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                {media.data.items.map((asset) => (
                  <MediaCard key={asset.id} asset={asset} projectId={projectId} />
                ))}
              </ul>
            ) : (
              <p className="mt-4 text-xs text-slate-500">
                {media.isLoading ? "Loading…" : "No media yet."}
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
