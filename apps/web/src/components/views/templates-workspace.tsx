"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, ApiError, type EditPlan, type EditTemplate, type MediaAsset } from "@/lib/api";
import { shortDuration } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { toDraft, toMusicRequest } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import { MediaBrowser } from "@/components/media-browser";
import { SlotStrip, templateLabel } from "@/components/inspector/templates-panel";
import {
  Badge,
  Button,
  ErrorNote,
  Field,
  Panel,
  PanelHeader,
  Select,
  Spinner,
  TextInput,
} from "@/components/ui";

/**
 * The Templates workspace (Phase 12).
 *
 * A place of its own, because a template is something you come back to. The
 * library and every template you have made are laid out as cards, each drawn
 * as the row of shots it is; picking one and pressing Use builds the edit from
 * whatever is in the project -- photos and videos alike -- and opens it in the
 * editor.
 *
 * The media browser sits beside the gallery rather than on another screen,
 * because the other half of "use this template" is "on these files": a folder
 * can be added, analysed and cut from this one view.
 */
export function TemplatesWorkspace({
  projectId,
  media,
  loading,
  analyzedIds,
  onUploaded,
  onAnalyzeAll,
  onPlanned,
}: {
  projectId: string;
  media: MediaAsset[];
  loading: boolean;
  analyzedIds: Set<string>;
  onUploaded: () => void;
  onAnalyzeAll: () => void;
  onPlanned: (plan: EditPlan) => void;
}) {
  const t = useT();
  const queryClient = useQueryClient();

  const resolution = useEditorStore((s) => s.resolution);
  const encoder = useEditorStore((s) => s.encoder);
  const fps = useEditorStore((s) => s.fps);
  const quality = useEditorStore((s) => s.quality);
  const musicBed = useEditorStore((s) => s.music);
  const beatSync = useEditorStore((s) => s.beatSync);
  const setOutput = useEditorStore((s) => s.setOutput);
  const setClips = useEditorStore((s) => s.setClips);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);
  const setView = useEditorStore((s) => s.setView);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [sourceId, setSourceId] = useState("");
  const [name, setName] = useState("");
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  const templates = useQuery({ queryKey: ["templates"], queryFn: api.templates });
  const library = templates.data?.library ?? [];
  const mine = templates.data?.mine ?? [];
  const selected = [...library, ...mine].find((item) => item.id === selectedId) ?? null;

  const visual = media.filter(
    (asset) => (asset.kind === "video" || asset.kind === "image") && asset.status === "ready",
  );
  const photos = visual.filter((asset) => asset.kind === "image").length;
  const unanalysed = visual.filter((asset) => !analyzedIds.has(asset.id)).length;
  const videos = visual.filter((asset) => asset.kind === "video");

  const reportError = (caught: unknown) =>
    setError(
      caught instanceof ApiError
        ? { message: caught.message, hint: caught.hint }
        : { message: t("template.error"), hint: null },
    );

  const invalidate = () => void queryClient.invalidateQueries({ queryKey: ["templates"] });

  /** Build the edit from the chosen template and take it to the editor. */
  const use = useMutation({
    mutationFn: (template: EditTemplate) =>
      api.createEditPlan(projectId, {
        mode: "rules",
        template_id: template.id,
        aspect_ratio: template.aspect,
        resolution,
        encoder,
        fps,
        quality,
        min_clips: 1,
        music: toMusicRequest(musicBed),
        beat_sync: beatSync,
      }),
    onSuccess: (plan, template) => {
      setError(null);
      setOutput({ aspect: template.aspect });
      const draft = toDraft(plan);
      setClips(draft.clips, plan.id, plan.id, draft.music, draft.subtitles);
      setPreviewSource("program");
      void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
      setView("editor");
      onPlanned(plan);
    },
    onError: reportError,
  });

  const measure = useMutation({
    mutationFn: () => api.measureTemplate(projectId, sourceId, name.trim()),
    onSuccess: (made) => {
      setError(null);
      setName("");
      setSelectedId(made.id);
      invalidate();
    },
    onError: reportError,
  });

  const rename = useMutation({
    mutationFn: (next: { id: string; name: string }) => api.renameTemplate(next.id, next.name),
    onSuccess: () => {
      setRenaming(null);
      invalidate();
    },
    onError: reportError,
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteTemplate(id),
    onSuccess: () => {
      setSelectedId(null);
      invalidate();
    },
    onError: reportError,
  });

  function Card({ item }: { item: EditTemplate }) {
    const current = item.id === selectedId;
    const editing = renaming?.id === item.id;
    return (
      <div
        role="button"
        tabIndex={0}
        aria-pressed={current}
        onClick={() => setSelectedId(item.id)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") setSelectedId(item.id);
        }}
        className={`flex cursor-pointer flex-col gap-2 rounded-lg p-3 text-left transition-[background-color,box-shadow] duration-fast ${
          current
            ? "bg-accent-soft shadow-raised ring-1 ring-accent"
            : "bg-elevated hover:bg-hover"
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          {editing ? (
            <TextInput
              autoFocus
              value={renaming.name}
              maxLength={60}
              aria-label={t("template.name")}
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => setRenaming({ id: item.id, name: event.target.value })}
              onKeyDown={(event) => {
                event.stopPropagation();
                if (event.key === "Enter" && renaming.name.trim()) {
                  rename.mutate({ id: item.id, name: renaming.name.trim() });
                }
                if (event.key === "Escape") setRenaming(null);
              }}
            />
          ) : (
            <span className="truncate text-sm font-medium text-fg">{templateLabel(item, t)}</span>
          )}
          <Badge>{item.aspect}</Badge>
        </div>
        <SlotStrip template={item} />
        <span className="text-2xs text-muted">
          {t("template.summary", {
            slots: item.slot_count,
            length: shortDuration(item.total_ms),
            stills: item.still_slots,
          })}
        </span>
        {item.source === "user" && current && !editing && (
          <div className="flex gap-1.5">
            <Button
              size="sm"
              tone="quiet"
              onClick={(event) => {
                event.stopPropagation();
                setRenaming({ id: item.id, name: item.name });
              }}
            >
              {t("template.rename")}
            </Button>
            <Button
              size="sm"
              tone="quiet"
              disabled={remove.isPending}
              onClick={(event) => {
                event.stopPropagation();
                if (window.confirm(t("template.deleteConfirm", { name: item.name }))) {
                  remove.mutate(item.id);
                }
              }}
            >
              {t("template.delete")}
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="vf-view flex min-h-0 flex-1 gap-2 px-2 pb-1">
      {/* ---- the files the template will be filled from ---- */}
      <div className="flex w-[340px] shrink-0 flex-col 2xl:w-[380px]">
        <MediaBrowser
          projectId={projectId}
          media={media}
          loading={loading}
          analyzedIds={analyzedIds}
          onUploaded={onUploaded}
        />
      </div>

      {/* ---- the gallery ---- */}
      <Panel className="min-w-0 flex-1 overflow-hidden rounded-xl shadow-panel">
        <PanelHeader title={t("template.library")} icon="grid" />
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-panel pb-panel">
          <p className="mb-3 max-w-[70ch] text-xs leading-relaxed text-muted">
            {t("template.libraryIntro")}
          </p>

          {templates.isLoading && <Spinner />}

          <h3 className="mb-2 text-2xs font-semibold uppercase tracking-[0.08em] text-faint">
            {t("template.builtIn")}
          </h3>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-2">
            {library.map((item) => (
              <Card key={item.id} item={item} />
            ))}
          </div>

          <h3 className="mb-2 mt-5 text-2xs font-semibold uppercase tracking-[0.08em] text-faint">
            {t("template.mine")}
          </h3>
          {mine.length === 0 ? (
            <p className="text-xs text-faint">{t("template.mineEmpty")}</p>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-2">
              {mine.map((item) => (
                <Card key={item.id} item={item} />
              ))}
            </div>
          )}

          {/* ---- make one ---- */}
          <div className="mt-6 flex max-w-[520px] flex-col gap-2 rounded-lg bg-sunken/60 p-3">
            <p className="text-xs font-medium text-fg">{t("template.make")}</p>
            <p className="text-2xs leading-snug text-faint">{t("template.makeHint")}</p>
            <div className="grid grid-cols-2 gap-2">
              <Field label={t("template.fromVideo")}>
                <Select
                  value={sourceId}
                  aria-label={t("template.fromVideo")}
                  onChange={(event) => setSourceId(event.target.value)}
                >
                  <option value="">{t("template.pickVideo")}</option>
                  {videos.map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.original_filename}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label={t("template.name")}>
                <TextInput
                  value={name}
                  maxLength={60}
                  placeholder={t("template.namePlaceholder")}
                  onChange={(event) => setName(event.target.value)}
                />
              </Field>
            </div>
            <Button
              size="sm"
              className="self-start"
              disabled={!sourceId || !name.trim() || measure.isPending}
              onClick={() => measure.mutate()}
            >
              {measure.isPending ? t("template.measuring") : t("template.measure")}
            </Button>
          </div>
        </div>
      </Panel>

      {/* ---- use it ---- */}
      <Panel className="w-[300px] shrink-0 overflow-hidden rounded-xl shadow-panel 2xl:w-[340px]">
        <PanelHeader title={t("template.use")} icon="spark" />
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-panel pb-panel">
          {selected ? (
            <>
              <div className="flex flex-col gap-1.5">
                <span className="text-sm font-medium text-fg">{templateLabel(selected, t)}</span>
                <SlotStrip template={selected} />
                <p className="text-2xs leading-snug text-faint">{t("template.fills")}</p>
                {selected.extraction && (
                  <p className="text-2xs leading-snug text-faint">
                    {t("template.measured", {
                      shots: selected.extraction.shots,
                      merged: selected.extraction.merged,
                      dropped: selected.extraction.dropped,
                    })}
                  </p>
                )}
              </div>

              <div className="rounded-md bg-sunken/60 p-2.5 text-2xs leading-snug text-muted">
                {t("template.footage", {
                  videos: visual.length - photos,
                  photos,
                })}
                {selected.still_slots > 0 && photos === 0 && (
                  <span className="mt-1 block text-warning">{t("template.noPhotos")}</span>
                )}
              </div>

              {unanalysed > 0 && (
                <div className="flex flex-col gap-1.5 rounded-md bg-warning/10 p-2.5">
                  <span className="text-2xs leading-snug text-warning">
                    {t("template.unanalysed", { count: unanalysed })}
                  </span>
                  <Button size="sm" className="self-start" onClick={onAnalyzeAll}>
                    {t("template.analyseNow")}
                  </Button>
                </div>
              )}

              <p className="text-2xs leading-snug text-faint">
                {t("template.output", {
                  resolution: `${resolution} · ${t(`encoder.${encoder}`)}`,
                  fps,
                  quality: t(`export.quality.${quality}`),
                })}
              </p>

              <Button
                tone="primary"
                disabled={use.isPending || visual.length === 0}
                onClick={() => use.mutate(selected)}
              >
                {use.isPending ? t("template.building") : t("template.useThis")}
              </Button>
            </>
          ) : (
            <p className="text-xs leading-relaxed text-faint">{t("template.pickOne")}</p>
          )}

          {error && <ErrorNote hint={error.hint}>{error.message}</ErrorNote>}
        </div>
      </Panel>
    </div>
  );
}
