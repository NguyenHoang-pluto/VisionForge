"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, ApiError, type EditTemplate, type MediaAsset } from "@/lib/api";
import { shortDuration } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { Button, ErrorNote, Field, SectionTitle, Select, TextInput } from "@/components/ui";

/**
 * Templates, in the AI Edit tab (Phase 12).
 *
 * A section rather than a tab, for the reason the reference is one: "cut it
 * like this" is a way of shaping an edit, beside style and length, and the
 * chosen template travels with the ordinary Generate button.
 *
 * Two things this section must make visible:
 *
 * **The structure being copied.** A template is a row of slots, so it is drawn
 * as one: width is length, height is energy, and a slot that wants a photo is
 * marked. A user choosing between two templates is choosing between rhythms,
 * and a name alone does not show a rhythm.
 *
 * **What a measured template could not measure.** Joins are read as cuts, and
 * a video with more shots than an edit may hold is truncated. The server says
 * so; this says it again where the user decides.
 */

const LIBRARY_LABEL: Record<string, MessageKey> = {
  travel: "template.travel",
  memories: "template.memories",
  beat_highlight: "template.beat_highlight",
  slideshow: "template.slideshow",
  day_vlog: "template.day_vlog",
  vertical_reel: "template.vertical_reel",
};

export function templateLabel(
  template: Pick<EditTemplate, "id" | "name" | "source">,
  t: ReturnType<typeof useT>,
): string {
  const key = template.source === "builtin" ? LIBRARY_LABEL[template.id] : undefined;
  return key ? t(key) : template.name;
}

/** The template's slots as a strip: width is length, height is energy. */
export function SlotStrip({ template }: { template: EditTemplate }) {
  const t = useT();
  const total = template.slots.reduce((sum, slot) => sum + slot.duration_ms, 0) || 1;
  return (
    <div
      className="flex h-9 items-end gap-px overflow-hidden rounded bg-sunken px-px pt-1"
      aria-label={t("template.strip")}
      role="img"
    >
      {template.slots.map((slot, index) => (
        <span
          key={index}
          title={t("template.slotTitle", {
            index: index + 1,
            length: shortDuration(slot.duration_ms),
            role: t(`editorial.role.${slot.role}` as MessageKey),
          })}
          className={`block min-w-[3px] rounded-t-sm ${
            slot.prefer === "still"
              ? "bg-accent-strong"
              : slot.prefer === "video"
                ? "bg-fg/70"
                : "bg-fg/35"
          }`}
          style={{
            width: `${(slot.duration_ms / total) * 100}%`,
            height: `${Math.round(25 + slot.energy * 75)}%`,
          }}
        />
      ))}
    </div>
  );
}

function Chip({
  selected,
  label,
  title,
  onClick,
}: {
  selected: boolean;
  label: string;
  title?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      title={title}
      onClick={onClick}
      className={`h-control-sm max-w-full truncate rounded-full px-3 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast ${
        selected
          ? "bg-accent-strong text-accent-fg shadow-raised"
          : "bg-hover text-muted hover:text-fg"
      }`}
    >
      {label}
    </button>
  );
}

export function TemplatesPanel({
  projectId,
  mediaList,
  value,
  onChange,
}: {
  projectId: string;
  mediaList: MediaAsset[];
  value: string | null;
  onChange: (template: EditTemplate | null) => void;
}) {
  const t = useT();
  const queryClient = useQueryClient();
  const [sourceId, setSourceId] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  const templates = useQuery({ queryKey: ["templates"], queryFn: api.templates });
  const all = [...(templates.data?.library ?? []), ...(templates.data?.mine ?? [])];
  const selected = all.find((item) => item.id === value) ?? null;

  const videos = mediaList.filter((asset) => asset.kind === "video" && asset.status === "ready");

  const reportError = (caught: unknown) =>
    setError(
      caught instanceof ApiError
        ? { message: caught.message, hint: caught.hint }
        : { message: t("template.error"), hint: null },
    );

  const measure = useMutation({
    mutationFn: () => api.measureTemplate(projectId, sourceId, name.trim()),
    onSuccess: (made) => {
      setError(null);
      setName("");
      void queryClient.invalidateQueries({ queryKey: ["templates"] });
      onChange(made);
    },
    onError: reportError,
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteTemplate(id),
    onSuccess: () => {
      onChange(null);
      void queryClient.invalidateQueries({ queryKey: ["templates"] });
    },
    onError: reportError,
  });

  const describe = (item: EditTemplate) =>
    t("template.summary", {
      slots: item.slot_count,
      length: shortDuration(item.total_ms),
      stills: item.still_slots,
    });

  return (
    <section>
      <SectionTitle description={t("template.description")}>{t("template.title")}</SectionTitle>

      <div role="radiogroup" aria-label={t("template.title")} className="flex flex-wrap gap-1.5">
        <Chip selected={value === null} label={t("template.none")} onClick={() => onChange(null)} />
        {(templates.data?.library ?? []).map((item) => (
          <Chip
            key={item.id}
            selected={value === item.id}
            label={templateLabel(item, t)}
            title={describe(item)}
            onClick={() => onChange(item)}
          />
        ))}
      </div>

      {(templates.data?.mine.length ?? 0) > 0 && (
        <>
          <p className="mb-1.5 mt-3 text-2xs font-medium text-faint">{t("template.mine")}</p>
          <div
            role="radiogroup"
            aria-label={t("template.mine")}
            className="flex flex-wrap gap-1.5"
          >
            {templates.data?.mine.map((item) => (
              <Chip
                key={item.id}
                selected={value === item.id}
                label={item.name}
                title={describe(item)}
                onClick={() => onChange(item)}
              />
            ))}
          </div>
        </>
      )}

      {selected && (
        <div className="mt-3 flex flex-col gap-1.5">
          <SlotStrip template={selected} />
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-2xs text-muted">{describe(selected)}</span>
            {selected.source === "user" && (
              <Button
                size="sm"
                tone="ghost"
                disabled={remove.isPending}
                onClick={() => remove.mutate(selected.id)}
              >
                {t("template.delete")}
              </Button>
            )}
          </div>
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
      )}

      {/* ------------------------------------------------ make one ---------- */}
      <div className="mt-4 flex flex-col gap-2 rounded-md bg-sunken/60 p-2.5">
        <p className="text-2xs font-medium text-muted">{t("template.make")}</p>
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
        <Button
          size="sm"
          disabled={!sourceId || !name.trim() || measure.isPending}
          onClick={() => measure.mutate()}
        >
          {measure.isPending ? t("template.measuring") : t("template.measure")}
        </Button>
        <p className="text-2xs leading-snug text-faint">{t("template.makeHint")}</p>
      </div>

      {error && <ErrorNote hint={error.hint}>{error.message}</ErrorNote>}
    </section>
  );
}
