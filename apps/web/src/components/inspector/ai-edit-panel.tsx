"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  api,
  ApiError,
  type AspectRatio,
  type ClipOrder,
  type EditPlan,
  type EditStyle,
  type PlannerMode,
  type QualityPreset,
} from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { toDraft, toMusicRequest } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Button,
  ErrorNote,
  Field,
  Glyph,
  NumberInput,
  SectionTitle,
  SegmentedControl,
  Select,
  TextInput,
} from "@/components/ui";

/**
 * The AI edit panel.
 *
 * A tool panel, not a pitch. The model is one of three planners the user can
 * choose between, sitting in an inspector tab beside clip properties and
 * export -- which is the accurate description of what it is. What the panel
 * says about the server is whatever `/planner/capabilities` reports, including
 * "there is no model configured here", because a control that claims a
 * capability the deployment lacks is worse than no control.
 */

const MODES: { value: PlannerMode; label: MessageKey; note: MessageKey }[] = [
  { value: "automatic", label: "ai.mode.automatic", note: "ai.mode.automaticNote" },
  { value: "rules", label: "ai.mode.rules", note: "ai.mode.rulesNote" },
  { value: "ai", label: "ai.mode.ai", note: "ai.mode.aiNote" },
];

const ORDERS: { value: ClipOrder; label: MessageKey }[] = [
  { value: "score_desc", label: "ai.order.score" },
  { value: "sequence", label: "ai.order.sequence" },
];

const QUALITIES: { value: QualityPreset; label: MessageKey }[] = [
  { value: "draft", label: "export.quality.draft" },
  { value: "balanced", label: "export.quality.balanced" },
  { value: "high", label: "export.quality.high" },
];

export function AiEditPanel({
  projectId,
  readyCount,
  onPlanned,
}: {
  projectId: string;
  readyCount: number;
  onPlanned: (plan: EditPlan) => void;
}) {
  const t = useT();
  const queryClient = useQueryClient();

  const aspect = useEditorStore((s) => s.aspect);
  const fps = useEditorStore((s) => s.fps);
  const quality = useEditorStore((s) => s.quality);
  const setOutput = useEditorStore((s) => s.setOutput);
  const setClips = useEditorStore((s) => s.setClips);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);
  const musicBed = useEditorStore((s) => s.music);
  const beatSync = useEditorStore((s) => s.beatSync);

  const [mode, setMode] = useState<PlannerMode>("automatic");
  const [style, setStyle] = useState<EditStyle | "">("");
  const [requestText, setRequestText] = useState("");
  const [targetSeconds, setTargetSeconds] = useState(25);
  const [maxClips, setMaxClips] = useState(6);
  const [order, setOrder] = useState<ClipOrder>("score_desc");
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  // Server-declared, so every control reflects what this deployment can do.
  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });

  const aiAvailable = capabilities.data?.ai_available ?? false;
  const styles = capabilities.data?.styles ?? [];
  const fpsPresets = capabilities.data?.fps_presets ?? [24, 30, 60];
  const activeMode: PlannerMode = mode === "ai" && !aiAvailable ? "rules" : mode;

  const generate = useMutation({
    mutationFn: () =>
      api.createEditPlan(projectId, {
        mode: activeMode,
        style: style || null,
        request_text: requestText.trim() || null,
        target_duration_ms: targetSeconds * 1000,
        max_clips: maxClips,
        min_clips: 1,
        aspect_ratio: aspect,
        fps,
        quality,
        order,
        // The planner needs the bed to align the cue to the first beat, and the
        // flag to decide whether the tempo may move a cut at all.
        music: toMusicRequest(musicBed),
        beat_sync: beatSync,
      }),
    onSuccess: (plan) => {
      setError(null);
      // The plan becomes the timeline. From here it is an ordinary draft: it
      // can be trimmed, reordered and cut like one the user assembled, because
      // that is exactly what it now is.
      const draft = toDraft(plan);
      // The plan may have chosen its own cue -- trimmed to the cut, started on
      // the first beat. Accepting the plan takes that too, or the edit reviewed
      // is not the edit produced.
      setClips(draft.clips, plan.id, plan.id, draft.music);
      setPreviewSource("program");
      onPlanned(plan);
      void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["llm-runs", projectId] });
    },
    onError: (caught: unknown) =>
      setError(
        caught instanceof ApiError
          ? { message: caught.message, hint: caught.hint }
          : { message: t("ai.error"), hint: null },
      ),
  });

  const selectedStyle = styles.find((item) => item.value === style);
  const activeNote = MODES.find((item) => item.value === mode)?.note;

  return (
    <div className="flex flex-col gap-3 p-panel">
      <SectionTitle>{t("ai.title")}</SectionTitle>

      <Field label={t("ai.mode")} hint={activeNote ? t(activeNote) : undefined}>
        <SegmentedControl
          label={t("ai.mode.label")}
          value={mode}
          onChange={setMode}
          className="w-full [&>button]:flex-1"
          options={MODES.map((item) => ({
            value: item.value,
            label: t(item.label),
            title:
              item.value === "ai" && !aiAvailable ? t("ai.mode.aiUnavailable") : t(item.note),
            disabled: item.value === "ai" && !aiAvailable,
          }))}
        />
      </Field>
      {activeNote && <p className="-mt-1.5 text-2xs leading-snug text-dim">{t(activeNote)}</p>}

      <Field label={t("ai.style")} hint={selectedStyle?.description}>
        <Select
          value={style}
          aria-label={t("ai.style.label")}
          onChange={(event) => {
            const next = event.target.value as EditStyle | "";
            setStyle(next);
            // A style carries its own sensible length and shape. Applying them
            // as the control's visible value keeps the panel honest about what
            // the server will actually do.
            const profile = styles.find((item) => item.value === next);
            if (profile) {
              setTargetSeconds(Math.round(profile.default_duration_ms / 1000));
              setOutput({ aspect: profile.default_aspect });
            }
          }}
        >
          <option value="">{t("ai.style.none")}</option>
          {styles.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </Select>
      </Field>
      {selectedStyle && (
        <p className="-mt-1.5 text-2xs leading-snug text-dim">{selectedStyle.description}</p>
      )}

      <Field label={t("ai.request")} hint={t("ai.request.hint")}>
        <TextInput
          value={requestText}
          maxLength={capabilities.data?.max_request_chars ?? 500}
          placeholder={t("ai.request.placeholder")}
          aria-label={t("ai.request.label")}
          onChange={(event) => setRequestText(event.target.value)}
        />
      </Field>

      <div className="grid grid-cols-2 gap-2">
        <Field label={t("ai.duration")}>
          <NumberInput
            min={2}
            max={120}
            value={targetSeconds}
            aria-label={t("ai.duration.label")}
            onChange={(event) => setTargetSeconds(Number(event.target.value))}
          />
        </Field>
        <Field label={t("ai.maxClips")}>
          <NumberInput
            min={1}
            max={20}
            value={maxClips}
            aria-label={t("ai.maxClips.label")}
            onChange={(event) => setMaxClips(Number(event.target.value))}
          />
        </Field>
        <Field label={t("ai.aspect")}>
          <Select
            value={aspect}
            aria-label={t("ai.aspect.label")}
            onChange={(event) => setOutput({ aspect: event.target.value as AspectRatio })}
          >
            {(capabilities.data?.aspect_ratios ?? []).map((item) => (
              <option key={item.value} value={item.value}>
                {item.value} · {item.width}×{item.height}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("ai.fps")}>
          <Select
            value={fps}
            aria-label={t("ai.fps.label")}
            onChange={(event) => setOutput({ fps: Number(event.target.value) })}
          >
            {fpsPresets.map((value) => (
              <option key={value} value={value}>
                {value} fps
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("ai.quality")}>
          <Select
            value={quality}
            aria-label={t("ai.quality.label")}
            onChange={(event) => setOutput({ quality: event.target.value as QualityPreset })}
          >
            {QUALITIES.map((item) => (
              <option key={item.value} value={item.value}>
                {t(item.label)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("ai.order")}>
          <Select
            value={order}
            aria-label={t("ai.order.label")}
            onChange={(event) => setOrder(event.target.value as ClipOrder)}
          >
            {ORDERS.map((item) => (
              <option key={item.value} value={item.value}>
                {t(item.label)}
              </option>
            ))}
          </Select>
        </Field>
      </div>

      <Button
        tone="primary"
        disabled={readyCount === 0 || generate.isPending}
        onClick={() => generate.mutate()}
      >
        <Glyph name="spark" size={11} />
        {generate.isPending ? t("ai.generating") : t("ai.generate")}
      </Button>

      {error && <ErrorNote hint={error.hint}>{error.message}</ErrorNote>}

      {readyCount === 0 && <p className="text-2xs leading-snug text-dim">{t("ai.noMedia")}</p>}

      {/* What this server actually has. No claim beyond it. */}
      <p className="border-t border-line pt-2 font-mono text-2xs leading-snug text-dim">
        {aiAvailable
          ? capabilities.data?.is_stub
            ? t("ai.planner.stub", { provider: capabilities.data.provider ?? "—" })
            : t("ai.planner.model", {
                provider: capabilities.data?.provider ?? "—",
                model: capabilities.data?.model ?? "—",
              })
          : t("ai.planner.rules")}
      </p>
      {mode === "ai" && !aiAvailable && (
        <p className="text-2xs leading-snug text-warn">{t("ai.planner.willFallBack")}</p>
      )}
    </div>
  );
}
