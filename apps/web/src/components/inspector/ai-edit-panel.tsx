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
import { toDraft } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Button,
  ErrorNote,
  Field,
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

const MODES: { value: PlannerMode; label: string; note: string }[] = [
  {
    value: "automatic",
    label: "Automatic",
    note: "Picks the planner from what you ask for.",
  },
  {
    value: "rules",
    label: "Rules",
    note: "Deterministic scoring. Same footage, same edit, every time.",
  },
  {
    value: "ai",
    label: "AI",
    note: "A model chooses the clips. Falls back to the rules engine if it fails.",
  },
];

const ORDERS: { value: ClipOrder; label: string }[] = [
  { value: "score_desc", label: "Strongest first" },
  { value: "sequence", label: "Upload order" },
];

const QUALITIES: { value: QualityPreset; label: string }[] = [
  { value: "draft", label: "Draft" },
  { value: "balanced", label: "Balanced" },
  { value: "high", label: "High" },
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
  const queryClient = useQueryClient();

  const aspect = useEditorStore((s) => s.aspect);
  const fps = useEditorStore((s) => s.fps);
  const quality = useEditorStore((s) => s.quality);
  const setOutput = useEditorStore((s) => s.setOutput);
  const setClips = useEditorStore((s) => s.setClips);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);

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
      }),
    onSuccess: (plan) => {
      setError(null);
      // The plan becomes the timeline. From here it is an ordinary draft: it
      // can be trimmed, reordered and cut like one the user assembled, because
      // that is exactly what it now is.
      const draft = toDraft(plan);
      setClips(draft.clips, plan.id);
      setPreviewSource("program");
      onPlanned(plan);
      void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["llm-runs", projectId] });
    },
    onError: (caught: unknown) =>
      setError(
        caught instanceof ApiError
          ? { message: caught.message, hint: caught.hint }
          : { message: "Could not generate an edit.", hint: null },
      ),
  });

  const selectedStyle = styles.find((item) => item.value === style);

  return (
    <div className="flex flex-col gap-3 p-2.5">
      <SectionTitle>Automatic edit</SectionTitle>

      <Field label="Mode" hint={MODES.find((m) => m.value === mode)?.note}>
        <SegmentedControl
          label="Planner mode"
          value={mode}
          onChange={setMode}
          className="w-full [&>button]:flex-1"
          options={MODES.map((item) => ({
            value: item.value,
            label: item.label,
            title:
              item.value === "ai" && !aiAvailable
                ? "No AI provider is configured on this server."
                : item.note,
            disabled: item.value === "ai" && !aiAvailable,
          }))}
        />
      </Field>

      <Field label="Style" hint={selectedStyle?.description}>
        <Select
          value={style}
          aria-label="Edit style"
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
          <option value="">None</option>
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

      <Field label="Request" hint="What you want, in your own words.">
        <TextInput
          value={requestText}
          maxLength={capabilities.data?.max_request_chars ?? 500}
          placeholder="Fast 25 second football highlight, best moments"
          aria-label="Describe the edit"
          onChange={(event) => setRequestText(event.target.value)}
        />
      </Field>

      <div className="grid grid-cols-2 gap-2">
        <Field label="Duration (s)">
          <NumberInput
            min={2}
            max={120}
            value={targetSeconds}
            aria-label="Target duration in seconds"
            onChange={(event) => setTargetSeconds(Number(event.target.value))}
          />
        </Field>
        <Field label="Max clips">
          <NumberInput
            min={1}
            max={20}
            value={maxClips}
            aria-label="Maximum clips"
            onChange={(event) => setMaxClips(Number(event.target.value))}
          />
        </Field>
        <Field label="Aspect">
          <Select
            value={aspect}
            aria-label="Aspect ratio"
            onChange={(event) => setOutput({ aspect: event.target.value as AspectRatio })}
          >
            {(capabilities.data?.aspect_ratios ?? []).map((item) => (
              <option key={item.value} value={item.value}>
                {item.value} · {item.width}×{item.height}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Frame rate">
          <Select
            value={fps}
            aria-label="Frame rate"
            onChange={(event) => setOutput({ fps: Number(event.target.value) })}
          >
            {fpsPresets.map((value) => (
              <option key={value} value={value}>
                {value} fps
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Quality">
          <Select
            value={quality}
            aria-label="Quality preset"
            onChange={(event) => setOutput({ quality: event.target.value as QualityPreset })}
          >
            {QUALITIES.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Order">
          <Select
            value={order}
            aria-label="Clip order"
            onChange={(event) => setOrder(event.target.value as ClipOrder)}
          >
            {ORDERS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
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
        {generate.isPending ? "Generating…" : "Generate edit"}
      </Button>

      {error && <ErrorNote hint={error.hint}>{error.message}</ErrorNote>}

      {readyCount === 0 && (
        <p className="text-2xs leading-snug text-dim">
          No analysed media in this project yet. Import clips and run Analyse first.
        </p>
      )}

      {/* What this server actually has. No claim beyond it. */}
      <p className="border-t border-line pt-2 font-mono text-2xs leading-snug text-dim">
        {aiAvailable
          ? capabilities.data?.is_stub
            ? `planner: ${capabilities.data.provider} — deterministic stub, not a model`
            : `planner: ${capabilities.data?.provider} · ${capabilities.data?.model}`
          : "planner: rules engine only — no AI provider configured"}
      </p>
      {mode === "ai" && !aiAvailable && (
        <p className="text-2xs leading-snug text-warn">
          AI is unavailable on this server, so the rules engine will plan instead.
        </p>
      )}
    </div>
  );
}
