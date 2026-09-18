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
  type EditorialVariantPreview,
  type MediaAsset,
  type PlannerMode,
  type PolicyId,
  type QualityPreset,
  type VariantId,
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
  Spinner,
  TextArea,
} from "@/components/ui";
import { CoEditorPanel } from "@/components/inspector/co-editor-panel";
import { EditorialPlanPanel } from "@/components/inspector/editorial-plan";
import { ReferencePanel } from "@/components/inspector/reference-panel";
import { TemplatesPanel } from "@/components/inspector/templates-panel";
import { OutputSizeFields } from "@/components/inspector/output-fields";

/**
 * The AI edit panel.
 *
 * A tool panel, not a pitch. The model is one of three planners the user can
 * choose between, sitting in an inspector tab beside clip properties and
 * export -- which is the accurate description of what it is. What the panel
 * says about the server is whatever `/planner/capabilities` reports, including
 * "there is no model configured here", because a control that claims a
 * capability the deployment lacks is worse than no control.
 *
 * Two modes, one tab (Phase 10). **Create** plans an edit from the library;
 * **Refine** changes the edit that already exists. They are the same activity
 * at two moments rather than two features, and a seventh tab on a strip that is
 * already six wide would have said otherwise -- an inspector whose tab strip is
 * wider than its panel has stopped being an inspector.
 */

type AiMode = "create" | "refine";

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
  { value: "max", label: "export.quality.max" },
];

export function AiEditPanel({
  projectId,
  readyCount,
  media,
  mediaList,
  onPlanned,
  onAnalyze,
}: {
  projectId: string;
  readyCount: number;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
  onPlanned: (plan: EditPlan) => void;
  onAnalyze: (mediaId: string) => void;
}) {
  const t = useT();
  const queryClient = useQueryClient();

  const aspect = useEditorStore((s) => s.aspect);
  const fps = useEditorStore((s) => s.fps);
  const resolution = useEditorStore((s) => s.resolution);
  const encoder = useEditorStore((s) => s.encoder);
  const quality = useEditorStore((s) => s.quality);
  const setOutput = useEditorStore((s) => s.setOutput);
  const setClips = useEditorStore((s) => s.setClips);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);
  const musicBed = useEditorStore((s) => s.music);
  const beatSync = useEditorStore((s) => s.beatSync);
  const styleStrength = useEditorStore((s) => s.styleStrength);
  const setInspectorTab = useEditorStore((s) => s.setInspectorTab);

  const [aiMode, setAiMode] = useState<AiMode>("create");
  const [mode, setMode] = useState<PlannerMode>("automatic");
  const [style, setStyle] = useState<EditStyle | "">("");
  const [requestText, setRequestText] = useState("");
  const [targetSeconds, setTargetSeconds] = useState(25);
  const [maxClips, setMaxClips] = useState(6);
  const [order, setOrder] = useState<ClipOrder>("score_desc");
  const [policy, setPolicy] = useState<PolicyId | "">("");
  const [templateId, setTemplateId] = useState<string | null>(null);
  const [variant, setVariant] = useState<VariantId | null>(null);
  const [variants, setVariants] = useState<EditorialVariantPreview[] | null>(null);
  const [plan, setPlan] = useState<EditPlan | null>(null);
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
  const heavyOutput = resolution === "2160p" || fps > 60;
  const activeMode: PlannerMode = mode === "ai" && !aiAvailable ? "rules" : mode;

  /**
   * Everything the server needs to decide an edit, assembled once.
   *
   * Both the plan request and the variant preview send exactly this, which is
   * what makes the preview honest: a variant previewed with a different target
   * duration from the one that will be planned is a preview of a different
   * edit.
   */
  const planOptions = (chosen: VariantId | null) => ({
    mode: activeMode,
    style: style || null,
    request_text: requestText.trim() || null,
    target_duration_ms: targetSeconds * 1000,
    max_clips: maxClips,
    min_clips: 1,
    aspect_ratio: aspect,
    resolution,
    encoder,
    fps,
    quality,
    order,
    // The planner needs the bed to align the cue to the first beat, and the
    // flag to decide whether the tempo may move a cut at all.
    music: toMusicRequest(musicBed),
    beat_sync: beatSync,
    // How much of the reference to apply. Which clip the reference is stays
    // server-side; this is only the dial.
    style_strength: styleStrength,
    editorial_policy: policy || null,
    variant: chosen,
    template_id: templateId,
  });

  const reportError = (caught: unknown) =>
    setError(
      caught instanceof ApiError
        ? { message: caught.message, hint: caught.hint }
        : { message: t("ai.error"), hint: null },
    );

  const generate = useMutation({
    mutationFn: (chosen: VariantId | null) => api.createEditPlan(projectId, planOptions(chosen)),
    onSuccess: (generated) => {
      setError(null);
      setPlan(generated);
      // The plan becomes the timeline. From here it is an ordinary draft: it
      // can be trimmed, reordered and cut like one the user assembled, because
      // that is exactly what it now is.
      const draft = toDraft(generated);
      // The plan may have chosen its own cue -- trimmed to the cut, started on
      // the first beat. Accepting the plan takes that too, or the edit reviewed
      // is not the edit produced.
      setClips(draft.clips, generated.id, generated.id, draft.music, draft.subtitles);
      setPreviewSource("program");
      void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["llm-runs", projectId] });
    },
    onError: reportError,
  });

  /**
   * The alternatives, previewed rather than stored.
   *
   * Four editorial plans come back and nothing is written. Choosing one asks
   * for it through the ordinary plan route, which -- because planning is
   * deterministic -- produces exactly the edit that was previewed.
   */
  const compare = useMutation({
    mutationFn: () => api.editVariants(projectId, planOptions(null)),
    onSuccess: (response) => {
      setError(null);
      setVariants(response.items);
    },
    onError: reportError,
  });

  const selectedStyle = styles.find((item) => item.value === style);
  const activeNote = MODES.find((item) => item.value === mode)?.note;

  if (aiMode === "refine") {
    return (
      <div className="flex flex-col">
        <div className="px-panel pt-panel">
          <SegmentedControl
            label={t("coedit.mode.label")}
            value={aiMode}
            onChange={setAiMode}
            className="w-full [&>button]:flex-1"
            options={[
              { value: "create" as const, label: t("coedit.mode.create") },
              { value: "refine" as const, label: t("coedit.mode.refine") },
            ]}
          />
        </div>
        <CoEditorPanel projectId={projectId} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      <SegmentedControl
        label={t("coedit.mode.label")}
        value={aiMode}
        onChange={setAiMode}
        className="w-full [&>button]:flex-1"
        options={[
          { value: "create" as const, label: t("coedit.mode.create") },
          { value: "refine" as const, label: t("coedit.mode.refine") },
        ]}
      />

      {/* ------------------------------------------------------ the brief ----
          The prompt first, and given room. Everything under it narrows what the
          planner may do with the brief; putting the six numeric controls above
          it made the panel read as a settings form that happened to accept a
          sentence. */}
      <section>
        <div className="mb-2 flex items-center gap-2">
          <span
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent-strong"
            aria-hidden
          >
            <Glyph name="wand" size={14} />
          </span>
          <h3 className="text-xs font-semibold tracking-tight text-fg">{t("ai.title")}</h3>
        </div>

        <TextArea
          rows={4}
          value={requestText}
          maxLength={capabilities.data?.max_request_chars ?? 500}
          placeholder={t("ai.request.placeholder")}
          aria-label={t("ai.request.label")}
          onChange={(event) => setRequestText(event.target.value)}
        />
        <p className="mt-1.5 text-2xs leading-snug text-faint">{t("ai.request.hint")}</p>

        <Button
          tone="primary"
          size="lg"
          className="mt-3 w-full"
          disabled={readyCount === 0 || generate.isPending}
          onClick={() => {
            setVariant(null);
            setVariants(null);
            generate.mutate(null);
          }}
        >
          {generate.isPending ? <Spinner size={13} /> : <Glyph name="spark" size={13} />}
          {generate.isPending ? t("ai.generating") : t("ai.generate")}
        </Button>

        {error && (
          <div className="mt-2">
            <ErrorNote hint={error.hint}>{error.message}</ErrorNote>
          </div>
        )}
        {readyCount === 0 && (
          <p className="mt-2 text-2xs leading-snug text-warning">{t("ai.noMedia")}</p>
        )}
      </section>

      {/* ------------------------------------------------ editorial plan ----
          Directly under the brief, because it is the answer to the question the
          button above just asked. Everything below it narrows what the next
          generation may do; this is what the last one decided. */}
      {plan?.editorial ? (
        <EditorialPlanPanel
          record={plan.editorial}
          media={media}
          variants={variants}
          activeVariant={variant}
          loadingVariants={compare.isPending}
          onChooseVariant={(chosen) => {
            // The first click on "Variants" has nothing to choose between yet,
            // so it fetches; afterwards a click is a choice and re-plans.
            if (!variants) {
              compare.mutate();
              return;
            }
            setVariant(chosen);
            generate.mutate(chosen);
          }}
          onRegenerate={() => generate.mutate(variant)}
          onAccept={() => {
            onPlanned(plan);
            setInspectorTab("export");
          }}
          onRender={() => setInspectorTab("export")}
          busy={generate.isPending}
        />
      ) : (
        <p className="text-2xs leading-snug text-faint">{t("ai.plan.none")}</p>
      )}

      {/* ------------------------------------------------------ reference ----
          Between the brief and the mode: it shapes the edit like a style does,
          and it is the thing the user came to this tab to set when they have a
          clip they want copied. */}
      <ReferencePanel
        projectId={projectId}
        media={media}
        mediaList={mediaList}
        onAnalyze={onAnalyze}
      />

      {/* A template fixes the slots, their lengths and their joins; the engine
          still decides which photo or video goes in each. Its shape becomes
          the output's, visibly, rather than being applied behind the panel. */}
      <TemplatesPanel
        projectId={projectId}
        mediaList={mediaList}
        value={templateId}
        onChange={(chosen) => {
          setTemplateId(chosen?.id ?? null);
          if (chosen) setOutput({ aspect: chosen.aspect });
        }}
      />

      {/* ----------------------------------------------------------- mode ---- */}
      <section>
        <SectionTitle description={activeNote ? t(activeNote) : undefined}>
          {t("ai.mode")}
        </SectionTitle>
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
        {mode === "ai" && !aiAvailable && (
          <p className="mt-2 text-2xs leading-snug text-warning">{t("ai.planner.willFallBack")}</p>
        )}
      </section>

      {/* ---------------------------------------------------------- style ----
          Chips rather than a dropdown. There are six of them, they are the most
          consequential choice on the panel, and a dropdown hides five of six
          options behind a click for no gain in space at this width. */}
      <section>
        <SectionTitle
          description={
            selectedStyle && t(`style.${selectedStyle.value}.description` as MessageKey)
          }
        >
          {t("ai.style")}
        </SectionTitle>

        <div role="radiogroup" aria-label={t("ai.style.label")} className="flex flex-wrap gap-1.5">
          {[{ value: "" as const, label: t("ai.style.none"), description: "" }, ...styles].map(
            (item) => {
              const selected = style === item.value;
              return (
                <button
                  key={item.value || "none"}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  title={
                    item.value
                      ? t(`style.${item.value}.description` as MessageKey)
                      : undefined
                  }
                  onClick={() => {
                    const next = item.value as EditStyle | "";
                    setStyle(next);
                    // A style carries its own sensible length and shape.
                    // Applying them as the controls' visible values keeps the
                    // panel honest about what the server will actually do.
                    const profile = styles.find((entry) => entry.value === next);
                    if (profile) {
                      setTargetSeconds(Math.round(profile.default_duration_ms / 1000));
                      setOutput({ aspect: profile.default_aspect });
                    }
                  }}
                  className={`h-control-sm rounded-full px-3 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast ${
                    selected
                      ? "bg-accent-strong text-accent-fg shadow-raised"
                      : "bg-hover text-muted hover:text-fg"
                  }`}
                >
                  {item.value ? t(`style.${item.value}` as MessageKey) : item.label}
                </button>
              );
            },
          )}
        </div>
      </section>

      {/* --------------------------------------------------------- target ---- */}
      <section>
        <SectionTitle>{t("ai.target")}</SectionTitle>

        <div className="grid grid-cols-2 gap-2.5">
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
                  {item.value}
                </option>
              ))}
            </Select>
          </Field>
          <OutputSizeFields
            prefix="ai"
            capabilities={capabilities.data}
            hint={heavyOutput ? t("output.heavyHint") : undefined}
          />
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
          {/* The genre policy. A dropdown rather than chips: there are eleven,
              most users want the one their style already implies, and the
              default says so rather than pretending nothing is chosen. */}
          <Field label={t("ai.policy")}>
            <Select
              value={policy}
              aria-label={t("ai.policy.label")}
              onChange={(event) => setPolicy(event.target.value as PolicyId | "")}
            >
              <option value="">{t("ai.policy.auto")}</option>
              {(capabilities.data?.editorial_policies ?? []).map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                  title={t(`policy.${item.id}.description` as MessageKey)}
                >
                  {t(`policy.${item.id}` as MessageKey)}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      </section>

      {/* What this server actually has. No claim beyond it. */}
      <p className="font-mono text-2xs leading-snug text-faint">
        {aiAvailable
          ? capabilities.data?.is_stub
            ? t("ai.planner.stub", { provider: capabilities.data.provider ?? "—" })
            : t("ai.planner.model", {
                provider: capabilities.data?.provider ?? "—",
                model: capabilities.data?.model ?? "—",
              })
          : t("ai.planner.rules")}
      </p>
    </div>
  );
}
