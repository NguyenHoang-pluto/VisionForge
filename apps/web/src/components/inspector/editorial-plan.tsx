"use client";

import { useState } from "react";

import {
  type EditorialRecord,
  type EditorialSegment,
  type EditorialVariantPreview,
  type MediaAsset,
  type StoryRole,
  type VariantId,
} from "@/lib/api";
import { seconds, timecode } from "@/lib/format";
import { useT, type MessageKey, type Translate } from "@/lib/i18n";
import { Badge, Button, Glyph, Meter, SectionTitle, Spinner } from "@/components/ui";
import { templateLabel } from "@/components/inspector/templates-panel";

/**
 * The editorial plan, drawn as the arc the engine actually built.
 *
 *     HOOK  ↓  SETUP  ↓  BUILD  ↓  PEAK  ↓  REACTION  ↓  ENDING
 *
 * This is the panel the phase exists for. Before it, an automatic edit arrived
 * as a list of clips and the user's only question -- "why these?" -- had no
 * answer in the interface. Every row here carries what the engine decided and
 * the short, structured reasons it decided so: the role, the length, how busy
 * the shot is against how busy the curve wanted it, whether the cut landed on a
 * beat, and what argued for the clip.
 *
 * Three things it deliberately does not show. There is no model output: the
 * reasons are tokens from a closed server-side vocabulary, rendered here in the
 * user's own language, and no chain of thought exists to leak. There is no
 * single quality score, because the server does not compute one -- eight
 * measurements are shown side by side and each can be argued with. And there is
 * no dashboard: this is a panel in the inspector beside clip properties and
 * export, which is the accurate description of what it is.
 */

/** Role to a colour and an icon. The arc reads as a shape before it reads as text. */
const ROLE_TONE: Record<StoryRole, string> = {
  hook: "bg-accent/70",
  setup: "bg-info/55",
  build: "bg-info/70",
  peak: "bg-warning/80",
  reaction: "bg-success/60",
  ending: "bg-accent/40",
};

const ROLE_LABEL: Record<StoryRole, MessageKey> = {
  hook: "editorial.role.hook",
  setup: "editorial.role.setup",
  build: "editorial.role.build",
  peak: "editorial.role.peak",
  reaction: "editorial.role.reaction",
  ending: "editorial.role.ending",
};

/**
 * Reason token to a sentence.
 *
 * A lookup rather than a formatter, because the server sends tokens precisely so
 * that the wording is a client decision. A token with no entry falls back to its
 * own text with the underscores removed -- visible and slightly ugly, which is
 * the right outcome for a vocabulary the server grew and the client has not
 * caught up with, and much better than hiding a reason the user was given.
 */
const REASON_LABEL: Record<string, MessageKey> = {
  high_quality: "editorial.reason.high_quality",
  high_motion: "editorial.reason.high_motion",
  peak_moment: "editorial.reason.peak_moment",
  unique_content: "editorial.reason.unique_content",
  role_fit: "editorial.reason.role_fit",
  energy_fit: "editorial.reason.energy_fit",
  style_match: "editorial.reason.style_match",
  semantic_relevance: "editorial.reason.semantic_relevance",
  faces_present: "editorial.reason.faces_present",
  only_candidate: "editorial.reason.only_candidate",
  weak_quality: "editorial.reason.weak_quality",
  duplicate_content: "editorial.reason.duplicate_content",
  too_similar: "editorial.reason.too_similar",
  not_needed: "editorial.reason.not_needed",
  too_short: "editorial.reason.too_short",
  weak_role_fit: "editorial.reason.weak_role_fit",
  pacing_shortens: "editorial.reason.pacing_shortens",
  pacing_lengthens: "editorial.reason.pacing_lengthens",
  policy_emphasis: "editorial.reason.policy_emphasis",
  beat_aligned: "editorial.reason.beat_aligned",
  off_beat: "editorial.reason.off_beat",
  avoids_internal_cut: "editorial.reason.avoids_internal_cut",
  centred_on_action: "editorial.reason.centred_on_action",
  centre_trim: "editorial.reason.centre_trim",
  source_too_short: "editorial.reason.source_too_short",
  chronology: "editorial.reason.chronology",
  narrative_order: "editorial.reason.narrative_order",
  style_policy: "editorial.reason.style_policy",
  likely_speech: "editorial.reason.likely_speech",
};

const METRIC_LABEL: Record<string, MessageKey> = {
  content_diversity: "editorial.metric.content_diversity",
  repetition: "editorial.metric.repetition",
  pacing_consistency: "editorial.metric.pacing_consistency",
  energy_progression: "editorial.metric.energy_progression",
  beat_alignment: "editorial.metric.beat_alignment",
  style_adherence: "editorial.metric.style_adherence",
  story_completeness: "editorial.metric.story_completeness",
  quality: "editorial.metric.quality",
};

function reasonText(t: Translate, token: string): string {
  const key = REASON_LABEL[token];
  return key ? t(key) : token.replace(/_/g, " ");
}

function sourceName(media: Map<string, MediaAsset>, mediaId: string): string {
  return media.get(mediaId)?.original_filename ?? mediaId.slice(0, 8);
}

/** Contiguous runs of one role, so the arc is drawn per part rather than per clip. */
function groupByRole(segments: EditorialSegment[]): { role: StoryRole; items: EditorialSegment[] }[] {
  const groups: { role: StoryRole; items: EditorialSegment[] }[] = [];
  for (const segment of segments) {
    const last = groups[groups.length - 1];
    if (last && last.role === segment.role) last.items.push(segment);
    else groups.push({ role: segment.role, items: [segment] });
  }
  return groups;
}

export function EditorialPlanPanel({
  record,
  media,
  variants,
  activeVariant,
  loadingVariants,
  onChooseVariant,
  onRegenerate,
  onAccept,
  onRender,
  busy = false,
}: {
  record: EditorialRecord;
  media: Map<string, MediaAsset>;
  /** Previewed alternatives, when the user has asked for them. */
  variants: EditorialVariantPreview[] | null;
  activeVariant: VariantId | null;
  loadingVariants: boolean;
  onChooseVariant: (variant: VariantId | null) => void;
  onRegenerate: () => void;
  onAccept: () => void;
  onRender: () => void;
  busy?: boolean;
}) {
  const t = useT();
  const [showRejected, setShowRejected] = useState(false);

  const groups = groupByRole(record.segments);
  const total = record.pacing.total_ms || 1;

  return (
    <section className="flex flex-col gap-3">
      <SectionTitle description={t(`policy.${record.policy_detail.id}.description` as MessageKey)}>
        {t("editorial.title")}
      </SectionTitle>

      {/* ------------------------------------------------------ the summary -- */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone="accent">{t(`policy.${record.policy_detail.id}` as MessageKey)}</Badge>
        <Badge>{t(`editorial.pacing.${record.pacing.curve.shape}` as MessageKey)}</Badge>
        {record.variant && <Badge tone="warn">{t("editorial.variantBadge")}</Badge>}
        {record.fit.template && (
          <Badge tone="accent">
            {t("template.badge", { name: templateLabel(record.fit.template, t) })}
          </Badge>
        )}
        <span className="ml-auto font-mono text-2xs tabular-nums text-faint">
          {t("editorial.summary", {
            clips: record.segments.length,
            duration: timecode(record.pacing.total_ms, false),
            density: record.pacing.shot_density.toFixed(2),
          })}
        </span>
      </div>

      {/* The shortfall, when the footage could not carry what was asked for.
          Stated rather than left for the user to notice a short video. */}
      {record.fit.shortfall_ms > 500 && (
        <p className="border-l-2 border-warning pl-2 text-2xs leading-snug text-warning">
          {t("editorial.shortfall", {
            wanted: seconds(record.fit.target_ms, 0),
            got: seconds(record.fit.achieved_ms, 0),
            clips: record.fit.achieved_clips,
          })}
        </p>
      )}

      {/* --------------------------------------------------------- variants -- */}
      {variants && variants.length > 1 && (
        <div role="radiogroup" aria-label={t("editorial.variants")} className="flex flex-wrap gap-1.5">
          {variants.map((item) => {
            const selected = (item.variant ?? null) === activeVariant;
            return (
              <button
                key={item.variant ?? "base"}
                type="button"
                role="radio"
                aria-checked={selected}
                title={
                  item.variant
                    ? t(`variant.${item.variant}.description` as MessageKey)
                    : t(`policy.${item.policy}.description` as MessageKey)
                }
                disabled={busy}
                onClick={() => onChooseVariant(item.variant)}
                className={`h-control-sm rounded-full px-3 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast disabled:opacity-50 ${
                  selected
                    ? "bg-accent-strong text-accent-fg shadow-raised"
                    : "bg-hover text-muted hover:text-fg"
                }`}
              >
                {t(item.variant ? (`variant.${item.variant}` as MessageKey) : "variant.base")}
                <span className="ml-1.5 font-mono tabular-nums opacity-70">
                  {item.clip_count}·{seconds(item.total_duration_ms, 0)}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* -------------------------------------------------------- the arc --- */}
      <ol className="flex flex-col">
        {groups.map((group, index) => (
          <li key={`${group.role}-${index}`}>
            <div className="mb-1 flex items-center gap-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${ROLE_TONE[group.role]}`}
                aria-hidden
              />
              <span className="text-2xs font-semibold uppercase tracking-wide text-muted">
                {t(ROLE_LABEL[group.role])}
              </span>
              <span className="font-mono text-2xs tabular-nums text-faint">
                {seconds(group.items.reduce((sum, item) => sum + item.output_ms, 0), 1)}
              </span>
            </div>

            <div className="ml-[3px] flex flex-col gap-1 border-l border-strong pl-3">
              {group.items.map((segment) => (
                <SegmentRow key={segment.slot} segment={segment} media={media} total={total} />
              ))}
            </div>

            {index < groups.length - 1 && (
              <div className="ml-[3px] py-1 text-faint" aria-hidden>
                <Glyph name="chevron-down" size={12} />
              </div>
            )}
          </li>
        ))}
      </ol>

      {/* --------------------------------------------------------- metrics -- */}
      <div>
        <SectionTitle description={t("editorial.metrics.note")}>
          {t("editorial.metrics")}
        </SectionTitle>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1">
          {Object.values(record.metrics).map((metric) => (
            <div key={metric.name} className="flex items-baseline justify-between gap-2">
              <dt className="truncate text-2xs text-muted" title={t("editorial.metric.samples", {
                count: metric.sample_size,
              })}>
                {METRIC_LABEL[metric.name]
                  ? t(METRIC_LABEL[metric.name])
                  : metric.name.replace(/_/g, " ")}
              </dt>
              <dd className="shrink-0 font-mono text-2xs tabular-nums text-fg">
                {metric.value === null ? (
                  <span
                    className="text-faint"
                    title={metric.unmeasurable ?? undefined}
                  >
                    {t("editorial.metric.unmeasured")}
                  </span>
                ) : (
                  (metric.value * 100).toFixed(0) + "%"
                )}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {/* -------------------------------------------------------- rejected -- */}
      {record.rejected.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowRejected((open) => !open)}
            className="flex items-center gap-1.5 font-mono text-2xs text-faint hover:text-muted"
            aria-expanded={showRejected}
          >
            <Glyph name={showRejected ? "chevron-down" : "chevron-right"} size={10} />
            {t.plural("editorial.rejected", record.rejected.length)}
          </button>
          {showRejected && (
            <ul className="mt-1 flex flex-col gap-0.5">
              {record.rejected.map((item) => (
                <li key={item.media_id} className="flex gap-2 font-mono text-2xs">
                  <span className="w-28 truncate text-muted">
                    {sourceName(media, item.media_id)}
                  </span>
                  <span className="truncate text-faint">
                    {item.reasons.map((token) => reasonText(t, token)).join(", ")}
                    {item.similar_to &&
                      ` — ${t("editorial.rejected.similarTo", {
                        name: sourceName(media, item.similar_to),
                      })}`}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* --------------------------------------------------------- actions -- */}
      <div className="flex flex-wrap gap-2">
        <Button tone="primary" disabled={busy} onClick={onAccept}>
          <Glyph name="check" size={11} />
          {t("editorial.accept")}
        </Button>
        <Button disabled={busy} onClick={onRegenerate}>
          {busy ? <Spinner size={11} /> : <Glyph name="refresh" size={11} />}
          {t("editorial.regenerate")}
        </Button>
        <Button
          disabled={busy || loadingVariants}
          onClick={() => onChooseVariant(activeVariant)}
          title={t("editorial.variants.hint")}
        >
          {loadingVariants ? <Spinner size={11} /> : <Glyph name="layers" size={11} />}
          {t("editorial.variants")}
        </Button>
        <Button className="ml-auto" disabled={busy} onClick={onRender}>
          <Glyph name="export" size={11} />
          {t("editorial.render")}
        </Button>
      </div>
    </section>
  );
}

/**
 * One clip, with everything the engine decided about it.
 *
 * The energy meter shows *two* numbers on one bar: the clip's own energy as the
 * fill, and where the pacing curve wanted it as a tick. That is the comparison
 * worth seeing -- a clip is not too calm in the absolute, it is calmer than the
 * moment in the edit it was put into.
 */
function SegmentRow({
  segment,
  media,
  total,
}: {
  segment: EditorialSegment;
  media: Map<string, MediaAsset>;
  total: number;
}) {
  const t = useT();
  const share = Math.max(0.02, segment.output_ms / total);

  return (
    <div className="rounded-md bg-hover/40 px-2 py-1.5">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-2xs tabular-nums text-faint">{segment.slot + 1}</span>
        <span className="min-w-0 flex-1 truncate text-2xs text-fg">
          {sourceName(media, segment.media_id)}
        </span>
        <span className="shrink-0 font-mono text-2xs tabular-nums text-muted">
          {seconds(segment.output_ms, 1)}
        </span>
      </div>

      <div className="mt-1 flex items-center gap-2">
        <div className="relative h-[4px] flex-1">
          <Meter value={segment.energy} max={1} tone={segment.energy > 0.66 ? "warn" : "accent"} />
          {/* Where the curve wanted this position to sit. */}
          <span
            className="absolute top-[-2px] h-[8px] w-px bg-fg/50"
            style={{ left: `${Math.min(100, segment.target_energy * 100)}%` }}
            aria-hidden
            title={t("editorial.energy.target", {
              value: (segment.target_energy * 100).toFixed(0),
            })}
          />
        </div>
        <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
          {(segment.energy * 100).toFixed(0)}%
        </span>
        <span
          className="shrink-0 font-mono text-2xs tabular-nums text-faint"
          title={t("editorial.share")}
        >
          {(share * 100).toFixed(0)}%
        </span>
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-1">
        {segment.on_beat && (
          <Badge tone="accent">
            {t("editorial.onBeat", { beats: segment.beats ?? 0 })}
          </Badge>
        )}
        {segment.transition !== "cut" && <Badge>{segment.transition.replace(/_/g, " ")}</Badge>}
        {segment.effects.map((effect) => (
          <Badge key={effect.kind} tone="warn">
            {effect.kind.replace(/_/g, " ")}
          </Badge>
        ))}
        {segment.reasons.slice(0, 4).map((token) => (
          <span key={token} className="text-2xs leading-snug text-faint">
            · {reasonText(t, token)}
          </span>
        ))}
      </div>
    </div>
  );
}
