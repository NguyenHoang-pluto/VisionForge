"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type EditPlan, type MediaAsset } from "@/lib/api";
import { seconds, timecode } from "@/lib/format";
import { useT, type MessageKey, type Translate } from "@/lib/i18n";
import { toDraft } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import { Badge, Button, Dialog, Glyph, Row } from "@/components/ui";

/**
 * The generated plan, before it is rendered.
 *
 * Opens after a plan is generated so the edit can be read before anything is
 * encoded. What it shows is the plan document and the selection the planner
 * recorded -- which clips, in what order, trimmed where, and which clips were
 * dropped and why.
 *
 * There is no raw model output here and there is nowhere to put one: the API
 * never returns a prompt or a completion, only the structured plan and a
 * provenance record. What the model "said" is the plan.
 */

/** Fallback reasons, in words a user can act on. */
const FALLBACK_TEXT: Record<string, MessageKey> = {
  provider_disabled: "plan.fallback.provider_disabled",
  provider_unavailable: "plan.fallback.provider_unavailable",
  provider_error: "plan.fallback.provider_error",
  invalid_output: "plan.fallback.invalid_output",
  invalid_plan: "plan.fallback.invalid_plan",
  no_usable_media: "plan.fallback.no_usable_media",
  unexpected_error: "plan.fallback.unexpected_error",
};

/** Distinct hues so adjacent clips separate in the proportional strip. */
const STRIP_COLOURS = [
  "bg-accent/70",
  "bg-info/60",
  "bg-success/50",
  "bg-warning/45",
  "bg-accent/45",
  "bg-info/40",
];

export function PlanReview({
  projectId,
  planId,
  media,
  open,
  onClose,
}: {
  projectId: string;
  planId: string | null;
  media: Map<string, MediaAsset>;
  open: boolean;
  onClose: () => void;
}) {
  const t = useT();
  const setClips = useEditorStore((s) => s.setClips);
  const setInspectorTab = useEditorStore((s) => s.setInspectorTab);

  // The listing carries no plan document, so the detail is fetched for the one
  // plan being reviewed rather than for all twenty.
  const detail = useQuery({
    queryKey: ["edit-plan", planId],
    queryFn: () => api.getEditPlan(projectId, planId!),
    enabled: Boolean(open && planId),
    staleTime: Infinity,
  });

  const plan = detail.data ?? null;

  return (
    <Dialog open={open} onClose={onClose} title={t("plan.title")} size="lg">
      {!plan ? (
        <p className="py-6 text-center text-xs text-faint">
          {detail.isError ? t("plan.loadFailed") : t("plan.loading")}
        </p>
      ) : (
        <div className="flex max-h-[70vh] flex-col gap-3 overflow-y-auto overscroll-contain">
          <PlanSummary plan={plan} />
          <Strip plan={plan} media={media} />
          <Segments plan={plan} media={media} />
          <Rejections plan={plan} media={media} />
          <Provenance plan={plan} />

          <div className="flex gap-1 border-t border-subtle pt-2">
            <Button
              tone="primary"
              onClick={() => {
                const draft = toDraft(plan);
                setClips(draft.clips, plan.id, plan.id, draft.music);
                setInspectorTab("export");
                onClose();
              }}
            >
              <Glyph name="check" size={10} />
              {t("plan.accept")}
            </Button>
            <Button
              onClick={() => {
                setInspectorTab("ai");
                onClose();
              }}
            >
              <Glyph name="refresh" size={10} />
              {t("plan.adjust")}
            </Button>
            <Button className="ml-auto" onClick={onClose}>
              {t("plan.close")}
            </Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}

function PlanSummary({ plan }: { plan: EditPlan }) {
  const t = useT();
  const output = plan.plan.output;
  return (
    <section>
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={plan.planner === "manual" ? "neutral" : "accent"}>{plan.planner}</Badge>
        {plan.mode && <Badge>{plan.mode.mode}</Badge>}
        {plan.llm?.fallback_reason && <Badge tone="warn">{t("plan.fellBack")}</Badge>}
      </div>
      <div className="mt-1.5">
        <Row label={t("plan.clips")} value={String(plan.segment_count)} />
        <Row label={t("plan.duration")} value={timecode(plan.total_duration_ms)} />
        <Row
          label={t("plan.output")}
          value={t("plan.outputValue", {
            width: output.width,
            height: output.height,
            fps: output.fps,
          })}
        />
        <Row label={t("plan.aspect")} value={output.aspect_ratio} />
        <Row label={t("plan.quality")} value={output.quality} />
        <Row label={t("plan.audio")} value={output.audio} />
        <Row label={t("plan.fit")} value={output.fit} />
      </div>
    </section>
  );
}

/** Proportional strip: how the duration is distributed across the cut. */
function Strip({ plan, media }: { plan: EditPlan; media: Map<string, MediaAsset> }) {
  const total = plan.total_duration_ms || 1;
  return (
    <div className="flex h-6 w-full overflow-hidden rounded-sm border border-subtle">
      {plan.plan.segments.map((segment, index) => (
        <div
          key={`${segment.media_id}-${segment.order}`}
          className={`flex items-center justify-center overflow-hidden border-r border-surface last:border-r-0 ${
            STRIP_COLOURS[index % STRIP_COLOURS.length]
          }`}
          style={{ width: `${(segment.duration_ms / total) * 100}%` }}
          title={`${media.get(segment.media_id)?.original_filename ?? segment.media_id.slice(0, 8)} · ${segment.source_in_ms}–${segment.source_out_ms} ms`}
        >
          <span className="truncate px-1 font-mono text-2xs text-fg/90 tabular-nums">
            {seconds(segment.duration_ms)}
          </span>
        </div>
      ))}
    </div>
  );
}

function Segments({ plan, media }: { plan: EditPlan; media: Map<string, MediaAsset> }) {
  const t = useT();
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-2xs tabular-nums">
        <thead className="text-faint">
          <tr className="border-b border-strong">
            <th className="py-1 pr-2 text-left font-normal">{t("plan.column.index")}</th>
            <th className="py-1 pr-2 text-left font-normal">{t("plan.column.source")}</th>
            <th className="py-1 pr-2 text-right font-normal">{t("plan.column.in")}</th>
            <th className="py-1 pr-2 text-right font-normal">{t("plan.column.out")}</th>
            <th className="py-1 pr-2 text-right font-normal">{t("plan.column.duration")}</th>
            <th className="py-1 text-right font-normal">{t("plan.column.cut")}</th>
          </tr>
        </thead>
        <tbody className="text-muted">
          {plan.plan.segments.map((segment) => (
            <tr
              key={`${segment.media_id}-${segment.order}`}
              className="border-b border-subtle/50 last:border-b-0"
            >
              <td className="py-0.5 pr-2 text-faint">{segment.order + 1}</td>
              <td className="max-w-[12rem] truncate py-0.5 pr-2 text-fg">
                {media.get(segment.media_id)?.original_filename ??
                  segment.media_id.slice(0, 8)}
              </td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.source_in_ms)}</td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.source_out_ms)}</td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.duration_ms)}</td>
              <td className="py-0.5 text-right text-faint">{segment.transition_in}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Why clips were dropped. An automatic edit that cannot explain itself is not reviewable. */
function Rejections({ plan, media }: { plan: EditPlan; media: Map<string, MediaAsset> }) {
  const t = useT();
  const rejected = plan.selection?.rejected ?? [];
  if (rejected.length === 0) return null;

  return (
    <details>
      <summary className="cursor-pointer font-mono text-2xs text-faint marker:text-strong hover:text-muted">
        {t.plural("plan.rejected", rejected.length)}
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5">
        {rejected.map((item) => (
          <li key={item.media_id} className="flex gap-2 font-mono text-2xs">
            <span className="w-32 truncate text-muted">
              {media.get(item.media_id)?.original_filename ?? item.media_id.slice(0, 8)}
            </span>
            <span className="shrink-0 text-warning">{item.reason.replace(/_/g, " ")}</span>
            <span className="truncate text-faint">{item.detail}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

/**
 * Where the plan came from: which planner, which model, what it cost, and
 * whether it fell back. A fallback nobody can see is indistinguishable from a
 * feature that silently does nothing.
 */
function Provenance({ plan }: { plan: EditPlan }) {
  const t: Translate = useT();
  const llm = plan.llm;
  const fell = llm?.fallback_reason ?? null;

  const fields: [string, string][] = [
    [t("plan.provenance.planner"), `${plan.plan.planner}@${plan.plan.planner_version}`],
  ];
  if (plan.mode) fields.push([t("plan.provenance.mode"), plan.mode.mode]);
  if (llm?.provider) {
    fields.push([
      t("plan.provenance.provider"),
      llm.model ? `${llm.provider} · ${llm.model}` : llm.provider,
    ]);
    fields.push([t("plan.provenance.prompt"), llm.prompt_version]);
    if (llm.latency_ms) {
      fields.push([t("plan.provenance.latency"), `${Math.round(llm.latency_ms)} ms`]);
    }
    const tokens = (llm.input_tokens ?? 0) + (llm.output_tokens ?? 0);
    if (tokens > 0) {
      fields.push([
        t("plan.provenance.tokens"),
        t("plan.provenance.tokensValue", {
          input: llm.input_tokens ?? 0,
          output: llm.output_tokens ?? 0,
        }),
      ]);
    }
    if (llm.attempts > 1) fields.push([t("plan.provenance.attempts"), String(llm.attempts)]);
  }
  if (typeof plan.plan.metadata?.derived_from_edit_plan_id === "string") {
    fields.push([
      t("plan.provenance.derivedFrom"),
      String(plan.plan.metadata.derived_from_edit_plan_id).slice(0, 8),
    ]);
  }

  return (
    <section className="border-t border-subtle pt-2">
      <dl className="flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-2xs tabular-nums">
        {fields.map(([label, value]) => (
          <div key={label} className="flex gap-1.5">
            <dt className="text-faint">{label}</dt>
            <dd className="text-muted">{value}</dd>
          </div>
        ))}
      </dl>

      {fell && (
        <p className="mt-1 border-l-2 border-warning pl-2 text-2xs leading-snug text-warning">
          {t("plan.fallback", {
            reason: FALLBACK_TEXT[fell] ? t(FALLBACK_TEXT[fell]) : fell.replace(/_/g, " "),
          })}
          {llm?.fallback_detail && (
            <span className="block truncate text-faint">{llm.fallback_detail}</span>
          )}
        </p>
      )}

      {plan.mode?.reason && !fell && (
        <p className="mt-1 text-2xs leading-snug text-faint">{plan.mode.reason}.</p>
      )}

      {typeof plan.plan.metadata?.rationale === "string" &&
        plan.plan.metadata.rationale.length > 0 && (
          <p className="mt-1 text-2xs leading-snug text-muted">
            {String(plan.plan.metadata.rationale)}
          </p>
        )}
    </section>
  );
}
