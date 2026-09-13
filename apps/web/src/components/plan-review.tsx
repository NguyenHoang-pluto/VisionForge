"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type EditPlan, type MediaAsset } from "@/lib/api";
import { seconds, timecode } from "@/lib/format";
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
const FALLBACK_TEXT: Record<string, string> = {
  provider_disabled: "no AI provider is configured",
  provider_unavailable: "the AI provider was unreachable",
  provider_error: "the AI provider returned an error",
  invalid_output: "the model's answer could not be used",
  invalid_plan: "the model's edit did not pass validation",
  no_usable_media: "there was not enough usable footage",
  unexpected_error: "an unexpected error in the AI planner",
};

/** Distinct hues so adjacent clips separate in the proportional strip. */
const STRIP_COLOURS = [
  "bg-accent/70",
  "bg-info/60",
  "bg-ok/50",
  "bg-warn/45",
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
    <Dialog open={open} onClose={onClose} title="Edit plan">
      {!plan ? (
        <p className="py-6 text-center text-xs text-dim">
          {detail.isError ? "Could not load this plan." : "Loading plan…"}
        </p>
      ) : (
        <div className="flex max-h-[70vh] flex-col gap-3 overflow-y-auto">
          <PlanSummary plan={plan} />
          <Strip plan={plan} media={media} />
          <Segments plan={plan} media={media} />
          <Rejections plan={plan} media={media} />
          <Provenance plan={plan} />

          <div className="flex gap-1 border-t border-line pt-2">
            <Button
              tone="primary"
              onClick={() => {
                const draft = toDraft(plan);
                setClips(draft.clips, plan.id);
                setInspectorTab("export");
                onClose();
              }}
            >
              <Glyph name="check" size={10} />
              Accept into timeline
            </Button>
            <Button
              onClick={() => {
                setInspectorTab("ai");
                onClose();
              }}
            >
              <Glyph name="refresh" size={10} />
              Adjust and regenerate
            </Button>
            <Button className="ml-auto" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}

function PlanSummary({ plan }: { plan: EditPlan }) {
  const output = plan.plan.output;
  return (
    <section>
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={plan.planner === "manual" ? "neutral" : "accent"}>{plan.planner}</Badge>
        {plan.mode && <Badge>{plan.mode.mode}</Badge>}
        {plan.llm?.fallback_reason && <Badge tone="warn">fell back</Badge>}
      </div>
      <div className="mt-1.5">
        <Row label="Clips" value={String(plan.segment_count)} />
        <Row label="Duration" value={timecode(plan.total_duration_ms)} />
        <Row label="Output" value={`${output.width}×${output.height} @ ${output.fps} fps`} />
        <Row label="Aspect" value={output.aspect_ratio} />
        <Row label="Quality" value={output.quality} />
        <Row label="Audio" value={output.audio} />
        <Row label="Fit" value={output.fit} />
      </div>
    </section>
  );
}

/** Proportional strip: how the duration is distributed across the cut. */
function Strip({ plan, media }: { plan: EditPlan; media: Map<string, MediaAsset> }) {
  const total = plan.total_duration_ms || 1;
  return (
    <div className="flex h-6 w-full overflow-hidden rounded-sm border border-line">
      {plan.plan.segments.map((segment, index) => (
        <div
          key={`${segment.media_id}-${segment.order}`}
          className={`flex items-center justify-center overflow-hidden border-r border-panel last:border-r-0 ${
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
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-2xs tabular-nums">
        <thead className="text-dim">
          <tr className="border-b border-line-strong">
            <th className="py-1 pr-2 text-left font-normal">#</th>
            <th className="py-1 pr-2 text-left font-normal">Source</th>
            <th className="py-1 pr-2 text-right font-normal">In</th>
            <th className="py-1 pr-2 text-right font-normal">Out</th>
            <th className="py-1 pr-2 text-right font-normal">Dur</th>
            <th className="py-1 text-right font-normal">Cut</th>
          </tr>
        </thead>
        <tbody className="text-muted">
          {plan.plan.segments.map((segment) => (
            <tr
              key={`${segment.media_id}-${segment.order}`}
              className="border-b border-line/50 last:border-b-0"
            >
              <td className="py-0.5 pr-2 text-dim">{segment.order + 1}</td>
              <td className="max-w-[12rem] truncate py-0.5 pr-2 text-fg">
                {media.get(segment.media_id)?.original_filename ??
                  segment.media_id.slice(0, 8)}
              </td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.source_in_ms)}</td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.source_out_ms)}</td>
              <td className="py-0.5 pr-2 text-right">{seconds(segment.duration_ms)}</td>
              <td className="py-0.5 text-right text-dim">{segment.transition_in}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Why clips were dropped. An automatic edit that cannot explain itself is not reviewable. */
function Rejections({ plan, media }: { plan: EditPlan; media: Map<string, MediaAsset> }) {
  const rejected = plan.selection?.rejected ?? [];
  if (rejected.length === 0) return null;

  return (
    <details>
      <summary className="cursor-pointer font-mono text-2xs text-dim marker:text-line-strong hover:text-muted">
        {rejected.length} clip{rejected.length === 1 ? "" : "s"} not used
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5">
        {rejected.map((item) => (
          <li key={item.media_id} className="flex gap-2 font-mono text-2xs">
            <span className="w-32 truncate text-muted">
              {media.get(item.media_id)?.original_filename ?? item.media_id.slice(0, 8)}
            </span>
            <span className="shrink-0 text-warn">{item.reason.replace(/_/g, " ")}</span>
            <span className="truncate text-dim">{item.detail}</span>
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
  const llm = plan.llm;
  const fell = llm?.fallback_reason ?? null;

  const fields: [string, string][] = [
    ["Planner", `${plan.plan.planner}@${plan.plan.planner_version}`],
  ];
  if (plan.mode) fields.push(["Mode", plan.mode.mode]);
  if (llm?.provider) {
    fields.push(["Provider", llm.model ? `${llm.provider} · ${llm.model}` : llm.provider]);
    fields.push(["Prompt", llm.prompt_version]);
    if (llm.latency_ms) fields.push(["Latency", `${Math.round(llm.latency_ms)} ms`]);
    const tokens = (llm.input_tokens ?? 0) + (llm.output_tokens ?? 0);
    if (tokens > 0) {
      fields.push(["Tokens", `${llm.input_tokens ?? 0} in / ${llm.output_tokens ?? 0} out`]);
    }
    if (llm.attempts > 1) fields.push(["Attempts", String(llm.attempts)]);
  }
  if (typeof plan.plan.metadata?.derived_from_edit_plan_id === "string") {
    fields.push(["Cut from", String(plan.plan.metadata.derived_from_edit_plan_id).slice(0, 8)]);
  }

  return (
    <section className="border-t border-line pt-2">
      <dl className="flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-2xs tabular-nums">
        {fields.map(([label, value]) => (
          <div key={label} className="flex gap-1.5">
            <dt className="text-dim">{label}</dt>
            <dd className="text-muted">{value}</dd>
          </div>
        ))}
      </dl>

      {fell && (
        <p className="mt-1 border-l-2 border-warn pl-2 text-2xs leading-snug text-warn">
          Planned by the rules engine — {FALLBACK_TEXT[fell] ?? fell.replace(/_/g, " ")}.
          {llm?.fallback_detail && (
            <span className="block truncate text-dim">{llm.fallback_detail}</span>
          )}
        </p>
      )}

      {plan.mode?.reason && !fell && (
        <p className="mt-1 text-2xs leading-snug text-dim">{plan.mode.reason}.</p>
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
