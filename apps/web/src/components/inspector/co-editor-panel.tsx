"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  api,
  ApiError,
  type CoEditPreview,
  type EditPlan,
  type EditVersion,
  type PlanDiff,
} from "@/lib/api";
import { useT } from "@/lib/i18n";
import { toDraft } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Button,
  EmptyState,
  ErrorNote,
  Glyph,
  SectionTitle,
  Spinner,
  TextArea,
} from "@/components/ui";

/**
 * The AI co-editor.
 *
 * An editing tool, not a chat window. There is no conversation, no message
 * history and no assistant persona: there is a field that takes a change, a
 * diff that says what it would do, and a list of the versions it produced.
 * What the user reads back is the *edit*, not a reply about the edit.
 *
 * The shape follows what the server can prove:
 *
 *   - a change the server's own rules resolved is applied immediately, because
 *     "set the music to 40%" has nothing to confirm;
 *   - a change a model proposed is previewed first, because it involved
 *     judgement and the user should see the judgement before it lands;
 *   - undo and redo are server calls, not local state, because the versions
 *     live on the server and a second browser must see the same head.
 *
 * Applying never renders. The Render button is here because it is the next
 * thing the user wants, not because a change implies an encode.
 */

/** Recent requests, kept in the component. Convenience, never sent anywhere. */
const MAX_RECENT = 5;

export function CoEditorPanel({ projectId }: { projectId: string }) {
  const t = useT();
  const queryClient = useQueryClient();
  const setClips = useEditorStore((s) => s.setClips);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);

  const [request, setRequest] = useState("");
  const [recent, setRecent] = useState<string[]>([]);
  const [pending, setPending] = useState<CoEditPreview | null>(null);
  const [applied, setApplied] = useState<PlanDiff | null>(null);
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  const versions = useQuery({
    queryKey: ["edit-versions", projectId],
    queryFn: () => api.listEditVersions(projectId),
    staleTime: 5_000,
  });

  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });

  const current = versions.data?.items.find((item) => item.is_current) ?? null;
  const hasEdit = (versions.data?.total ?? 0) > 0;

  function remember(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;
    setRecent((previous) =>
      [trimmed, ...previous.filter((item) => item !== trimmed)].slice(0, MAX_RECENT),
    );
  }

  function refresh() {
    void queryClient.invalidateQueries({ queryKey: ["edit-versions", projectId] });
    void queryClient.invalidateQueries({ queryKey: ["edit-plans", projectId] });
  }

  /**
   * Load a plan the server has accepted into the timeline on screen.
   *
   * No review dialog: reviewing belongs to *generating* an edit, where there is
   * something to accept or reject. A patch has already been shown as a diff and
   * is already stored, so the only thing left is to draw it.
   */
  function adopt(plan: EditPlan) {
    const draft = toDraft(plan);
    setClips(draft.clips, plan.id, plan.id, draft.music, draft.subtitles);
    setPreviewSource("program");
  }

  function fail(caught: unknown) {
    setError(
      caught instanceof ApiError
        ? { message: caught.message, hint: caught.hint }
        : { message: t("coedit.error"), hint: null },
    );
  }

  /**
   * Ask what a change would do.
   *
   * A request the server's rules resolved comes back with
   * `needs_confirmation: false` and is applied straight away -- showing a
   * confirmation for "music to 40%" is ceremony, and the user can undo.
   */
  const propose = useMutation({
    mutationFn: () => api.previewCoEdit(projectId, { request_text: request.trim() }),
    onSuccess: (preview) => {
      setError(null);
      if (!preview.ok) {
        setPending(null);
        setError({ message: failureMessage(preview, t), hint: preview.detail || null });
        return;
      }
      if (!preview.needs_confirmation) {
        commit.mutate(preview);
        return;
      }
      setPending(preview);
      setApplied(null);
    },
    onError: fail,
  });

  const commit = useMutation({
    mutationFn: (preview: CoEditPreview) =>
      api.applyCoEdit(projectId, {
        // The operations go back exactly as they came; the server re-parses and
        // re-applies them rather than trusting the round trip.
        operations: preview.operations,
        base_version_id: preview.base_version_id,
      }),
    onSuccess: (result) => {
      remember(request);
      setRequest("");
      setPending(null);
      setApplied(result.diff);
      setError(null);
      adopt(result.plan);
      refresh();
    },
    onError: (caught) => {
      setPending(null);
      fail(caught);
    },
  });

  const move = useMutation({
    mutationFn: (direction: "undo" | "redo") =>
      direction === "undo"
        ? api.undoEditVersion(projectId)
        : api.redoEditVersion(projectId),
    onSuccess: async (version: EditVersion) => {
      setError(null);
      setPending(null);
      setApplied(null);
      // The restored version points at a plan that already exists; loading it
      // is a read, not a re-derivation.
      adopt(await api.getEditPlan(projectId, version.edit_plan_id));
      refresh();
    },
    onError: fail,
  });

  /**
   * Render the current version.
   *
   * Straight to `createRender`: the version already points at a stored,
   * validated plan, so there is nothing to commit first. Applying a change
   * never starts this -- an encode is something the user asks for.
   */
  const render = useMutation({
    mutationFn: (editPlanId: string) => api.createRender(projectId, editPlanId),
    onSuccess: () => {
      setError(null);
      setPreviewSource("render");
      void queryClient.invalidateQueries({ queryKey: ["renders", projectId] });
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
    onError: fail,
  });

  const busy = propose.isPending || commit.isPending || move.isPending;

  if (versions.isLoading) {
    return (
      <div className="flex items-center justify-center p-panel">
        <Spinner size={16} />
      </div>
    );
  }

  if (!hasEdit) {
    return (
      <EmptyState icon="wand" title={t("coedit.empty.title")}>
        {t("coedit.empty.body")}
      </EmptyState>
    );
  }

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      {/* ------------------------------------------------------ the request --- */}
      <section>
        <div className="mb-2 flex items-center gap-2">
          <span
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent-strong"
            aria-hidden
          >
            <Glyph name="wand" size={14} />
          </span>
          <h3 className="text-xs font-semibold tracking-tight text-fg">{t("coedit.title")}</h3>
          {current && (
            <span className="ml-auto font-mono text-2xs text-faint">
              {t("coedit.version", { version: String(current.version) })}
            </span>
          )}
        </div>

        <TextArea
          rows={3}
          value={request}
          maxLength={capabilities.data?.max_request_chars ?? 500}
          placeholder={t("coedit.placeholder")}
          aria-label={t("coedit.label")}
          onChange={(event) => setRequest(event.target.value)}
          onKeyDown={(event) => {
            // Enter applies; Shift+Enter is a newline. A change request is one
            // sentence far more often than it is a paragraph.
            if (event.key === "Enter" && !event.shiftKey && request.trim() && !busy) {
              event.preventDefault();
              propose.mutate();
            }
          }}
        />

        <Button
          tone="primary"
          size="lg"
          className="mt-2 w-full"
          disabled={!request.trim() || busy}
          onClick={() => propose.mutate()}
        >
          {propose.isPending || commit.isPending ? (
            <Spinner size={13} />
          ) : (
            <Glyph name="spark" size={13} />
          )}
          {propose.isPending
            ? t("coedit.thinking")
            : commit.isPending
              ? t("coedit.applying")
              : t("coedit.apply")}
        </Button>

        {error && (
          <div className="mt-2">
            <ErrorNote hint={error.hint}>{error.message}</ErrorNote>
          </div>
        )}
      </section>

      {/* -------------------------------------------------------- the diff --- */}
      {pending && (
        <section className="rounded-lg border border-accent-strong/30 bg-accent-soft/40 p-3">
          <SectionTitle>{t("coedit.proposed")}</SectionTitle>
          {pending.rationale && (
            <p className="mb-2 text-2xs leading-snug text-muted">{pending.rationale}</p>
          )}
          <DiffList diff={pending.diff} />
          <div className="mt-3 flex gap-2">
            <Button
              tone="primary"
              className="flex-1"
              disabled={busy}
              onClick={() => commit.mutate(pending)}
            >
              <Glyph name="check" size={12} />
              {t("coedit.confirm")}
            </Button>
            <Button className="flex-1" disabled={busy} onClick={() => setPending(null)}>
              {t("common.cancel")}
            </Button>
          </div>
          {pending.source === "llm" && pending.model && (
            <p className="mt-2 font-mono text-2xs text-faint">
              {t("coedit.by", { provider: pending.provider, model: pending.model })}
            </p>
          )}
        </section>
      )}

      {/* ----------------------------------------------------- what landed --- */}
      {applied && !pending && (
        <section>
          <SectionTitle>{t("coedit.applied")}</SectionTitle>
          <ul className="flex flex-col gap-1">
            {(applied.applied.length ? applied.applied : applied.entries.map(summarise)).map(
              (line, index) => (
                <li key={index} className="flex items-start gap-1.5 text-2xs text-fg">
                  <span className="mt-[2px] text-success" aria-hidden>
                    <Glyph name="check" size={11} />
                  </span>
                  {line}
                </li>
              ),
            )}
          </ul>
          {applied.adjustments.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1">
              {applied.adjustments.map((line, index) => (
                <li key={index} className="flex items-start gap-1.5 text-2xs text-warning">
                  <span className="mt-[2px]" aria-hidden>
                    <Glyph name="warning" size={11} />
                  </span>
                  {line}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* --------------------------------------------------------- history --- */}
      <section>
        <SectionTitle description={t("coedit.history.note")}>
          {t("coedit.history")}
        </SectionTitle>

        <div className="mb-2 flex gap-2">
          <Button
            className="flex-1"
            disabled={!versions.data?.can_undo || busy}
            onClick={() => move.mutate("undo")}
          >
            <Glyph name="chevron-left" size={12} />
            {t("coedit.undo")}
          </Button>
          <Button
            className="flex-1"
            disabled={!versions.data?.can_redo || busy}
            onClick={() => move.mutate("redo")}
          >
            {t("coedit.redo")}
            <Glyph name="chevron-right" size={12} />
          </Button>
          <Button
            tone="primary"
            className="flex-1"
            disabled={!current || render.isPending || busy}
            onClick={() => current && render.mutate(current.edit_plan_id)}
          >
            {render.isPending ? <Spinner size={12} /> : <Glyph name="export" size={12} />}
            {t("coedit.render")}
          </Button>
        </div>

        <ol className="flex flex-col gap-1">
          {(versions.data?.items ?? []).slice(0, 8).map((version) => (
            <VersionRow key={version.id} version={version} />
          ))}
        </ol>
      </section>

      {/* ------------------------------------------------- recent requests --- */}
      {recent.length > 0 && (
        <section>
          <SectionTitle>{t("coedit.recent")}</SectionTitle>
          <div className="flex flex-wrap gap-1.5">
            {recent.map((text) => (
              <button
                key={text}
                type="button"
                title={text}
                onClick={() => setRequest(text)}
                className="h-control-sm max-w-full truncate rounded-full bg-hover px-3 text-2xs text-muted transition-colors duration-fast hover:text-fg"
              >
                {text}
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/** The before/after, as the server formatted it. */
function DiffList({ diff }: { diff: PlanDiff }) {
  const t = useT();
  if (diff.entries.length === 0) {
    return <p className="text-2xs text-faint">{t("coedit.noChange")}</p>;
  }
  return (
    <dl className="flex flex-col gap-1.5">
      {diff.entries.map((entry, index) => (
        <div key={index} className="flex items-baseline justify-between gap-3">
          <dt className="w-[40%] shrink-0 truncate text-2xs text-muted">{entry.label}</dt>
          <dd className="flex min-w-0 items-baseline gap-1.5 font-mono text-2xs tabular-nums">
            <span className="truncate text-faint line-through">{entry.before}</span>
            <span className="text-faint" aria-hidden>
              →
            </span>
            <span className="truncate text-fg">{entry.after}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function VersionRow({ version }: { version: EditVersion }) {
  const t = useT();
  const label =
    version.applied.length > 0
      ? version.applied.join(", ")
      : t(`coedit.origin.${version.origin}` as never);
  return (
    <li
      className={`flex items-baseline gap-2 rounded-md px-2 py-1 text-2xs ${
        version.is_current ? "bg-accent-soft text-fg" : "text-muted"
      }`}
    >
      <span className="shrink-0 font-mono tabular-nums text-faint">v{version.version}</span>
      <span className="min-w-0 flex-1 truncate" title={label}>
        {label}
      </span>
      {version.render_status === "ready" && (
        <span className="shrink-0 text-success" title={t("coedit.rendered")} aria-hidden>
          <Glyph name="check" size={10} />
        </span>
      )}
    </li>
  );
}

function summarise(entry: { label: string; before: string; after: string }): string {
  return `${entry.label}: ${entry.before} → ${entry.after}`;
}

/**
 * A failure, in the words of what went wrong.
 *
 * Every branch names a specific, actionable state -- "no provider is
 * configured" is a different problem from "the model could not express that",
 * and a panel that says "something went wrong" for both is a panel nobody can
 * act on.
 */
function failureMessage(preview: CoEditPreview, t: ReturnType<typeof useT>): string {
  switch (preview.failure) {
    case "provider_disabled":
    case "not_understood":
      return t("coedit.fail.notUnderstood");
    case "provider_unavailable":
    case "provider_error":
      return t("coedit.fail.provider");
    case "unreadable":
    case "no_usable_operations":
      return t("coedit.fail.unreadable");
    case "empty_request":
      return t("coedit.fail.empty");
    default:
      return preview.violations[0]?.message ?? t("coedit.fail.rejected");
  }
}
