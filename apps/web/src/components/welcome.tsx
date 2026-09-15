"use client";

import { useT } from "@/lib/i18n";
import type { Project } from "@/lib/api";
import { relativeTime } from "@/lib/format";
import { BrandMark, Button, ErrorNote, Glyph, type GlyphName } from "@/components/ui";

/**
 * What the application shows before a project is open.
 *
 * The previous version of this screen was a sentence in the middle of an
 * otherwise black 1920×1080 viewport, which reads as a failure state whether or
 * not anything has failed. This is the same information laid out as a starting
 * point: what this is, what the three steps are, and the actual list of
 * projects to click.
 *
 * Deliberately restrained. No hero, no illustration, no gradient. A tool's
 * front door should look like the tool, not like its marketing site — so the
 * type, the hairlines and the spacing are the ones used everywhere else, only
 * with more room around them.
 */

const STEPS: { icon: GlyphName; title: "welcome.step1.title" | "welcome.step2.title" | "welcome.step3.title"; body: "welcome.step1.body" | "welcome.step2.body" | "welcome.step3.body" }[] = [
  { icon: "folder", title: "welcome.step1.title", body: "welcome.step1.body" },
  { icon: "film", title: "welcome.step2.title", body: "welcome.step2.body" },
  { icon: "split", title: "welcome.step3.title", body: "welcome.step3.body" },
];

export function Welcome({
  projects,
  loading,
  unreachable,
  onOpen,
  onCreate,
  onRetry,
}: {
  projects: Project[];
  loading: boolean;
  unreachable: boolean;
  onOpen: (projectId: string) => void;
  onCreate: () => void;
  onRetry: () => void;
}) {
  const t = useT();

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto bg-ground p-8">
      <div className="w-full max-w-[680px]">
        {/* ---- what this is ---- */}
        <header className="flex items-center gap-2.5">
          <BrandMark size={22} />
          <div className="min-w-0">
            <h1 className="text-xl font-semibold tracking-tight text-fg">{t("app.name")}</h1>
            <p className="text-xs text-faint">{t("app.tagline")}</p>
          </div>
        </header>

        <p className="mt-4 max-w-[58ch] text-sm leading-relaxed text-muted">{t("welcome.lead")}</p>

        {/* ---- the three steps ---- */}
        <ol className="mt-5 grid grid-cols-3 gap-px overflow-hidden rounded border border-subtle bg-subtle">
          {STEPS.map((step, index) => (
            <li key={step.title} className="flex flex-col gap-1.5 bg-surface p-panel">
              <span className="flex items-center gap-1.5 text-faint">
                <Glyph name={step.icon} size={13} />
                <span className="font-mono text-2xs tabular-nums">{index + 1}</span>
              </span>
              <h2 className="text-xs font-semibold text-fg">{t(step.title)}</h2>
              <p className="text-2xs leading-relaxed text-faint">{t(step.body)}</p>
            </li>
          ))}
        </ol>

        {/* ---- the projects themselves ---- */}
        <section className="mt-5">
          <header className="flex items-baseline justify-between gap-2 border-b border-strong pb-1">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
              {t("welcome.recent")}
            </h2>
            <Button size="sm" tone="primary" onClick={onCreate}>
              <Glyph name="plus" size={10} />
              {t("welcome.create")}
            </Button>
          </header>

          {unreachable ? (
            <div className="mt-2">
              <ErrorNote hint={t("welcome.unreachableHint")}>{t("welcome.unreachable")}</ErrorNote>
              <div className="mt-2">
                <Button size="sm" onClick={onRetry}>
                  <Glyph name="refresh" size={10} />
                  {t("welcome.retry")}
                </Button>
              </div>
            </div>
          ) : loading ? (
            <p className="py-6 text-center text-xs text-faint">{t("welcome.loading")}</p>
          ) : projects.length === 0 ? (
            <p className="py-6 text-center text-xs text-faint">{t("welcome.empty")}</p>
          ) : (
            <ul className="mt-1 max-h-[38vh] overflow-y-auto">
              {projects.map((project) => (
                <li key={project.id}>
                  <button
                    type="button"
                    onClick={() => onOpen(project.id)}
                    className="group flex w-full items-center gap-3 border-b border-subtle/60 px-1 py-1.5 text-left transition-colors last:border-b-0 hover:bg-accent-soft"
                  >
                    <span className="min-w-0 flex-1 truncate text-xs text-fg">{project.title}</span>
                    <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
                      {t.plural("top.project.mediaCount", project.media_count)}
                    </span>
                    <span className="w-16 shrink-0 text-right font-mono text-2xs text-faint">
                      {relativeTime(project.created_at)}
                    </span>
                    <span className="shrink-0 text-faint transition-colors group-hover:text-accent-strong">
                      <Glyph name="chevron-right" size={12} />
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
