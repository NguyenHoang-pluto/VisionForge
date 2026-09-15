"use client";

import type { Project } from "@/lib/api";
import { relativeTime, shortDuration } from "@/lib/format";
import { useT } from "@/lib/i18n";
import { useThumbnailUrl } from "@/lib/media-urls";
import {
  aspectLabel,
  formatLabel,
  useProjectSummaries,
  type ProjectSummary,
} from "@/lib/project-summary";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  BrandMark,
  Button,
  Card,
  EmptyState,
  ErrorNote,
  Glyph,
  Spinner,
} from "@/components/ui";

/**
 * Project Home.
 *
 * What the application shows when nothing is open, and -- this is the part that
 * changed -- somewhere worth coming back to when something is. The previous
 * screen was a column of text in the middle of a black viewport with a list of
 * project titles: correct, and indistinguishable from an error page.
 *
 * Everything on a card is derived from that project's own media listing (see
 * `project-summary.ts`). Nothing is fabricated: a project with no ready footage
 * shows a placeholder frame and no duration, because the honest answer to "how
 * long is this project" before anything has been ingested is "there is nothing
 * to measure yet".
 */

function CoverFrame({ summary }: { summary: ProjectSummary }) {
  const t = useT();
  const thumbnail = useThumbnailUrl(summary.project.id, summary.cover);
  const url = thumbnail.data?.url;

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-sunken">
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt=""
          loading="lazy"
          decoding="async"
          className="h-full w-full object-cover transition-transform duration-base group-hover:scale-[1.03]"
        />
      ) : (
        <span
          className="flex h-full w-full items-center justify-center text-faint/50"
          aria-hidden
        >
          <Glyph name={summary.loaded ? "film" : "clock"} size={24} />
        </span>
      )}

      {/* The one place a gradient is defensible: it is a scrim that makes white
          text legible over an unknown frame, not decoration. */}
      {summary.footageMs !== null && (
        <span className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/70 to-transparent px-2.5 pb-2 pt-6">
          <span className="font-mono text-2xs tabular-nums text-white/90">
            {shortDuration(summary.footageMs)}
          </span>
          {summary.audio > 0 && (
            <span className="text-white/75" title={t("media.filter.audio")} aria-hidden>
              <Glyph name="audio" size={12} />
            </span>
          )}
        </span>
      )}
    </div>
  );
}

function ProjectTile({
  summary,
  onOpen,
}: {
  summary: ProjectSummary;
  onOpen: (projectId: string) => void;
}) {
  const t = useT();
  const { project } = summary;
  const format = formatLabel(summary.width, summary.height);
  const aspect = aspectLabel(summary.width, summary.height);

  return (
    <Card
      interactive
      className="group cursor-pointer p-2.5"
      role="button"
      tabIndex={0}
      onClick={() => onOpen(project.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(project.id);
        }
      }}
      aria-label={project.title}
    >
      <CoverFrame summary={summary} />

      <div className="mt-2.5 min-w-0 px-0.5 pb-0.5">
        <h3 className="truncate text-sm font-semibold tracking-tight text-fg" title={project.title}>
          {project.title}
        </h3>

        <p className="mt-1 flex items-center gap-1.5 truncate text-2xs text-faint">
          <span>{t.plural("top.project.mediaCount", project.media_count)}</span>
          {format && (
            <>
              <span aria-hidden>·</span>
              <span className="font-mono tabular-nums">{format}</span>
            </>
          )}
          {aspect && (
            <>
              <span aria-hidden>·</span>
              <span className="font-mono tabular-nums">{aspect}</span>
            </>
          )}
        </p>

        <p
          className="mt-1.5 truncate text-2xs text-faint"
          title={new Date(summary.activeAt).toLocaleString(t.lang)}
        >
          {t("home.active", { when: relativeTime(summary.activeAt, t.lang) })}
        </p>
      </div>
    </Card>
  );
}

export function ProjectHome({
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
  const setView = useEditorStore((s) => s.setView);
  const summaries = useProjectSummaries(projects);

  /** Import needs somewhere to import *into*. */
  const newest = projects[0] ?? null;

  function importInto(project: Project) {
    onOpen(project.id);
    setView("assets");
  }

  return (
    <div className="vf-view min-h-0 flex-1 overflow-y-auto bg-ground">
      <div className="mx-auto w-full max-w-[1180px] px-8 py-10">
        {/* ------------------------------------------------------ masthead */}
        <header className="flex flex-wrap items-end justify-between gap-6">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <BrandMark size={26} />
              <h1 className="text-2xl font-semibold tracking-tight text-fg">
                {t("app.name")}
              </h1>
            </div>
            <p className="mt-2 max-w-[54ch] text-sm leading-relaxed text-muted">
              {t("home.lead")}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button
              size="lg"
              tone="quiet"
              disabled={!newest}
              title={
                newest ? t("home.importInto", { project: newest.title }) : t("home.importNeeds")
              }
              onClick={() => newest && importInto(newest)}
            >
              <Glyph name="upload" size={14} />
              {t("home.import")}
            </Button>
            <Button size="lg" tone="primary" onClick={onCreate}>
              <Glyph name="plus" size={13} />
              {t("home.create")}
            </Button>
          </div>
        </header>

        {/* ------------------------------------------------------- projects */}
        <section className="mt-9">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="text-xs font-semibold uppercase tracking-[0.08em] text-faint">
              {t("home.recent")}
            </h2>
            {projects.length > 0 && (
              <Badge>{t.plural("home.projectCount", projects.length)}</Badge>
            )}
          </div>

          <div className="mt-4">
            {unreachable ? (
              <div className="max-w-[60ch]">
                <ErrorNote hint={t("welcome.unreachableHint")}>
                  {t("welcome.unreachable")}
                </ErrorNote>
                <div className="mt-3">
                  <Button onClick={onRetry}>
                    <Glyph name="refresh" size={12} />
                    {t("welcome.retry")}
                  </Button>
                </div>
              </div>
            ) : loading ? (
              <div className="flex items-center gap-2.5 py-16 text-xs text-faint">
                <Spinner />
                {t("welcome.loading")}
              </div>
            ) : projects.length === 0 ? (
              <Card className="py-4">
                <EmptyState
                  icon="folder"
                  title={t("home.empty.title")}
                  action={
                    <Button tone="primary" onClick={onCreate}>
                      <Glyph name="plus" size={12} />
                      {t("home.create")}
                    </Button>
                  }
                >
                  {t("home.empty.body")}
                </EmptyState>
              </Card>
            ) : (
              <ul className="grid grid-cols-2 gap-4 lg:grid-cols-3 2xl:grid-cols-4">
                {summaries.map((summary) => (
                  <li key={summary.project.id} className="min-w-0">
                    <ProjectTile summary={summary} onOpen={onOpen} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        {/* --------------------------------------------------- how it works */}
        <section className="mt-12">
          <h2 className="text-xs font-semibold uppercase tracking-[0.08em] text-faint">
            {t("home.how")}
          </h2>

          <ol className="mt-4 grid grid-cols-3 gap-4">
            {(
              [
                { icon: "upload", title: "welcome.step1.title", body: "welcome.step1.body" },
                { icon: "analyse", title: "welcome.step2.title", body: "welcome.step2.body" },
                { icon: "wand", title: "welcome.step3.title", body: "welcome.step3.body" },
              ] as const
            ).map((step, index) => (
              <li key={step.title}>
                <Card className="h-full p-4">
                  <div className="flex items-center gap-2.5">
                    <span
                      className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-soft text-accent-strong"
                      aria-hidden
                    >
                      <Glyph name={step.icon} size={15} />
                    </span>
                    <span className="font-mono text-2xs tabular-nums text-faint">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <h3 className="mt-3 text-xs font-semibold text-fg">{t(step.title)}</h3>
                  <p className="mt-1.5 text-2xs leading-relaxed text-faint">{t(step.body)}</p>
                </Card>
              </li>
            ))}
          </ol>
        </section>

        <p className="mt-10 text-2xs text-faint">{t("home.local")}</p>
      </div>
    </div>
  );
}
