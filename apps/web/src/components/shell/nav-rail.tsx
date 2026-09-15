"use client";

import { useT, type MessageKey } from "@/lib/i18n";
import { LANGUAGES } from "@/lib/i18n";
import { usePreferences } from "@/stores/preferences-store";
import { useEditorStore, type AppView } from "@/stores/editor-store";
import { BrandMark, Glyph, type GlyphName } from "@/components/ui";

/**
 * The navigation rail.
 *
 * The structural half of the redesign. The application used to be one screen
 * with panels toggled on and off, which is why it read as a dashboard: there
 * was nowhere to *be*, only things to show and hide. The rail names six places
 * and keeps exactly one of them current.
 *
 * It is a rail and not a sidebar on purpose -- 56px of chrome, not 220. An
 * editor's horizontal space belongs to the footage; navigation that permanently
 * occupies a sixth of it is navigation that is being used once an hour and paid
 * for continuously. The labels appear on hover, in a tooltip the platform draws.
 *
 * Grouped the way the work is: the two project-independent destinations at the
 * top, the four that need an open project below them, and the global chrome at
 * the bottom where a desktop application keeps it.
 */

interface NavItem {
  view: AppView;
  icon: GlyphName;
  label: MessageKey;
  /** Needs an open project to mean anything. */
  scoped?: boolean;
}

const PRIMARY: NavItem[] = [{ view: "home", icon: "home", label: "nav.home" }];

const PROJECT: NavItem[] = [
  { view: "editor", icon: "film", label: "nav.editor", scoped: true },
  { view: "assets", icon: "layers", label: "nav.assets", scoped: true },
  { view: "audio", icon: "beat", label: "nav.audio", scoped: true },
  { view: "export", icon: "export", label: "nav.exports", scoped: true },
];

function RailButton({
  item,
  current,
  disabled,
  onSelect,
}: {
  item: NavItem;
  current: boolean;
  disabled: boolean;
  onSelect: () => void;
}) {
  const t = useT();
  const label = t(item.label);

  return (
    <button
      type="button"
      title={disabled ? t("nav.needsProject", { view: label }) : label}
      aria-label={label}
      aria-current={current ? "page" : undefined}
      disabled={disabled}
      onClick={onSelect}
      className={`group relative flex h-[42px] w-[42px] items-center justify-center rounded-lg transition-[background-color,color,transform] duration-fast active:scale-[0.94] disabled:pointer-events-none disabled:opacity-30 ${
        current
          ? "bg-accent-soft text-accent-strong"
          : "text-muted hover:bg-hover hover:text-fg"
      }`}
    >
      <Glyph name={item.icon} size={17} />

      {/*
       * The current marker is a bar on the rail's edge rather than a border
       * around the button: it reads from the corner of the eye, and it does not
       * add a fifth rectangle to a column that already has four.
       */}
      {current && (
        <span
          aria-hidden
          className="absolute -left-[11px] h-5 w-[3px] rounded-full bg-accent"
        />
      )}
    </button>
  );
}

export function NavRail({
  onShowShortcuts,
  onShowPreferences,
}: {
  onShowShortcuts: () => void;
  onShowPreferences: () => void;
}) {
  const t = useT();
  const view = useEditorStore((s) => s.view);
  const setView = useEditorStore((s) => s.setView);
  const projectId = useEditorStore((s) => s.projectId);

  const theme = usePreferences((s) => s.theme);
  const setTheme = usePreferences((s) => s.setTheme);
  const language = usePreferences((s) => s.language);
  const setLanguage = usePreferences((s) => s.setLanguage);

  /** The other language, because with two the control is a switch, not a list. */
  const other = LANGUAGES.find((item) => item.value !== language) ?? LANGUAGES[0];

  function render(items: NavItem[]) {
    return items.map((item) => (
      <RailButton
        key={item.view}
        item={item}
        current={view === item.view}
        disabled={Boolean(item.scoped) && !projectId}
        onSelect={() => setView(item.view)}
      />
    ));
  }

  return (
    <nav
      aria-label={t("nav.title")}
      className="flex w-rail shrink-0 flex-col items-center gap-1 bg-ground px-2 py-3"
    >
      <button
        type="button"
        title={t("app.name")}
        aria-label={t("nav.home")}
        onClick={() => setView("home")}
        className="mb-2 flex h-[42px] w-[42px] items-center justify-center rounded-lg transition-transform duration-fast hover:scale-105 active:scale-95"
      >
        <BrandMark size={23} />
      </button>

      {render(PRIMARY)}

      {/* The divider is the one line in the rail, and it separates "anywhere"
          from "inside a project" -- a real boundary, not decoration. */}
      <span aria-hidden className="my-2 h-px w-6 bg-subtle" />

      {render(PROJECT)}

      <div className="flex-1" />

      {/* ---- global chrome ---- */}
      <button
        type="button"
        title={t(theme === "dark" ? "prefs.theme.light" : "prefs.theme.dark")}
        aria-label={t(theme === "dark" ? "prefs.theme.light" : "prefs.theme.dark")}
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="flex h-[38px] w-[38px] items-center justify-center rounded-lg text-muted transition-[background-color,color,transform] duration-fast hover:bg-hover hover:text-fg active:scale-[0.94]"
      >
        <Glyph name={theme === "dark" ? "sun" : "moon"} size={16} />
      </button>

      <button
        type="button"
        title={t("prefs.language")}
        aria-label={t("prefs.language")}
        onClick={() => setLanguage(other.value)}
        className="flex h-[38px] w-[38px] items-center justify-center rounded-lg text-2xs font-semibold uppercase tracking-wide text-muted transition-[background-color,color,transform] duration-fast hover:bg-hover hover:text-fg active:scale-[0.94]"
      >
        {language}
      </button>

      <button
        type="button"
        title={t("shortcuts.title")}
        aria-label={t("shortcuts.title")}
        onClick={onShowShortcuts}
        className="flex h-[38px] w-[38px] items-center justify-center rounded-lg font-mono text-sm text-muted transition-[background-color,color,transform] duration-fast hover:bg-hover hover:text-fg active:scale-[0.94]"
      >
        ?
      </button>

      <RailButton
        item={{ view: "settings", icon: "settings", label: "nav.settings" }}
        current={view === "settings"}
        disabled={false}
        onSelect={onShowPreferences}
      />
    </nav>
  );
}
