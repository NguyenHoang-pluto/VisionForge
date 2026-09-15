"use client";

import { useT, LANGUAGES } from "@/lib/i18n";
import {
  ACCENTS,
  usePreferences,
  type Accent,
  type Density,
  type Language,
  type Theme,
} from "@/stores/preferences-store";
import { Button, Dialog, Glyph, SectionTitle, SegmentedControl } from "@/components/ui";

/**
 * Preferences.
 *
 * Four settings, applied live. There is no Apply button and no Cancel, because
 * every one of these is instantly reversible and instantly visible -- the
 * dialog is a preview of itself. A confirmation step would only add a decision
 * to something the user can simply look at.
 *
 * All four are stored in this browser and nowhere else. The panel says so,
 * because a settings screen that does not tell you where your settings went is
 * asking to be distrusted.
 */

const ACCENT_LABEL: Record<Accent, "prefs.accent.azure" | "prefs.accent.violet" | "prefs.accent.teal" | "prefs.accent.amber" | "prefs.accent.crimson"> = {
  azure: "prefs.accent.azure",
  violet: "prefs.accent.violet",
  teal: "prefs.accent.teal",
  amber: "prefs.accent.amber",
  crimson: "prefs.accent.crimson",
};

function Setting({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-1.5">
      <SectionTitle>{label}</SectionTitle>
      {children}
      <p className="text-2xs leading-snug text-dim">{hint}</p>
    </section>
  );
}

export function PreferencesDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useT();

  const language = usePreferences((s) => s.language);
  const theme = usePreferences((s) => s.theme);
  const accent = usePreferences((s) => s.accent);
  const density = usePreferences((s) => s.density);
  const setLanguage = usePreferences((s) => s.setLanguage);
  const setTheme = usePreferences((s) => s.setTheme);
  const setAccent = usePreferences((s) => s.setAccent);
  const setDensity = usePreferences((s) => s.setDensity);
  const reset = usePreferences((s) => s.reset);

  return (
    <Dialog open={open} onClose={onClose} title={t("prefs.title")}>
      <div className="flex flex-col gap-panel-gap">
        <Setting label={t("prefs.language")} hint={t("prefs.language.hint")}>
          <SegmentedControl<Language>
            label={t("prefs.language")}
            value={language}
            onChange={setLanguage}
            className="w-full [&>button]:flex-1"
            options={LANGUAGES.map((item) => ({ value: item.value, label: item.label }))}
          />
        </Setting>

        <Setting label={t("prefs.theme")} hint={t("prefs.theme.hint")}>
          <SegmentedControl<Theme>
            label={t("prefs.theme")}
            value={theme}
            onChange={setTheme}
            className="w-full [&>button]:flex-1"
            options={[
              { value: "dark", label: t("prefs.theme.dark") },
              { value: "light", label: t("prefs.theme.light") },
            ]}
          />
        </Setting>

        <Setting label={t("prefs.accent")} hint={t("prefs.accent.hint")}>
          {/*
            Swatches rather than a dropdown: the thing being chosen is a colour,
            and a list of colour *names* makes the user try each one to find out
            what it looks like. The tick, not the ring alone, is what marks the
            current choice -- an accent ring around an accent swatch is exactly
            the state that disappears for anyone who cannot see the hue.
          */}
          <div role="radiogroup" aria-label={t("prefs.accent")} className="flex gap-1.5">
            {ACCENTS.map((option) => {
              const selected = option === accent;
              return (
                <button
                  key={option}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  title={t(ACCENT_LABEL[option])}
                  onClick={() => setAccent(option)}
                  data-accent={option}
                  className={`flex h-7 flex-1 items-center justify-center rounded border transition-colors ${
                    selected ? "border-fg" : "border-line-strong hover:border-line-strong"
                  }`}
                  style={{ background: "rgb(var(--accent))" }}
                >
                  <span className="sr-only">{t(ACCENT_LABEL[option])}</span>
                  {selected && (
                    <span className="text-accent-fg">
                      <Glyph name="check" size={12} />
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </Setting>

        <Setting label={t("prefs.density")} hint={t("prefs.density.hint")}>
          <SegmentedControl<Density>
            label={t("prefs.density")}
            value={density}
            onChange={setDensity}
            className="w-full [&>button]:flex-1"
            options={[
              { value: "comfortable", label: t("prefs.density.comfortable") },
              { value: "compact", label: t("prefs.density.compact") },
            ]}
          />
        </Setting>

        <footer className="flex items-center gap-2 border-t border-line pt-2">
          <p className="min-w-0 flex-1 text-2xs leading-snug text-dim">{t("prefs.stored")}</p>
          <Button size="sm" onClick={reset}>
            {t("prefs.reset")}
          </Button>
          <Button size="sm" tone="primary" onClick={onClose}>
            {t("prefs.done")}
          </Button>
        </footer>
      </div>
    </Dialog>
  );
}
