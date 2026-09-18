import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * Interface preferences.
 *
 * Not editor state and not server state: this is how one person likes to look
 * at the application, so it lives in their browser and nowhere else. Nothing
 * here is ever sent to the API, and a second machine is entitled to a different
 * answer.
 *
 * All four settings resolve to data attributes on <html>, which is what makes
 * them free at runtime: the theme, the accent and the density are pure CSS
 * cascade, so switching one repaints rather than re-rendering the tree. Only
 * the language actually passes through React, because only the language changes
 * the text nodes.
 */

export type Language = "en" | "vi";
export type Theme = "dark" | "light";
export type Accent = "azure" | "violet" | "teal" | "amber" | "crimson";
export type Density = "comfortable" | "compact";

export const ACCENTS: Accent[] = ["azure", "violet", "teal", "amber", "crimson"];

export interface Preferences {
  language: Language;
  theme: Theme;
  accent: Accent;
  density: Density;
}

export const DEFAULT_PREFERENCES: Preferences = {
  language: "en",
  theme: "dark",
  accent: "azure",
  density: "comfortable",
};

/** Shared with the inline bootstrap script in the root layout. */
export const PREFERENCES_KEY = "visionforge.preferences";

interface PreferencesState extends Preferences {
  setLanguage: (language: Language) => void;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  setAccent: (accent: Accent) => void;
  setDensity: (density: Density) => void;
  reset: () => void;
}

export const usePreferences = create<PreferencesState>()(
  persist(
    (set) => ({
      ...DEFAULT_PREFERENCES,
      setLanguage: (language) => set({ language }),
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((state) => ({ theme: state.theme === "dark" ? "light" : "dark" })),
      setAccent: (accent) => set({ accent }),
      setDensity: (density) => set({ density }),
      reset: () => set({ ...DEFAULT_PREFERENCES }),
    }),
    {
      name: PREFERENCES_KEY,
      storage: createJSONStorage(() => localStorage),
      version: 1,
      /*
       * Rehydration is deferred to the provider.
       *
       * Left automatic, zustand reads localStorage at module load — before
       * React hydrates — and the first client render would disagree with the
       * server-rendered HTML for anyone who is not on the defaults. Reading it
       * from a layout effect instead means the two agree, and the corrected
       * values are committed before the browser paints, so there is no flash
       * either. The colours never flicker at all: the bootstrap script has
       * already put them on <html> before React exists.
       */
      skipHydration: true,
      partialize: ({ language, theme, accent, density }) => ({
        language,
        theme,
        accent,
        density,
      }),
    },
  ),
);

/**
 * The bootstrap script, inlined into <head>.
 *
 * Runs before first paint and before React, so the application's very first
 * frame is already in the right theme, accent and density. Without it the page
 * paints dark, hydrates, and then flips to light in front of anyone who chose
 * light — the single most obvious "this is a web page" tell there is.
 *
 * Deliberately defensive: private mode throws on localStorage, and a
 * half-written value must degrade to the defaults rather than to a blank page.
 */
export const PREFERENCES_BOOTSTRAP = `
(function(){
  try {
    var raw = localStorage.getItem(${JSON.stringify(PREFERENCES_KEY)});
    var p = raw ? (JSON.parse(raw) || {}).state || {} : {};
    var d = document.documentElement;
    d.dataset.theme = p.theme === "light" ? "light" : "dark";
    d.dataset.accent = ${JSON.stringify(ACCENTS)}.indexOf(p.accent) >= 0 ? p.accent : "azure";
    d.dataset.density = p.density === "compact" ? "compact" : "comfortable";
  } catch (e) {
    var el = document.documentElement;
    el.dataset.theme = "dark";
    el.dataset.accent = "azure";
    el.dataset.density = "comfortable";
  }
})();
`.trim();
