"use client";

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { I18nProvider } from "@/lib/i18n";
import { usePreferences } from "@/stores/preferences-store";

/**
 * `useLayoutEffect` on the client, `useEffect` on the server.
 *
 * The distinction matters here rather than being boilerplate: rehydrating
 * preferences has to happen after React has matched the server's HTML but
 * before the browser paints, and a layout effect is the only hook that sits in
 * that gap. React warns if one is called during SSR, hence the swap.
 */
const useIsomorphicLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

/**
 * Puts stored preferences into effect.
 *
 * Two jobs, in this order:
 *
 *   1. Rehydrate the store from localStorage, before paint. The colours are
 *      already correct — the bootstrap script in <head> set them long before
 *      this — but the language lives in React and has to be corrected here.
 *   2. Keep <html> in step from then on, so that every subsequent change is one
 *      attribute write and a repaint, with no component re-rendering because a
 *      colour changed.
 */
export function PreferencesProvider({ children }: { children: ReactNode }) {
  const theme = usePreferences((s) => s.theme);
  const accent = usePreferences((s) => s.accent);
  const density = usePreferences((s) => s.density);
  const language = usePreferences((s) => s.language);

  // Rendered once with the server's values, then with the stored ones. Tracked
  // so that nothing downstream has to guess whether storage has been read.
  const [hydrated, setHydrated] = useState(false);
  const started = useRef(false);

  useIsomorphicLayoutEffect(() => {
    if (started.current) return;
    started.current = true;
    void Promise.resolve(usePreferences.persist.rehydrate()).finally(() => setHydrated(true));
  }, []);

  useIsomorphicLayoutEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.dataset.accent = accent;
    root.dataset.density = density;
  }, [theme, accent, density]);

  useEffect(() => {
    // Not a layout effect: `lang` is read by assistive technology and the
    // spellchecker, neither of which is waiting on this frame, and writing it
    // during hydration would fight React over an attribute it also renders.
    document.documentElement.lang = language;
  }, [language]);

  return (
    <div data-preferences-hydrated={hydrated || undefined} className="contents">
      <I18nProvider>{children}</I18nProvider>
    </div>
  );
}
