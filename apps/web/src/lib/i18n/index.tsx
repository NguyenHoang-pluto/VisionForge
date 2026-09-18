"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";

import { en, type MessageKey } from "@/lib/i18n/messages.en";
import { vi } from "@/lib/i18n/messages.vi";
import { usePreferences, type Language } from "@/stores/preferences-store";

export type { MessageKey } from "@/lib/i18n/messages.en";

/**
 * Translation.
 *
 * Small on purpose. The application needs three things from an i18n layer —
 * look a key up, put a number into a sentence, and pick between one and many —
 * and every library that does those three also brings a message compiler, an
 * ICU parser and a plugin system. Forty lines here costs less than the
 * dependency and is easier to read than the dependency's docs.
 *
 * The important property is not the size, it is that component code contains
 * no English. A component asks for `media.empty.title`; the fact that there are
 * two languages, or five, is not something it can see.
 */

/**
 * The languages on offer.
 *
 * Labels are endonyms and are deliberately *not* translated: someone who has
 * landed in the wrong language needs to recognise their own, and "Vietnamese"
 * spelled in English is no help to the person looking for "Tiếng Việt".
 */
export const LANGUAGES: { value: Language; label: string }[] = [
  { value: "en", label: "English" },
  { value: "vi", label: "Tiếng Việt" },
];

const DICTIONARIES: Record<Language, Record<MessageKey, string>> = { en, vi };

/** A key whose `.one` / `.other` pair can be chosen between by count. */
export type PluralKey = {
  [K in MessageKey]: K extends `${infer Base}.one` ? Base : never;
}[MessageKey];

type Vars = Record<string, string | number>;

export interface Translate {
  /** Look up a message, filling any `{name}` slots from `vars`. */
  (key: MessageKey, vars?: Vars): string;
  /**
   * Pick between `<base>.one` and `<base>.other` by `count`, which is also
   * available to the message as `{count}`.
   *
   * English is the only language here that inflects, but the call site should
   * not have to know that — it says "this is a count", and each dictionary
   * decides for itself whether that changes the words.
   */
  plural(base: PluralKey, count: number, vars?: Vars): string;
  /** The active language, for `Intl` formatting and the `lang` attribute. */
  lang: Language;
}

const SLOT = /\{(\w+)\}/g;

function fill(template: string, vars?: Vars): string {
  if (!vars) return template;
  return template.replace(SLOT, (match, name: string) =>
    name in vars ? String(vars[name]) : match,
  );
}

function build(lang: Language): Translate {
  const dictionary = DICTIONARIES[lang] ?? en;

  const translate = ((key: MessageKey, vars?: Vars) =>
    fill(dictionary[key] ?? en[key] ?? key, vars)) as Translate;

  translate.plural = (base, count, vars) =>
    translate(`${base}.${count === 1 ? "one" : "other"}` as MessageKey, {
      count,
      ...vars,
    });

  translate.lang = lang;
  return translate;
}

/** English at module scope, so a component rendered outside the provider — a
    test, a Storybook-less snapshot — still produces words rather than keys. */
const I18nContext = createContext<Translate>(build("en"));

export function I18nProvider({ children }: { children: ReactNode }) {
  const language = usePreferences((s) => s.language);
  const translate = useMemo(() => build(language), [language]);
  return <I18nContext.Provider value={translate}>{children}</I18nContext.Provider>;
}

export function useT(): Translate {
  return useContext(I18nContext);
}
