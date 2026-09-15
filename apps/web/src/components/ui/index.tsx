"use client";

import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

import { useT } from "@/lib/i18n";

/**
 * The workstation's component vocabulary.
 *
 * Every control in the application is one of these. They are small, dense and
 * quiet by design: a button in an editor is pressed a thousand times a day, and
 * anything that draws attention to itself at that frequency becomes noise.
 *
 * Nothing here spells a colour or a height. Colours come from the semantic
 * Tailwind names, which resolve to the theme's custom properties; heights come
 * from `h-control` and friends, which resolve to the density preference. That
 * is what makes theme, accent and density switchable at runtime without a
 * single component knowing they exist.
 *
 * Kept in one module rather than a file each. The set is small, the pieces are
 * short, and the alternative -- twenty files of twenty lines -- makes it harder
 * to see the whole vocabulary at once and easier to add a twenty-first variant
 * nobody notices is a duplicate.
 */

// ------------------------------------------------------------------ primitives
type ButtonTone = "default" | "primary" | "danger" | "ghost";
type ButtonSize = "sm" | "md";

const BUTTON_TONE: Record<ButtonTone, string> = {
  default:
    "border-line-strong bg-control text-fg hover:bg-control-hover hover:border-line-strong active:bg-control",
  /* The one emphatic fill in the application. `accent-strong` against
     `accent-fg` clears 7:1 in both themes, which a mid-accent fill does not. */
  primary:
    "border-accent-strong bg-accent-strong text-accent-fg hover:bg-accent hover:border-accent",
  danger: "border-line-strong bg-control text-danger hover:border-danger/70 hover:bg-danger/10",
  ghost: "border-transparent bg-transparent text-muted hover:bg-control hover:text-fg",
};

const BUTTON_SIZE: Record<ButtonSize, string> = {
  sm: "h-control-sm px-2 text-2xs",
  md: "h-control px-2.5 text-xs",
};

export function Button({
  tone = "default",
  size = "md",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: ButtonTone; size?: ButtonSize }) {
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded border font-medium transition-colors disabled:pointer-events-none disabled:opacity-35 ${BUTTON_TONE[tone]} ${BUTTON_SIZE[size]} ${className}`}
    />
  );
}

/**
 * A square button carrying a glyph.
 *
 * `label` is mandatory and becomes both the accessible name and the tooltip: an
 * icon-only control with no name is unusable with a screen reader and merely
 * cryptic with one.
 *
 * The active state is not colour alone. A pressed toggle gets the accent fill
 * *and* `aria-pressed`, so it is legible to a screen reader and to anyone who
 * cannot separate the accent from the surface behind it.
 */
export function IconButton({
  label,
  active,
  size = "md",
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  active?: boolean;
  size?: ButtonSize;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active}
      {...props}
      className={`inline-flex shrink-0 items-center justify-center rounded border transition-colors disabled:pointer-events-none disabled:opacity-35 ${
        size === "sm" ? "h-control-sm w-control-sm" : "h-control w-control"
      } ${
        active
          ? "border-accent/60 bg-accent-soft text-accent-strong"
          : "border-transparent text-muted hover:bg-control hover:text-fg"
      } ${className}`}
    >
      {children}
    </button>
  );
}

/** Radio semantics for a small closed set. */
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  label,
  className = "",
}: {
  value: T;
  options: { value: T; label: string; title?: string; disabled?: boolean }[];
  onChange: (value: T) => void;
  label: string;
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={`inline-flex h-control-sm shrink-0 overflow-hidden rounded border border-line-strong ${className}`}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={option.disabled}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={`h-full border-r border-line-strong px-2 text-2xs font-medium transition-colors last:border-r-0 disabled:cursor-not-allowed disabled:text-dim/50 ${
              selected
                ? "bg-accent-soft text-accent-strong"
                : "bg-control text-muted hover:bg-control-hover hover:text-fg"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

const CONTROL_CLASS =
  "h-control w-full rounded border border-line-strong bg-control px-1.5 text-xs text-fg transition-colors hover:border-line-strong focus:border-accent disabled:cursor-not-allowed disabled:opacity-40";

/** The id the enclosing `Field` labelled, if there is one. */
const FieldIdContext = createContext<string | null>(null);

function useFieldId(explicit?: string): string | undefined {
  const inherited = useContext(FieldIdContext);
  return explicit ?? inherited ?? undefined;
}

export function Select({
  className = "",
  children,
  id,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select id={useFieldId(id)} {...props} className={`${CONTROL_CLASS} ${className}`}>
      {children}
    </select>
  );
}

export function TextInput({
  className = "",
  id,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      id={useFieldId(id)}
      {...props}
      className={`${CONTROL_CLASS} placeholder:text-dim ${className}`}
    />
  );
}

export function NumberInput({
  className = "",
  id,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      type="number"
      id={useFieldId(id)}
      {...props}
      className={`${CONTROL_CLASS} tabular-nums ${className}`}
    />
  );
}

/**
 * A range input that shows how far along it is.
 *
 * The filled portion of the track is drawn by the stylesheet from `--fill`,
 * which only the caller's value can supply. Doing it this way rather than with
 * a second element behind the input keeps the whole control one hit target and
 * one focus stop.
 */
export function Slider({
  className = "",
  id,
  value,
  min = 0,
  max = 100,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, "type"> & {
  value: number;
  min?: number;
  max?: number;
}) {
  const span = Number(max) - Number(min);
  const fraction = span > 0 ? (value - Number(min)) / span : 0;
  return (
    <input
      type="range"
      id={useFieldId(id)}
      value={value}
      min={min}
      max={max}
      {...props}
      style={{
        ...props.style,
        ["--fill" as string]: `${Math.max(0, Math.min(1, fraction)) * 100}%`,
      }}
      className={`shrink-0 ${className}`}
    />
  );
}

/**
 * Label above control. The only form layout in the application.
 *
 * The label's `htmlFor` target is generated here and handed to the control
 * through context, so the two cannot drift apart and a caller cannot ship a
 * label pointing at nothing.
 */
export function Field({
  label,
  hint,
  children,
  className = "",
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  const id = useId();
  return (
    <div className={`flex min-w-0 flex-col gap-1 ${className}`}>
      <label
        htmlFor={id}
        title={hint}
        className="select-none truncate font-mono text-2xs uppercase tracking-wider text-dim"
      >
        {label}
      </label>
      <FieldIdContext.Provider value={id}>{children}</FieldIdContext.Provider>
    </div>
  );
}

// ---------------------------------------------------------------------- panels
export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`flex min-h-0 min-w-0 flex-col bg-panel ${className}`}>{children}</section>
  );
}

/** A panel's title strip. One height everywhere, so panels align across the app. */
export function PanelHeader({
  title,
  children,
  className = "",
}: {
  title: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <header
      className={`flex h-header shrink-0 items-center gap-2 border-b border-line bg-raised px-2 ${className}`}
    >
      <h2 className="select-none whitespace-nowrap font-mono text-2xs font-medium uppercase tracking-wider text-dim">
        {title}
      </h2>
      {children}
    </header>
  );
}

/**
 * A group of related controls in a toolbar.
 *
 * Grouping is the whole job of a workstation's top bar: fifteen loose buttons
 * are unreadable, five groups of three are a menu you can learn. The separator
 * is drawn by the group rather than sprinkled between buttons, so a group
 * cannot end up with two or none.
 */
export function ToolGroup({
  label,
  children,
  className = "",
}: {
  label?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className={`flex shrink-0 items-center gap-1 border-l border-line pl-2 first:border-l-0 first:pl-0 ${className}`}
    >
      {children}
    </div>
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  tabs,
  label,
}: {
  value: T;
  onChange: (value: T) => void;
  tabs: { value: T; label: string }[];
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex h-header shrink-0 border-b border-line">
      {tabs.map((tab) => {
        const selected = tab.value === value;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(tab.value)}
            className={`relative h-full flex-1 px-1.5 text-2xs font-medium uppercase tracking-wider transition-colors ${
              selected
                ? "bg-panel text-fg"
                : "bg-raised text-dim hover:bg-raised hover:text-muted"
            }`}
          >
            {tab.label}
            {/* The underline, not the fill, is what reads as "current" at a
                glance; the fill alone is too close to the panel beside it. */}
            {selected && (
              <span className="absolute inset-x-0 bottom-0 h-[2px] bg-accent" aria-hidden />
            )}
          </button>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------------ menu
/**
 * A menu button.
 *
 * Desktop applications keep their rarely-used verbs behind a menu rather than
 * in the toolbar, and the alternative -- a toolbar with every verb in it -- is
 * the thing that makes a web app look like a web app.
 *
 * Closes on Escape, on outside pointer-down and on choosing an item; arrow keys
 * walk the items. Written here rather than pulled in because the application
 * needs one of these and a popover library is forty kilobytes of positioning
 * engine for a panel that opens under a button.
 */
export function Menu({
  label,
  items,
  align = "start",
  children,
  className = "",
}: {
  label: string;
  items: { key: string; label: string; onSelect: () => void; disabled?: boolean }[];
  align?: "start" | "end";
  children: ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function onPointerDown(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
        (root.current?.firstElementChild as HTMLElement | null)?.focus();
      }
    }

    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    // Focus the first item, so the menu is usable without reaching for a mouse.
    list.current?.querySelector<HTMLElement>("[role=menuitem]:not([disabled])")?.focus();
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  function step(from: HTMLElement, delta: number) {
    const all = Array.from(
      list.current?.querySelectorAll<HTMLElement>("[role=menuitem]:not([disabled])") ?? [],
    );
    const next = all[(all.indexOf(from) + delta + all.length) % all.length];
    next?.focus();
  }

  return (
    <div ref={root} className={`relative shrink-0 ${className}`}>
      <button
        type="button"
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className={`inline-flex h-control items-center gap-1.5 rounded border px-1.5 transition-colors ${
          open
            ? "border-line-strong bg-control text-fg"
            : "border-transparent text-muted hover:bg-control hover:text-fg"
        }`}
      >
        {children}
      </button>

      {open && (
        <div
          ref={list}
          role="menu"
          aria-label={label}
          className={`absolute top-[calc(100%+3px)] z-50 min-w-[190px] rounded-lg border border-line-strong bg-raised py-1 shadow-menu ${
            align === "end" ? "right-0" : "left-0"
          }`}
        >
          {items.map((item) => (
            <button
              key={item.key}
              type="button"
              role="menuitem"
              disabled={item.disabled}
              onClick={() => {
                setOpen(false);
                item.onSelect();
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  step(event.currentTarget, 1);
                }
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  step(event.currentTarget, -1);
                }
              }}
              className="flex w-full items-center gap-2 px-2.5 py-1 text-left text-xs text-muted transition-colors hover:bg-accent-soft hover:text-fg focus:bg-accent-soft focus:text-fg disabled:pointer-events-none disabled:text-dim/50"
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------- indicators
type BadgeTone = "neutral" | "ok" | "warn" | "danger" | "info" | "accent";

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral: "border-line-strong bg-control text-dim",
  ok: "border-ok/40 bg-ok/10 text-ok",
  warn: "border-warn/40 bg-warn/10 text-warn",
  danger: "border-danger/40 bg-danger/10 text-danger",
  info: "border-info/40 bg-info/10 text-info",
  accent: "border-accent/50 bg-accent-soft text-accent-strong",
};

/**
 * A status word in a box.
 *
 * Sentence case, not upper. Upper-casing buys a little more "chrome" in English
 * and costs about a third more width, which is what pushes a status out of a
 * column in a 230px panel -- and Vietnamese, where every other vowel carries a
 * diacritic, is harder to read upper-cased, not easier.
 */
export function Badge({
  tone = "neutral",
  children,
  title,
}: {
  tone?: BadgeTone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex h-[15px] shrink-0 items-center whitespace-nowrap rounded-sm border px-1 font-mono text-2xs font-medium ${BADGE_TONE[tone]}`}
    >
      {children}
    </span>
  );
}

/**
 * A state indicator that is never colour alone.
 *
 * The dot is the fast read and the word is the reliable one. Anything that ships
 * only the dot is illegible to roughly one in twelve men.
 */
export function StatusDot({
  tone,
  children,
  title,
}: {
  tone: "ok" | "warn" | "danger" | "neutral";
  children: ReactNode;
  title?: string;
}) {
  const fill = {
    ok: "bg-ok",
    warn: "bg-warn",
    danger: "bg-danger",
    neutral: "bg-dim",
  }[tone];
  return (
    <span title={title} className="flex shrink-0 items-center gap-1.5">
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${fill}`} />
      {children}
    </span>
  );
}

/**
 * A labelled bar for a bounded measurement.
 *
 * Used for analysis values that have a meaningful range (sharpness against a
 * blur threshold, exposure against mid-grey). Values without a range stay as
 * numbers -- a bar with an arbitrary maximum implies a comparison that does not
 * exist.
 */
export function Meter({
  value,
  max,
  tone = "accent",
}: {
  value: number;
  max: number;
  tone?: "accent" | "ok" | "warn" | "danger";
}) {
  const fraction = Math.max(0, Math.min(1, max > 0 ? value / max : 0));
  const fill = {
    accent: "bg-accent",
    ok: "bg-ok",
    warn: "bg-warn",
    danger: "bg-danger",
  }[tone];
  return (
    <div
      className="h-[3px] w-full overflow-hidden rounded-sm bg-control"
      role="meter"
      aria-valuenow={Math.round(fraction * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className={`h-full ${fill}`} style={{ width: `${fraction * 100}%` }} />
    </div>
  );
}

export function ProgressBar({
  fraction,
  tone = "accent",
  label,
}: {
  fraction: number;
  tone?: "accent" | "ok" | "warn" | "danger";
  label?: string;
}) {
  const percent = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  const fill = { accent: "bg-accent", ok: "bg-ok", warn: "bg-warn", danger: "bg-danger" }[tone];
  return (
    <div
      className="h-[2px] w-full overflow-hidden rounded-sm bg-control"
      role="progressbar"
      aria-label={label}
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className={`h-full transition-[width] ${fill}`} style={{ width: `${percent}%` }} />
    </div>
  );
}

/**
 * One label/value line. The unit the inspector is built from.
 *
 * The label column is fixed rather than shrink-to-fit, so that a column of
 * these lines up down the panel instead of ragging with the length of each
 * value -- which is the difference between a readout and a list.
 */
export function Row({
  label,
  value,
  tone,
  title,
}: {
  label: string;
  value: ReactNode;
  tone?: "warn" | "ok" | "danger";
  title?: string;
}) {
  const valueClass =
    tone === "warn"
      ? "text-warn"
      : tone === "ok"
        ? "text-ok"
        : tone === "danger"
          ? "text-danger"
          : "text-fg";
  return (
    <div
      title={title}
      className="flex min-h-row items-baseline justify-between gap-3 border-b border-line/60 py-[3px] last:border-b-0"
    >
      <span className="w-[42%] shrink-0 truncate text-xs text-muted">{label}</span>
      <span className={`truncate text-right font-mono text-xs tabular-nums ${valueClass}`}>
        {value}
      </span>
    </div>
  );
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <header className="flex items-baseline justify-between gap-2 border-b border-line-strong pb-1">
      <h3 className="truncate text-xs font-semibold uppercase tracking-wider text-muted">
        {children}
      </h3>
      {aside}
    </header>
  );
}

/**
 * An empty panel that says what to do about it.
 *
 * Takes a title and a body rather than one blob of prose: "No media yet" read
 * in a quarter of a second is worth more than a paragraph that is read once and
 * then skipped forever.
 */
export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: GlyphName;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex h-full min-h-[90px] flex-col items-center justify-center gap-1.5 px-4 py-6 text-center">
      {icon && (
        <span className="mb-0.5 text-dim/60" aria-hidden>
          <Glyph name={icon} size={22} />
        </span>
      )}
      {title && <p className="text-xs font-semibold text-muted">{title}</p>}
      {children && <p className="max-w-[36ch] text-2xs leading-relaxed text-dim">{children}</p>}
      {action && <div className="mt-1.5">{action}</div>}
    </div>
  );
}

export function ErrorNote({
  children,
  hint,
}: {
  children: ReactNode;
  hint?: string | null;
}) {
  return (
    <p
      role="alert"
      className="rounded-sm border-l-2 border-danger bg-danger/10 px-2 py-1 text-2xs leading-snug text-danger"
    >
      {children}
      {hint && <span className="mt-0.5 block text-danger/80">{hint}</span>}
    </p>
  );
}

export function Divider({ vertical }: { vertical?: boolean }) {
  return vertical ? (
    <span className="mx-1 h-4 w-px shrink-0 bg-line-strong" aria-hidden />
  ) : (
    <span className="my-1 h-px w-full bg-line" aria-hidden />
  );
}

/** A key on a keyboard, drawn as one. */
export function KeyCap({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-[17px] min-w-[17px] items-center justify-center rounded-sm border border-line-strong bg-control px-1 font-mono text-2xs text-fg">
      {children}
    </kbd>
  );
}

// ---------------------------------------------------------------------- dialog
/**
 * A modal.
 *
 * Uses `<dialog>` so that focus trapping, Escape and the backdrop come from the
 * platform rather than from an effect that has to be maintained.
 */
export function Dialog({
  open,
  onClose,
  title,
  size = "md",
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  size?: "md" | "lg";
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const t = useT();

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => {
        // Clicking the backdrop closes; clicking the panel must not.
        if (event.target === ref.current) onClose();
      }}
      className={`rounded-lg border border-line-strong bg-panel p-0 text-fg shadow-menu backdrop:bg-[rgb(var(--scrim))] ${
        size === "lg" ? "w-[min(620px,94vw)]" : "w-[min(440px,94vw)]"
      }`}
    >
      <header className="flex h-header items-center justify-between border-b border-line bg-raised px-2">
        <h2 className="font-mono text-2xs font-medium uppercase tracking-wider text-dim">
          {title}
        </h2>
        <IconButton label={t("common.close")} size="sm" onClick={onClose}>
          <Glyph name="close" size={12} />
        </IconButton>
      </header>
      <div className="p-panel">{children}</div>
    </dialog>
  );
}

// ---------------------------------------------------------------------- brand
/**
 * The application mark.
 *
 * An aperture: six blades around an opening, which is what a camera does and
 * what the product is named for. Flat, monochrome and drawn on the same 16px
 * grid as every other glyph, so it sits in a toolbar rather than on top of one.
 */
export function BrandMark({ size = 16 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      aria-hidden
      focusable="false"
      className="shrink-0 text-accent"
    >
      <circle cx="8" cy="8" r="6.4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path d="M8 1.9 L11.4 7.2 H4.6 Z" fill="currentColor" opacity="0.9" />
      <path d="M13.5 10.6 L7.4 10.6 L10.8 5.3 Z" fill="currentColor" opacity="0.55" />
      <path d="M2.5 10.6 L5.9 5.3 L9.3 10.6 Z" fill="currentColor" opacity="0.75" />
    </svg>
  );
}

// ---------------------------------------------------------------------- glyphs
/**
 * The icon set, as inline SVG.
 *
 * Drawn here rather than pulled from a package: the editor needs about thirty
 * shapes, all of them on a 16px grid, and a dependency would ship several
 * hundred more plus a tree-shaking problem. They are deliberately plain --
 * transport controls, not decoration. There are no emoji anywhere in the
 * application, and this is why: an icon here is a drawing whose weight and grid
 * are the same as its neighbours'.
 */
export type GlyphName =
  | "play"
  | "pause"
  | "skip-back"
  | "skip-forward"
  | "step-back"
  | "step-forward"
  | "volume"
  | "mute"
  | "fullscreen"
  | "grid"
  | "list"
  | "rows"
  | "close"
  | "plus"
  | "trash"
  | "split"
  | "zoom-in"
  | "zoom-out"
  | "fit"
  | "chevron-left"
  | "chevron-right"
  | "chevron-down"
  | "chevron-up"
  | "video"
  | "audio"
  | "image"
  | "check"
  | "refresh"
  | "download"
  | "settings"
  | "globe"
  | "sun"
  | "moon"
  | "folder"
  | "search"
  | "panel-left"
  | "panel-right"
  | "spark"
  | "warning"
  | "film"
  | "analyse";

const STROKE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.3,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

const PATHS: Record<GlyphName, ReactNode> = {
  play: <path d="M5 3.5v9l7.5-4.5z" />,
  pause: <path d="M5 3.5h2.2v9H5zm3.8 0H11v9H8.8z" />,
  "skip-back": <path d="M4 3.5h1.6v9H4zm2.6 4.5L13 3.5v9z" />,
  "skip-forward": <path d="M10.4 3.5H12v9h-1.6zM9.4 8L3 12.5v-9z" />,
  "step-back": <path d="M4.2 8l6-4.5v9z" />,
  "step-forward": <path d="M11.8 8l-6 4.5v-9z" />,
  volume: (
    <>
      <path d="M3 6.2h2L7.8 4v8L5 9.8H3z" />
      <path d="M10 5.8a3 3 0 010 4.4M11.8 4a5.4 5.4 0 010 8" {...STROKE} />
    </>
  ),
  mute: (
    <>
      <path d="M3 6.2h2L7.8 4v8L5 9.8H3z" />
      <path d="M10.2 6.2l3.4 3.6M13.6 6.2l-3.4 3.6" {...STROKE} />
    </>
  ),
  fullscreen: <path d="M3 6V3h3M13 6V3h-3M3 10v3h3M13 10v3h-3" {...STROKE} />,
  grid: <path d="M2.5 2.5h5v5h-5zm6 0h5v5h-5zm-6 6h5v5h-5zm6 0h5v5h-5z" />,
  list: (
    <>
      <path d="M2.5 3.4h3v3h-3zm0 6.2h3v3h-3z" />
      <path d="M7 4.2h6.5M7 5.8h4M7 10.4h6.5M7 12h4" {...STROKE} strokeWidth={1.2} />
    </>
  ),
  rows: <path d="M2.5 3.6h11v1.5h-11zm0 3.6h11v1.5h-11zm0 3.6h11v1.5h-11z" />,
  close: <path d="M4 4l8 8M12 4l-8 8" {...STROKE} />,
  plus: <path d="M8 3.5v9M3.5 8h9" {...STROKE} />,
  trash: (
    <>
      <path d="M3.5 4.5h9l-.8 8.2a1 1 0 01-1 .8H5.3a1 1 0 01-1-.8z" />
      <path d="M6 2.6h4v1.4H6z" />
    </>
  ),
  split: (
    <>
      <path d="M7.3 2.5h1.4v11H7.3z" />
      <path d="M2 5h3.5v6H2zm8.5 0H14v6h-3.5z" opacity="0.5" />
    </>
  ),
  "zoom-in": (
    <>
      <circle cx="7" cy="7" r="4" {...STROKE} />
      <path d="M7 5.2v3.6M5.2 7h3.6M10 10l3 3" {...STROKE} />
    </>
  ),
  "zoom-out": (
    <>
      <circle cx="7" cy="7" r="4" {...STROKE} />
      <path d="M5.2 7h3.6M10 10l3 3" {...STROKE} />
    </>
  ),
  fit: <path d="M2.5 5V2.5h3M13.5 5V2.5h-3M2.5 11v2.5h3M13.5 11v2.5h-3M5 8h6" {...STROKE} />,
  "chevron-left": <path d="M10 3.5L5.5 8l4.5 4.5" {...STROKE} strokeWidth={1.4} />,
  "chevron-right": <path d="M6 3.5L10.5 8 6 12.5" {...STROKE} strokeWidth={1.4} />,
  "chevron-down": <path d="M3.5 6L8 10.5 12.5 6" {...STROKE} strokeWidth={1.4} />,
  "chevron-up": <path d="M3.5 10L8 5.5 12.5 10" {...STROKE} strokeWidth={1.4} />,
  video: (
    <>
      <rect x="2" y="4" width="8" height="8" rx="1" />
      <path d="M10.6 7.2L14 5.2v5.6l-3.4-2z" />
    </>
  ),
  audio: <path d="M3 6.5h1.6v3H3zm3 -2h1.6v7H6zm3-1.6h1.6v10.2H9zm3 2.4h1.6v5.4H12z" />,
  image: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1" {...STROKE} strokeWidth={1.2} />
      <path d="M3.6 11.4l3-3.4 2.2 2.4 1.8-2 1.8 3z" />
    </>
  ),
  film: (
    <>
      <rect x="1.8" y="3" width="12.4" height="10" rx="1" {...STROKE} strokeWidth={1.2} />
      <path d="M4.6 3v10M11.4 3v10" {...STROKE} strokeWidth={1.1} />
    </>
  ),
  check: <path d="M3.5 8.3l3 3 6-6.6" {...STROKE} strokeWidth={1.6} />,
  refresh: <path d="M13 8a5 5 0 11-1.6-3.7M13 2.6V5.4h-2.8" {...STROKE} />,
  download: (
    <>
      <path d="M7.2 2.5h1.6v6.2l2.2-2.2 1.1 1.1L8 12.2 3.9 7.6 5 6.5l2.2 2.2z" />
      <path d="M3 12.4h10v1.4H3z" />
    </>
  ),
  settings: (
    <>
      <circle cx="8" cy="8" r="2.1" {...STROKE} />
      <path
        d="M8 1.6v1.8M8 12.6v1.8M14.4 8h-1.8M3.4 8H1.6M12.5 3.5l-1.3 1.3M4.8 11.2l-1.3 1.3M12.5 12.5l-1.3-1.3M4.8 4.8L3.5 3.5"
        {...STROKE}
        strokeWidth={1.2}
      />
    </>
  ),
  globe: (
    <>
      <circle cx="8" cy="8" r="5.6" {...STROKE} strokeWidth={1.2} />
      <path d="M2.4 8h11.2M8 2.4c1.5 1.7 2.3 3.6 2.3 5.6S9.5 12 8 13.6C6.5 12 5.7 10 5.7 8S6.5 4.1 8 2.4z" {...STROKE} strokeWidth={1.2} />
    </>
  ),
  sun: (
    <>
      <circle cx="8" cy="8" r="2.8" {...STROKE} strokeWidth={1.2} />
      <path
        d="M8 1.4v1.6M8 13v1.6M14.6 8H13M3 8H1.4M12.7 3.3l-1.1 1.1M4.4 11.6l-1.1 1.1M12.7 12.7l-1.1-1.1M4.4 4.4L3.3 3.3"
        {...STROKE}
        strokeWidth={1.2}
      />
    </>
  ),
  moon: <path d="M13 9.6A5.6 5.6 0 016.4 3a5.6 5.6 0 106.6 6.6z" {...STROKE} strokeWidth={1.2} />,
  folder: (
    <path
      d="M1.9 12.4V4.2a.8.8 0 01.8-.8h3l1.5 1.8h6.1a.8.8 0 01.8.8v6.4a.8.8 0 01-.8.8H2.7a.8.8 0 01-.8-.8z"
      {...STROKE}
      strokeWidth={1.2}
    />
  ),
  search: (
    <>
      <circle cx="7.2" cy="7.2" r="4.1" {...STROKE} strokeWidth={1.2} />
      <path d="M10.3 10.3L13.4 13.4" {...STROKE} />
    </>
  ),
  "panel-left": (
    <>
      <rect x="1.8" y="2.8" width="12.4" height="10.4" rx="1.2" {...STROKE} strokeWidth={1.2} />
      <path d="M6.2 2.8v10.4" {...STROKE} strokeWidth={1.2} />
      <path d="M1.8 4a1.2 1.2 0 011.2-1.2h3.2v10.4H3A1.2 1.2 0 011.8 12z" />
    </>
  ),
  "panel-right": (
    <>
      <rect x="1.8" y="2.8" width="12.4" height="10.4" rx="1.2" {...STROKE} strokeWidth={1.2} />
      <path d="M9.8 2.8v10.4" {...STROKE} strokeWidth={1.2} />
      <path d="M9.8 2.8H13A1.2 1.2 0 0114.2 4v8a1.2 1.2 0 01-1.2 1.2H9.8z" />
    </>
  ),
  spark: <path d="M8 1.6l1.5 4.3 4.3 1.5-4.3 1.5L8 13.2 6.5 8.9 2.2 7.4l4.3-1.5z" />,
  warning: (
    <>
      <path d="M8 2.2l6 10.4H2z" {...STROKE} strokeWidth={1.2} />
      <path d="M8 6.3v3.1" {...STROKE} strokeWidth={1.4} />
      <circle cx="8" cy="11.1" r="0.75" fill="currentColor" stroke="none" />
    </>
  ),
  analyse: (
    <>
      <path d="M2 13.2V2.8" {...STROKE} strokeWidth={1.2} />
      <path d="M2 13.2h12" {...STROKE} strokeWidth={1.2} />
      <path d="M4.4 10.6l2.6-3.4 2.4 2 3-4" {...STROKE} strokeWidth={1.4} />
    </>
  ),
};

export function Glyph({ name, size = 14 }: { name: GlyphName; size?: number }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden
      focusable="false"
      className="shrink-0"
    >
      {PATHS[name]}
    </svg>
  );
}

// -------------------------------------------------------------------- keyboard
/**
 * A global shortcut, registered once.
 *
 * Ignores keystrokes aimed at a text field: an editor's single-key shortcuts
 * ("s" to split, space to play) would otherwise make every input unusable.
 */
export function useShortcuts(
  bindings: Record<string, (event: KeyboardEvent) => void>,
  enabled = true,
): void {
  const latest = useRef(bindings);
  latest.current = bindings;

  useEffect(() => {
    if (!enabled) return;

    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        target?.isContentEditable
      ) {
        return;
      }

      const parts = [
        event.ctrlKey || event.metaKey ? "mod" : null,
        event.shiftKey ? "shift" : null,
        event.altKey ? "alt" : null,
        event.key.length === 1 ? event.key.toLowerCase() : event.key,
      ].filter(Boolean);

      const handler = latest.current[parts.join("+")];
      if (handler) {
        event.preventDefault();
        handler(event);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled]);
}
