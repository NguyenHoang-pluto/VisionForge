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
 * Every control in the application is one of these. They are quiet by design: a
 * button in an editor is pressed a thousand times a day, and anything that
 * draws attention to itself at that frequency becomes noise.
 *
 * Quiet is not the same as rigid, which is what the previous revision of this
 * file got wrong. Everything was a 3px rectangle with a hairline around it, and
 * forty such rectangles in one viewport read as a wiring diagram. The shapes
 * here are softer, the hierarchy is carried by surface level and weight rather
 * than by outlines, and a control responds to the pointer — but the palette,
 * the density and the restraint are unchanged.
 *
 * Nothing here spells a colour or a height. Colours come from the semantic
 * Tailwind names, which resolve to the theme's custom properties; heights come
 * from `h-control` and friends, which resolve to the density preference. That
 * is what makes theme, accent and density switchable at runtime without a
 * single component knowing they exist.
 *
 * Kept in one module rather than a file each. The set is small, the pieces are
 * short, and the alternative -- thirty files of twenty lines -- makes it harder
 * to see the whole vocabulary at once and easier to add a thirty-first variant
 * nobody notices is a duplicate.
 */

// ------------------------------------------------------------------ primitives
type ButtonTone = "default" | "primary" | "danger" | "ghost" | "quiet";
type ButtonSize = "sm" | "md" | "lg";

/**
 * Tones.
 *
 * `default` is a filled control, `quiet` is the same shape with no fill until
 * hovered, `ghost` has no shape at all until hovered. Three levels of presence
 * for what is structurally the same button, so a toolbar can be dense without
 * being loud and a panel can still have one obvious action.
 */
const BUTTON_TONE: Record<ButtonTone, string> = {
  default: "bg-elevated text-fg shadow-raised hover:bg-hover active:bg-elevated",
  /* The one emphatic fill in the application. `accent-strong` against
     `accent-fg` clears 7:1 in both themes, which a mid-accent fill does not. */
  primary:
    "bg-accent-strong text-accent-fg shadow-raised hover:bg-accent active:bg-accent-strong",
  danger: "bg-danger/12 text-danger hover:bg-danger/20 active:bg-danger/12",
  quiet: "bg-transparent text-muted hover:bg-hover hover:text-fg active:bg-elevated",
  ghost: "bg-transparent text-muted hover:text-fg",
};

const BUTTON_SIZE: Record<ButtonSize, string> = {
  sm: "h-control-sm gap-1.5 rounded px-2.5 text-2xs",
  md: "h-control gap-2 rounded-md px-3 text-xs",
  lg: "h-[calc(var(--h-control)+8px)] gap-2 rounded-lg px-4 text-sm",
};

/**
 * `active:scale-[0.97]` is the whole of the press feedback: a transform, so it
 * costs nothing to animate, and small enough to be felt rather than seen.
 */
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
      className={`inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap font-medium transition-[background-color,color,transform,box-shadow] duration-fast active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40 ${BUTTON_TONE[tone]} ${BUTTON_SIZE[size]} ${className}`}
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
      className={`inline-flex shrink-0 items-center justify-center rounded-md transition-[background-color,color,transform] duration-fast active:scale-[0.94] disabled:pointer-events-none disabled:opacity-40 ${
        size === "sm" ? "h-control-sm w-control-sm" : "h-control w-control"
      } ${
        active
          ? "bg-accent-soft text-accent-strong"
          : "text-muted hover:bg-hover hover:text-fg"
      } ${className}`}
    >
      {children}
    </button>
  );
}

/**
 * Radio semantics for a small closed set.
 *
 * A recessed track with the selected option raised out of it, rather than a row
 * of bordered cells. The selection is a shape you can see at a glance from
 * across the desk, which a border colour is not.
 */
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
      className={`inline-flex h-control shrink-0 items-center gap-0.5 rounded-lg bg-sunken/70 p-[3px] ${className}`}
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
            className={`h-full whitespace-nowrap rounded px-2.5 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast disabled:cursor-not-allowed disabled:opacity-40 ${
              selected
                ? "bg-elevated text-fg shadow-raised"
                : "text-muted hover:text-fg"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The shared shape of anything you type into or pick from.
 *
 * A ring rather than a border on focus, so the control does not change size,
 * and `bg-elevated` rather than an outline to say "this is editable" — the
 * surface level does the work an outline used to.
 */
const CONTROL_CLASS =
  "h-control w-full rounded-md border border-subtle bg-elevated px-2.5 text-xs text-fg shadow-raised transition-[border-color,box-shadow] duration-fast hover:border-strong focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-45";

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
      className={`${CONTROL_CLASS} placeholder:text-faint ${className}`}
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

/** A multi-line input, for the one place the application takes prose. */
export function TextArea({
  className = "",
  id,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      id={useFieldId(id)}
      {...props}
      className={`w-full resize-none rounded-lg border border-subtle bg-elevated px-3 py-2.5 text-sm leading-relaxed text-fg shadow-raised transition-[border-color,box-shadow] duration-fast placeholder:text-faint hover:border-strong focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-45 ${className}`}
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

/** A labelled switch, for a binary that is not a form field. */
export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={`group flex items-start gap-2.5 ${
        disabled ? "cursor-not-allowed opacity-45" : "cursor-pointer"
      }`}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative mt-[1px] h-[18px] w-[30px] shrink-0 rounded-full transition-colors duration-fast disabled:pointer-events-none ${
          checked ? "bg-accent" : "bg-strong"
        }`}
      >
        <span
          aria-hidden
          className={`absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow-raised transition-transform duration-fast ${
            checked ? "translate-x-[14px]" : "translate-x-[2px]"
          }`}
        />
      </button>
      <span className="min-w-0">
        <span className="block text-xs text-fg">{label}</span>
        {hint && <span className="mt-0.5 block text-2xs leading-snug text-faint">{hint}</span>}
      </span>
    </label>
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
    <div className={`flex min-w-0 flex-col gap-1.5 ${className}`}>
      <label
        htmlFor={id}
        title={hint}
        className="select-none truncate text-2xs font-medium text-muted"
      >
        {label}
      </label>
      <FieldIdContext.Provider value={id}>{children}</FieldIdContext.Provider>
    </div>
  );
}

// ---------------------------------------------------------------------- panels
/**
 * A region of the workspace.
 *
 * Panels are surfaces, not boxes: the level and the radius say where one ends
 * and the next begins, so the layout can drop almost every internal border it
 * used to draw. `inset` rounds all four corners for a panel that floats in the
 * ground; the default rounds none, for a panel that fills its column.
 */
export function Panel({
  children,
  className = "",
  inset = false,
}: {
  children: ReactNode;
  className?: string;
  inset?: boolean;
}) {
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col bg-surface ${
        inset ? "rounded-xl shadow-panel" : ""
      } ${className}`}
    >
      {children}
    </section>
  );
}

/**
 * A panel's title strip. One height everywhere, so panels align across the app.
 *
 * Sentence case in the text colour rather than a monospace micro-caps label:
 * the old treatment made every header look like a field name in a database
 * tool, and it was the single most "internal admin panel" thing in the UI.
 */
export function PanelHeader({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon?: GlyphName;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <header
      className={`flex h-header shrink-0 items-center gap-2 px-panel ${className}`}
    >
      {icon && (
        <span className="shrink-0 text-faint" aria-hidden>
          <Glyph name={icon} size={14} />
        </span>
      )}
      <h2 className="select-none truncate whitespace-nowrap text-xs font-semibold tracking-tight text-fg">
        {title}
      </h2>
      {children}
    </header>
  );
}

/**
 * A raised rectangle of content: a project on the home screen, a plan summary,
 * a render in the export list.
 *
 * `interactive` adds the hover response — a lift of one shadow step and a
 * one-pixel rise. Both are transforms and shadows, nothing that reflows.
 */
export function Card({
  children,
  className = "",
  interactive = false,
  selected = false,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & {
  interactive?: boolean;
  selected?: boolean;
}) {
  return (
    <div
      {...props}
      className={`rounded-xl bg-elevated shadow-raised transition-[box-shadow,transform,background-color] duration-base ${
        interactive ? "hover:-translate-y-px hover:bg-hover hover:shadow-panel" : ""
      } ${selected ? "ring-2 ring-accent" : ""} ${className}`}
    >
      {children}
    </div>
  );
}

/**
 * A group of related controls in a toolbar.
 *
 * Grouping is the whole job of a workstation's top bar: fifteen loose buttons
 * are unreadable, five groups of three are a menu you can learn. The gap
 * between groups is what separates them — the previous version drew a hairline
 * between every pair, and a toolbar with six vertical rules in it is a
 * calculator.
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
      className={`flex shrink-0 items-center gap-1 ${className}`}
    >
      {children}
    </div>
  );
}

/**
 * A tab strip that stays one row tall.
 *
 * Pills in a recessed track rather than a row of bordered cells with an
 * underline: the same shape language as `SegmentedControl`, because they are
 * the same idea at two scales, and one fewer rectangle in a panel that has
 * plenty.
 *
 * `whitespace-nowrap` with `truncate` rather than letting labels wrap: a strip
 * whose height depends on the longest translation is a strip that changes the
 * panel's geometry when the language changes, and every panel below it would
 * shift.
 */
export function Tabs<T extends string>({
  value,
  onChange,
  tabs,
  label,
  className = "",
}: {
  value: T;
  onChange: (value: T) => void;
  tabs: { value: T; label: string; icon?: GlyphName }[];
  label: string;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={label}
      className={`flex shrink-0 items-center gap-0.5 rounded-lg bg-sunken/60 p-[3px] ${className}`}
    >
      {tabs.map((tab) => {
        const selected = tab.value === value;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={selected}
            // The label may be clipped -- a five-tab strip cannot fit every
            // translation -- so the full text is always available on hover,
            // the same rule every other truncated string in the app follows.
            title={tab.label}
            onClick={() => onChange(tab.value)}
            className={`flex h-[calc(var(--h-control)-4px)] min-w-0 flex-1 items-center justify-center gap-1.5 truncate whitespace-nowrap rounded px-2 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast ${
              selected
                ? "bg-elevated text-fg shadow-raised"
                : "text-muted hover:text-fg"
            }`}
          >
            {tab.icon && <Glyph name={tab.icon} size={12} />}
            <span className="truncate">{tab.label}</span>
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
  items: {
    key: string;
    label: string;
    onSelect: () => void;
    disabled?: boolean;
    icon?: GlyphName;
  }[];
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
        className={`inline-flex h-control items-center gap-1.5 rounded-md px-2 transition-colors duration-fast ${
          open ? "bg-hover text-fg" : "text-muted hover:bg-hover hover:text-fg"
        }`}
      >
        {children}
      </button>

      {open && (
        <div
          ref={list}
          role="menu"
          aria-label={label}
          className={`vf-animate-pop absolute top-[calc(100%+6px)] z-50 min-w-[210px] origin-top rounded-xl border border-subtle bg-elevated p-1.5 shadow-menu ${
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
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-xs text-muted transition-colors duration-fast hover:bg-accent-soft hover:text-fg focus:bg-accent-soft focus:text-fg focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40"
            >
              {item.icon && (
                <span className="shrink-0 text-faint" aria-hidden>
                  <Glyph name={item.icon} size={13} />
                </span>
              )}
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
  neutral: "bg-hover text-muted",
  ok: "bg-success/14 text-success",
  warn: "bg-warning/14 text-warning",
  danger: "bg-danger/14 text-danger",
  info: "bg-info/14 text-info",
  accent: "bg-accent-soft text-accent-strong",
};

/**
 * A status word in a pill.
 *
 * Sentence case, not upper. Upper-casing buys a little more "chrome" in English
 * and costs about a third more width, which is what pushes a status out of a
 * column in a narrow panel -- and Vietnamese, where every other vowel carries a
 * diacritic, is harder to read upper-cased, not easier.
 *
 * A tint with no border: a badge is a highlight, and outlining it as well makes
 * it a component.
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
      className={`inline-flex h-[18px] shrink-0 items-center whitespace-nowrap rounded-full px-2 text-2xs font-medium ${BADGE_TONE[tone]}`}
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
  pulse = false,
}: {
  tone: "ok" | "warn" | "danger" | "neutral";
  children: ReactNode;
  title?: string;
  pulse?: boolean;
}) {
  const fill = {
    ok: "bg-success",
    warn: "bg-warning",
    danger: "bg-danger",
    neutral: "bg-faint",
  }[tone];
  return (
    <span title={title} className="flex shrink-0 items-center gap-2">
      <span aria-hidden className="relative flex h-1.5 w-1.5 shrink-0">
        {pulse && (
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${fill}`} />
        )}
        <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${fill}`} />
      </span>
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
    ok: "bg-success",
    warn: "bg-warning",
    danger: "bg-danger",
  }[tone];
  return (
    <div
      className="h-[4px] w-full overflow-hidden rounded-full bg-sunken"
      role="meter"
      aria-valuenow={Math.round(fraction * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={`h-full rounded-full transition-[width] duration-base ${fill}`}
        style={{ width: `${fraction * 100}%` }}
      />
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
  const fill = {
    accent: "bg-accent",
    ok: "bg-success",
    warn: "bg-warning",
    danger: "bg-danger",
  }[tone];
  return (
    <div
      className="h-[3px] w-full overflow-hidden rounded-full bg-sunken"
      role="progressbar"
      aria-label={label}
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={`h-full rounded-full transition-[width] duration-base ${fill}`}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

/**
 * One label/value line. The unit the inspector is built from.
 *
 * No rule between rows. A readout of twelve facts with twelve hairlines through
 * it is a table nobody asked for; the rhythm of the baseline and the contrast
 * between a muted label and a foreground value separate them perfectly well,
 * and the panel gets quieter for free.
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
      ? "text-warning"
      : tone === "ok"
        ? "text-success"
        : tone === "danger"
          ? "text-danger"
          : "text-fg";
  return (
    <div
      title={title}
      className="flex min-h-row items-baseline justify-between gap-3 py-[3px]"
    >
      <span className="w-[42%] shrink-0 truncate text-xs text-muted">{label}</span>
      <span className={`truncate text-right font-mono text-xs tabular-nums ${valueClass}`}>
        {value}
      </span>
    </div>
  );
}

/**
 * A heading inside a panel.
 *
 * Weight and size carry it, not a rule underneath. An optional `description`
 * exists so a section can explain itself in one line instead of putting the
 * explanation in a tooltip nobody hovers.
 */
export function SectionTitle({
  children,
  aside,
  description,
}: {
  children: ReactNode;
  aside?: ReactNode;
  description?: string;
}) {
  return (
    <header className="mb-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="truncate text-2xs font-semibold uppercase tracking-[0.08em] text-faint">
          {children}
        </h3>
        {aside}
      </div>
      {description && (
        <p className="mt-1 text-2xs leading-snug text-faint">{description}</p>
      )}
    </header>
  );
}

/**
 * An empty panel that says what to do about it.
 *
 * Takes a title and a body rather than one blob of prose: "No media yet" read
 * in a quarter of a second is worth more than a paragraph that is read once and
 * then skipped forever.
 *
 * The icon sits in a soft disc rather than floating loose, which is what stops
 * an empty panel from reading as a broken one.
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
    <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2 px-6 py-8 text-center">
      {icon && (
        <span
          className="mb-1 flex h-11 w-11 items-center justify-center rounded-full bg-hover text-faint"
          aria-hidden
        >
          <Glyph name={icon} size={20} />
        </span>
      )}
      {title && <p className="text-sm font-semibold text-fg">{title}</p>}
      {children && (
        <p className="max-w-[40ch] text-xs leading-relaxed text-faint">{children}</p>
      )}
      {action && <div className="mt-2">{action}</div>}
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
      className="rounded-lg bg-danger/12 px-3 py-2 text-2xs leading-snug text-danger"
    >
      {children}
      {hint && <span className="mt-1 block text-danger/80">{hint}</span>}
    </p>
  );
}

export function Divider({ vertical }: { vertical?: boolean }) {
  return vertical ? (
    <span className="mx-1.5 h-4 w-px shrink-0 bg-subtle" aria-hidden />
  ) : (
    <span className="my-2 h-px w-full bg-subtle" aria-hidden />
  );
}

/** A key on a keyboard, drawn as one. */
export function KeyCap({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-[20px] min-w-[20px] items-center justify-center rounded border border-subtle bg-elevated px-1.5 font-mono text-2xs text-fg shadow-raised">
      {children}
    </kbd>
  );
}

/** A quiet, indefinite "working on it". */
export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <span
      role="status"
      aria-hidden
      style={{ width: size, height: size }}
      className="inline-block shrink-0 animate-spin rounded-full border-2 border-strong border-t-accent"
    />
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
  description,
  size = "md",
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  size?: "md" | "lg" | "xl";
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

  const width =
    size === "xl"
      ? "w-[min(880px,94vw)]"
      : size === "lg"
        ? "w-[min(640px,94vw)]"
        : "w-[min(460px,94vw)]";

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => {
        // Clicking the backdrop closes; clicking the panel must not.
        if (event.target === ref.current) onClose();
      }}
      className={`vf-animate-pop rounded-2xl border border-subtle bg-surface p-0 text-fg shadow-float backdrop:bg-[rgb(var(--scrim))] backdrop:backdrop-blur-[2px] ${width}`}
    >
      <header className="flex items-start justify-between gap-4 px-panel pb-3 pt-panel">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold tracking-tight text-fg">{title}</h2>
          {description && (
            <p className="mt-1 text-2xs leading-snug text-faint">{description}</p>
          )}
        </div>
        <IconButton label={t("common.close")} size="sm" onClick={onClose}>
          <Glyph name="close" size={13} />
        </IconButton>
      </header>
      <div className="px-panel pb-panel">{children}</div>
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
  | "analyse"
  | "home"
  | "upload"
  | "export"
  | "layers"
  | "clock"
  | "beat"
  | "wand"
  | "more";

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
  home: <path d="M8 2.2 1.9 7.1h1.5v6.1h3.1V9.4h3v3.8h3.1V7.1h1.5z" />,
  upload: (
    <>
      <path d="M8 10.8V2.9M4.9 6 8 2.9 11.1 6" {...STROKE} strokeWidth={1.4} />
      <path d="M2.6 10.4v2.2h10.8v-2.2" {...STROKE} strokeWidth={1.3} />
    </>
  ),
  export: (
    <>
      <path d="M2.4 3.2h7.2v3.1M2.4 3.2v9.6h7.2V9.7" {...STROKE} strokeWidth={1.3} />
      <path d="M6.3 8h7.3M11 5.4 13.6 8 11 10.6" {...STROKE} strokeWidth={1.4} />
    </>
  ),
  layers: (
    <>
      <path d="M8 2.2 1.9 5.4 8 8.6l6.1-3.2z" />
      <path d="m2.2 8.4 5.8 3 5.8-3M2.2 11.2l5.8 3 5.8-3" {...STROKE} strokeWidth={1.2} />
    </>
  ),
  clock: (
    <>
      <circle cx="8" cy="8" r="5.7" {...STROKE} strokeWidth={1.3} />
      <path d="M8 4.6V8l2.5 1.6" {...STROKE} strokeWidth={1.3} />
    </>
  ),
  /* Beat: a pulse on a baseline. The audio counterpart to `analyse`. */
  beat: <path d="M1.6 8h2.6l1.5-4.4 2.2 8.8 1.6-5.6 1 1.2h4" {...STROKE} strokeWidth={1.4} />,
  wand: (
    <>
      <path d="M3 13 11.2 4.8" {...STROKE} strokeWidth={1.5} />
      <path d="M12.4 1.6 13 3.4l1.8.6-1.8.6-.6 1.8-.6-1.8-1.8-.6 1.8-.6z" />
      <path d="M4.2 2.2 4.6 3.4l1.2.4-1.2.4-.4 1.2-.4-1.2-1.2-.4 1.2-.4z" />
    </>
  ),
  more: (
    <>
      <circle cx="3.4" cy="8" r="1.25" />
      <circle cx="8" cy="8" r="1.25" />
      <circle cx="12.6" cy="8" r="1.25" />
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
