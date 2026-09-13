"use client";

import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

/**
 * The workstation's component vocabulary.
 *
 * Every control in the application is one of these. They are small, dense and
 * quiet by design: a button in an editor is pressed a thousand times a day, and
 * anything that draws attention to itself at that frequency becomes noise.
 *
 * Kept in one module rather than a file each. The set is small, the pieces are
 * short, and the alternative -- fifteen files of twenty lines -- makes it
 * harder to see the whole vocabulary at once and easier to add a sixteenth
 * variant nobody notices is a duplicate.
 */

// ------------------------------------------------------------------ primitives
type ButtonTone = "default" | "primary" | "danger" | "ghost";
type ButtonSize = "sm" | "md";

const BUTTON_TONE: Record<ButtonTone, string> = {
  default: "border-line-strong bg-control text-fg hover:bg-control-hover",
  primary: "border-accent bg-accent-soft text-accent-fg hover:bg-accent/40",
  danger: "border-line-strong bg-control text-danger hover:border-danger/60",
  ghost: "border-transparent bg-transparent text-muted hover:bg-control hover:text-fg",
};

const BUTTON_SIZE: Record<ButtonSize, string> = {
  sm: "h-[22px] px-2 text-2xs",
  md: "h-[26px] px-2.5 text-xs",
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
      className={`inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded border transition-colors disabled:pointer-events-none disabled:opacity-35 ${BUTTON_TONE[tone]} ${BUTTON_SIZE[size]} ${className}`}
    />
  );
}

/**
 * A square button carrying a glyph.
 *
 * `label` is mandatory and becomes the accessible name and the tooltip: an
 * icon-only control with no name is unusable with a screen reader and merely
 * cryptic with one.
 */
export function IconButton({
  label,
  active,
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active}
      {...props}
      className={`inline-flex h-[26px] w-[26px] shrink-0 items-center justify-center rounded border transition-colors disabled:pointer-events-none disabled:opacity-35 ${
        active
          ? "border-accent bg-accent-soft text-accent-fg"
          : "border-transparent text-muted hover:bg-control hover:text-fg"
      } ${className}`}
    >
      {children}
    </button>
  );
}

/** Radio semantics for a small closed set. Arrow keys move between options. */
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
      id={useFieldId()}
      className={`inline-flex overflow-hidden rounded border border-line-strong ${className}`}
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
            className={`h-[24px] border-r border-line-strong px-2 text-2xs transition-colors last:border-r-0 disabled:cursor-not-allowed disabled:text-dim/60 ${
              selected
                ? "bg-accent-soft text-accent-fg"
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
  "h-[26px] w-full rounded border border-line-strong bg-control px-1.5 text-xs text-fg transition-colors hover:border-line-strong/80 focus:border-accent disabled:opacity-40";

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
        className="select-none font-mono text-2xs uppercase tracking-wider text-dim"
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

/** A panel's title strip. Fixed height everywhere, so panels align across the app. */
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
      className={`flex h-[30px] shrink-0 items-center gap-2 border-b border-line bg-raised px-2 ${className}`}
    >
      <h2 className="select-none whitespace-nowrap font-mono text-2xs uppercase tracking-wider text-dim">
        {title}
      </h2>
      {children}
    </header>
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
    <div role="tablist" aria-label={label} className="flex h-[30px] shrink-0 border-b border-line">
      {tabs.map((tab) => {
        const selected = tab.value === value;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(tab.value)}
            className={`relative h-full flex-1 px-2 text-2xs uppercase tracking-wider transition-colors ${
              selected ? "bg-panel text-fg" : "bg-raised text-dim hover:text-muted"
            }`}
          >
            {tab.label}
            {selected && (
              <span className="absolute inset-x-0 bottom-0 h-px bg-accent" aria-hidden />
            )}
          </button>
        );
      })}
    </div>
  );
}

// ----------------------------------------------------------------- indicators
type BadgeTone = "neutral" | "ok" | "warn" | "danger" | "info" | "accent";

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral: "border-line-strong text-dim",
  ok: "border-ok/40 text-ok",
  warn: "border-warn/40 text-warn",
  danger: "border-danger/40 text-danger",
  info: "border-info/40 text-info",
  accent: "border-accent/50 text-accent-strong",
};

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
      className={`inline-flex h-[15px] items-center whitespace-nowrap rounded border px-1 font-mono text-2xs uppercase tracking-wide ${BADGE_TONE[tone]}`}
    >
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
      className="h-[2px] w-full overflow-hidden bg-control"
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

/** One label/value line. The unit the inspector is built from. */
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
      className="flex items-baseline justify-between gap-3 border-b border-line/60 py-[3px] last:border-b-0"
    >
      <span className="shrink-0 text-xs text-muted">{label}</span>
      <span className={`truncate font-mono text-xs tabular-nums ${valueClass}`}>{value}</span>
    </div>
  );
}

export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <header className="flex items-baseline justify-between gap-2 border-b border-line-strong pb-1">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">{children}</h3>
      {aside}
    </header>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-[80px] items-center justify-center px-4 py-6 text-center">
      <p className="max-w-[34ch] text-xs leading-relaxed text-dim">{children}</p>
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
      className="border-l-2 border-danger bg-danger/5 px-2 py-1 text-2xs leading-snug text-danger"
    >
      {children}
      {hint && <span className="mt-0.5 block text-danger/70">{hint}</span>}
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
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

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
      className="w-[min(420px,92vw)] rounded border border-line-strong bg-panel p-0 text-fg backdrop:bg-black/60"
    >
      <header className="flex h-[30px] items-center justify-between border-b border-line bg-raised px-2">
        <h2 className="font-mono text-2xs uppercase tracking-wider text-dim">{title}</h2>
        <IconButton label="Close" onClick={onClose}>
          <Glyph name="close" />
        </IconButton>
      </header>
      <div className="p-3">{children}</div>
    </dialog>
  );
}

// ---------------------------------------------------------------------- glyphs
/**
 * The icon set, as inline SVG.
 *
 * Drawn here rather than pulled from a package: the editor needs about a dozen
 * shapes, all of them on a 16px grid, and a dependency would ship several
 * hundred more plus a tree-shaking problem. They are deliberately plain --
 * transport controls, not decoration.
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
  | "close"
  | "plus"
  | "trash"
  | "split"
  | "zoom-in"
  | "zoom-out"
  | "chevron-left"
  | "chevron-right"
  | "video"
  | "audio"
  | "image"
  | "check"
  | "refresh"
  | "download";

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
      <path
        d="M10 5.8a3 3 0 010 4.4M11.8 4a5.4 5.4 0 010 8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </>
  ),
  mute: (
    <>
      <path d="M3 6.2h2L7.8 4v8L5 9.8H3z" />
      <path
        d="M10.2 6.2l3.4 3.6M13.6 6.2l-3.4 3.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </>
  ),
  fullscreen: (
    <path
      d="M3 6V3h3M13 6V3h-3M3 10v3h3M13 10v3h-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
    />
  ),
  grid: <path d="M2.5 2.5h5v5h-5zm6 0h5v5h-5zm-6 6h5v5h-5zm6 0h5v5h-5z" />,
  list: <path d="M2.5 3.5h11v1.6h-11zm0 3.7h11v1.6h-11zm0 3.7h11v1.6h-11z" />,
  close: (
    <path
      d="M4 4l8 8M12 4l-8 8"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
    />
  ),
  plus: (
    <path d="M8 3.5v9M3.5 8h9" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
  ),
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
      <circle cx="7" cy="7" r="4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M7 5.2v3.6M5.2 7h3.6M10 10l3 3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </>
  ),
  "zoom-out": (
    <>
      <circle cx="7" cy="7" r="4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M5.2 7h3.6M10 10l3 3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </>
  ),
  "chevron-left": (
    <path d="M10 3.5L5.5 8l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
  ),
  "chevron-right": (
    <path d="M6 3.5L10.5 8 6 12.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
  ),
  video: (
    <>
      <rect x="2" y="4" width="8" height="8" rx="1" />
      <path d="M10.6 7.2L14 5.2v5.6l-3.4-2z" />
    </>
  ),
  audio: (
    <path d="M3 6.5h1.6v3H3zm3 -2h1.6v7H6zm3-1.6h1.6v10.2H9zm3 2.4h1.6v5.4H12z" />
  ),
  image: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M3.6 11.4l3-3.4 2.2 2.4 1.8-2 1.8 3z" />
    </>
  ),
  check: (
    <path
      d="M3.5 8.3l3 3 6-6.6"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
    />
  ),
  refresh: (
    <path
      d="M13 8a5 5 0 11-1.6-3.7M13 2.6V5.4h-2.8"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
    />
  ),
  download: (
    <>
      <path d="M7.2 2.5h1.6v6.2l2.2-2.2 1.1 1.1L8 12.2 3.9 7.6 5 6.5l2.2 2.2z" />
      <path d="M3 12.4h10v1.4H3z" />
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
