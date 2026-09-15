import type { Config } from "tailwindcss";

/**
 * Semantic names only.
 *
 * Components say `bg-panel` and `border-line`, never `bg-slate-900`. The values
 * live in globals.css as custom properties, so the application can be re-toned
 * in one place — and re-themed, re-accented and re-densified at runtime —
 * while a component remains unable to invent a colour that is off-palette.
 *
 * Colours are declared in the `rgb(var(--x) / <alpha-value>)` form rather than
 * as a bare `var(--x)`. That is what lets `bg-accent/40` and `border-danger/30`
 * actually carry their alpha; with a bare var, Tailwind 3 emits the opaque
 * colour and drops the modifier without a warning.
 */
const channel = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ground: channel("surface-0"),
        panel: channel("surface-1"),
        raised: channel("surface-2"),
        control: channel("surface-3"),
        "control-hover": channel("surface-4"),

        line: channel("line"),
        "line-strong": channel("line-strong"),

        fg: channel("text"),
        muted: channel("text-muted"),
        dim: channel("text-dim"),

        accent: channel("accent"),
        "accent-strong": channel("accent-strong"),
        "accent-fg": channel("accent-fg"),
        /* Mixed per theme, so it carries no alpha channel of its own. */
        "accent-soft": "var(--accent-soft)",

        ok: channel("ok"),
        warn: channel("warn"),
        danger: channel("danger"),
        info: channel("info"),

        "track-video": "var(--track-video)",
        "track-video-selected": "var(--track-video-selected)",
        "track-audio": "var(--track-audio)",
        playhead: channel("playhead"),
        ruler: channel("ruler"),
        lane: channel("lane"),
      },

      /* Density, as sizing tokens. A control that spells its own height cannot
         follow the preference; one that says `h-control` follows it for free. */
      height: {
        control: "var(--h-control)",
        "control-sm": "var(--h-control-sm)",
        header: "var(--h-header)",
        strip: "var(--h-strip)",
        row: "var(--h-row)",
      },
      minHeight: {
        control: "var(--h-control)",
      },
      width: {
        control: "var(--h-control)",
        "control-sm": "var(--h-control-sm)",
      },
      spacing: {
        panel: "var(--pad-panel)",
        "panel-gap": "var(--gap-panel)",
      },

      borderRadius: {
        // Workstation radius: present enough to soften an edge, small enough
        // that nothing reads as a card.
        DEFAULT: "3px",
        sm: "2px",
        md: "4px",
        lg: "6px",
      },

      boxShadow: {
        menu: "var(--shadow-menu)",
      },

      fontSize: {
        // A dense scale. 11px is the working size of this application; the
        // larger steps are for the few places a heading is genuinely needed.
        "2xs": ["10px", { lineHeight: "14px" }],
        xs: ["11px", { lineHeight: "15px" }],
        sm: ["12px", { lineHeight: "16px" }],
        base: ["13px", { lineHeight: "18px" }],
        lg: ["15px", { lineHeight: "20px" }],
        xl: ["18px", { lineHeight: "24px" }],
      },

      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "Segoe UI",
          "Inter",
          "Roboto",
          "Helvetica Neue",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Cascadia Mono",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },

      transitionDuration: {
        DEFAULT: "120ms",
      },
    },
  },
  plugins: [],
} satisfies Config;
