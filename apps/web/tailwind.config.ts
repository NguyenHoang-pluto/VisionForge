import type { Config } from "tailwindcss";

/**
 * Semantic names only.
 *
 * Components say `bg-surface` and `border-subtle`, never `bg-slate-900`. The
 * values live in globals.css as custom properties, so the application can be
 * re-toned in one place — and re-themed, re-accented and re-densified at
 * runtime — while a component remains unable to invent a colour that is
 * off-palette.
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
        /* Four surface levels and a well. Hierarchy is carried by the level
           first, a shadow second and a line only where two things abut. */
        ground: channel("background"),
        surface: channel("surface"),
        elevated: channel("surface-elevated"),
        hover: channel("surface-hover"),
        sunken: channel("surface-sunken"),

        subtle: channel("border-subtle"),
        strong: channel("border-strong"),

        fg: channel("foreground"),
        muted: channel("foreground-muted"),
        faint: channel("foreground-subtle"),

        accent: channel("accent"),
        "accent-strong": channel("accent-strong"),
        "accent-fg": channel("accent-fg"),
        /* Mixed per theme, so it carries no alpha channel of its own. */
        "accent-soft": "var(--accent-soft)",

        success: channel("success"),
        warning: channel("warning"),
        danger: channel("danger"),
        info: channel("info"),

        "track-video": "var(--track-video)",
        "track-video-selected": "var(--track-video-selected)",
        "track-audio": "var(--track-audio)",
        "track-audio-selected": "var(--track-audio-selected)",
        "track-subtitle": "var(--track-subtitle)",
        "track-subtitle-selected": "var(--track-subtitle-selected)",
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
        rail: "var(--h-rail)",
      },
      minHeight: {
        control: "var(--h-control)",
        row: "var(--h-row)",
      },
      width: {
        control: "var(--h-control)",
        "control-sm": "var(--h-control-sm)",
        rail: "var(--h-rail)",
      },
      spacing: {
        panel: "var(--pad-panel)",
        "panel-gap": "var(--gap-panel)",
      },

      /*
       * Radius.
       *
       * The single largest contributor to how rigid the application felt: the
       * previous scale stopped at 6px, so a panel and a checkbox were the same
       * shape. This one runs far enough that a surface can be visibly softer
       * than the control sitting on it, which is what lets a layout read as
       * layered rather than as a grid of boxes.
       *
       * `DEFAULT` is 6 — controls, tiles, clips. Panels take `xl`/`2xl`.
       */
      borderRadius: {
        none: "0",
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
        xl: "14px",
        "2xl": "18px",
        "3xl": "24px",
      },

      boxShadow: {
        raised: "var(--shadow-raised)",
        panel: "var(--shadow-panel)",
        menu: "var(--shadow-menu)",
        float: "var(--shadow-float)",
      },

      /*
       * Type.
       *
       * Raised a step across the board from the previous scale, where 11px did
       * the work of a body size and everything below it was 10px. The steps are
       * now far enough apart to build a hierarchy from: a section title, a
       * label and a value are three visibly different things without needing
       * three different colours to say so.
       */
      fontSize: {
        "2xs": ["11px", { lineHeight: "15px", letterSpacing: "0.01em" }],
        xs: ["12px", { lineHeight: "17px" }],
        sm: ["13px", { lineHeight: "19px" }],
        base: ["14px", { lineHeight: "21px" }],
        lg: ["16px", { lineHeight: "23px", letterSpacing: "-0.005em" }],
        xl: ["19px", { lineHeight: "26px", letterSpacing: "-0.012em" }],
        "2xl": ["23px", { lineHeight: "30px", letterSpacing: "-0.018em" }],
        "3xl": ["30px", { lineHeight: "36px", letterSpacing: "-0.022em" }],
      },

      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "Segoe UI Variable Display",
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

      transitionTimingFunction: {
        DEFAULT: "var(--ease)",
        ease: "var(--ease)",
      },
      transitionDuration: {
        DEFAULT: "var(--t-fast)",
        fast: "var(--t-fast)",
        base: "var(--t-base)",
      },
    },
  },
  plugins: [],
} satisfies Config;
