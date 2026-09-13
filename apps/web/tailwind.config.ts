import type { Config } from "tailwindcss";

/**
 * Semantic names only.
 *
 * Components say `bg-panel` and `border-line`, never `bg-slate-900`. The values
 * live in globals.css as custom properties, so the application can be re-toned
 * in one place and a component cannot invent a colour that is off-palette.
 */
export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ground: "var(--surface-0)",
        panel: "var(--surface-1)",
        raised: "var(--surface-2)",
        control: "var(--surface-3)",
        "control-hover": "var(--surface-4)",

        line: "var(--line)",
        "line-strong": "var(--line-strong)",

        fg: "var(--text)",
        muted: "var(--text-muted)",
        dim: "var(--text-dim)",

        accent: "var(--accent)",
        "accent-strong": "var(--accent-strong)",
        "accent-soft": "var(--accent-soft)",
        "accent-fg": "var(--accent-fg)",

        ok: "var(--ok)",
        warn: "var(--warn)",
        danger: "var(--danger)",
        info: "var(--info)",

        "track-video": "var(--track-video)",
        "track-video-selected": "var(--track-video-selected)",
        "track-audio": "var(--track-audio)",
        playhead: "var(--playhead)",
        ruler: "var(--ruler)",
      },
      borderRadius: {
        // Workstation radius: present enough to soften an edge, small enough
        // that nothing reads as a card.
        DEFAULT: "2px",
        sm: "2px",
        md: "3px",
      },
      fontSize: {
        // A dense scale. 11px is the working size of this application; the
        // larger steps are for the few places a heading is genuinely needed.
        "2xs": ["10px", { lineHeight: "14px" }],
        xs: ["11px", { lineHeight: "15px" }],
        sm: ["12px", { lineHeight: "16px" }],
        base: ["13px", { lineHeight: "18px" }],
        lg: ["15px", { lineHeight: "20px" }],
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
    },
  },
  plugins: [],
} satisfies Config;
