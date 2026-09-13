/** Display formatting. One definition each, so two panels never disagree. */

/**
 * Timecode as an editor reads it.
 *
 * `mm:ss.mmm` under an hour, `h:mm:ss.mmm` over it. Milliseconds rather than
 * frames: the frame rate of the *timeline* differs from the frame rate of the
 * source, and showing a frame number that means neither would be worse than
 * showing none.
 */
export function timecode(ms: number | null | undefined, withMillis = true): string {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "--:--";
  const clamped = Math.max(0, Math.round(ms));
  const hours = Math.floor(clamped / 3_600_000);
  const minutes = Math.floor((clamped % 3_600_000) / 60_000);
  const seconds = Math.floor((clamped % 60_000) / 1000);
  const millis = clamped % 1000;

  const body =
    (hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}` : String(minutes)) +
    `:${String(seconds).padStart(2, "0")}`;
  return withMillis ? `${body}.${String(millis).padStart(3, "0")}` : body;
}

/** Short duration, for dense rows: `4.2s`, `1:05`. */
export function shortDuration(ms: number | null | undefined): string {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "—";
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`;
  return timecode(ms, false);
}

export function seconds(ms: number | null | undefined, digits = 1): string {
  return typeof ms === "number" ? `${(ms / 1000).toFixed(digits)}s` : "—";
}

export function bytes(value: number | null | undefined): string {
  if (typeof value !== "number" || value <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function resolution(
  width: number | null | undefined,
  height: number | null | undefined,
): string {
  return width && height ? `${width}×${height}` : "—";
}

export function fps(value: number | null | undefined): string {
  if (typeof value !== "number") return "—";
  // 29.97 must not print as 30: it is the difference between a file that
  // conforms and one that drifts.
  return Number.isInteger(value) ? `${value}` : value.toFixed(2);
}

/** A number, or an em dash. Never `NaN`, never `0` standing in for absent. */
export function num(value: unknown, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function int(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? String(Math.round(value)) : "—";
}

/** A duration in analyzer metrics: sub-second in ms, otherwise seconds. */
export function elapsed(value: unknown): string {
  if (typeof value !== "number") return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const delta = Date.now() - Date.parse(iso);
  if (!Number.isFinite(delta)) return "—";
  const minutes = Math.round(delta / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
