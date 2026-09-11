import type { ComponentStatus } from "@/lib/api";

const STYLES: Record<ComponentStatus | "unknown", string> = {
  ok: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30",
  degraded: "bg-amber-500/10 text-amber-400 ring-amber-500/30",
  failed: "bg-rose-500/10 text-rose-400 ring-rose-500/30",
  unknown: "bg-slate-500/10 text-slate-400 ring-slate-500/30",
};

const LABELS: Record<ComponentStatus | "unknown", string> = {
  ok: "Online",
  degraded: "Degraded",
  failed: "Unreachable",
  unknown: "Unknown",
};

export function StatusBadge({
  status,
}: {
  status: ComponentStatus | "unknown";
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${STYLES[status]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden />
      {LABELS[status]}
    </span>
  );
}
