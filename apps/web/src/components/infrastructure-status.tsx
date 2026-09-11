"use client";

import { useQuery } from "@tanstack/react-query";

import { api, APP_VERSION, API_BASE_URL } from "@/lib/api";
import { StatusBadge } from "@/components/status-badge";
import { useUiStore } from "@/stores/ui-store";

const COMPONENT_LABELS: Record<string, string> = {
  postgres: "PostgreSQL 16 + pgvector",
  redis: "Redis 7",
  storage: "Object storage (MinIO)",
};

export function InfrastructureStatus() {
  const showDetail = useUiStore((s) => s.showComponentDetail);
  const toggleDetail = useUiStore((s) => s.toggleComponentDetail);

  const readiness = useQuery({
    queryKey: ["health", "ready"],
    queryFn: api.readiness,
    refetchInterval: 10_000,
  });

  const version = useQuery({
    queryKey: ["version"],
    queryFn: api.version,
  });

  const apiReachable = !readiness.isError && !version.isError;

  return (
    <div className="flex flex-col gap-6">
      {/* ---- API connection ---- */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">
              API connection
            </h2>
            <p className="mt-0.5 font-mono text-xs text-slate-500">
              {API_BASE_URL}
            </p>
          </div>
          <StatusBadge
            status={
              readiness.isLoading
                ? "unknown"
                : apiReachable
                  ? "ok"
                  : "failed"
            }
          />
        </div>
        {!apiReachable && !readiness.isLoading && (
          <p className="mt-3 text-xs text-rose-400">
            Cannot reach the API. Start it with{" "}
            <code className="rounded bg-slate-800 px-1.5 py-0.5">
              .\scripts\vf.ps1 api
            </code>
            .
          </p>
        )}
      </section>

      {/* ---- Infrastructure ---- */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-slate-200">
            Infrastructure
          </h2>
          <button
            type="button"
            onClick={toggleDetail}
            className="rounded px-2 py-1 text-xs text-slate-400 transition hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            {showDetail ? "Hide latency" : "Show latency"}
          </button>
        </div>

        <ul className="mt-4 flex flex-col divide-y divide-slate-800">
          {readiness.data?.components.map((component) => (
            <li
              key={component.name}
              className="flex flex-wrap items-center justify-between gap-3 py-3 first:pt-0 last:pb-0"
            >
              <div>
                <p className="text-sm text-slate-300">
                  {COMPONENT_LABELS[component.name] ?? component.name}
                </p>
                {showDetail && (
                  <p className="mt-0.5 font-mono text-xs text-slate-500 tabular-nums">
                    {component.latency_ms !== null
                      ? `${component.latency_ms} ms`
                      : (component.detail ?? "—")}
                  </p>
                )}
              </div>
              <StatusBadge status={component.status} />
            </li>
          ))}

          {readiness.isLoading && (
            <li className="py-3 text-sm text-slate-500">Checking…</li>
          )}
          {readiness.isError && (
            <li className="py-3 text-sm text-slate-500">
              Unavailable while the API is unreachable.
            </li>
          )}
        </ul>
      </section>

      {/* ---- Version ---- */}
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
        <h2 className="text-sm font-semibold text-slate-200">Version</h2>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
          <div>
            <dt className="text-slate-500">Web</dt>
            <dd className="mt-0.5 font-mono text-slate-300">{APP_VERSION}</dd>
          </div>
          <div>
            <dt className="text-slate-500">API</dt>
            <dd className="mt-0.5 font-mono text-slate-300">
              {version.data?.version ?? "—"}
            </dd>
          </div>
          <div>
            <dt className="text-slate-500">Environment</dt>
            <dd className="mt-0.5 font-mono text-slate-300">
              {version.data?.environment ?? "—"}
            </dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
