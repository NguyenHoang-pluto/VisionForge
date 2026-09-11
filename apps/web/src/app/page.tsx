import { InfrastructureStatus } from "@/components/infrastructure-status";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-10 px-5 py-16">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-sky-400">
          Phase 1 · Foundation
        </p>
        <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-100">
          VisionForge
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-400">
          AI-assisted media creation platform. This page exists to prove the
          foundation works end to end: the web app reaches the API, and the API
          reaches PostgreSQL, Redis and object storage.
        </p>
      </header>

      <InfrastructureStatus />

      <footer className="border-t border-slate-800 pt-6 text-xs text-slate-500">
        No AI, media processing or editing features are implemented yet — those
        begin in Phase 2.
      </footer>
    </main>
  );
}
