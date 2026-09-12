import { InfrastructureStatus } from "@/components/infrastructure-status";
import { MediaLibrary } from "@/components/media-library";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-10 px-5 py-12">
      <header>
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-sky-400">
          Phase 2 · Media Ingest
        </p>
        <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-100">
          VisionForge
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
          Upload a folder of images, video or audio. Each file goes straight to
          object storage, then a worker probes it with ffprobe, hashes it,
          generates a thumbnail and — for anything above 720p — a proxy. Progress
          below is live, from real pipeline steps.
        </p>
      </header>

      <MediaLibrary />

      <details className="rounded-lg border border-slate-800 bg-slate-900/40">
        <summary className="cursor-pointer px-5 py-4 text-sm font-semibold text-slate-300 marker:text-slate-600">
          Infrastructure
        </summary>
        <div className="border-t border-slate-800 p-5">
          <InfrastructureStatus />
        </div>
      </details>

      <footer className="border-t border-slate-800 pt-6 text-xs text-slate-500">
        AI analysis, editing and rendering arrive in later phases.
      </footer>
    </main>
  );
}
