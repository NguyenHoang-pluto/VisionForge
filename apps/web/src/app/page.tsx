import { InfrastructureStatus } from "@/components/infrastructure-status";
import { MediaLibrary } from "@/components/media-library";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[1400px] flex-col gap-4 px-4 py-5">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-base font-semibold tracking-tight text-slate-100">
            VisionForge
          </h1>
          <span className="font-mono text-[10px] uppercase tracking-wider text-slate-600">
            Media Intelligence
          </span>
        </div>
        <p className="font-mono text-[10px] text-slate-600">
          quality · scenes · phash · clip · faces
        </p>
      </header>

      <MediaLibrary />

      <details className="border border-slate-800">
        <summary className="cursor-pointer px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-600 marker:text-slate-700 hover:text-slate-400">
          Infrastructure
        </summary>
        <div className="border-t border-slate-800 p-3">
          <InfrastructureStatus />
        </div>
      </details>
    </main>
  );
}
