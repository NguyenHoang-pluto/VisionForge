"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, uploadToStorage, type MediaAsset, type MediaStatus } from "@/lib/api";
import { useThumbnailUrl } from "@/lib/media-urls";
import { bytes, fps as formatFps, resolution, shortDuration } from "@/lib/format";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  EmptyState,
  ErrorNote,
  Glyph,
  IconButton,
  Panel,
  PanelHeader,
  ProgressBar,
  TextInput,
  type GlyphName,
} from "@/components/ui";

/**
 * The media browser.
 *
 * Two views over the same list, because the two questions an editor asks of a
 * library are different: "which shot is this" wants thumbnails, "which of these
 * is 60fps" wants columns. Selection, filtering and the analysis badge are
 * shared; only the row rendering differs.
 */

const STATUS_TONE: Record<MediaStatus, "neutral" | "info" | "warn" | "ok" | "danger"> = {
  pending_upload: "neutral",
  uploaded: "info",
  processing: "warn",
  ready: "ok",
  failed: "danger",
};

const STATUS_LABEL: Record<MediaStatus, string> = {
  pending_upload: "pending",
  uploaded: "queued",
  processing: "proc",
  ready: "ready",
  failed: "failed",
};

const KIND_GLYPH: Record<MediaAsset["kind"], GlyphName> = {
  video: "video",
  audio: "audio",
  image: "image",
};

type Filter = "all" | "video" | "audio" | "image" | "unanalyzed";

interface UploadFailure {
  filename: string;
  message: string;
  hint: string | null;
}

// ------------------------------------------------------------------ virtualization
/**
 * A window over a long list.
 *
 * A library of several hundred clips mounts several hundred `<video>`-adjacent
 * tiles, each with its own presigned-URL query. Rendering only what is near the
 * viewport keeps that bounded. Written here rather than pulled in as a
 * dependency: the editor needs one fixed-row-height window, which is thirty
 * lines, and the alternative is a package plus its own scroll container.
 */
function useWindowedRange(
  scrollRef: React.RefObject<HTMLElement | null>,
  rowCount: number,
  rowHeight: number,
  overscan = 3,
): { first: number; last: number; padTop: number; padBottom: number } {
  const [range, setRange] = useState({ start: 0, end: 40 });

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    let frame = 0;
    const measure = () => {
      frame = 0;
      const start = Math.floor(element.scrollTop / rowHeight);
      const visible = Math.ceil(element.clientHeight / rowHeight);
      setRange({ start, end: start + visible });
    };
    const onScroll = () => {
      // Coalesce to one measurement per frame: scroll fires far faster than
      // React can usefully re-render.
      if (!frame) frame = requestAnimationFrame(measure);
    };

    measure();
    element.addEventListener("scroll", onScroll, { passive: true });
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => {
      element.removeEventListener("scroll", onScroll);
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
    };
  }, [scrollRef, rowHeight]);

  const first = Math.max(0, range.start - overscan);
  const last = Math.min(rowCount, range.end + overscan);
  return {
    first,
    last,
    padTop: first * rowHeight,
    padBottom: Math.max(0, (rowCount - last) * rowHeight),
  };
}

// ------------------------------------------------------------------- thumbnail
function Thumb({
  asset,
  projectId,
  className = "",
}: {
  asset: MediaAsset;
  projectId: string;
  className?: string;
}) {
  const thumbnail = useThumbnailUrl(projectId, asset);

  return (
    <div className={`relative flex items-center justify-center overflow-hidden bg-ground ${className}`}>
      {thumbnail.data?.url ? (
        /* eslint-disable-next-line @next/next/no-img-element --
           presigned URL on a per-environment host; next/image would need a
           remotePatterns entry for every deployment. */
        <img
          src={thumbnail.data.url}
          alt=""
          loading="lazy"
          decoding="async"
          className="h-full w-full object-cover"
        />
      ) : (
        <span className="text-dim">
          <Glyph name={KIND_GLYPH[asset.kind]} size={18} />
        </span>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ grid tile
function Tile({
  asset,
  projectId,
  selected,
  active,
  analyzed,
  onSelect,
  onOpen,
}: {
  asset: MediaAsset;
  projectId: string;
  selected: boolean;
  active: boolean;
  analyzed: boolean;
  onSelect: (event: React.MouseEvent) => void;
  onOpen: () => void;
}) {
  return (
    <li>
      <div
        role="option"
        aria-selected={selected}
        tabIndex={0}
        onClick={onSelect}
        onDoubleClick={onOpen}
        onKeyDown={(event) => {
          if (event.key === "Enter") onOpen();
          if (event.key === " ") {
            event.preventDefault();
            onSelect(event as unknown as React.MouseEvent);
          }
        }}
        className={`group flex cursor-default flex-col overflow-hidden rounded border transition-colors ${
          active
            ? "border-accent bg-accent-soft/40"
            : selected
              ? "border-line-strong bg-raised"
              : "border-line bg-panel hover:border-line-strong"
        }`}
      >
        <Thumb asset={asset} projectId={projectId} className="aspect-video bg-ground" />

        <div className="flex flex-col gap-0.5 border-t border-line px-1.5 py-1">
          <p className="truncate text-xs text-fg" title={asset.original_filename}>
            {asset.original_filename}
          </p>
          <div className="flex items-center justify-between gap-1 font-mono text-2xs text-dim tabular-nums">
            <span className="truncate">
              {asset.duration_ms ? shortDuration(asset.duration_ms) : asset.kind}
              {asset.width ? ` · ${asset.width}×${asset.height}` : ""}
            </span>
            <span className="flex shrink-0 items-center gap-1">
              {analyzed && (
                <span className="text-ok" title="Analysed">
                  <Glyph name="check" size={9} />
                </span>
              )}
              <Badge tone={STATUS_TONE[asset.status]}>{STATUS_LABEL[asset.status]}</Badge>
            </span>
          </div>
        </div>
      </div>
    </li>
  );
}

// -------------------------------------------------------------------- list row
function ListRow({
  asset,
  projectId,
  selected,
  active,
  analyzed,
  onSelect,
  onOpen,
}: {
  asset: MediaAsset;
  projectId: string;
  selected: boolean;
  active: boolean;
  analyzed: boolean;
  onSelect: (event: React.MouseEvent) => void;
  onOpen: () => void;
}) {
  return (
    <tr
      role="row"
      aria-selected={selected}
      tabIndex={0}
      onClick={onSelect}
      onDoubleClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter") onOpen();
      }}
      className={`cursor-default border-b border-line/50 transition-colors ${
        active ? "bg-accent-soft/50" : selected ? "bg-raised" : "hover:bg-raised/60"
      }`}
    >
      <td className="w-[46px] p-1">
        <Thumb asset={asset} projectId={projectId} className="h-[26px] w-[42px] rounded-sm" />
      </td>
      <td className="max-w-0 truncate px-1 text-xs text-fg" title={asset.original_filename}>
        {analyzed && (
          <span className="mr-1 inline-block align-middle text-ok" title="Analysed">
            <Glyph name="check" size={9} />
          </span>
        )}
        {asset.original_filename}
      </td>
      <td className="w-[52px] px-1 text-right font-mono text-2xs text-muted tabular-nums">
        {shortDuration(asset.duration_ms)}
      </td>
      <td className="w-[72px] px-1 text-right font-mono text-2xs text-dim tabular-nums">
        {resolution(asset.width, asset.height)}
      </td>
      <td className="w-[38px] px-1 text-right font-mono text-2xs text-dim tabular-nums">
        {formatFps(asset.fps)}
      </td>
      <td className="w-[56px] px-1 text-right font-mono text-2xs text-dim tabular-nums">
        {bytes(asset.bytes_size)}
      </td>
      <td className="w-[46px] px-1 text-right">
        <Badge tone={STATUS_TONE[asset.status]}>{STATUS_LABEL[asset.status]}</Badge>
      </td>
    </tr>
  );
}

// --------------------------------------------------------------------- browser
export function MediaBrowser({
  projectId,
  media,
  loading,
  analyzedIds,
  onUploaded,
}: {
  projectId: string;
  media: MediaAsset[];
  loading: boolean;
  analyzedIds: Set<string>;
  onUploaded: () => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const view = useEditorStore((s) => s.browserView);
  const setView = useEditorStore((s) => s.setBrowserView);
  const selectedIds = useEditorStore((s) => s.selectedMediaIds);
  const activeId = useEditorStore((s) => s.activeMediaId);
  const selectMedia = useEditorStore((s) => s.selectMedia);
  const addMedia = useEditorStore((s) => s.addMedia);
  const setPreviewSource = useEditorStore((s) => s.setPreviewSource);

  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [uploading, setUploading] = useState<{ done: number; total: number } | null>(null);
  const [failures, setFailures] = useState<UploadFailure[]>([]);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return media.filter((asset) => {
      if (filter === "unanalyzed" && analyzedIds.has(asset.id)) return false;
      if (filter !== "all" && filter !== "unanalyzed" && asset.kind !== filter) return false;
      if (needle && !asset.original_filename.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [media, filter, search, analyzedIds]);

  const visibleIds = useMemo(() => visible.map((asset) => asset.id), [visible]);

  /**
   * Upload sequentially.
   *
   * One at a time on purpose: this runs on a laptop, a stampede of parallel
   * PUTs makes the progress count meaningless, and a failure in the middle of a
   * folder should leave the earlier files landed rather than all of them in
   * doubt.
   */
  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      const list = Array.from(files);
      setFailures([]);
      setUploading({ done: 0, total: list.length });

      const problems: UploadFailure[] = [];
      for (const [index, file] of list.entries()) {
        try {
          const ticket = await api.createUploadUrl(projectId, file.name, file.size);
          await uploadToStorage(ticket.upload_url, file);
          await api.completeUpload(projectId, ticket.media_id);
        } catch (error) {
          problems.push({
            filename: file.name,
            message: error instanceof ApiError ? error.message : "Upload failed",
            hint: error instanceof ApiError ? error.hint : null,
          });
        }
        setUploading({ done: index + 1, total: list.length });
      }

      setFailures(problems);
      setUploading(null);
      onUploaded();
    },
    [projectId, onUploaded],
  );

  function onSelect(asset: MediaAsset, event: React.MouseEvent | React.KeyboardEvent) {
    const modifier = event.ctrlKey || event.metaKey ? "toggle" : event.shiftKey ? "range" : "replace";
    selectMedia(asset.id, modifier, visibleIds);
  }

  function openInSource(asset: MediaAsset) {
    selectMedia(asset.id, "replace", visibleIds);
    setPreviewSource("source");
  }

  const selectedAssets = media.filter((asset) => selectedIds.includes(asset.id));
  const insertable = selectedAssets.filter(
    (asset) => asset.kind === "video" && asset.status === "ready",
  );

  const ROW_HEIGHT = 30;
  const rows = useWindowedRange(scrollRef, view === "list" ? visible.length : 0, ROW_HEIGHT);

  return (
    <Panel className="h-full border-r border-line">
      <PanelHeader title="Media">
        <span className="ml-auto flex items-center gap-1">
          <IconButton
            label="Grid view"
            active={view === "grid"}
            onClick={() => setView("grid")}
          >
            <Glyph name="grid" />
          </IconButton>
          <IconButton
            label="List view"
            active={view === "list"}
            onClick={() => setView("list")}
          >
            <Glyph name="list" />
          </IconButton>
        </span>
      </PanelHeader>

      {/* ---- import ---- */}
      <div className="flex shrink-0 items-center gap-1 border-b border-line px-2 py-1.5">
        <Button size="sm" onClick={() => fileInput.current?.click()} disabled={Boolean(uploading)}>
          <Glyph name="plus" size={10} />
          Files
        </Button>
        <Button size="sm" onClick={() => folderInput.current?.click()} disabled={Boolean(uploading)}>
          Folder
        </Button>
        <Button
          size="sm"
          tone="primary"
          className="ml-auto"
          disabled={insertable.length === 0}
          title={
            insertable.length === 0
              ? "Select one or more ready video clips."
              : `Append ${insertable.length} clip(s) to the timeline`
          }
          onClick={() => addMedia(insertable)}
        >
          To timeline
          {insertable.length > 1 && (
            <span className="font-mono tabular-nums">{insertable.length}</span>
          )}
        </Button>

        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          accept="image/*,video/*,audio/*"
          onChange={(event) => void handleFiles(event.target.files)}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          hidden
          {...{ webkitdirectory: "", directory: "" }}
          onChange={(event) => void handleFiles(event.target.files)}
        />
      </div>

      {/* ---- filter ---- */}
      <div className="flex shrink-0 items-center gap-1 border-b border-line px-2 py-1.5">
        <TextInput
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Filter by name"
          aria-label="Filter media by filename"
          className="h-[22px] text-2xs"
        />
        <select
          value={filter}
          onChange={(event) => setFilter(event.target.value as Filter)}
          aria-label="Filter media by kind"
          className="h-[22px] shrink-0 rounded border border-line-strong bg-control px-1 text-2xs text-fg focus:border-accent"
        >
          <option value="all">All</option>
          <option value="video">Video</option>
          <option value="audio">Audio</option>
          <option value="image">Image</option>
          <option value="unanalyzed">Unanalysed</option>
        </select>
      </div>

      {uploading && (
        <div className="shrink-0 border-b border-line px-2 py-1">
          <div className="flex justify-between font-mono text-2xs text-muted tabular-nums">
            <span>Uploading</span>
            <span>
              {uploading.done}/{uploading.total}
            </span>
          </div>
          <div className="mt-1">
            <ProgressBar fraction={uploading.done / uploading.total} label="Upload progress" />
          </div>
        </div>
      )}

      {failures.length > 0 && (
        <ul className="shrink-0 border-b border-line">
          {failures.map((failure) => (
            <li key={failure.filename} className="px-2 py-1">
              <ErrorNote hint={failure.hint}>
                <span className="font-mono">{failure.filename}</span> — {failure.message}
              </ErrorNote>
            </li>
          ))}
        </ul>
      )}

      {/* ---- the library ---- */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <EmptyState>
            {loading
              ? "Loading media…"
              : media.length === 0
                ? "No media in this project. Import files or a folder to begin."
                : "Nothing matches this filter."}
          </EmptyState>
        ) : view === "grid" ? (
          <ul
            role="listbox"
            aria-label="Media library"
            aria-multiselectable
            className="grid grid-cols-2 gap-1.5 p-1.5 2xl:grid-cols-3"
          >
            {visible.map((asset) => (
              <Tile
                key={asset.id}
                asset={asset}
                projectId={projectId}
                selected={selectedIds.includes(asset.id)}
                active={asset.id === activeId}
                analyzed={analyzedIds.has(asset.id)}
                onSelect={(event) => onSelect(asset, event)}
                onOpen={() => openInSource(asset)}
              />
            ))}
          </ul>
        ) : (
          <table
            role="grid"
            aria-label="Media library"
            aria-rowcount={visible.length}
            className="w-full table-fixed border-collapse"
          >
            <tbody>
              {rows.padTop > 0 && (
                <tr style={{ height: rows.padTop }} aria-hidden>
                  <td colSpan={7} />
                </tr>
              )}
              {visible.slice(rows.first, rows.last).map((asset) => (
                <ListRow
                  key={asset.id}
                  asset={asset}
                  projectId={projectId}
                  selected={selectedIds.includes(asset.id)}
                  active={asset.id === activeId}
                  analyzed={analyzedIds.has(asset.id)}
                  onSelect={(event) => onSelect(asset, event)}
                  onOpen={() => openInSource(asset)}
                />
              ))}
              {rows.padBottom > 0 && (
                <tr style={{ height: rows.padBottom }} aria-hidden>
                  <td colSpan={7} />
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      <footer className="flex h-[22px] shrink-0 items-center justify-between border-t border-line px-2 font-mono text-2xs text-dim tabular-nums">
        <span>
          {visible.length}
          {visible.length !== media.length ? ` / ${media.length}` : ""} items
        </span>
        <span>
          {selectedIds.length > 0 ? `${selectedIds.length} selected · ` : ""}
          {analyzedIds.size} analysed
        </span>
      </footer>
    </Panel>
  );
}
