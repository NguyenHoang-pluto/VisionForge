"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, uploadToStorage, type MediaAsset, type MediaStatus } from "@/lib/api";
import { useT, type MessageKey } from "@/lib/i18n";
import { useThumbnailUrl } from "@/lib/media-urls";
import { bytes, fps as formatFps, resolution, shortDuration } from "@/lib/format";
import { useEditorStore, type BrowserView } from "@/stores/editor-store";
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
 * Three views over the same list, because the questions an editor asks of a
 * library are different in kind: "which shot is this" wants thumbnails, "which
 * of these is 60fps" wants columns, and "where is the one called clip_047"
 * wants as many filenames on screen at once as will fit. Selection, filtering
 * and the analysis mark are shared; only the row rendering differs.
 */

const STATUS_TONE: Record<MediaStatus, "neutral" | "info" | "warn" | "ok" | "danger"> = {
  pending_upload: "neutral",
  uploaded: "info",
  processing: "warn",
  ready: "ok",
  failed: "danger",
};

const STATUS_LABEL: Record<MediaStatus, MessageKey> = {
  pending_upload: "media.status.pending_upload",
  uploaded: "media.status.uploaded",
  processing: "media.status.processing",
  ready: "media.status.ready",
  failed: "media.status.failed",
};

const KIND_GLYPH: Record<MediaAsset["kind"], GlyphName> = {
  video: "video",
  audio: "audio",
  image: "image",
};

const KIND_LABEL: Record<MediaAsset["kind"], MessageKey> = {
  video: "media.kind.video",
  audio: "media.kind.audio",
  image: "media.kind.image",
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
 * A library of several hundred clips mounts several hundred tiles, each with
 * its own presigned-URL query. Rendering only what is near the viewport keeps
 * that bounded. Written here rather than pulled in as a dependency: the editor
 * needs one fixed-row-height window, which is thirty lines, and the alternative
 * is a package plus its own scroll container.
 */
function useWindowedRange(
  scrollRef: React.RefObject<HTMLElement | null>,
  rowCount: number,
  rowHeight: number,
  overscan = 4,
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
  glyphSize = 18,
}: {
  asset: MediaAsset;
  projectId: string;
  className?: string;
  glyphSize?: number;
}) {
  const thumbnail = useThumbnailUrl(projectId, asset);

  return (
    <div
      className={`relative flex items-center justify-center overflow-hidden bg-ground ${className}`}
    >
      {thumbnail.data?.url ? (
        /* eslint-disable-next-line @next/next/no-img-element --
           presigned URL on a per-environment host; next/image would need a
           remotePatterns entry for every deployment. */
        <img
          src={thumbnail.data.url}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
          className="h-full w-full object-cover"
        />
      ) : (
        <span className="text-faint/70">
          <Glyph name={KIND_GLYPH[asset.kind]} size={glyphSize} />
        </span>
      )}
    </div>
  );
}

/** The analysis mark: a tick, with a name, never the colour alone. */
function AnalysedMark({ analysed, label }: { analysed: boolean; label: string }) {
  if (!analysed) return null;
  return (
    <span className="text-success" title={label} aria-label={label}>
      <Glyph name="check" size={9} />
    </span>
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
  const t = useT();
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
        className={`group flex cursor-default flex-col overflow-hidden rounded-lg p-1 transition-[background-color,box-shadow,transform] duration-base hover:-translate-y-px ${
          active
            ? "bg-accent-soft ring-2 ring-accent"
            : selected
              ? "bg-elevated ring-1 ring-accent/60"
              : "bg-elevated/50 hover:bg-elevated hover:shadow-raised"
        }`}
      >
        <div className="relative overflow-hidden rounded">
          <Thumb
            asset={asset}
            projectId={projectId}
            className="aspect-video transition-transform duration-base group-hover:scale-[1.04]"
          />

          {/* Duration on the frame, as every media browser in this category
              puts it: it is the one number you read before the name. */}
          {asset.duration_ms != null && (
            <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/70 px-1.5 font-mono text-2xs tabular-nums text-white">
              {shortDuration(asset.duration_ms)}
            </span>
          )}

          <span
            className="pointer-events-none absolute left-1 top-1 flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-white"
            title={t(KIND_LABEL[asset.kind])}
          >
            <Glyph name={KIND_GLYPH[asset.kind]} size={9} />
          </span>

          {/* Selection is marked, not merely tinted, so the state survives for
              anyone who cannot pick the accent out of the surface behind it.
              The mark is filled for the item the inspector is following and
              outlined for the rest of a multiple selection -- which is the
              distinction the tint alone was being asked to carry. */}
          {(selected || active) && (
            <span
              className={`pointer-events-none absolute right-1 top-1 flex h-[16px] w-[16px] items-center justify-center rounded-full shadow-raised ${
                active
                  ? "bg-accent-strong text-accent-fg"
                  : "bg-ground/85 text-accent-strong ring-1 ring-accent"
              }`}
            >
              <Glyph name="check" size={9} />
            </span>
          )}
        </div>

        <div className="flex flex-col gap-0.5 px-1 pb-0.5 pt-1.5">
          <p className="truncate text-xs text-fg" title={asset.original_filename}>
            {asset.original_filename}
          </p>
          <div className="flex items-center justify-between gap-1 font-mono text-2xs tabular-nums text-faint">
            <span className="truncate">
              {asset.width ? `${asset.width}×${asset.height}` : t(KIND_LABEL[asset.kind])}
              {asset.fps ? ` · ${formatFps(asset.fps)}` : ""}
            </span>
            <span className="flex shrink-0 items-center gap-1">
              <AnalysedMark analysed={analyzed} label={t("media.analysed")} />
              <Badge tone={STATUS_TONE[asset.status]}>{t(STATUS_LABEL[asset.status])}</Badge>
            </span>
          </div>
        </div>
      </div>
    </li>
  );
}

// -------------------------------------------------------------------- list row
/**
 * Which metric columns the panel is wide enough for.
 *
 * The browser is a 230-290px panel, and six fixed columns do not fit in it: the
 * filename -- the only flexible column and the one actually used to find
 * things -- is what gets squeezed to nothing. So the metrics are dropped from
 * the least useful inwards as the panel narrows, and duration and status, which
 * every row is read for, are never dropped.
 *
 * There are no column headings. At this width a heading would have to be an
 * abbreviation, and in a language with longer words than English it would have
 * to be a worse one; a duration, a resolution and a frame rate are already
 * self-describing, and the inspector names every field in full.
 */
function columnsFor(width: number) {
  return {
    resolution: width >= 275,
    fps: width >= 330,
    size: width >= 380,
  };
}

function ListRow({
  asset,
  projectId,
  selected,
  active,
  analyzed,
  columns,
  onSelect,
  onOpen,
}: {
  asset: MediaAsset;
  projectId: string;
  selected: boolean;
  active: boolean;
  analyzed: boolean;
  columns: ReturnType<typeof columnsFor>;
  onSelect: (event: React.MouseEvent) => void;
  onOpen: () => void;
}) {
  const t = useT();
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
      className={`cursor-default border-b border-subtle/40 transition-colors duration-fast ${
        active ? "bg-accent-soft" : selected ? "bg-elevated" : "hover:bg-elevated"
      }`}
    >
      {/* The current row is marked on the left edge as well as by its fill, so
          it stays identifiable in a list of forty tinted-or-not rows. */}
      {/* A left edge in two weights: solid for the item the inspector follows,
          half for the rest of a multiple selection, nothing otherwise. */}
      <td className="w-[3px] p-0">
        <span
          className={`block h-[30px] w-[3px] ${
            active ? "bg-accent" : selected ? "bg-accent/45" : "bg-transparent"
          }`}
        />
      </td>
      <td className="w-[38px] py-1 pl-1 pr-0">
        <Thumb
          asset={asset}
          projectId={projectId}
          glyphSize={11}
          className="h-[22px] w-[34px] rounded"
        />
      </td>
      <td className="max-w-0 truncate px-1.5 text-xs text-fg" title={asset.original_filename}>
        <span className="mr-1 inline-block align-middle">
          <AnalysedMark analysed={analyzed} label={t("media.analysed")} />
        </span>
        {asset.original_filename}
      </td>
      <td className="w-[44px] px-1 text-right font-mono text-2xs tabular-nums text-muted">
        {shortDuration(asset.duration_ms)}
      </td>
      {columns.resolution && (
        <td className="w-[62px] px-1 text-right font-mono text-2xs tabular-nums text-faint">
          {resolution(asset.width, asset.height)}
        </td>
      )}
      {columns.fps && (
        <td className="w-[32px] px-1 text-right font-mono text-2xs tabular-nums text-faint">
          {formatFps(asset.fps)}
        </td>
      )}
      {columns.size && (
        <td className="w-[48px] px-1 text-right font-mono text-2xs tabular-nums text-faint">
          {bytes(asset.bytes_size)}
        </td>
      )}
      {/* Fixed width, clipped rather than allowed to push the filename out:
          under `table-fixed` an over-wide badge steals from the only flexible
          column, which is the one the list is read for. */}
      <td className="w-[58px] overflow-hidden px-1 pr-1.5 text-right">
        <Badge tone={STATUS_TONE[asset.status]}>{t(STATUS_LABEL[asset.status])}</Badge>
      </td>
    </tr>
  );
}

// ----------------------------------------------------------------- compact row
/** Name and duration, one line. For finding a filename in a library of hundreds. */
function CompactRow({
  asset,
  selected,
  active,
  analyzed,
  onSelect,
  onOpen,
}: {
  asset: MediaAsset;
  selected: boolean;
  active: boolean;
  analyzed: boolean;
  onSelect: (event: React.MouseEvent) => void;
  onOpen: () => void;
}) {
  const t = useT();
  return (
    <li
      role="option"
      aria-selected={selected}
      tabIndex={0}
      onClick={onSelect}
      onDoubleClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter") onOpen();
      }}
      className={`flex h-[22px] cursor-default items-center gap-1.5 border-l-2 px-1.5 transition-colors ${
        active
          ? "border-l-accent bg-accent-soft"
          : selected
            ? "border-l-accent/45 bg-elevated"
            : "border-l-transparent hover:bg-elevated"
      }`}
    >
      <span className="shrink-0 text-faint" title={t(KIND_LABEL[asset.kind])}>
        <Glyph name={KIND_GLYPH[asset.kind]} size={10} />
      </span>
      <span className="min-w-0 flex-1 truncate text-xs text-fg" title={asset.original_filename}>
        {asset.original_filename}
      </span>
      <AnalysedMark analysed={analyzed} label={t("media.analysed")} />
      <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
        {shortDuration(asset.duration_ms)}
      </span>
      {asset.status !== "ready" && (
        <Badge tone={STATUS_TONE[asset.status]}>{t(STATUS_LABEL[asset.status])}</Badge>
      )}
    </li>
  );
}

// --------------------------------------------------------------------- browser
export function MediaBrowser({
  projectId,
  media,
  loading,
  analyzedIds,
  onUploaded,
  wide = false,
}: {
  projectId: string;
  media: MediaAsset[];
  loading: boolean;
  analyzedIds: Set<string>;
  onUploaded: () => void;
  /**
   * Whether the browser has the viewport rather than a column.
   *
   * The Assets workspace passes this. It changes the tile grid from two columns
   * to as many as fit, and puts the import and filter controls on one row --
   * the same component doing the same job with room to do it in, not a second
   * implementation.
   */
  wide?: boolean;
}) {
  const t = useT();
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
  const [panelWidth, setPanelWidth] = useState(0);

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
  const filtered = visible.length !== media.length;

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
    const modifier =
      event.ctrlKey || event.metaKey ? "toggle" : event.shiftKey ? "range" : "replace";
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

  /** The panel's own width, which decides how many columns the list can carry. */
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setPanelWidth(element.clientWidth));
    observer.observe(element);
    setPanelWidth(element.clientWidth);
    return () => observer.disconnect();
  }, []);

  const columns = columnsFor(panelWidth);
  const columnCount = 4 + Number(columns.resolution) + Number(columns.fps) + Number(columns.size);

  const ROW_HEIGHT = view === "compact" ? 22 : 30;
  const rows = useWindowedRange(
    scrollRef,
    view === "grid" ? 0 : visible.length,
    ROW_HEIGHT,
  );

  return (
    <Panel className="h-full overflow-hidden rounded-xl shadow-panel">
      <PanelHeader title={t("media.title")} icon="layers">
        <span className="ml-auto flex items-center gap-0.5">
          {(
            [
              ["grid", "media.view.grid", "grid"],
              ["list", "media.view.list", "list"],
              ["compact", "media.view.compact", "rows"],
            ] as [BrowserView, MessageKey, GlyphName][]
          ).map(([value, label, glyph]) => (
            <IconButton
              key={value}
              size="sm"
              label={t(label)}
              active={view === value}
              onClick={() => setView(value)}
            >
              <Glyph name={glyph} size={12} />
            </IconButton>
          ))}
        </span>
      </PanelHeader>

      {/* ---- import ---- */}
      <div className="flex shrink-0 items-center gap-1.5 px-panel pb-2">
        <Button
          size="sm"
          title={t("media.import.filesHint")}
          onClick={() => fileInput.current?.click()}
          disabled={Boolean(uploading)}
        >
          <Glyph name="plus" size={10} />
          {t("media.import.files")}
        </Button>
        <Button
          size="sm"
          title={t("media.import.folderHint")}
          onClick={() => folderInput.current?.click()}
          disabled={Boolean(uploading)}
        >
          <Glyph name="folder" size={10} />
          {t("media.import.folder")}
        </Button>
        <Button
          size="sm"
          tone="primary"
          className="ml-auto"
          disabled={insertable.length === 0}
          title={
            insertable.length === 0
              ? t("media.toTimeline.none")
              : t("media.toTimeline.hint", { count: insertable.length })
          }
          onClick={() => addMedia(insertable)}
        >
          {t("media.toTimeline")}
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
      <div className="flex shrink-0 items-center gap-1.5 px-panel pb-2.5">
        <div className={`relative min-w-0 ${wide ? "w-[360px] shrink-0" : "flex-1"}`}>
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint">
            <Glyph name="search" size={12} />
          </span>
          <TextInput
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t("media.search")}
            aria-label={t("media.search.label")}
            className="pl-7"
          />
        </div>
        <select
          value={filter}
          onChange={(event) => setFilter(event.target.value as Filter)}
          aria-label={t("media.filter.label")}
          className="h-control shrink-0 rounded-md border border-subtle bg-elevated px-2 text-2xs text-fg shadow-raised transition-colors hover:border-strong focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30"
        >
          <option value="all">{t("media.filter.all")}</option>
          <option value="video">{t("media.filter.video")}</option>
          <option value="audio">{t("media.filter.audio")}</option>
          <option value="image">{t("media.filter.image")}</option>
          <option value="unanalyzed">{t("media.filter.unanalyzed")}</option>
        </select>
      </div>

      {uploading && (
        <div className="shrink-0 px-panel pb-2">
          <div className="flex justify-between font-mono text-2xs tabular-nums text-muted">
            <span>{t("media.uploading")}</span>
            <span>
              {uploading.done}/{uploading.total}
            </span>
          </div>
          <div className="mt-1">
            <ProgressBar fraction={uploading.done / uploading.total} label={t("media.uploading")} />
          </div>
        </div>
      )}

      {failures.length > 0 && (
        <ul className="max-h-24 shrink-0 overflow-y-auto">
          {failures.map((failure) => (
            <li key={failure.filename} className="px-panel pb-1.5">
              <ErrorNote hint={failure.hint}>
                <span className="font-mono">{failure.filename}</span> — {failure.message}
              </ErrorNote>
            </li>
          ))}
        </ul>
      )}

      {/* ---- the library ----
          `overscroll-contain` stops a flick at the end of the list from
          scrolling the page behind it, which in a fixed-viewport application
          means scrolling nothing while looking broken. */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        {visible.length === 0 ? (
          loading ? (
            <EmptyState icon="film">{t("media.loading")}</EmptyState>
          ) : media.length === 0 ? (
            <EmptyState
              icon="film"
              title={t("media.empty.title")}
              action={
                <Button size="sm" onClick={() => fileInput.current?.click()}>
                  <Glyph name="plus" size={10} />
                  {t("media.import.files")}
                </Button>
              }
            >
              {t("media.empty.body")}
            </EmptyState>
          ) : (
            <EmptyState
              icon="search"
              title={t("media.empty.filtered.title")}
              action={
                <Button
                  size="sm"
                  onClick={() => {
                    setSearch("");
                    setFilter("all");
                  }}
                >
                  {t("media.clearFilter")}
                </Button>
              }
            >
              {t("media.empty.filtered.body")}
            </EmptyState>
          )
        ) : view === "grid" ? (
          <ul
            role="listbox"
            aria-label={t("media.library")}
            aria-multiselectable
            className={`grid gap-2 px-panel pb-panel ${
              wide
                ? "grid-cols-3 md:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6"
                : "grid-cols-2"
            }`}
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
        ) : view === "compact" ? (
          <ul role="listbox" aria-label={t("media.library")} aria-multiselectable>
            {rows.padTop > 0 && <li style={{ height: rows.padTop }} aria-hidden />}
            {visible.slice(rows.first, rows.last).map((asset) => (
              <CompactRow
                key={asset.id}
                asset={asset}
                selected={selectedIds.includes(asset.id)}
                active={asset.id === activeId}
                analyzed={analyzedIds.has(asset.id)}
                onSelect={(event) => onSelect(asset, event)}
                onOpen={() => openInSource(asset)}
              />
            ))}
            {rows.padBottom > 0 && <li style={{ height: rows.padBottom }} aria-hidden />}
          </ul>
        ) : (
          <table
            role="grid"
            aria-label={t("media.library")}
            aria-rowcount={visible.length}
            className="w-full table-fixed border-collapse"
          >
            <tbody>
              {rows.padTop > 0 && (
                <tr style={{ height: rows.padTop }} aria-hidden>
                  <td colSpan={columnCount} />
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
                  columns={columns}
                  onSelect={(event) => onSelect(asset, event)}
                  onOpen={() => openInSource(asset)}
                />
              ))}
              {rows.padBottom > 0 && (
                <tr style={{ height: rows.padBottom }} aria-hidden>
                  <td colSpan={columnCount} />
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      <footer className="flex h-row shrink-0 items-center justify-between gap-2 px-panel pb-1 font-mono text-2xs tabular-nums text-faint">
        <span>
          {filtered
            ? t("media.count", { visible: visible.length, total: media.length })
            : t.plural("media.countAll", media.length)}
        </span>
        <span className="truncate">
          {selectedIds.length > 0 && `${t("media.selected", { count: selectedIds.length })} · `}
          {t("media.analysedCount", { count: analyzedIds.size })}
        </span>
      </footer>
    </Panel>
  );
}
