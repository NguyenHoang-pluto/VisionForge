"use client";

import type { MediaAsset } from "@/lib/api";
import { bytes, fps as formatFps, resolution, timecode } from "@/lib/format";
import { clipDuration, MAX_CLIP_MS, MIN_CLIP_MS } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Button,
  EmptyState,
  Field,
  Glyph,
  NumberInput,
  Row,
  SectionTitle,
} from "@/components/ui";

/**
 * Properties of whatever is selected.
 *
 * Two things can be: a clip on the timeline, or an asset in the browser. The
 * timeline wins when both are, because a selected clip is what the user is
 * working on and its trim is the only thing here that is editable.
 */
export function ClipProperties({ media }: { media: Map<string, MediaAsset> }) {
  const clips = useEditorStore((s) => s.clips);
  const selectedClipId = useEditorStore((s) => s.selectedClipId);
  const activeMediaId = useEditorStore((s) => s.activeMediaId);
  const trim = useEditorStore((s) => s.trim);
  const removeClip = useEditorStore((s) => s.removeClip);
  const setPlayhead = useEditorStore((s) => s.setPlayhead);

  const index = clips.findIndex((clip) => clip.id === selectedClipId);
  const clip = index >= 0 ? clips[index] : null;
  const asset = clip
    ? media.get(clip.mediaId)
    : activeMediaId
      ? media.get(activeMediaId)
      : undefined;

  if (!asset) {
    return (
      <EmptyState>
        Select a clip on the timeline or an item in the media browser to inspect it.
      </EmptyState>
    );
  }

  const startMs = clip ? clips.slice(0, index).reduce((sum, c) => sum + clipDuration(c), 0) : 0;

  return (
    <div className="flex flex-col gap-4 p-2.5">
      {clip && (
        <section>
          <SectionTitle
            aside={
              <span className="font-mono text-2xs text-dim">
                clip {index + 1} / {clips.length}
              </span>
            }
          >
            Timeline clip
          </SectionTitle>

          <div className="mt-1.5 grid grid-cols-2 gap-2">
            <Field label="In (ms)" hint="Trim point within the source">
              <NumberInput
                value={Math.round(clip.inMs)}
                min={0}
                step={100}
                max={Math.round(clip.outMs - MIN_CLIP_MS)}
                aria-label="Clip in point in milliseconds"
                onChange={(event) =>
                  trim(clip.id, "in", Number(event.target.value), asset.duration_ms)
                }
              />
            </Field>
            <Field label="Out (ms)">
              <NumberInput
                value={Math.round(clip.outMs)}
                min={Math.round(clip.inMs + MIN_CLIP_MS)}
                step={100}
                max={asset.duration_ms ?? undefined}
                aria-label="Clip out point in milliseconds"
                onChange={(event) =>
                  trim(clip.id, "out", Number(event.target.value), asset.duration_ms)
                }
              />
            </Field>
          </div>

          <div className="mt-1.5">
            <Row label="Duration" value={timecode(clipDuration(clip))} />
            <Row label="Starts at" value={timecode(startMs)} />
            <Row
              label="Limits"
              value={`${MIN_CLIP_MS} ms – ${MAX_CLIP_MS / 1000} s`}
              title="Enforced by the server on every plan."
            />
          </div>

          <div className="mt-2 flex gap-1">
            <Button size="sm" onClick={() => setPlayhead(startMs)}>
              Go to clip
            </Button>
            <Button size="sm" tone="danger" onClick={() => removeClip(clip.id)}>
              <Glyph name="trash" size={10} />
              Remove
            </Button>
          </div>
        </section>
      )}

      <section>
        <SectionTitle>Source</SectionTitle>
        <div className="mt-1">
          <Row label="File" value={asset.original_filename} title={asset.original_filename} />
          <Row label="Kind" value={asset.kind} />
          <Row label="Status" value={asset.status} tone={asset.status === "failed" ? "danger" : undefined} />
          <Row label="Duration" value={timecode(asset.duration_ms)} />
          <Row label="Resolution" value={resolution(asset.width, asset.height)} />
          <Row label="Frame rate" value={formatFps(asset.fps)} />
          <Row label="Codec" value={asset.codec ?? "—"} />
          <Row label="Pixel format" value={asset.pix_fmt ?? "—"} />
          <Row label="Container" value={asset.container_format ?? "—"} />
          <Row
            label="Audio"
            value={asset.channels ? `${asset.channels} ch · ${asset.sample_rate ?? "—"} Hz` : "none"}
          />
          <Row label="Bit rate" value={asset.bit_rate ? `${Math.round(asset.bit_rate / 1000)} kbps` : "—"} />
          <Row label="Size" value={bytes(asset.bytes_size)} />
          <Row label="Proxy" value={asset.has_proxy ? "720p" : "none"} />
        </div>

        {asset.error?.message && (
          <p className="mt-2 border-l-2 border-danger px-2 py-1 text-2xs leading-snug text-danger">
            {asset.error.message}
            {asset.error.hint && <span className="block text-danger/70">{asset.error.hint}</span>}
          </p>
        )}
      </section>
    </div>
  );
}
