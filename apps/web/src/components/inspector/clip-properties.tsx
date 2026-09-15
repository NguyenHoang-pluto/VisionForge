"use client";

import type { MediaAsset } from "@/lib/api";
import { bytes, fps as formatFps, resolution, timecode } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { clipDuration, MAX_CLIP_MS, MIN_CLIP_MS } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Button,
  EmptyState,
  ErrorNote,
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
 *
 * Laid out as label/value pairs on a fixed column rather than as free-flowing
 * text, because a readout is scanned rather than read: you come here to check
 * one number, and a ragged right edge makes you find it again every time.
 */

const KIND_LABEL: Record<MediaAsset["kind"], MessageKey> = {
  video: "media.kind.video",
  audio: "media.kind.audio",
  image: "media.kind.image",
};

const STATUS_LABEL: Record<MediaAsset["status"], MessageKey> = {
  pending_upload: "media.status.pending_upload",
  uploaded: "media.status.uploaded",
  processing: "media.status.processing",
  ready: "media.status.ready",
  failed: "media.status.failed",
};

export function ClipProperties({ media }: { media: Map<string, MediaAsset> }) {
  const t = useT();
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
    return <EmptyState icon="film">{t("inspector.noSelection")}</EmptyState>;
  }

  const startMs = clip ? clips.slice(0, index).reduce((sum, c) => sum + clipDuration(c), 0) : 0;

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      {clip && (
        <section>
          <SectionTitle
            aside={
              <span className="shrink-0 font-mono text-2xs tabular-nums text-dim">
                {t("clip.position", { index: index + 1, total: clips.length })}
              </span>
            }
          >
            {t("clip.section.timeline")}
          </SectionTitle>

          <div className="mt-1.5 grid grid-cols-2 gap-2">
            <Field label={t("clip.in")} hint={t("clip.inHint")}>
              <NumberInput
                value={Math.round(clip.inMs)}
                min={0}
                step={100}
                max={Math.round(clip.outMs - MIN_CLIP_MS)}
                aria-label={t("clip.inLabel")}
                onChange={(event) =>
                  trim(clip.id, "in", Number(event.target.value), asset.duration_ms)
                }
              />
            </Field>
            <Field label={t("clip.out")}>
              <NumberInput
                value={Math.round(clip.outMs)}
                min={Math.round(clip.inMs + MIN_CLIP_MS)}
                step={100}
                max={asset.duration_ms ?? undefined}
                aria-label={t("clip.outLabel")}
                onChange={(event) =>
                  trim(clip.id, "out", Number(event.target.value), asset.duration_ms)
                }
              />
            </Field>
          </div>

          <div className="mt-1.5">
            <Row label={t("clip.duration")} value={timecode(clipDuration(clip))} />
            <Row label={t("clip.startsAt")} value={timecode(startMs)} />
            <Row
              label={t("clip.limits")}
              value={`${MIN_CLIP_MS} ms – ${MAX_CLIP_MS / 1000} s`}
              title={t("clip.limitsHint")}
            />
          </div>

          <div className="mt-2 flex gap-1">
            <Button size="sm" onClick={() => setPlayhead(startMs)}>
              {t("clip.goTo")}
            </Button>
            <Button size="sm" tone="danger" onClick={() => removeClip(clip.id)}>
              <Glyph name="trash" size={10} />
              {t("clip.remove")}
            </Button>
          </div>
        </section>
      )}

      <section>
        <SectionTitle>{t("clip.section.source")}</SectionTitle>
        <div className="mt-1">
          <Row
            label={t("clip.file")}
            value={asset.original_filename}
            title={asset.original_filename}
          />
          <Row label={t("clip.kind")} value={t(KIND_LABEL[asset.kind])} />
          <Row
            label={t("clip.status")}
            value={t(STATUS_LABEL[asset.status])}
            tone={asset.status === "failed" ? "danger" : undefined}
          />
          <Row label={t("clip.duration")} value={timecode(asset.duration_ms)} />
          <Row label={t("clip.resolution")} value={resolution(asset.width, asset.height)} />
          <Row label={t("clip.fps")} value={formatFps(asset.fps)} />
          <Row label={t("clip.codec")} value={asset.codec ?? t("common.dash")} />
          <Row label={t("clip.pixFmt")} value={asset.pix_fmt ?? t("common.dash")} />
          <Row label={t("clip.container")} value={asset.container_format ?? t("common.dash")} />
          <Row
            label={t("clip.audio")}
            value={
              asset.channels
                ? t("clip.audioValue", {
                    channels: asset.channels,
                    rate: asset.sample_rate ?? t("common.dash"),
                  })
                : t("clip.audioNone")
            }
          />
          <Row
            label={t("clip.bitRate")}
            value={asset.bit_rate ? `${Math.round(asset.bit_rate / 1000)} kbps` : t("common.dash")}
          />
          <Row label={t("clip.size")} value={bytes(asset.bytes_size)} />
          <Row
            label={t("clip.proxy")}
            value={asset.has_proxy ? t("clip.proxy720") : t("clip.proxyNone")}
          />
        </div>

        {asset.error?.message && (
          <div className="mt-2">
            <ErrorNote hint={asset.error.hint ?? null}>{asset.error.message}</ErrorNote>
          </div>
        )}
      </section>
    </div>
  );
}
