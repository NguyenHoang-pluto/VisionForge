"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  api,
  STYLE_STRENGTHS,
  type MediaAsset,
  type Measurement,
  type ReferenceProfile,
  type StyleStrength,
} from "@/lib/api";
import { shortDuration } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { useThumbnailUrl } from "@/lib/media-urls";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  Glyph,
  Meter,
  Row,
  SectionTitle,
  SegmentedControl,
  Spinner,
  StatusDot,
} from "@/components/ui";

/**
 * The reference video, in the AI Edit tab.
 *
 * Phase 8 in the workstation, and deliberately a section rather than a mode:
 * "cut it like this clip" is a way of shaping an edit, in the panel where edits
 * are shaped, beside style and target length.
 *
 * Two things this panel must get right, because the feature is worthless
 * otherwise:
 *
 * **Show the measurements, with their confidence.** A user who cannot see that
 * the shot length was read from four cuts has no way to judge why the dial did
 * little. Every figure here is paired with the confidence the server measured
 * it at, and an unmeasured feature is shown as unmeasured rather than as zero.
 *
 * **Say when there is nothing to show yet.** A nominated reference that has not
 * been analysed has an empty profile, and the honest response to that is a
 * button that runs the analysis, not a panel of dashes.
 */

const PACING_LABEL: Record<string, MessageKey> = {
  slow: "reference.pacing.slow",
  measured: "reference.pacing.measured",
  brisk: "reference.pacing.brisk",
  rapid: "reference.pacing.rapid",
};

/** A measured 0..1 feature, as a bar plus its confidence. */
function Feature({
  label,
  measurement,
  format,
}: {
  label: string;
  measurement: Measurement | null;
  format?: (value: number) => string;
}) {
  const t = useT();

  if (!measurement) {
    // Absent, not zero. A dash with a reason beats a bar at zero, which would
    // read as "measured, and it was nothing".
    return <Row label={label} value={t("reference.unmeasured")} />;
  }

  const shown = format ? format(measurement.value) : `${Math.round(measurement.value * 100)}%`;
  return (
    <div className="flex min-h-row flex-col justify-center gap-1 py-[3px]">
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate text-xs text-muted">{label}</span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-fg">{shown}</span>
      </div>
      <div className="flex items-center gap-2">
        <Meter value={measurement.value} max={1} />
        <span
          className="w-8 shrink-0 text-right font-mono text-2xs tabular-nums text-faint"
          title={t("reference.confidenceHint")}
        >
          {Math.round(measurement.confidence * 100)}%
        </span>
      </div>
    </div>
  );
}

function ReferenceThumb({ projectId, asset }: { projectId: string; asset: MediaAsset | undefined }) {
  const thumbnail = useThumbnailUrl(projectId, asset ?? null);
  const url = thumbnail.data?.url;

  return (
    <div className="relative aspect-video w-[84px] shrink-0 overflow-hidden rounded bg-sunken">
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt="" loading="lazy" className="h-full w-full object-cover" />
      ) : (
        <span className="flex h-full w-full items-center justify-center text-faint/60" aria-hidden>
          <Glyph name="film" size={14} />
        </span>
      )}
    </div>
  );
}

function Detected({ profile }: { profile: ReferenceProfile }) {
  const t = useT();
  const pacing = profile.pacing ? t(PACING_LABEL[profile.pacing]) : null;

  return (
    <>
      <div className="mt-1">
        <Row
          label={t("reference.pacing")}
          value={
            pacing && profile.shot_ms
              ? t("reference.pacingValue", {
                  pacing,
                  shot: shortDuration(profile.shot_ms.value),
                })
              : t("reference.unmeasured")
          }
        />
        <Row
          label={t("reference.cutRate")}
          value={
            profile.cut_rate
              ? t("reference.cutRateValue", { rate: profile.cut_rate.value.toFixed(1) })
              : t("reference.unmeasured")
          }
        />
        <Row
          label={t("reference.shots")}
          value={profile.scene_count !== null ? String(profile.scene_count) : t("common.dash")}
        />
        {profile.bpm !== null && profile.bpm > 0 && (
          <Row
            label={t("audio.bpm")}
            value={t("audio.bpmValue", { bpm: profile.bpm.toFixed(1) })}
          />
        )}
      </div>

      <div className="mt-2 flex flex-col">
        <Feature label={t("reference.motion")} measurement={profile.motion} />
        <Feature label={t("reference.saturation")} measurement={profile.saturation} />
        <Feature label={t("reference.luminance")} measurement={profile.luminance} />
        <Feature label={t("reference.contrast")} measurement={profile.contrast} />
        <Feature label={t("reference.beatSync")} measurement={profile.beat_sync} />
      </div>

      {profile.suggests_beat_sync && (
        <p className="mt-2 text-2xs leading-snug text-faint">{t("reference.beatSyncHint")}</p>
      )}
    </>
  );
}

export function ReferencePanel({
  projectId,
  media,
  mediaList,
  onAnalyze,
}: {
  projectId: string;
  media: Map<string, MediaAsset>;
  mediaList: MediaAsset[];
  onAnalyze: (mediaId: string) => void;
}) {
  const t = useT();
  const queryClient = useQueryClient();

  const strength = useEditorStore((s) => s.styleStrength);
  const setStrength = useEditorStore((s) => s.setStyleStrength);
  const activeMediaId = useEditorStore((s) => s.activeMediaId);

  const reference = useQuery({
    queryKey: ["reference", projectId],
    queryFn: () => api.reference(projectId),
    staleTime: 30_000,
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["reference", projectId] });
  };

  const choose = useMutation({
    mutationFn: (mediaId: string) => api.setReference(projectId, mediaId),
    onSuccess: invalidate,
  });
  const clear = useMutation({
    mutationFn: () => api.clearReference(projectId),
    onSuccess: invalidate,
  });

  const state = reference.data;
  const profile = state?.profile ?? null;
  const asset = state?.media_id ? media.get(state.media_id) : undefined;

  /** The clip the browser is pointing at, if it could be a reference. */
  const candidate = activeMediaId ? media.get(activeMediaId) : undefined;
  const choosable =
    candidate && candidate.kind === "video" && candidate.status === "ready" ? candidate : undefined;

  const videos = mediaList.filter((item) => item.kind === "video" && item.status === "ready");
  const pending = state?.pending_analyzers ?? [];

  return (
    <section>
      <SectionTitle
        description={t("reference.hint")}
        aside={
          state?.media_id ? (
            <Button
              size="sm"
              tone="danger"
              title={t("reference.clear")}
              disabled={clear.isPending}
              onClick={() => clear.mutate()}
            >
              <Glyph name="trash" size={10} />
            </Button>
          ) : undefined
        }
      >
        {t("reference.title")}
      </SectionTitle>

      {/* ---------------- nothing chosen ---------------- */}
      {!state?.media_id ? (
        videos.length === 0 ? (
          <p className="py-1 text-2xs leading-snug text-faint">{t("reference.noVideos")}</p>
        ) : (
          <div className="flex flex-col gap-1">
            {choosable && (
              <Button
                tone="primary"
                className="w-full"
                disabled={choose.isPending}
                onClick={() => choose.mutate(choosable.id)}
              >
                <Glyph name="wand" size={12} />
                {t("reference.useSelected")}
              </Button>
            )}
            {/* The project's own videos, offered directly. Telling a user to go
                and select something in another panel is an instruction they can
                follow only if that panel happens to be open. */}
            <ul className="flex max-h-[168px] flex-col gap-1 overflow-y-auto">
              {videos.slice(0, 20).map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    title={item.original_filename}
                    disabled={choose.isPending}
                    onClick={() => choose.mutate(item.id)}
                    className="group flex w-full items-center gap-2.5 rounded-lg bg-elevated p-1.5 text-left transition-[background-color,transform] duration-fast hover:bg-hover active:scale-[0.99] disabled:opacity-50"
                  >
                    <ReferenceThumb projectId={projectId} asset={item} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs text-fg">
                        {item.original_filename}
                      </span>
                      <span className="mt-0.5 block font-mono text-2xs tabular-nums text-faint">
                        {shortDuration(item.duration_ms)}
                      </span>
                    </span>
                    <span className="shrink-0 text-faint transition-colors duration-fast group-hover:text-accent-strong">
                      <Glyph name="chevron-right" size={12} />
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )
      ) : (
        <>
          {/* ---------------- chosen ---------------- */}
          <div className="flex items-center gap-2.5 rounded-lg bg-elevated p-1.5">
            <ReferenceThumb projectId={projectId} asset={asset} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-xs text-fg" title={asset?.original_filename}>
                {asset?.original_filename ?? state.media_id.slice(0, 8)}
              </span>
              <span className="mt-0.5 block font-mono text-2xs tabular-nums text-faint">
                {shortDuration(asset?.duration_ms ?? profile?.duration_ms ?? null)}
              </span>
            </span>
            {profile && (
              <StatusDot tone={profile.usable ? "ok" : "warn"}>
                <span className="font-mono text-2xs text-faint">
                  {Math.round(profile.confidence * 100)}%
                </span>
              </StatusDot>
            )}
          </div>

          {/* ---------------- what it measured ---------------- */}
          {reference.isLoading ? (
            <p className="flex items-center gap-2 py-2 text-2xs text-faint">
              <Spinner size={11} />
              {t("reference.reading")}
            </p>
          ) : !profile || !profile.usable ? (
            <div className="mt-2">
              <p className="text-2xs leading-snug text-warning">
                {pending.length > 0 ? t("reference.needsAnalysis") : t("reference.unusable")}
              </p>
              {pending.length > 0 && (
                <Button
                  className="mt-2 w-full"
                  onClick={() => state.media_id && onAnalyze(state.media_id)}
                >
                  <Glyph name="analyse" size={12} />
                  {t("reference.analyse")}
                </Button>
              )}
            </div>
          ) : (
            <Detected profile={profile} />
          )}

          {/* ---------------- the dial ---------------- */}
          <div className="mt-3">
            <div className="mb-1.5 flex items-baseline justify-between gap-2">
              <span className="text-2xs font-medium text-muted">{t("reference.strength")}</span>
              {strength === "0" && <Badge>{t("reference.off")}</Badge>}
            </div>
            <SegmentedControl<StyleStrength>
              label={t("reference.strength")}
              value={strength}
              onChange={setStrength}
              className="w-full [&>button]:flex-1"
              options={STYLE_STRENGTHS.map((value) => ({
                value,
                label: `${value}%`,
                title: t("reference.strengthHint", { percent: value }),
                disabled: !profile?.usable && value !== "0",
              }))}
            />
            <p className="mt-1.5 text-2xs leading-snug text-faint">
              {strength === "0" ? t("reference.zeroHint") : t("reference.appliedHint")}
            </p>
          </div>
        </>
      )}
    </section>
  );
}
