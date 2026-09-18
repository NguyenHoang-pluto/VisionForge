"use client";

import { useState } from "react";

import { useMutation, useQuery } from "@tanstack/react-query";

import {
  ApiError,
  api,
  CUE_BOUNDS,
  SUBTITLE_POSITIONS,
  SUBTITLE_STYLES,
  type SubtitleFailure,
  type SubtitlePosition,
  type SubtitleStyle,
} from "@/lib/api";
import { timecode } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import { cueId, cueProblems, sortedCues, totalDuration } from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import {
  Badge,
  Button,
  ErrorNote,
  Field,
  Glyph,
  NumberInput,
  Row,
  SectionTitle,
  Select,
  TextArea,
} from "@/components/ui";

/**
 * Subtitles.
 *
 * One track over the whole programme, which is what `EditPlan` can express. Cue
 * times are *timeline* positions, not offsets inside a clip, because that is
 * what the server stores -- a cue belongs to the edit, not to whichever shot
 * happens to be under it, and pinning one to a clip would make it drift the
 * moment the edit was re-paced.
 *
 * The look is two dropdowns of ids. There is no font field, no colour picker
 * and no x/y here, and their absence is the feature: the server's preset table
 * decides all of them, so a client cannot name a font path. The panel shows
 * what each preset resolves to -- the font, size and weight the server reports
 * -- so the choice is informed without being editable.
 */

const STYLE_LABEL: Record<SubtitleStyle, MessageKey> = {
  clean: "subs.style.clean",
  bold: "subs.style.bold",
  minimal: "subs.style.minimal",
  cinematic: "subs.style.cinematic",
  social: "subs.style.social",
};

const POSITION_LABEL: Record<SubtitlePosition, MessageKey> = {
  bottom: "subs.position.bottom",
  center: "subs.position.center",
  top: "subs.position.top",
  bottom_left: "subs.position.bottom_left",
  bottom_right: "subs.position.bottom_right",
};

/** Each failure the server can report, said in words a user can act on. */
const FAILURE_LABEL: Record<SubtitleFailure, MessageKey> = {
  provider_disabled: "subs.ai.fail.disabled",
  provider_unavailable: "subs.ai.fail.unavailable",
  provider_error: "subs.ai.fail.error",
  unreadable: "subs.ai.fail.unreadable",
  no_usable_cues: "subs.ai.fail.noCues",
  timeline_too_short: "subs.ai.fail.tooShort",
};

export function SubtitlesPanel({ projectId }: { projectId: string }) {
  const t = useT();

  const clips = useEditorStore((s) => s.clips);
  const subtitles = useEditorStore((s) => s.subtitles);
  const selectedCueId = useEditorStore((s) => s.selectedCueId);
  const selectCue = useEditorStore((s) => s.selectCue);
  const addCue = useEditorStore((s) => s.addCue);
  const updateCue = useEditorStore((s) => s.updateCue);
  const removeCue = useEditorStore((s) => s.removeCue);
  const setCues = useEditorStore((s) => s.setCues);
  const setStyle = useEditorStore((s) => s.setSubtitleStyle);
  const setPosition = useEditorStore((s) => s.setSubtitlePosition);
  const clearSubtitles = useEditorStore((s) => s.clearSubtitles);
  const setPlayhead = useEditorStore((s) => s.setPlayhead);
  const committedPlanId = useEditorStore((s) => s.committedPlanId);
  const dirty = useEditorStore((s) => s.dirty);

  const [request, setRequest] = useState("");
  const [failure, setFailure] = useState<SubtitleFailure | "request_failed" | null>(null);

  const capabilities = useQuery({
    queryKey: ["planner-capabilities"],
    queryFn: api.plannerCapabilities,
    staleTime: 5 * 60 * 1000,
  });

  const totalMs = totalDuration(clips);
  const cues = sortedCues(subtitles?.cues ?? []);
  const problems = cueProblems(subtitles, totalMs);
  const style = subtitles?.style ?? "clean";
  const position = subtitles?.position ?? "bottom";
  const preset = capabilities.data?.subtitle_styles.find((item) => item.id === style);

  // The route runs against a *stored* plan, so the model is shown the timing
  // that will actually be rendered. An unsaved draft has no such plan, and
  // offering the button anyway would be offering a request that cannot be made.
  const planReady = Boolean(committedPlanId) && !dirty;
  const aiAvailable = capabilities.data?.ai_available ?? false;

  const suggest = useMutation({
    mutationFn: () =>
      api.suggestSubtitles(projectId, committedPlanId!, {
        request_text: request.trim() || null,
        style,
        position,
      }),
    onSuccess: (result) => {
      if (!result.ok || !result.subtitles) {
        // No cues, and nothing invented to fill the gap.
        setFailure(result.failure ?? "unreadable");
        return;
      }
      setFailure(null);
      setCues(
        result.subtitles.cues.map((cue) => ({
          id: cueId(),
          startMs: cue.start_ms,
          endMs: cue.end_ms,
          text: cue.text,
        })),
      );
      setStyle(result.subtitles.style);
      setPosition(result.subtitles.position);
    },
    onError: (caught: unknown) =>
      setFailure(caught instanceof ApiError ? "request_failed" : "request_failed"),
  });

  return (
    <div className="flex flex-col gap-panel-gap p-panel">
      {/* ------------------------------------------------------ the look --- */}
      <section>
        <SectionTitle
          aside={
            <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
              {t.plural("subs.cueCount", cues.length)}
            </span>
          }
        >
          {t("subs.section.look")}
        </SectionTitle>

        <div className="mt-1.5 grid grid-cols-2 gap-2">
          <Field label={t("subs.style")} hint={t("subs.styleHint")}>
            <Select
              value={style}
              aria-label={t("subs.style")}
              onChange={(event) => setStyle(event.target.value as SubtitleStyle)}
            >
              {(capabilities.data?.subtitle_styles.map((item) => item.id) ?? SUBTITLE_STYLES).map(
                (id) => (
                  <option key={id} value={id}>
                    {t(STYLE_LABEL[id])}
                  </option>
                ),
              )}
            </Select>
          </Field>

          <Field label={t("subs.position")} hint={t("subs.positionHint")}>
            <Select
              value={position}
              aria-label={t("subs.position")}
              onChange={(event) => setPosition(event.target.value as SubtitlePosition)}
            >
              {(capabilities.data?.subtitle_positions ?? SUBTITLE_POSITIONS).map((id) => (
                <option key={id} value={id}>
                  {t(POSITION_LABEL[id])}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        {/* What the preset resolves to, reported by the server. Shown, never
            edited: the client names a preset and the server decides the rest,
            which is what keeps a font path out of the render. */}
        {preset && (
          <div className="mt-1.5">
            <Row label={t("subs.font")} value={preset.font} title={t("subs.fontHint")} />
            <Row
              label={t("subs.size")}
              value={t("subs.sizeValue", { size: preset.size })}
              title={t("subs.sizeHint")}
            />
            <Row
              label={t("subs.weight")}
              value={preset.bold ? t("subs.weightBold") : t("subs.weightRegular")}
            />
          </div>
        )}
      </section>

      {/* ------------------------------------------------------- the cues --- */}
      <section>
        <SectionTitle
          aside={
            <Button
              size="sm"
              onClick={() => {
                const created = addCue();
                if (created === null) setFailure("timeline_too_short");
              }}
              disabled={clips.length === 0 || cues.length >= CUE_BOUNDS.maxCues}
            >
              <Glyph name="plus" size={10} />
              {t("subs.add")}
            </Button>
          }
        >
          {t("subs.section.cues")}
        </SectionTitle>

        {cues.length === 0 ? (
          <p className="mt-1.5 text-2xs leading-snug text-faint">
            {clips.length === 0 ? t("subs.emptyTimeline") : t("subs.empty")}
          </p>
        ) : (
          <ul className="mt-1.5 flex flex-col gap-1.5">
            {cues.map((cue, index) => {
              const selected = cue.id === selectedCueId;
              return (
                <li
                  key={cue.id}
                  onPointerDown={() => selectCue(cue.id)}
                  className={`rounded-md p-2 transition-[background-color,box-shadow] duration-fast ${
                    selected
                      ? "bg-elevated shadow-raised ring-1 ring-accent-strong"
                      : "bg-sunken/60 hover:bg-sunken"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-2xs tabular-nums text-faint">
                      {index + 1}
                    </span>
                    <button
                      type="button"
                      onClick={() => setPlayhead(cue.startMs)}
                      title={t("subs.goTo")}
                      className="font-mono text-2xs tabular-nums text-muted transition-colors duration-fast hover:text-fg"
                    >
                      {timecode(cue.startMs, false)} – {timecode(cue.endMs, false)}
                    </button>
                    <Badge tone={cue.text.length > CUE_BOUNDS.maxChars ? "danger" : "neutral"}>
                      {cue.text.length}/{CUE_BOUNDS.maxChars}
                    </Badge>
                    <button
                      type="button"
                      onClick={() => removeCue(cue.id)}
                      aria-label={t("subs.removeCue", { index: index + 1 })}
                      className="ml-auto text-faint transition-colors duration-fast hover:text-danger"
                    >
                      <Glyph name="trash" size={10} />
                    </button>
                  </div>

                  <TextArea
                    className="mt-1.5"
                    rows={2}
                    value={cue.text}
                    maxLength={CUE_BOUNDS.maxChars}
                    placeholder={t("subs.textPlaceholder")}
                    aria-label={t("subs.textLabel", { index: index + 1 })}
                    onChange={(event) => updateCue(cue.id, { text: event.target.value })}
                  />

                  {selected && (
                    <div className="mt-1.5 grid grid-cols-2 gap-2">
                      <Field label={t("subs.from")}>
                        <NumberInput
                          value={Math.round(cue.startMs)}
                          min={0}
                          max={Math.max(0, Math.round(cue.endMs) - CUE_BOUNDS.minMs)}
                          step={100}
                          aria-label={t("subs.from")}
                          onChange={(event) =>
                            updateCue(cue.id, { startMs: Number(event.target.value) })
                          }
                        />
                      </Field>
                      <Field label={t("subs.to")}>
                        <NumberInput
                          value={Math.round(cue.endMs)}
                          min={Math.round(cue.startMs) + CUE_BOUNDS.minMs}
                          max={Math.round(totalMs)}
                          step={100}
                          aria-label={t("subs.to")}
                          onChange={(event) =>
                            updateCue(cue.id, { endMs: Number(event.target.value) })
                          }
                        />
                      </Field>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        {problems.length > 0 && (
          <ul className="mt-2 flex flex-col gap-1">
            {problems.slice(0, 4).map((problem) => (
              <li key={problem.message} className="text-2xs leading-snug text-danger">
                {problem.message}
              </li>
            ))}
          </ul>
        )}

        {cues.length > 0 && (
          <Button size="sm" tone="danger" className="mt-2" onClick={clearSubtitles}>
            {t("subs.clear")}
          </Button>
        )}
      </section>

      {/* --------------------------------------------------------- the AI --- */}
      <section>
        <SectionTitle>{t("subs.section.ai")}</SectionTitle>

        <p className="mt-1 text-2xs leading-snug text-faint">{t("subs.aiHint")}</p>

        <TextArea
          className="mt-1.5"
          rows={2}
          value={request}
          maxLength={capabilities.data?.max_request_chars ?? 600}
          placeholder={t("subs.aiPlaceholder")}
          aria-label={t("subs.aiLabel")}
          onChange={(event) => setRequest(event.target.value)}
          disabled={!aiAvailable}
        />

        <Button
          size="sm"
          tone="primary"
          className="mt-1.5"
          disabled={!aiAvailable || !planReady || suggest.isPending}
          onClick={() => {
            setFailure(null);
            suggest.mutate();
          }}
        >
          <Glyph name="wand" size={10} />
          {suggest.isPending ? t("subs.aiWorking") : t("subs.aiWrite")}
        </Button>

        {/* Honest about why the button is off, rather than a disabled control
            with no explanation. */}
        {!aiAvailable && (
          <p className="mt-1.5 text-2xs leading-snug text-faint">{t("subs.ai.fail.disabled")}</p>
        )}
        {aiAvailable && !planReady && (
          <p className="mt-1.5 text-2xs leading-snug text-faint">{t("subs.aiNeedsPlan")}</p>
        )}

        {failure && (
          <div className="mt-2">
            <ErrorNote hint={suggest.data?.detail || null}>
              {failure === "request_failed"
                ? t("subs.ai.fail.request")
                : t(FAILURE_LABEL[failure])}
            </ErrorNote>
          </div>
        )}
      </section>
    </div>
  );
}
