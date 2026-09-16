"use client";

import {
  EFFECT_BOUNDS,
  TRANSITION_BOUNDS,
  TRANSITIONS,
  type EffectKind,
  type TransitionKind,
} from "@/lib/api";
import { timecode } from "@/lib/format";
import { useT, type MessageKey } from "@/lib/i18n";
import {
  clipPlaybackMs,
  clipSpeed,
  effectAmount,
  maxTransitionMs,
  transitionOf,
  type TimelineClip,
} from "@/lib/timeline";
import { useEditorStore } from "@/stores/editor-store";
import { Button, Field, NumberInput, Row, SectionTitle, SegmentedControl, Slider } from "@/components/ui";

/**
 * Transitions and effects for the selected clip.
 *
 * Both sections are sliders over closed vocabularies, and that is the point:
 * there is no text field here, no filter box, no "advanced" escape hatch,
 * because the server accepts a kind from an enum and a number inside a range
 * and nothing else. A control that could express more than that would be a
 * control whose value the server has to refuse.
 *
 * Every bound drawn here is mirrored from the server's, so a slider cannot be
 * dragged to a value that produces a 422. The mirror is a courtesy: the plan
 * validator re-derives all of it from the real media rows.
 */

const TRANSITION_LABEL: Record<TransitionKind, MessageKey> = {
  cut: "transition.cut",
  crossfade: "transition.crossfade",
  fade_in: "transition.fade_in",
  fade_to_black: "transition.fade_to_black",
};

const TRANSITION_HINT: Record<TransitionKind, MessageKey> = {
  cut: "transition.cut.hint",
  crossfade: "transition.crossfade.hint",
  fade_in: "transition.fade_in.hint",
  fade_to_black: "transition.fade_to_black.hint",
};

const EFFECT_LABEL: Record<EffectKind, MessageKey> = {
  zoom_in: "effect.zoom_in",
  zoom_out: "effect.zoom_out",
  slow_motion: "effect.slow_motion",
  speed_up: "effect.speed_up",
  brightness: "effect.brightness",
  contrast: "effect.contrast",
  saturation: "effect.saturation",
};

/** The colour effects, which the compiler applies with one `eq` pass. */
const COLOUR: EffectKind[] = ["brightness", "contrast", "saturation"];
/** The framing effects, which are a crop ramp and must span the whole clip. */
const FRAMING: EffectKind[] = ["zoom_in", "zoom_out"];

// -------------------------------------------------------------------- pieces
/**
 * One effect, as a slider.
 *
 * The neutral value is the off position, and releasing there removes the
 * effect rather than sending a no-op instruction. That is why there is no
 * separate on/off switch: the range already contains "not applied".
 */
function EffectSlider({
  clip,
  kind,
  disabled,
  format,
}: {
  clip: TimelineClip;
  kind: EffectKind;
  disabled?: boolean;
  format: (amount: number) => string;
}) {
  const t = useT();
  const setEffect = useEditorStore((s) => s.setEffect);
  const bounds = EFFECT_BOUNDS[kind];
  const amount = effectAmount(clip, kind);
  const active = amount !== bounds.neutral;

  return (
    <div className="flex items-center gap-2">
      <span
        className={`w-[72px] shrink-0 truncate text-2xs ${active ? "text-fg" : "text-muted"}`}
        title={t(EFFECT_LABEL[kind])}
      >
        {t(EFFECT_LABEL[kind])}
      </span>
      <Slider
        className="min-w-0 flex-1"
        aria-label={t(EFFECT_LABEL[kind])}
        value={amount}
        min={bounds.min}
        max={bounds.max}
        step={bounds.step}
        disabled={disabled}
        onChange={(event) =>
          setEffect(clip.id, { kind, amount: Number(event.target.value) })
        }
      />
      <span
        className={`w-[52px] shrink-0 text-right font-mono text-2xs tabular-nums ${
          active ? "text-fg" : "text-faint"
        }`}
      >
        {format(amount)}
      </span>
      <button
        type="button"
        disabled={!active || disabled}
        onClick={() => setEffect(clip.id, { kind, amount: bounds.neutral })}
        title={t("effect.reset")}
        aria-label={t("effect.resetOne", { name: t(EFFECT_LABEL[kind]) })}
        className="shrink-0 text-2xs text-faint transition-colors duration-fast hover:text-fg disabled:opacity-25 disabled:hover:text-faint"
      >
        ×
      </button>
    </div>
  );
}

// ----------------------------------------------------------------- transition
export function TransitionSection({ clip, index }: { clip: TimelineClip; index: number }) {
  const t = useT();
  const clips = useEditorStore((s) => s.clips);
  const setTransition = useEditorStore((s) => s.setTransition);

  const { kind, ms } = transitionOf(clip);
  const ceiling = maxTransitionMs(clips, index);
  const first = index === 0;

  return (
    <section>
      <SectionTitle
        aside={
          <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
            {kind === "crossfade" ? t("transition.shortens", { ms }) : ""}
          </span>
        }
      >
        {t("transition.section")}
      </SectionTitle>

      <div className="mt-1.5 grid grid-cols-2 gap-1">
        {TRANSITIONS.map((option) => {
          // A crossfade needs something to fade *from*, so the first clip is
          // not offered one -- the domain refuses it and an enabled button
          // that always fails is worse than no button.
          const unavailable =
            (option === "crossfade" && first) || (option !== "cut" && ceiling === 0);
          const selected = option === kind;
          return (
            <button
              key={option}
              type="button"
              disabled={unavailable}
              title={t(TRANSITION_HINT[option])}
              aria-pressed={selected}
              onClick={() => setTransition(clip.id, option)}
              className={`flex h-control items-center justify-center rounded-md px-2 text-2xs font-medium transition-[background-color,color,box-shadow] duration-fast disabled:cursor-not-allowed disabled:opacity-35 ${
                selected
                  ? "bg-elevated text-fg shadow-raised ring-1 ring-accent-strong"
                  : "bg-sunken/70 text-muted hover:text-fg"
              }`}
            >
              {t(TRANSITION_LABEL[option])}
            </button>
          );
        })}
      </div>

      {kind !== "cut" && (
        <div className="mt-2 flex items-center gap-2">
          <Slider
            className="min-w-0 flex-1"
            aria-label={t("transition.duration")}
            value={ms}
            min={TRANSITION_BOUNDS.min}
            max={Math.max(ceiling, TRANSITION_BOUNDS.min)}
            step={20}
            onChange={(event) => setTransition(clip.id, kind, Number(event.target.value))}
          />
          <span className="w-[56px] shrink-0 text-right font-mono text-2xs tabular-nums text-fg">
            {ms} ms
          </span>
        </div>
      )}

      <div className="mt-1.5">
        <Row
          label={t("transition.limit")}
          value={ceiling ? `${TRANSITION_BOUNDS.min}–${ceiling} ms` : t("transition.tooShort")}
          title={t("transition.limitHint")}
          tone={ceiling ? undefined : "danger"}
        />
      </div>

      <p className="mt-1.5 text-2xs leading-snug text-faint">{t(TRANSITION_HINT[kind])}</p>
    </section>
  );
}

// -------------------------------------------------------------------- effects
export function EffectsSection({ clip }: { clip: TimelineClip }) {
  const t = useT();
  const setEffect = useEditorStore((s) => s.setEffect);
  const clearEffect = useEditorStore((s) => s.clearEffect);

  const speed = clipSpeed(clip);
  const slowed = speed < 1;
  // One rate per clip, expressed as two ranges that meet at 1.0. Which slider
  // is live follows the current rate, so the control the user reaches for is
  // the one that already holds the value.
  const speedKind: EffectKind = slowed ? "slow_motion" : "speed_up";
  const zoomKind: EffectKind = effectAmount(clip, "zoom_out") > 0 ? "zoom_out" : "zoom_in";

  return (
    <>
      <section>
        <SectionTitle
          aside={
            <span className="shrink-0 font-mono text-2xs tabular-nums text-faint">
              {timecode(clipPlaybackMs(clip), false)}
            </span>
          }
        >
          {t("effect.section.speed")}
        </SectionTitle>

        <div className="mt-1.5 flex items-center gap-2">
          <SegmentedControl<EffectKind>
            label={t("effect.speedDirection")}
            value={speedKind}
            options={[
              { value: "slow_motion", label: t("effect.slow_motion") },
              { value: "speed_up", label: t("effect.speed_up") },
            ]}
            onChange={(next) => {
              clearEffect(clip.id, next === "slow_motion" ? "speed_up" : "slow_motion");
              setEffect(clip.id, { kind: next, amount: EFFECT_BOUNDS[next].neutral });
            }}
          />
          <span className="ml-auto font-mono text-2xs tabular-nums text-fg">
            {speed.toFixed(2)}×
          </span>
        </div>

        <div className="mt-2">
          <EffectSlider clip={clip} kind={speedKind} format={(v) => `${v.toFixed(2)}×`} />
        </div>

        <p className="mt-1.5 text-2xs leading-snug text-faint">
          {t("effect.speedHint", { min: EFFECT_BOUNDS.slow_motion.min, max: EFFECT_BOUNDS.speed_up.max })}
        </p>
      </section>

      <section>
        <SectionTitle>{t("effect.section.transform")}</SectionTitle>
        <div className="mt-1.5 flex items-center gap-2">
          <SegmentedControl<EffectKind>
            label={t("effect.zoomDirection")}
            value={zoomKind}
            options={FRAMING.map((kind) => ({ value: kind, label: t(EFFECT_LABEL[kind]) }))}
            onChange={(next) => {
              clearEffect(clip.id, next === "zoom_in" ? "zoom_out" : "zoom_in");
              setEffect(clip.id, { kind: next, amount: EFFECT_BOUNDS[next].neutral });
            }}
          />
        </div>
        <div className="mt-2">
          <EffectSlider
            clip={clip}
            kind={zoomKind}
            format={(v) => `${Math.round(v * 100)}%`}
          />
        </div>
        <p className="mt-1.5 text-2xs leading-snug text-faint">{t("effect.zoomHint")}</p>
      </section>

      <section>
        <SectionTitle>{t("effect.section.colour")}</SectionTitle>
        <div className="mt-1.5 flex flex-col gap-1.5">
          {COLOUR.map((kind) => (
            <EffectSlider key={kind} clip={clip} kind={kind} format={(v) => v.toFixed(2)} />
          ))}
        </div>

        {/* The window is an offset *inside the clip*, and only the colour
            effects have one: a zoom ramp or a speed change that applied to part
            of a clip would need a second segment the plan cannot express. */}
        <ColourWindow clip={clip} />
      </section>
    </>
  );
}

/**
 * When the colour grade applies, within the clip.
 *
 * Blank means the whole clip, which is the common case and the default. The
 * two numbers are offsets from the clip's own start, not timeline positions,
 * because that is what the segment records and what survives the clip being
 * moved.
 */
function ColourWindow({ clip }: { clip: TimelineClip }) {
  const t = useT();
  const setEffect = useEditorStore((s) => s.setEffect);

  const graded = (clip.effects ?? []).filter((effect) => COLOUR.includes(effect.kind));
  if (graded.length === 0) return null;

  const windowed = graded.find((effect) => effect.start_ms != null || effect.end_ms != null);
  const length = clip.outMs - clip.inMs;
  const startMs = windowed?.start_ms ?? 0;
  const endMs = windowed?.end_ms ?? length;

  const retime = (patch: { start_ms?: number | null; end_ms?: number | null }) => {
    for (const effect of graded) setEffect(clip.id, { ...effect, ...patch });
  };

  return (
    <div className="mt-2">
      <div className="grid grid-cols-2 gap-2">
        <Field label={t("effect.windowFrom")} hint={t("effect.windowHint")}>
          <NumberInput
            value={Math.round(startMs)}
            min={0}
            max={Math.max(0, Math.round(endMs) - 1)}
            step={100}
            aria-label={t("effect.windowFrom")}
            onChange={(event) => retime({ start_ms: Number(event.target.value), end_ms: endMs })}
          />
        </Field>
        <Field label={t("effect.windowTo")}>
          <NumberInput
            value={Math.round(endMs)}
            min={Math.round(startMs) + 1}
            max={Math.round(length)}
            step={100}
            aria-label={t("effect.windowTo")}
            onChange={(event) => retime({ start_ms: startMs, end_ms: Number(event.target.value) })}
          />
        </Field>
      </div>
      {windowed && (
        <Button
          size="sm"
          className="mt-1.5"
          onClick={() => retime({ start_ms: null, end_ms: null })}
        >
          {t("effect.windowWhole")}
        </Button>
      )}
    </div>
  );
}
