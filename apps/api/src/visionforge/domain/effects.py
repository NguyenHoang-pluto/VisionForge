"""Visual effects: a closed vocabulary with per-kind parameter rules.

The shape of this module is set by one requirement, which is the same
requirement that shaped `MusicCue` in Phase 7 and `Segment` in Phase 4: there
must be no field through which a caller, or a model reading a field a caller
set, could influence what FFmpeg is handed. So an effect is a member of a closed
enum plus **one number**, and what that number means is decided here rather than
by whoever sent it.

There is no parameter dictionary. A `dict[str, Any]` on a renderer instruction
is a hole in exactly the shape of an arbitrary filter argument, and the fact
that today's code only reads two keys out of it is not a guarantee about
tomorrow's.

Seven kinds, not twenty. Each one is a filter, a range, a timing rule and a test;
shipping twenty of them badly is worse than shipping seven that are exact.

**On `fade`.** It is not here. Fading in from black and out to black is how a
clip *begins and ends*, which is what `TransitionKind.FADE_IN` and
`FADE_TO_BLACK` already express -- and two spellings of one capability is how a
vocabulary rots. The capability ships once, in the place it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EffectKind(StrEnum):
    """What an effect does. The string is persisted, so values are stable."""

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    #: Slow drifts across the frame (Phase 12). With the zooms, these are the
    #: movement a still is given so that a photo does not read as a freeze.
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    SLOW_MOTION = "slow_motion"
    SPEED_UP = "speed_up"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"

    @property
    def changes_duration(self) -> bool:
        """Whether this effect makes the segment play for a different length.

        The question the timeline arithmetic asks. A property of the kind rather
        than a set each caller keeps, for the same reason
        ``TransitionKind.consumes_time`` is: an eighth member added without
        answering it would otherwise default to "no" and produce a plan whose
        reported duration is wrong.
        """
        return self in (EffectKind.SLOW_MOTION, EffectKind.SPEED_UP)

    @property
    def spans_whole_segment(self) -> bool:
        """Whether this kind may only apply to the entire segment.

        Colour can be ramped over part of a clip -- ``eq`` has timeline support,
        so the compiler can gate it with ``enable``. Zoom cannot: ``zoompan``
        has no timeline support and its expression is written against the
        segment's own frame count. Speed cannot either, and for a better reason
        than a filter limitation -- changing speed half way through a clip is
        two clips, and the honest way to express that is to cut it.
        """
        return self not in (
            EffectKind.BRIGHTNESS,
            EffectKind.CONTRAST,
            EffectKind.SATURATION,
        )


#: Inclusive bounds per kind: (minimum, maximum, neutral).
#:
#: ``neutral`` is the value at which the effect does nothing, and it is recorded
#: because "is this effect a no-op" is a question both the validator and the
#: compiler ask, and answering it from a literal in two places is how the two
#: eventually disagree.
EFFECT_BOUNDS: dict[EffectKind, tuple[float, float, float]] = {
    # A fraction of the frame, not a multiplier: 0.15 is a 15% push in. Capped
    # well below anything that would show interpolation on 720p footage.
    EffectKind.ZOOM_IN: (0.0, 0.30, 0.0),
    EffectKind.ZOOM_OUT: (0.0, 0.30, 0.0),
    # How far the view travels, as a fraction of the frame. The picture is
    # cropped by this much and the crop slides across it, so 0.15 is a drift
    # through 15% of the frame -- the same scale the zooms use.
    EffectKind.PAN_LEFT: (0.0, 0.30, 0.0),
    EffectKind.PAN_RIGHT: (0.0, 0.30, 0.0),
    EffectKind.PAN_UP: (0.0, 0.30, 0.0),
    EffectKind.PAN_DOWN: (0.0, 0.30, 0.0),
    # Playback rate. Below 0.25 the motion judders without frame interpolation,
    # which this phase does not do; above 4 the audio is unusable whatever
    # ``atempo`` is asked to do about it.
    EffectKind.SLOW_MOTION: (0.25, 1.0, 1.0),
    EffectKind.SPEED_UP: (1.0, 4.0, 1.0),
    # ``eq`` semantics exactly: brightness is an offset, the other two are
    # multipliers. Kept as the filter's own units so that nothing has to invent
    # a mapping the user would then have to learn twice.
    EffectKind.BRIGHTNESS: (-1.0, 1.0, 0.0),
    EffectKind.CONTRAST: (0.5, 1.5, 1.0),
    EffectKind.SATURATION: (0.0, 2.0, 1.0),
}

#: Shortest slice of a clip a colour effect may be applied to. Below this it is
#: a flicker rather than a grade.
MIN_EFFECT_MS = 200

#: How many effects one segment may carry. Enough for a push-in and a grade;
#: short of the point where the filter chain per clip stops being reviewable.
MAX_EFFECTS_PER_SEGMENT = 4


@dataclass(frozen=True, slots=True)
class Effect:
    """One effect on one segment.

    ``start_ms``/``end_ms`` are offsets *within the segment*, not timeline
    positions -- the same separation of coordinate systems ``TimelineClip`` and
    ``MusicCue`` keep, and for the same reason: conflating them is how an effect
    drifts away from the frames it was meant for. ``None`` means "the whole
    segment", which is the only thing zoom and speed are allowed to mean.
    """

    kind: EffectKind
    #: What the number means depends on the kind, and only on the kind. See
    #: ``EFFECT_BOUNDS``.
    amount: float
    start_ms: int | None = None
    end_ms: int | None = None

    @property
    def is_ranged(self) -> bool:
        return self.start_ms is not None or self.end_ms is not None

    @property
    def is_neutral(self) -> bool:
        """Whether this effect would do nothing if rendered."""
        return abs(self.amount - EFFECT_BOUNDS[self.kind][2]) < 1e-9

    def window(self, segment_duration_ms: int) -> tuple[int, int]:
        """The slice of the segment this applies to, resolved and clamped."""
        start = 0 if self.start_ms is None else max(0, self.start_ms)
        end = segment_duration_ms if self.end_ms is None else min(segment_duration_ms, self.end_ms)
        return start, end

    @property
    def speed(self) -> float:
        """The playback rate this effect implies. 1.0 for anything but speed."""
        return self.amount if self.kind.changes_duration else 1.0

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "amount": round(self.amount, 4),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @staticmethod
    def from_payload(raw: dict[str, Any]) -> Effect:
        """Rebuild from stored JSON. Unknown kinds raise rather than degrade."""
        return Effect(
            kind=EffectKind(raw["kind"]),
            amount=float(raw["amount"]),
            start_ms=_optional_int(raw.get("start_ms")),
            end_ms=_optional_int(raw.get("end_ms")),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def speed_of(effects: tuple[Effect, ...]) -> float:
    """The combined playback rate of a segment's effects.

    Multiplied rather than last-one-wins, so that a plan carrying both a slow
    motion and a speed up describes something coherent instead of depending on
    list order. In practice validation rejects that combination; this makes the
    arithmetic total anyway, because a function that is only correct when the
    validator ran is a function that will one day run before it.
    """
    rate = 1.0
    for effect in effects:
        rate *= effect.speed
    return rate


def output_duration_ms(source_duration_ms: int, effects: tuple[Effect, ...]) -> int:
    """How long a segment plays once its speed effects are applied.

    Slow motion makes a clip *longer*: half speed doubles it. The division is
    the whole of the arithmetic, and it lives here rather than in the timeline
    so that the plan, the timeline and the compiler cannot disagree about it.
    """
    rate = speed_of(effects)
    if rate <= 0:
        raise ValueError("playback rate must be positive")
    return int(round(source_duration_ms / rate))


__all__ = [
    "EFFECT_BOUNDS",
    "MAX_EFFECTS_PER_SEGMENT",
    "MIN_EFFECT_MS",
    "Effect",
    "EffectKind",
    "output_duration_ms",
    "speed_of",
]
