"""Pacing: an explicit energy curve, and the clip lengths it implies.

    shape + target duration + clip count + bounds  ->  PacingPlan
                                    (+ beat grid)      per-slot durations,
                                                       on or off the grid

Before Phase 11 an automatic edit divided its target duration evenly among its
clips. That is not pacing; it is arithmetic, and it is the single most visible
reason a generated edit reads as a slideshow. A real cut accelerates into its
strongest moment and breathes afterwards, and the lengths of its shots are the
mechanism by which it does that.

So pacing is modelled explicitly, as a **curve of energy against position**, and
clip length is *derived* from it. High energy means short shots; low energy
means long ones. Everything else in this module is the arithmetic that keeps
that derivation honest:

- the derived lengths must sum to what was asked for, or as close as the bounds
  allow, so "make me 30 seconds" is not quietly ignored;
- every length must stay inside the pacing bounds a style or a reference set,
  and then inside the plan's own renderable bounds, in that order;
- and when there is a trustworthy beat grid, the lengths become whole numbers of
  beats *measured cumulatively*, so cut forty is as close to its beat as cut one.

Nothing here knows about clips, roles, media or FFmpeg. It takes numbers and
returns numbers, which is what makes the whole pacing model testable without a
single frame of video.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from visionforge.domain.beats import BeatGrid
from visionforge.domain.editplan import MAX_SEGMENT_MS, MIN_SEGMENT_MS


class PacingShape(StrEnum):
    """How energy moves across an edit. A closed set of named curves.

    Five, and each is a different *story of attention* rather than a different
    number. A closed set rather than free control points for the same reason
    styles are named profiles: "0.2, 0.45, 0.9, 0.6" is not a decision anyone
    can make deliberately, and a name is something a user, a policy table and a
    model can all agree on.
    """

    #: slow -> medium -> fast -> peak -> release. The classic highlight shape.
    RAMP = "ramp"
    #: medium -> fast -> peak. No preamble; used where the material is already
    #: the interesting part and an establishing shot would cost the viewer.
    BUILD = "build"
    #: calm -> build -> calm. Breathing room at both ends.
    WAVE = "wave"
    #: flat. Not "no pacing" -- a deliberately even rhythm is a real choice, and
    #: it is what a product or fashion sequence usually wants.
    STEADY = "steady"
    #: starts at the peak and settles. For material whose best moment is first,
    #: which is most social video.
    DECAY = "decay"


#: Control points per shape: energy at evenly spaced positions from 0 to 1.
#:
#: Written as tables rather than as functions so that a shape is a diff in this
#: file -- reviewable, testable, and attributable -- rather than a formula whose
#: effect nobody can predict. Values are 0..1 where 0 is the calmest this system
#: will pace and 1 is the busiest.
PACING_CURVES: dict[PacingShape, tuple[float, ...]] = {
    PacingShape.RAMP: (0.15, 0.35, 0.60, 0.85, 1.00, 0.45),
    PacingShape.BUILD: (0.45, 0.65, 0.85, 1.00),
    PacingShape.WAVE: (0.20, 0.55, 0.85, 0.55, 0.20),
    PacingShape.STEADY: (0.50, 0.50, 0.50, 0.50),
    PacingShape.DECAY: (1.00, 0.75, 0.50, 0.30, 0.20),
}

#: How far clip length is allowed to swing between the calmest and busiest point
#: of a curve, as a ratio of the *mean* length.
#:
#: At 0.6 a maximum-energy shot is roughly 40% of the mean and a
#: minimum-energy one roughly 160% -- a four-to-one spread, which is
#: emphatic without being a gimmick. Above about 0.75 the long shots start
#: exceeding what any style's bounds permit and the whole curve gets clamped
#: flat, which is worse than a gentler curve that survives.
DEFAULT_PACING_SPREAD = 0.60

#: The most a single slot may be stretched for emphasis, on top of the curve.
#: A peak given twice its neighbours' length reads as emphasis; four times reads
#: as a mistake.
MAX_EMPHASIS = 2.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class PacingCurve:
    """Energy as a function of position through the edit.

    Piecewise-linear between the control points, which is both the simplest
    interpolation that produces a smooth ramp and the only one whose output a
    reader can predict from the table above.
    """

    shape: PacingShape
    points: tuple[float, ...]
    #: How hard the curve swings clip length. Carried on the curve rather than
    #: passed at every call so that a policy can say "this genre paces gently"
    #: in one place.
    spread: float = DEFAULT_PACING_SPREAD

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a pacing curve needs at least two control points")
        if any(not 0.0 <= point <= 1.0 for point in self.points):
            raise ValueError("pacing control points must be between 0 and 1")
        if not 0.0 <= self.spread <= 1.0:
            raise ValueError("pacing spread must be between 0 and 1")

    def energy_at(self, position: float) -> float:
        """Energy at a position through the edit, 0..1. Total and clamped."""
        position = _clamp(position)
        span = len(self.points) - 1
        exact = position * span
        low = int(exact)
        if low >= span:
            return self.points[-1]
        fraction = exact - low
        return round(self.points[low] + (self.points[low + 1] - self.points[low]) * fraction, 6)

    def energies(self, count: int) -> tuple[float, ...]:
        """Energy for each of ``count`` slots, evenly spaced.

        A single slot sits at the curve's peak rather than at its start: an edit
        of one clip has no arc to walk, and giving it the opening energy of a
        ramp would make a one-shot edit the slowest thing the system produces.
        """
        if count <= 0:
            return ()
        if count == 1:
            return (max(self.points),)
        return tuple(self.energy_at(index / (count - 1)) for index in range(count))

    @property
    def peak_position(self) -> float:
        """Where the curve is busiest, 0..1. Ties resolve to the earlier point."""
        best = max(range(len(self.points)), key=lambda i: (self.points[i], -i))
        return round(best / (len(self.points) - 1), 4)

    def as_payload(self) -> dict[str, Any]:
        return {
            "shape": self.shape.value,
            "points": [round(point, 3) for point in self.points],
            "spread": round(self.spread, 3),
            "peak_position": self.peak_position,
        }


def curve_for(shape: PacingShape, *, spread: float = DEFAULT_PACING_SPREAD) -> PacingCurve:
    return PacingCurve(shape=shape, points=PACING_CURVES[shape], spread=spread)


# ------------------------------------------------------------------- the plan
@dataclass(frozen=True, slots=True)
class PacingSlot:
    """One position in the edit: how energetic it is and how long it runs."""

    index: int
    position: float
    energy: float
    duration_ms: int
    #: Whole beats this slot occupies, when it was quantised. ``None`` means the
    #: length is not on a grid -- either there was no usable grid or this slot
    #: could not be fitted to one.
    beats: int | None = None
    #: Where this slot starts in the finished edit, in ms. Carried because beat
    #: alignment is a statement about a *position*, not about a length.
    start_ms: int = 0
    #: Extra stretch applied for emphasis, 1.0 when none.
    emphasis: float = 1.0

    @property
    def on_beat(self) -> bool:
        return self.beats is not None

    def as_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "position": round(self.position, 4),
            "energy": round(self.energy, 4),
            "duration_ms": self.duration_ms,
            "start_ms": self.start_ms,
            "beats": self.beats,
            "on_beat": self.on_beat,
            "emphasis": round(self.emphasis, 3),
        }


@dataclass(frozen=True, slots=True)
class PacingPlan:
    """The rhythm of an edit, before any footage is attached to it."""

    curve: PacingCurve
    slots: tuple[PacingSlot, ...]
    target_ms: int
    #: Why the plan is or is not on a grid, in the same shape the Phase 7
    #: planner records. A beat-synced edit that cannot say which tempo it was
    #: synced to is not reviewable.
    beat_sync: dict[str, Any] | None = None

    @property
    def total_ms(self) -> int:
        return sum(slot.duration_ms for slot in self.slots)

    @property
    def cut_count(self) -> int:
        return len(self.slots)

    @property
    def shot_density(self) -> float:
        """Cuts per second. The number a viewer actually feels."""
        total = self.total_ms
        return round(self.cut_count / (total / 1000.0), 4) if total > 0 else 0.0

    @property
    def is_beat_synced(self) -> bool:
        return any(slot.on_beat for slot in self.slots)

    def as_payload(self) -> dict[str, Any]:
        return {
            "curve": self.curve.as_payload(),
            "target_ms": self.target_ms,
            "total_ms": self.total_ms,
            "cut_count": self.cut_count,
            "shot_density": self.shot_density,
            "beat_sync": self.beat_sync,
            "slots": [slot.as_payload() for slot in self.slots],
        }


def plan_pacing(
    *,
    count: int,
    target_ms: int,
    curve: PacingCurve,
    min_clip_ms: int,
    max_clip_ms: int,
    emphasis: tuple[float, ...] = (),
    grid: BeatGrid | None = None,
    beat_sync: bool = False,
    source_limits_ms: tuple[int, ...] = (),
) -> PacingPlan:
    """Turn a curve and a budget into per-slot clip lengths.

    The arithmetic, in order, and the order is what makes it correct:

    1. **Shape.** Each slot gets a raw weight from the curve -- inversely, so
       high energy is a short shot. The weights are relative; nothing is in
       milliseconds yet.
    2. **Scale.** The weights are scaled so they sum to the target duration.
    3. **Emphasis.** A slot the story wants held gets stretched, and the rest
       are rescaled so the total is unchanged. Emphasis moves time *between*
       shots rather than adding it, because the user asked for a length.
    4. **Clamp.** Each length is clamped to the style's pacing bounds and then
       to the plan's renderable bounds, and the slack that clamping created is
       redistributed over the slots that are still free to move. This is
       iterative, and it is where an even split would have stopped.
    5. **Quantise.** With a trustworthy grid, each length becomes a whole number
       of beats measured as the difference of two cumulative positions -- the
       same arithmetic ``BeatGrid.span_for_beats`` exists for, so forty cuts
       later the edit is still on the beat.

    ``source_limits_ms`` caps individual slots at the real length of the footage
    that will occupy them, when the caller knows it. A pacing plan that asks for
    four seconds of a two-second clip is a plan that gets silently truncated
    later, and truncation after the fact is how a carefully shaped curve becomes
    an even split again.
    """
    if count <= 0:
        raise ValueError("a pacing plan needs at least one slot")
    if target_ms <= 0:
        raise ValueError("target_ms must be positive")

    floor = max(MIN_SEGMENT_MS, min(min_clip_ms, max_clip_ms))
    ceiling = min(MAX_SEGMENT_MS, max(min_clip_ms, max_clip_ms))
    if floor > ceiling:
        floor, ceiling = ceiling, floor

    energies = curve.energies(count)

    # --- 1. shape -------------------------------------------------------
    # weight = 1 at mean energy, (1 + spread) at zero energy, (1 - spread) at
    # full. Linear rather than exponential because the spread constant is meant
    # to be readable as "how much longer the calm shots are", and an exponent
    # makes that relationship something you have to work out.
    weights = [1.0 + curve.spread * (0.5 - energy) * 2.0 for energy in energies]

    # --- 2. scale -------------------------------------------------------
    total_weight = sum(weights) or float(count)
    lengths = [target_ms * weight / total_weight for weight in weights]

    # --- 3. emphasis ----------------------------------------------------
    if emphasis:
        stretched = [
            length * _clamp(emphasis[index] if index < len(emphasis) else 1.0, 1.0, MAX_EMPHASIS)
            for index, length in enumerate(lengths)
        ]
        scale = sum(lengths) / sum(stretched) if sum(stretched) > 0 else 1.0
        lengths = [length * scale for length in stretched]

    # --- 4. clamp and redistribute --------------------------------------
    caps = [
        min(ceiling, source_limits_ms[index]) if index < len(source_limits_ms) else ceiling
        for index in range(count)
    ]
    # A source shorter than the floor cannot be honoured at all; the caller is
    # responsible for not offering one, and the floor wins here so that the plan
    # stays renderable rather than emitting a segment the validator refuses.
    caps = [max(cap, floor) for cap in caps]
    lengths = _redistribute(lengths, floor=floor, caps=caps, budget=float(target_ms))

    rounded = [int(round(length)) for length in lengths]

    # --- 5. quantise ----------------------------------------------------
    quantised: list[int | None] = [None] * count
    sync: dict[str, Any] | None = None
    if beat_sync and grid is not None and grid.is_reliable() and grid.period_ms > 0:
        rounded, quantised, sync = _quantise(rounded, grid, floor=floor, caps=caps)
    elif beat_sync:
        sync = {
            "applied": False,
            "reason": (
                "no_beats"
                if grid is None
                else "low_confidence"
                if not grid.is_reliable()
                else "no_fitting_span"
            ),
            "confidence": grid.confidence if grid is not None else None,
        }

    slots: list[PacingSlot] = []
    cursor = 0
    for index in range(count):
        slots.append(
            PacingSlot(
                index=index,
                position=index / max(count - 1, 1),
                energy=round(energies[index], 4),
                duration_ms=rounded[index],
                beats=quantised[index],
                start_ms=cursor,
                emphasis=round(emphasis[index], 3) if index < len(emphasis) else 1.0,
            )
        )
        cursor += rounded[index]

    return PacingPlan(curve=curve, slots=tuple(slots), target_ms=target_ms, beat_sync=sync)


def _redistribute(
    lengths: list[float], *, floor: int, caps: list[int], budget: float
) -> list[float]:
    """Clamp every length to its bounds and spread the slack over the rest.

    Iterative, because clamping one slot frees or consumes time that changes
    what the others should be. Without this the shape survives only while no
    bound bites, which in practice means it survives on paper and not in any
    real project -- the first clip too short to hold its slot flattens the whole
    curve back to an even split.

    Terminates in at most ``len(lengths)`` passes: each pass either fixes at
    least one slot at a bound or converges, and a fixed slot never unfixes.
    """
    count = len(lengths)
    free = list(range(count))
    fixed: dict[int, float] = {}
    current = list(lengths)

    for _ in range(count + 1):
        remaining = budget - sum(fixed.values())
        movable = list(free)
        if not movable:
            break

        share = sum(current[index] for index in movable)
        if share <= 0:
            for index in movable:
                current[index] = remaining / len(movable)
        else:
            scale = remaining / share
            for index in movable:
                current[index] = current[index] * scale

        newly_fixed = False
        for index in list(movable):
            if current[index] < floor:
                fixed[index] = float(floor)
                current[index] = float(floor)
                free.remove(index)
                newly_fixed = True
            elif current[index] > caps[index]:
                fixed[index] = float(caps[index])
                current[index] = float(caps[index])
                free.remove(index)
                newly_fixed = True

        if not newly_fixed:
            break

    return current


def _quantise(
    lengths: list[int], grid: BeatGrid, *, floor: int, caps: list[int]
) -> tuple[list[int], list[int | None], dict[str, Any]]:
    """Round each length to a whole number of beats, without drifting.

    The length of slot *n* is ``span_for_beats(placed + k) - span_for_beats(placed)``
    where ``placed`` is the total beats laid so far. Rounding the cumulative
    position rather than each length independently is what keeps the last cut on
    the beat: separately-rounded lengths accumulate up to half a millisecond of
    error per cut, which at forty cuts is an audible twentieth of a beat.

    A slot that cannot be fitted to a whole beat inside its bounds keeps its
    unquantised length and is reported as not on the grid, rather than being
    forced to a length the style refused. Partial quantisation is honest: the
    plan says which cuts landed on beats and which did not.
    """
    period = grid.period_ms
    placed_beats = 0
    placed_ms = 0
    result: list[int] = []
    beats_used: list[int | None] = []
    fitted = 0

    for index, wanted in enumerate(lengths):
        target_beats = max(1, int(round(wanted / period)))
        chosen: int | None = None
        duration = wanted

        # Search outward from the wanted count so the first fit is the closest.
        for beats in _search_order(target_beats):
            span = grid.span_for_beats(placed_beats + beats) - placed_ms
            if floor <= span <= caps[index]:
                chosen, duration = beats, span
                break

        if chosen is None:
            beats_used.append(None)
            result.append(wanted)
            # The running totals advance by real time even for an unquantised
            # slot, so the *next* slot's beat positions are still measured from
            # where the edit actually is rather than from where it would have
            # been on a perfect grid.
            placed_ms += wanted
            placed_beats = max(placed_beats, int(placed_ms / period))
        else:
            beats_used.append(chosen)
            result.append(duration)
            placed_beats += chosen
            placed_ms += duration
            fitted += 1

    return (
        result,
        beats_used,
        {
            "applied": fitted > 0,
            "bpm": round(grid.bpm, 2),
            "confidence": round(grid.confidence, 4),
            "first_beat_ms": grid.first_beat_ms,
            "beat_period_ms": round(period, 4),
            "slots_on_beat": fitted,
            "slots": len(lengths),
            "reason": None if fitted else "no_fitting_span",
        },
    )


def _search_order(centre: int) -> list[int]:
    """Whole-beat counts to try, nearest the wanted count first.

    Bounded at sixteen steps: beyond that the span bears no relation to what the
    curve asked for, and keeping the unquantised length is the better answer.
    """
    order = [centre]
    for step in range(1, 17):
        if centre - step >= 1:
            order.append(centre - step)
        order.append(centre + step)
    return order


__all__ = [
    "DEFAULT_PACING_SPREAD",
    "MAX_EMPHASIS",
    "PACING_CURVES",
    "PacingCurve",
    "PacingPlan",
    "PacingShape",
    "PacingSlot",
    "curve_for",
    "plan_pacing",
]
