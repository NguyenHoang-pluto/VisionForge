"""Stills in an edit (Phase 12): how long a photo may be held, and how it moves.

A photo has no length, no head to avoid and no action to centre on. Everything a
planner asks of a video's duration has a different answer for a still, and this
module is where those answers live, so the rules engine, the editorial engine
and the directive compiler cannot each invent their own.

    hold_span_ms   how much a planner may take:  MAX_STILL_MS for a photo
    hold_start_ms  where the take begins:        always 0 for a photo
    still_motion   the drift a photo is given so it does not read as a freeze
"""

from __future__ import annotations

from visionforge.domain.editplan import MAX_STILL_MS
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.media import MediaKind
from visionforge.domain.selection import Candidate

#: How far a still drifts over its hold, as a fraction of the frame. Enough to
#: be seen as movement; small enough that a 12-megapixel photo is never
#: magnified past its own resolution at 1080p.
STILL_MOTION_AMOUNT = 0.12

#: The motions a run of stills cycles through. Alternating direction is what
#: stops a slideshow of five photos reading as one long zoom.
_LANDSCAPE_CYCLE = (
    EffectKind.ZOOM_IN,
    EffectKind.PAN_RIGHT,
    EffectKind.ZOOM_OUT,
    EffectKind.PAN_LEFT,
)
#: A portrait photo has its spare picture above and below, so it drifts
#: vertically rather than across.
_PORTRAIT_CYCLE = (
    EffectKind.ZOOM_IN,
    EffectKind.PAN_DOWN,
    EffectKind.ZOOM_OUT,
    EffectKind.PAN_UP,
)

#: Effects that already move the frame. A still carrying one of these has been
#: given its motion by someone else, and is left alone.
_GEOMETRY = frozenset(
    {
        EffectKind.ZOOM_IN,
        EffectKind.ZOOM_OUT,
        EffectKind.PAN_LEFT,
        EffectKind.PAN_RIGHT,
        EffectKind.PAN_UP,
        EffectKind.PAN_DOWN,
    }
)


def is_still(candidate: Candidate) -> bool:
    return candidate.kind is MediaKind.IMAGE


def hold_span_ms(candidate: Candidate) -> int:
    """How much of this source a planner may take.

    A video's duration; a still's longest permitted hold.
    """
    if is_still(candidate):
        return MAX_STILL_MS
    return candidate.duration_ms or 0


def hold_start_ms(candidate: Candidate, span_ms: int, take_ms: int) -> int:
    """Where a take of ``take_ms`` begins: centred for a video, zero for a still."""
    if is_still(candidate):
        return 0
    return max(0, (span_ms - take_ms) // 2)


def still_motion(position: int, *, portrait: bool = False) -> Effect:
    """The drift a still gets at this position in the run of stills.

    Deterministic in ``position``, so the same edit always moves the same way.
    """
    cycle = _PORTRAIT_CYCLE if portrait else _LANDSCAPE_CYCLE
    return Effect(cycle[position % len(cycle)], STILL_MOTION_AMOUNT)


def drifted(
    effects: tuple[Effect, ...], position: int, *, portrait: bool = False
) -> tuple[Effect, ...]:
    """A still's ``effects``, plus a drift if nothing moves it yet.

    Speed changes are dropped: a still has no motion to slow down, and the plan
    validator refuses one.
    """
    kept = tuple(effect for effect in effects if not effect.kind.changes_duration)
    if any(effect.kind in _GEOMETRY for effect in kept):
        return kept
    return (still_motion(position, portrait=portrait), *kept)


def with_still_motion(
    candidate: Candidate, effects: tuple[Effect, ...], position: int
) -> tuple[Effect, ...]:
    """``effects`` unchanged for a video; ``drifted`` for a still."""
    if not is_still(candidate):
        return effects
    portrait = (candidate.height or 0) > (candidate.width or 0)
    return drifted(effects, position, portrait=portrait)
