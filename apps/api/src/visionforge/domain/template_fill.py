"""A template's slots, as the pacing plan the editorial engine fills (Phase 12).

The engine already knows how to fill a row of pacing slots with clips. A
template *is* a row of slots, so filling one means handing the engine the
template's row instead of the one it would have computed -- and nothing about
selection, trimming or explanation has to be written twice.

Timing is the only real decision here. A slot counted in beats is re-timed to
the music the user chose, by the same cumulative arithmetic the Phase 7 planner
uses so the fortieth cut is as close to its beat as the first. Without trusted
music, a slot keeps the length the template gave it.
"""

from __future__ import annotations

from typing import Any

from visionforge.domain.beats import BeatGrid
from visionforge.domain.editplan import MAX_SEGMENT_MS, MIN_SEGMENT_MS
from visionforge.domain.pacing import PacingCurve, PacingPlan, PacingShape, PacingSlot
from visionforge.domain.template import EditTemplate


def _retimed(template: EditTemplate, grid: BeatGrid | None) -> list[tuple[int, int | None]]:
    """``(duration_ms, beats)`` per slot, on the music's grid where possible.

    A slot is only re-timed when every slot can be: a template half on the
    music's beats and half on its own is on neither, so a single slot with no
    beat count leaves the whole template at its own timing.
    """
    own: list[tuple[int, int | None]] = [(slot.duration_ms, None) for slot in template.slots]
    if grid is None or not all(slot.beats for slot in template.slots):
        return own

    timed: list[tuple[int, int | None]] = []
    placed_beats = 0
    placed_ms = 0
    for slot in template.slots:
        beats = slot.beats or 1
        end_ms = grid.span_for_beats(placed_beats + beats)
        duration = end_ms - placed_ms
        if not MIN_SEGMENT_MS <= duration <= MAX_SEGMENT_MS:
            return own
        timed.append((duration, beats))
        placed_beats += beats
        placed_ms = end_ms
    return timed


def template_pacing(
    template: EditTemplate,
    *,
    grid: BeatGrid | None,
    beat_sync: bool,
    shape: PacingShape,
) -> PacingPlan:
    """The pacing plan a template stands for."""
    trusted = grid if beat_sync and grid is not None and grid.is_reliable() else None
    timing = _retimed(template, trusted)

    count = len(template.slots)
    energies = [slot.energy for slot in template.slots]
    slots: list[PacingSlot] = []
    start = 0
    for index, (slot, (duration, beats)) in enumerate(zip(template.slots, timing, strict=True)):
        slots.append(
            PacingSlot(
                index=index,
                position=index / (count - 1) if count > 1 else 0.5,
                energy=slot.energy,
                duration_ms=duration,
                beats=beats,
                start_ms=start,
            )
        )
        start += duration

    beat_sync_payload: dict[str, Any]
    if trusted is not None and any(beats is not None for _, beats in timing):
        beat_sync_payload = {
            "applied": True,
            "bpm": round(trusted.bpm, 2),
            "confidence": round(trusted.confidence, 4),
            "source": "template",
        }
    else:
        beat_sync_payload = {
            "applied": False,
            "reason": "not_requested" if not beat_sync else "template_own_timing",
        }

    return PacingPlan(
        # The template's own energies are the curve: that is what it measured,
        # or what its author laid out. Two points at least, for a one-slot one.
        curve=PacingCurve(
            shape=shape,
            points=tuple(energies) if count > 1 else (energies[0], energies[0]),
            spread=0.0,
        ),
        slots=tuple(slots),
        target_ms=sum(duration for duration, _ in timing),
        beat_sync=beat_sync_payload,
    )


__all__ = ["template_pacing"]
