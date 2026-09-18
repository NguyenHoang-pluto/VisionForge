"""Measure a template from a video somebody else edited (Phase 12).

The input is the analysis VisionForge already runs on every video -- scenes,
dynamics and beats -- so making a template needs no new analyzer and no model:

    scenes    every detected shot becomes a slot, its length the shot's length
    dynamics  the motion measured inside a shot becomes that slot's energy
    beats     when the video's own music is trusted, a slot that spans a whole
              number of beats records that count, so the template re-times to
              other music later

What it does *not* measure, and says so rather than guessing: how shots are
joined. Scene detection finds cuts; telling a dissolve from a cut, or reading a
zoom that was added in the edit, needs analysis this build does not have. A
measured template's joins are cuts, and a still placed in one of its slots gets
the default drift.

Pure: payloads in, a template out. The service decides whether the video is
ready enough to measure; this module decides what the measurements mean.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from visionforge.domain.beats import BeatGrid, grid_from_payload
from visionforge.domain.editorial import MOTION_REFERENCE
from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MAX_STILL_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    TransitionKind,
)
from visionforge.domain.stills import still_motion
from visionforge.domain.story import StoryRole
from visionforge.domain.template import (
    EditTemplate,
    SlotPreference,
    TemplateInvalidError,
    TemplateSlot,
    TemplateSource,
    assert_valid_template,
)

#: Bumped when the same analysis would now produce a different template.
EXTRACT_VERSION = "1"

#: A slot at or above this energy asks for a video. A still has no motion of
#: its own, and a template's fast shots are where that shows.
VIDEO_ENERGY = 0.6

#: How close a shot's length must be to a whole number of beats to be counted
#: as that many beats: a quarter of a beat either way.
BEAT_TOLERANCE = 0.25

#: The energy a shot is given when no motion sample fell inside it. Middling,
#: and recorded as a guess in the report.
UNMEASURED_ENERGY = 0.4


@dataclass(frozen=True, slots=True)
class Extraction:
    """A measured template, and an honest account of how it was measured."""

    template: EditTemplate
    #: Shots found in the video.
    shots: int
    #: Shots shorter than a legal clip, folded into a neighbour.
    merged: int
    #: Shots longer than a legal clip, split into several slots.
    split: int
    #: Slots dropped because the video had more than a plan may hold.
    dropped: int
    #: Slots whose energy was not measured and was set to ``UNMEASURED_ENERGY``.
    unmeasured: int
    beat_synced: bool

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": EXTRACT_VERSION,
            "shots": self.shots,
            "merged": self.merged,
            "split": self.split,
            "dropped": self.dropped,
            "unmeasured": self.unmeasured,
            "beat_synced": self.beat_synced,
            "transitions": "cuts_only",
        }


def aspect_for(width: int | None, height: int | None) -> AspectRatio:
    """The offered shape closest to the video's own."""
    if not width or not height:
        return AspectRatio.LANDSCAPE_16_9
    ratio = width / height
    shapes: list[AspectRatio] = list(AspectRatio)
    return min(shapes, key=lambda aspect: abs(aspect.ratio - ratio))


def _shots(scenes: dict[str, Any] | None) -> list[tuple[int, int]]:
    """``(start_ms, end_ms)`` for every detected shot, in order."""
    raw = scenes.get("scenes") if scenes else None
    if not isinstance(raw, list):
        return []
    shots: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start, end = entry.get("start_ms"), entry.get("end_ms")
        if isinstance(start, int | float) and isinstance(end, int | float) and end > start:
            shots.append((int(start), int(end)))
    return sorted(shots)


def _legal(shots: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    """Fold short shots into a neighbour and split long ones.

    Returns the adjusted shots and how many were merged and split.
    """
    merged = 0
    folded: list[tuple[int, int]] = []
    for start, end in shots:
        if folded and end - start < MIN_SEGMENT_MS:
            folded[-1] = (folded[-1][0], end)
            merged += 1
        elif folded and folded[-1][1] - folded[-1][0] < MIN_SEGMENT_MS:
            # The previous shot was the video's first and too short to stand.
            folded[-1] = (folded[-1][0], end)
            merged += 1
        else:
            folded.append((start, end))

    split = 0
    legal: list[tuple[int, int]] = []
    for start, end in folded:
        length = end - start
        if length <= MAX_SEGMENT_MS:
            legal.append((start, end))
            continue
        parts = -(-length // MAX_SEGMENT_MS)
        step = length // parts
        for index in range(parts):
            legal.append(
                (start + index * step, end if index == parts - 1 else start + (index + 1) * step)
            )
        split += parts - 1
    return legal, merged, split


def _energy(samples: list[tuple[int, float]], start: int, end: int) -> float | None:
    """Median motion measured inside a shot, on the editorial 0..1 scale."""
    inside = [motion for at, motion in samples if start <= at < end]
    if not inside:
        return None
    return round(max(0.0, min(1.0, statistics.median(inside) / MOTION_REFERENCE)), 3)


def _samples(dynamics: dict[str, Any] | None) -> list[tuple[int, float]]:
    raw = dynamics.get("samples") if dynamics else None
    if not isinstance(raw, list):
        return []
    samples: list[tuple[int, float]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        at, motion = entry.get("timestamp_ms"), entry.get("motion")
        if isinstance(at, int | float) and isinstance(motion, int | float):
            samples.append((int(at), float(motion)))
    return samples


def _beats(length_ms: int, grid: BeatGrid | None) -> int | None:
    """Whole beats this shot spans, if it spans a whole number of them."""
    if grid is None or grid.period_ms <= 0:
        return None
    count = round(length_ms / grid.period_ms)
    if count < 1:
        return None
    if abs(length_ms - count * grid.period_ms) > grid.period_ms * BEAT_TOLERANCE:
        return None
    return count


def _roles(energies: list[float]) -> list[StoryRole]:
    """A narrative role per slot, from position and energy.

    The busiest slot that is neither first nor last is the peak. Before it, the
    first quarter sets up and the rest builds; after it comes the reaction; the
    first slot hooks and the last one ends.
    """
    count = len(energies)
    if count == 1:
        return [StoryRole.PEAK]
    if count == 2:
        return [StoryRole.HOOK, StoryRole.ENDING]
    middle = range(1, count - 1)
    peak = max(middle, key=lambda index: (energies[index], -index))
    roles: list[StoryRole] = []
    for index in range(count):
        if index == 0:
            roles.append(StoryRole.HOOK)
        elif index == count - 1:
            roles.append(StoryRole.ENDING)
        elif index == peak:
            roles.append(StoryRole.PEAK)
        elif index > peak:
            roles.append(StoryRole.REACTION)
        elif index <= max(1, peak // 4):
            roles.append(StoryRole.SETUP)
        else:
            roles.append(StoryRole.BUILD)
    return roles


def extract_template(
    *,
    template_id: str,
    name: str,
    width: int | None,
    height: int | None,
    scenes: dict[str, Any] | None,
    dynamics: dict[str, Any] | None = None,
    beats: dict[str, Any] | None = None,
) -> Extraction:
    """Measure a template. Raises ``TemplateInvalidError`` when there is none.

    A video with a single detected shot has no structure to copy: one long take
    is a fact about the camera, not an edit.
    """
    shots = _shots(scenes)
    if len(shots) < 2:
        raise TemplateInvalidError(
            "no cuts were detected in this video, so it has no shot structure to copy"
        )

    legal, merged, split = _legal(shots)
    dropped = max(0, len(legal) - MAX_SEGMENTS)
    legal = legal[:MAX_SEGMENTS]

    samples = _samples(dynamics)
    measured = [_energy(samples, start, end) for start, end in legal]
    unmeasured = sum(1 for energy in measured if energy is None)
    energies = [UNMEASURED_ENERGY if energy is None else energy for energy in measured]

    grid = grid_from_payload(beats)
    trusted = grid if grid is not None and grid.is_reliable() else None
    counts = [_beats(end - start, trusted) for start, end in legal]

    portrait = bool(width and height and height > width)
    roles = _roles(energies)
    slots = tuple(
        TemplateSlot(
            duration_ms=end - start,
            energy=energy,
            role=role,
            transition_in=TransitionKind.CUT,
            transition_ms=0,
            motion=still_motion(index, portrait=portrait).kind,
            # A fast shot wants motion a photo does not have, and a shot longer
            # than a photo may be held cannot be filled by one.
            prefer=(
                SlotPreference.VIDEO
                if energy >= VIDEO_ENERGY or end - start > MAX_STILL_MS
                else SlotPreference.ANY
            ),
            beats=count,
        )
        for index, ((start, end), energy, role, count) in enumerate(
            zip(legal, energies, roles, counts, strict=True)
        )
    )

    beat_synced = trusted is not None and any(count is not None for count in counts)
    template = assert_valid_template(
        EditTemplate(
            id=template_id,
            name=name.strip(),
            source=TemplateSource.USER,
            aspect=aspect_for(width, height),
            slots=slots,
            bpm=round(trusted.bpm, 2) if beat_synced and trusted is not None else None,
        )
    )
    return Extraction(
        template=template,
        shots=len(shots),
        merged=merged,
        split=split,
        dropped=dropped,
        unmeasured=unmeasured,
        beat_synced=beat_synced,
    )


__all__ = ["EXTRACT_VERSION", "Extraction", "aspect_for", "extract_template"]
