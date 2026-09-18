"""The templates VisionForge ships with (Phase 12).

Declared as data. Each slot is counted in beats at ``LIBRARY_BPM``, so with no
music a template runs at that tempo, and with music every slot is re-timed to
the same number of the music's own beats -- which is how a template made for
one song fits another.

The names here are English identifiers and fallbacks. The interface shows each
template by its id (``template.<id>``), so every language owns its own words.
"""

from __future__ import annotations

from visionforge.domain.editplan import AspectRatio, TransitionKind
from visionforge.domain.effects import EffectKind
from visionforge.domain.story import StoryRole
from visionforge.domain.template import (
    EditTemplate,
    SlotPreference,
    TemplateSlot,
    TemplateSource,
    assert_valid_template,
)

#: The tempo every library template is counted at. 120 BPM is a half-second
#: beat, which keeps the arithmetic in the tables below readable.
LIBRARY_BPM = 120.0
_BEAT_MS = 60_000 / LIBRARY_BPM

_CUT = TransitionKind.CUT
_DISSOLVE = TransitionKind.CROSSFADE
_FADE_IN = TransitionKind.FADE_IN
_FADE_OUT = TransitionKind.FADE_TO_BLACK

_ANY = SlotPreference.ANY
_VIDEO = SlotPreference.VIDEO
_STILL = SlotPreference.STILL

_IN = EffectKind.ZOOM_IN
_OUT = EffectKind.ZOOM_OUT
_LEFT = EffectKind.PAN_LEFT
_RIGHT = EffectKind.PAN_RIGHT

#: One row per slot: beats, energy, role, prefer, how it enters, how long the
#: entry runs (ms), and how a still in it moves.
_Row = tuple[int, float, StoryRole, SlotPreference, TransitionKind, int, EffectKind | None]


def _template(template_id: str, name: str, aspect: AspectRatio, rows: list[_Row]) -> EditTemplate:
    slots = tuple(
        TemplateSlot(
            duration_ms=int(round(beats * _BEAT_MS)),
            energy=energy,
            role=role,
            prefer=prefer,
            transition_in=transition,
            transition_ms=transition_ms,
            motion=motion,
            beats=beats,
        )
        for beats, energy, role, prefer, transition, transition_ms, motion in rows
    )
    return assert_valid_template(
        EditTemplate(
            id=template_id,
            name=name,
            source=TemplateSource.BUILTIN,
            aspect=aspect,
            slots=slots,
            bpm=LIBRARY_BPM,
        )
    )


_H, _S, _B, _P, _R, _E = (
    StoryRole.HOOK,
    StoryRole.SETUP,
    StoryRole.BUILD,
    StoryRole.PEAK,
    StoryRole.REACTION,
    StoryRole.ENDING,
)

TRAVEL = _template(
    "travel",
    "Travel",
    AspectRatio.LANDSCAPE_16_9,
    [
        # A quick glimpse of the best of it, then the calm arrival.
        (2, 0.70, _H, _VIDEO, _FADE_IN, 400, None),
        (6, 0.20, _S, _STILL, _DISSOLVE, 400, _IN),
        (6, 0.25, _S, _ANY, _DISSOLVE, 600, _RIGHT),
        (4, 0.40, _B, _VIDEO, _CUT, 0, None),
        (4, 0.45, _B, _STILL, _CUT, 0, _LEFT),
        (2, 0.60, _B, _VIDEO, _CUT, 0, None),
        (2, 0.65, _B, _ANY, _CUT, 0, _IN),
        (2, 0.75, _B, _VIDEO, _CUT, 0, None),
        (4, 0.90, _P, _VIDEO, _CUT, 0, None),
        (4, 0.50, _R, _ANY, _DISSOLVE, 500, _OUT),
        (8, 0.20, _E, _STILL, _DISSOLVE, 800, _OUT),
    ],
)

MEMORIES = _template(
    "memories",
    "Memories",
    AspectRatio.LANDSCAPE_16_9,
    [
        # Birthdays, anniversaries: mostly photos, every join a dissolve.
        (8, 0.15, _H, _STILL, _FADE_IN, 800, _IN),
        (6, 0.20, _S, _STILL, _DISSOLVE, 700, _RIGHT),
        (6, 0.20, _S, _STILL, _DISSOLVE, 700, _OUT),
        (6, 0.30, _B, _ANY, _DISSOLVE, 700, _LEFT),
        (6, 0.30, _B, _STILL, _DISSOLVE, 700, _IN),
        (6, 0.35, _B, _ANY, _DISSOLVE, 700, _RIGHT),
        (6, 0.50, _P, _VIDEO, _DISSOLVE, 700, None),
        (6, 0.30, _R, _STILL, _DISSOLVE, 700, _OUT),
        (10, 0.15, _E, _STILL, _DISSOLVE, 1000, _IN),
    ],
)

BEAT_HIGHLIGHT = _template(
    "beat_highlight",
    "Beat highlight",
    AspectRatio.LANDSCAPE_16_9,
    [
        # Cut on every other beat, then every beat, then hold on the peak.
        (2, 0.80, _H, _VIDEO, _CUT, 0, None),
        (2, 0.50, _S, _ANY, _CUT, 0, _IN),
        (2, 0.55, _S, _ANY, _CUT, 0, _RIGHT),
        (2, 0.60, _B, _VIDEO, _CUT, 0, None),
        (2, 0.65, _B, _ANY, _CUT, 0, _LEFT),
        (1, 0.70, _B, _VIDEO, _CUT, 0, None),
        (1, 0.75, _B, _ANY, _CUT, 0, _IN),
        (1, 0.80, _B, _VIDEO, _CUT, 0, None),
        (1, 0.85, _B, _ANY, _CUT, 0, _OUT),
        (1, 0.90, _B, _VIDEO, _CUT, 0, None),
        (1, 0.90, _B, _ANY, _CUT, 0, _RIGHT),
        (4, 1.00, _P, _VIDEO, _CUT, 0, None),
        (2, 0.70, _R, _ANY, _CUT, 0, _IN),
        (4, 0.40, _E, _ANY, _FADE_OUT, 600, _OUT),
    ],
)

SLIDESHOW = _template(
    "slideshow",
    "Photo slideshow",
    AspectRatio.LANDSCAPE_16_9,
    [
        # Photos only, each held four seconds, the drift alternating.
        (8, 0.15, _H, _STILL, _FADE_IN, 800, _IN),
        (8, 0.15, _S, _STILL, _DISSOLVE, 800, _RIGHT),
        (8, 0.15, _S, _STILL, _DISSOLVE, 800, _OUT),
        (8, 0.20, _B, _STILL, _DISSOLVE, 800, _LEFT),
        (8, 0.20, _B, _STILL, _DISSOLVE, 800, _IN),
        (8, 0.20, _B, _STILL, _DISSOLVE, 800, _RIGHT),
        (8, 0.25, _P, _STILL, _DISSOLVE, 800, _OUT),
        (8, 0.20, _R, _STILL, _DISSOLVE, 800, _LEFT),
        (8, 0.15, _E, _STILL, _DISSOLVE, 800, _IN),
    ],
)

DAY_VLOG = _template(
    "day_vlog",
    "A day in",
    AspectRatio.LANDSCAPE_16_9,
    [
        # Morning to night: people and places, conversational lengths.
        (4, 0.50, _H, _VIDEO, _CUT, 0, None),
        (6, 0.25, _S, _STILL, _CUT, 0, _IN),
        (6, 0.35, _S, _VIDEO, _CUT, 0, None),
        (4, 0.40, _B, _ANY, _CUT, 0, _RIGHT),
        (6, 0.45, _B, _VIDEO, _CUT, 0, None),
        (4, 0.40, _B, _STILL, _CUT, 0, _LEFT),
        (6, 0.60, _P, _VIDEO, _CUT, 0, None),
        (4, 0.45, _R, _ANY, _CUT, 0, _OUT),
        (8, 0.20, _E, _ANY, _FADE_OUT, 800, _OUT),
    ],
)

VERTICAL_REEL = _template(
    "vertical_reel",
    "Vertical reel",
    AspectRatio.PORTRAIT_9_16,
    [
        # Strongest shot first, very short holds, no preamble.
        (2, 0.90, _H, _VIDEO, _CUT, 0, None),
        (2, 0.60, _B, _ANY, _CUT, 0, _IN),
        (2, 0.65, _B, _VIDEO, _CUT, 0, None),
        (1, 0.70, _B, _ANY, _CUT, 0, _OUT),
        (1, 0.75, _B, _VIDEO, _CUT, 0, None),
        (2, 0.70, _B, _ANY, _CUT, 0, _IN),
        (3, 0.95, _P, _VIDEO, _CUT, 0, None),
        (2, 0.60, _R, _ANY, _CUT, 0, _OUT),
        (3, 0.40, _E, _ANY, _CUT, 0, _IN),
    ],
)

#: Every library template, in the order the interface lists them.
LIBRARY: tuple[EditTemplate, ...] = (
    TRAVEL,
    MEMORIES,
    BEAT_HIGHLIGHT,
    SLIDESHOW,
    DAY_VLOG,
    VERTICAL_REEL,
)

LIBRARY_BY_ID: dict[str, EditTemplate] = {template.id: template for template in LIBRARY}


def builtin_template(template_id: str) -> EditTemplate | None:
    return LIBRARY_BY_ID.get(template_id)


__all__ = ["LIBRARY", "LIBRARY_BPM", "LIBRARY_BY_ID", "builtin_template"]
