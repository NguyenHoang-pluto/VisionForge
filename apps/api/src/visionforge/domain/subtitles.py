"""Subtitles: validated cues, closed presets, and text that stays text.

The security question for this feature is sharper than for any other part of the
editor, because subtitles are the first user-supplied **string** that reaches the
renderer. Everything before them was a number or a member of an enum.

Two decisions answer it.

**Text never enters a filter graph.** The obvious implementation is FFmpeg's
``drawtext``, and it is the wrong one: the text becomes part of the filter
string, where ``:`` separates options, ``\\`` escapes, ``'`` quotes and ``%``
expands -- so a subtitle reading ``12:30`` is a syntax error and a subtitle
reading ``':drawbox=...`` is something worse. This module produces an ASS
document instead, which the compiler writes to the render's own scratch
directory and references by a path *it* generated. The text lives in a data file
that libass parses as text, and the only thing on the command line is a filename
nobody outside the server chose.

**Style is a preset id, never a parameter.** A client picks ``cinematic``. It
does not send a font path, a font size, a colour, an outline width, or anything
else that ends up inside the renderer -- because a font path is a filesystem
path, and this codebase has spent eight phases making sure those do not arrive
from outside.

What remains is ordinary validation: cues have bounds, they are ordered, they do
not overlap, and the text is length-capped and stripped of the control
characters that would break the document format regardless of intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

#: Bounds on one cue. A subtitle shorter than this cannot be read; one longer
#: than this has stopped being a subtitle and become a caption card.
MIN_CUE_MS = 400
MAX_CUE_MS = 10_000

#: Characters per cue. Two lines of comfortable reading width; beyond it the
#: text either overflows the safe area or shrinks past legibility, and neither
#: is something the renderer should be asked to decide at encode time.
MAX_CUE_CHARS = 120

#: Cues per plan. Enough for a fully subtitled three-minute edit at a natural
#: speaking rate; short of the point where the ASS document stops being
#: reviewable.
MAX_CUES = 300

#: Smallest gap required between consecutive cues. Zero would allow two cues to
#: touch, which renders as a flicker rather than as a change.
MIN_CUE_GAP_MS = 1


class SubtitleStyle(StrEnum):
    """The closed set of looks. The client sends one of these and nothing else."""

    CLEAN = "clean"
    BOLD = "bold"
    MINIMAL = "minimal"
    CINEMATIC = "cinematic"
    SOCIAL = "social"


class SubtitlePosition(StrEnum):
    """Where in the frame. A closed set, not coordinates.

    Pixel positions are deliberately absent. They would have to be validated
    against an output geometry the client does not choose, they would break the
    moment an edit is re-rendered at another aspect ratio, and they are the
    field through which "put the text at -10000" arrives.
    """

    BOTTOM = "bottom"
    CENTER = "center"
    TOP = "top"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


#: ASS alignment codes (numpad layout) per position.
ALIGNMENT: dict[SubtitlePosition, int] = {
    SubtitlePosition.BOTTOM: 2,
    SubtitlePosition.CENTER: 5,
    SubtitlePosition.TOP: 8,
    SubtitlePosition.BOTTOM_LEFT: 1,
    SubtitlePosition.BOTTOM_RIGHT: 3,
}


@dataclass(frozen=True, slots=True)
class StylePreset:
    """Everything the renderer needs to draw one look, decided server-side.

    Sizes are given for a 1080-line frame and scaled to the real output height
    by the compiler, so a preset looks the same on a 720p preview and a 1080p
    master rather than half the size on one of them.

    Colours are RGB tuples here and converted to ASS's ``&HBBGGRR`` at the last
    moment -- keeping a format quirk in the one place that has to know it.
    """

    style: SubtitleStyle
    label: str
    #: A family name, never a path. libass resolves it through fontconfig and
    #: falls back on its own if the machine does not have it, which is why the
    #: families chosen here are ones that exist nearly everywhere.
    font: str
    #: Points at 1080 lines.
    size: int
    bold: bool
    #: Fill, outline and shadow, as 0-255 RGB.
    primary: tuple[int, int, int]
    outline_colour: tuple[int, int, int]
    outline: float
    shadow: float
    #: Distance from the frame edge, in pixels at 1080 lines. The safe area.
    margin_v: int
    margin_h: int


#: The presets. Five, and each one is a decision rather than a variation.
STYLE_PRESETS: dict[SubtitleStyle, StylePreset] = {
    SubtitleStyle.CLEAN: StylePreset(
        style=SubtitleStyle.CLEAN,
        label="Clean",
        font="Arial",
        size=48,
        bold=False,
        primary=(255, 255, 255),
        outline_colour=(0, 0, 0),
        outline=2.0,
        shadow=0.0,
        margin_v=60,
        margin_h=80,
    ),
    SubtitleStyle.BOLD: StylePreset(
        style=SubtitleStyle.BOLD,
        label="Bold",
        font="Arial",
        size=58,
        bold=True,
        primary=(255, 255, 255),
        outline_colour=(0, 0, 0),
        outline=3.5,
        shadow=1.0,
        margin_v=64,
        margin_h=80,
    ),
    SubtitleStyle.MINIMAL: StylePreset(
        style=SubtitleStyle.MINIMAL,
        label="Minimal",
        font="Arial",
        size=40,
        bold=False,
        primary=(235, 235, 235),
        outline_colour=(0, 0, 0),
        outline=1.0,
        shadow=0.0,
        margin_v=52,
        margin_h=96,
    ),
    SubtitleStyle.CINEMATIC: StylePreset(
        style=SubtitleStyle.CINEMATIC,
        label="Cinematic",
        # Georgia is a serif that ships with Windows and macOS and is present in
        # most Linux font packages; libass falls back gracefully where it is not.
        font="Georgia",
        size=46,
        bold=False,
        primary=(245, 240, 232),
        outline_colour=(0, 0, 0),
        outline=1.5,
        shadow=1.5,
        # Sits higher than the others: cinema subtitles clear the lower third.
        margin_v=90,
        margin_h=120,
    ),
    SubtitleStyle.SOCIAL: StylePreset(
        style=SubtitleStyle.SOCIAL,
        label="Social",
        font="Arial",
        size=64,
        bold=True,
        primary=(255, 255, 255),
        outline_colour=(16, 16, 16),
        outline=4.0,
        shadow=0.0,
        # Vertical video puts the text near the middle, clear of the interface
        # chrome every social player draws over the bottom sixth.
        margin_v=240,
        margin_h=64,
    ),
}


def preset_for(style: SubtitleStyle) -> StylePreset:
    return STYLE_PRESETS[style]


#: Control characters, and the two that would break the document format.
#:
#: ASS is line-oriented and comma-separated, so a newline or a brace inside a
#: dialogue line changes the structure rather than the text. They are stripped
#: rather than escaped: a subtitle containing a literal ``{`` is vanishingly
#: rare, and silently rendering an override tag would be worse than dropping a
#: character.
_FORBIDDEN = re.compile(r"[\x00-\x1f\x7f{}\\]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Reduce arbitrary input to plain display text.

    Not an escape. Escaping implies the original is recoverable and that the
    renderer will interpret it; this removes the characters that carry meaning
    to anything downstream and collapses the rest to single spaces, so what
    reaches the document is words.
    """
    return _WHITESPACE.sub(" ", _FORBIDDEN.sub(" ", raw)).strip()


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """One line of text, on screen for one interval of the *output*.

    Timeline coordinates, not source coordinates: a cue belongs to the
    programme, not to the clip that happens to be under it. That matters as soon
    as a crossfade shortens the timeline -- a cue pinned to a segment would move
    when the edit was re-paced, and a viewer would see it drift off the line it
    was written for.
    """

    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def as_payload(self) -> dict[str, Any]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text}

    @staticmethod
    def from_payload(raw: dict[str, Any]) -> SubtitleCue:
        return SubtitleCue(
            start_ms=int(raw["start_ms"]),
            end_ms=int(raw["end_ms"]),
            text=str(raw["text"]),
        )


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    """Every cue in an edit, plus the one look they share.

    One style for the whole track rather than one per cue. Mixed typography
    inside a single edit is a design mistake far more often than it is a
    decision, and per-cue styling is a vocabulary that can be added later
    without changing what is stored now.
    """

    cues: tuple[SubtitleCue, ...]
    style: SubtitleStyle = SubtitleStyle.CLEAN
    position: SubtitlePosition = SubtitlePosition.BOTTOM

    @property
    def preset(self) -> StylePreset:
        return preset_for(self.style)

    @property
    def alignment(self) -> int:
        return ALIGNMENT[self.position]

    @property
    def ordered(self) -> tuple[SubtitleCue, ...]:
        return tuple(sorted(self.cues, key=lambda cue: (cue.start_ms, cue.end_ms)))

    def as_payload(self) -> dict[str, Any]:
        return {
            "style": self.style.value,
            "position": self.position.value,
            "cues": [cue.as_payload() for cue in self.ordered],
        }

    @staticmethod
    def from_payload(raw: dict[str, Any]) -> SubtitleTrack:
        return SubtitleTrack(
            cues=tuple(SubtitleCue.from_payload(cue) for cue in raw.get("cues", [])),
            style=SubtitleStyle(raw.get("style", "clean")),
            position=SubtitlePosition(raw.get("position", "bottom")),
        )


__all__ = [
    "ALIGNMENT",
    "MAX_CUES",
    "MAX_CUE_CHARS",
    "MAX_CUE_MS",
    "MIN_CUE_GAP_MS",
    "MIN_CUE_MS",
    "STYLE_PRESETS",
    "StylePreset",
    "SubtitleCue",
    "SubtitlePosition",
    "SubtitleStyle",
    "SubtitleTrack",
    "clean_text",
    "preset_for",
]
