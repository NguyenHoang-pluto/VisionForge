"""Edit templates (Phase 12): an edit's structure, with the footage left out.

A template is what is left of an edit when every clip is taken out of it: a row
of slots, each with a length, an energy, a narrative role, how it is entered and
how a still placed in it should move. Filling a template means choosing, for
each slot, the photo or video that fits it best -- which is the same question
the editorial engine already answers for its own pacing slots, asked of slots
that somebody else laid out.

Two sources, one type:

    builtin   shipped with VisionForge, declared as data in ``template_library``
    user      measured from a video the user uploaded (``template_extract``),
              stored per user so it can be reused in any project

Nothing here renders, plans or touches storage. A template is data, validated
against the same bounds a plan is, so that a template which passes here cannot
produce a plan the plan validator would refuse for structural reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_OUTPUT_MS,
    MIN_SEGMENT_MS,
    MIN_TRANSITION_MS,
    AspectRatio,
    TransitionKind,
)
from visionforge.domain.effects import EFFECT_BOUNDS, EffectKind
from visionforge.domain.story import StoryRole

#: Bumped when the meaning of a stored template changes.
TEMPLATE_VERSION = "1"

#: Longest name kept. A label, not a description.
MAX_TEMPLATE_NAME = 60

#: The motions a slot may ask a still for. The zooms and pans only: a slot
#: cannot ask for a speed change, because a still would have to refuse it.
SLOT_MOTIONS = frozenset(
    {
        EffectKind.ZOOM_IN,
        EffectKind.ZOOM_OUT,
        EffectKind.PAN_LEFT,
        EffectKind.PAN_RIGHT,
        EffectKind.PAN_UP,
        EffectKind.PAN_DOWN,
    }
)

#: How far a slot's motion travels. The still default, so a template's motion
#: looks like the drift an untemplated still gets.
SLOT_MOTION_AMOUNT = 0.12


class TemplateSource(StrEnum):
    BUILTIN = "builtin"
    USER = "user"


class SlotPreference(StrEnum):
    """What kind of media a slot is best filled with.

    A preference, not a rule. A slot that prefers a video is filled with a
    photo when the project has no video left -- an edit with every slot filled
    is worth more than a rule kept.
    """

    ANY = "any"
    VIDEO = "video"
    STILL = "still"


@dataclass(frozen=True, slots=True)
class TemplateSlot:
    """One position in a template."""

    duration_ms: int
    #: How much should be going on in the shot that fills it, 0..1. What the
    #: filler matches a clip's measured motion against.
    energy: float
    role: StoryRole
    transition_in: TransitionKind = TransitionKind.CUT
    transition_ms: int = 0
    #: How a still placed here moves. ``None`` leaves it to the default drift.
    #: Ignored for a video, which has its own motion.
    motion: EffectKind | None = None
    prefer: SlotPreference = SlotPreference.ANY
    #: Whole beats this slot occupies, when the template was laid out on a
    #: grid. With music, the slot is re-timed to that many of the music's beats.
    beats: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "energy": round(self.energy, 3),
            "role": self.role.value,
            "transition_in": self.transition_in.value,
            "transition_ms": self.transition_ms,
            "motion": self.motion.value if self.motion else None,
            "prefer": self.prefer.value,
            "beats": self.beats,
        }


@dataclass(frozen=True, slots=True)
class EditTemplate:
    """A whole template. Frozen: a stored template is a record."""

    id: str
    name: str
    source: TemplateSource
    aspect: AspectRatio
    slots: tuple[TemplateSlot, ...]
    #: The tempo the template's beats were counted at, when it has any.
    bpm: float | None = None
    version: str = TEMPLATE_VERSION

    @property
    def total_ms(self) -> int:
        """How long the finished edit runs: every slot, minus the dissolves."""
        total = sum(slot.duration_ms for slot in self.slots)
        overlaps = sum(
            slot.transition_ms for slot in self.slots[1:] if slot.transition_in.consumes_time
        )
        return total - overlaps

    @property
    def still_slots(self) -> int:
        return sum(1 for slot in self.slots if slot.prefer is SlotPreference.STILL)

    def as_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source.value,
            "aspect": self.aspect.value,
            "bpm": self.bpm,
            "version": self.version,
            "total_ms": self.total_ms,
            "slots": [slot.as_payload() for slot in self.slots],
        }


class TemplateInvalidError(ValueError):
    """A template that could not produce a valid plan."""


def validate_template(template: EditTemplate) -> list[str]:
    """Every reason this template could not become a plan. Empty means valid.

    The bounds are the plan validator's own, imported rather than restated, so
    the two cannot drift apart.
    """
    problems: list[str] = []
    if not template.name.strip() or len(template.name) > MAX_TEMPLATE_NAME:
        problems.append(f"name must be 1-{MAX_TEMPLATE_NAME} characters")
    if not 1 <= len(template.slots) <= MAX_SEGMENTS:
        problems.append(f"a template has 1-{MAX_SEGMENTS} slots, not {len(template.slots)}")
    if not MIN_OUTPUT_MS <= template.total_ms <= MAX_OUTPUT_MS:
        problems.append(f"total {template.total_ms} ms outside {MIN_OUTPUT_MS}-{MAX_OUTPUT_MS}")
    if template.bpm is not None and not 0 < template.bpm <= 300:
        problems.append(f"bpm {template.bpm} is not a tempo")

    for index, slot in enumerate(template.slots):
        where = f"slot {index + 1}"
        if not MIN_SEGMENT_MS <= slot.duration_ms <= MAX_SEGMENT_MS:
            problems.append(
                f"{where}: {slot.duration_ms} ms outside {MIN_SEGMENT_MS}-{MAX_SEGMENT_MS}"
            )
        if not 0.0 <= slot.energy <= 1.0:
            problems.append(f"{where}: energy {slot.energy} outside 0-1")
        if slot.beats is not None and slot.beats < 1:
            problems.append(f"{where}: beats must be at least 1")
        if slot.motion is not None and slot.motion not in SLOT_MOTIONS:
            problems.append(f"{where}: {slot.motion.value} is not a motion a still can have")

        kind = slot.transition_in
        if kind is TransitionKind.CUT:
            if slot.transition_ms != 0:
                problems.append(f"{where}: a cut has no duration")
            continue
        if kind.needs_previous and index == 0:
            problems.append(f"{where}: the first slot cannot dissolve from nothing")
        if not MIN_TRANSITION_MS <= slot.transition_ms <= MAX_TRANSITION_MS:
            problems.append(
                f"{where}: transition {slot.transition_ms} ms outside "
                f"{MIN_TRANSITION_MS}-{MAX_TRANSITION_MS}"
            )
        if kind.consumes_time:
            shortest = min(
                slot.duration_ms,
                template.slots[index - 1].duration_ms if index > 0 else slot.duration_ms,
            )
            if slot.transition_ms > shortest * MAX_TRANSITION_SHARE:
                problems.append(f"{where}: dissolve longer than half of a slot it joins")
    return problems


def assert_valid_template(template: EditTemplate) -> EditTemplate:
    problems = validate_template(template)
    if problems:
        raise TemplateInvalidError("; ".join(problems))
    return template


def template_from_payload(payload: dict[str, Any]) -> EditTemplate:
    """Rebuild a stored template. Raises ``TemplateInvalidError`` on bad data.

    Every enum is parsed, not trusted: a stored row is data, and a row edited by
    hand or written by an older build must fail here rather than in a render.
    """
    try:
        slots = tuple(
            TemplateSlot(
                duration_ms=int(raw["duration_ms"]),
                energy=float(raw["energy"]),
                role=StoryRole(raw["role"]),
                transition_in=TransitionKind(raw.get("transition_in", "cut")),
                transition_ms=int(raw.get("transition_ms", 0)),
                motion=EffectKind(raw["motion"]) if raw.get("motion") else None,
                prefer=SlotPreference(raw.get("prefer", "any")),
                beats=int(raw["beats"]) if raw.get("beats") is not None else None,
            )
            for raw in payload["slots"]
        )
        template = EditTemplate(
            id=str(payload["id"]),
            name=str(payload["name"]),
            source=TemplateSource(payload["source"]),
            aspect=AspectRatio(payload["aspect"]),
            slots=slots,
            bpm=float(payload["bpm"]) if payload.get("bpm") is not None else None,
            version=str(payload.get("version", TEMPLATE_VERSION)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TemplateInvalidError(f"stored template is malformed: {exc}") from exc
    return assert_valid_template(template)


def slot_motion_bounds_ok() -> bool:
    """Whether the slot motion amount is inside every motion's bounds."""
    return all(
        EFFECT_BOUNDS[kind][0] <= SLOT_MOTION_AMOUNT <= EFFECT_BOUNDS[kind][1]
        for kind in SLOT_MOTIONS
    )


__all__ = [
    "MAX_TEMPLATE_NAME",
    "SLOT_MOTIONS",
    "SLOT_MOTION_AMOUNT",
    "TEMPLATE_VERSION",
    "EditTemplate",
    "SlotPreference",
    "TemplateInvalidError",
    "TemplateSlot",
    "TemplateSource",
    "assert_valid_template",
    "slot_motion_bounds_ok",
    "template_from_payload",
    "validate_template",
]
