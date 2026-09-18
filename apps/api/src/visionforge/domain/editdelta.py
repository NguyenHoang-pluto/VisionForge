"""The EditDelta: the only thing a co-editor is allowed to say about an edit.

Phase 5 established that a model may not emit an ``EditPlan``, because a plan
carries UUIDs and a model that emits UUIDs can emit the wrong one. It emits an
``EditDirective`` instead -- handles and enums -- which deterministic code turns
into a plan.

Phase 10 asks a narrower question. There is already a plan; the user wants it
*changed*. Regenerating from a directive would throw away every hand edit and
re-decide everything the user has already accepted, so the answer is not another
directive. It is a **delta**: a short list of operations drawn from a closed
vocabulary, each addressing something the current plan already has.

    EditDelta
      ├── operations  [ {kind: CHANGE_MUSIC_VOLUME, value: 0.4},
      │                 {kind: TRIM_SEGMENT, segment: 0, source_out_ms: 3200} ]
      └── rationale   a short string, for the user, never interpreted

and deterministic code applies that to the plan:

    delta --parse----------> closed enum + typed fields, or a violation
          --validate-------> against the plan it will be applied to
          --patch----------> a new EditPlan, built field by field
          --validate_plan--> the same Phase 4 gate every plan passes

**Sixteen kinds, and no seventeenth arrives by accident.** Every operation is a
frozen dataclass whose fields are integers, floats, booleans and members of
enums this codebase already validates. There is no parameter dictionary, no
free-form options bag and no string field except subtitle text -- which goes
through the same ``clean_text`` a hand-typed cue does and ends up in an ASS
document, never in a filter expression.

So there is nowhere in this schema to put a filesystem path, a storage key, an
FFmpeg argument, a filter string, a URL or a shell command. That is a structural
guarantee rather than a validation rule: the parser does not reject those
things, it has no field to read them into.

**Segments are addressed by position, not by identity.** ``segment: 0`` is the
first clip of the plan being edited. A model never sees a media id and cannot
name one -- the same property the Phase 5 handle table gives the planner,
achieved here by not needing a table at all: an index that is out of range is a
violation, and an index that is in range can only ever mean the user's own clip
in the user's own plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias, TypeVar

from visionforge.domain.editplan import (
    MAX_FADE_MS,
    MAX_GAIN,
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_TRANSITION_MS,
    MIN_GAIN,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    QualityPreset,
    TransitionKind,
)
from visionforge.domain.effects import EFFECT_BOUNDS, EffectKind
from visionforge.domain.policy import StyleStrength
from visionforge.domain.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUE_MS,
    SubtitlePosition,
    SubtitleStyle,
    clean_text,
)

#: Operations accepted in one delta. A change request is a handful of edits; a
#: list longer than this is a regeneration wearing a delta's clothes, and the
#: cost of applying twenty-four operations to a forty-segment plan is already
#: more than anyone asked for in a sentence.
MAX_OPERATIONS = 24

#: Longest rationale kept, matching the planner's. Display text, never parsed.
MAX_RATIONALE_CHARS = 400


class OperationKind(StrEnum):
    """What one operation does. The whole vocabulary, and it is closed.

    Values are UPPER_SNAKE rather than lower, unlike every other enum here,
    because these are the tokens a model emits and a distinctive shape makes a
    hallucinated ``"remove clip"`` fail loudly instead of matching by accident.
    """

    # --- structure ---
    REMOVE_SEGMENT = "REMOVE_SEGMENT"
    REORDER_SEGMENT = "REORDER_SEGMENT"
    TRIM_SEGMENT = "TRIM_SEGMENT"
    CHANGE_DURATION = "CHANGE_DURATION"

    # --- look ---
    CHANGE_STYLE_STRENGTH = "CHANGE_STYLE_STRENGTH"
    CHANGE_TRANSITION = "CHANGE_TRANSITION"
    ADD_EFFECT = "ADD_EFFECT"
    REMOVE_EFFECT = "REMOVE_EFFECT"
    MODIFY_EFFECT = "MODIFY_EFFECT"

    # --- text ---
    ADD_SUBTITLE = "ADD_SUBTITLE"
    MODIFY_SUBTITLE = "MODIFY_SUBTITLE"
    REMOVE_SUBTITLE = "REMOVE_SUBTITLE"

    # --- sound ---
    CHANGE_MUSIC_VOLUME = "CHANGE_MUSIC_VOLUME"
    CHANGE_MUSIC_FADE = "CHANGE_MUSIC_FADE"
    CHANGE_BEAT_SYNC = "CHANGE_BEAT_SYNC"

    # --- output ---
    CHANGE_OUTPUT_PRESET = "CHANGE_OUTPUT_PRESET"


# --------------------------------------------------------------- the operations
#
# One frozen dataclass per kind. Each carries ``KIND`` as a ClassVar so an
# operation always knows what it is without a caller having to pass the tag
# alongside it, and ``describe()`` so the UI has a sentence for it even before
# the patcher has computed what it changed.


@dataclass(frozen=True, slots=True)
class RemoveSegment:
    """Drop one clip. The clips after it move up; orders are renumbered."""

    KIND: ClassVar[OperationKind] = OperationKind.REMOVE_SEGMENT
    segment: int

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "segment": self.segment}

    def describe(self) -> str:
        return f"Remove clip {self.segment + 1}"


@dataclass(frozen=True, slots=True)
class ReorderSegment:
    """Move one clip to another position in the sequence."""

    KIND: ClassVar[OperationKind] = OperationKind.REORDER_SEGMENT
    segment: int
    to_index: int

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "segment": self.segment, "to_index": self.to_index}

    def describe(self) -> str:
        return f"Move clip {self.segment + 1} to position {self.to_index + 1}"


@dataclass(frozen=True, slots=True)
class TrimSegment:
    """Change where a clip starts or ends *within its source*.

    Absolute offsets, not deltas: "start at 2000 ms" survives being applied
    twice, where "start 500 ms later" does not. Either end may be left alone by
    sending ``None``, which is how "trim the head" is expressed without having
    to restate the tail.
    """

    KIND: ClassVar[OperationKind] = OperationKind.TRIM_SEGMENT
    segment: int
    source_in_ms: int | None = None
    source_out_ms: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "source_in_ms": self.source_in_ms,
            "source_out_ms": self.source_out_ms,
        }

    def describe(self) -> str:
        return f"Retrim clip {self.segment + 1}"


@dataclass(frozen=True, slots=True)
class ChangeDuration:
    """How long a clip holds, or how long the whole edit runs.

    ``segment`` names a clip; ``None`` means the programme. Both are expressed
    in source milliseconds and applied by moving out points, never by changing
    speed -- "make it shorter" and "make it faster" are different edits and
    conflating them would produce one when the user asked for the other.

    The whole-edit form scales every clip by the same factor rather than
    truncating the tail, because "keep this edit but make it 20 seconds" means
    the same shots, tighter.
    """

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_DURATION
    duration_ms: int
    segment: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "duration_ms": self.duration_ms,
        }

    def describe(self) -> str:
        seconds = self.duration_ms / 1000
        if self.segment is None:
            return f"Set the edit to {seconds:.1f}s"
        return f"Hold clip {self.segment + 1} for {seconds:.1f}s"


@dataclass(frozen=True, slots=True)
class ChangeStyleStrength:
    """How much of the project's reference video the next plan should apply.

    Recorded on the plan, not retro-applied to it. Reference strength decides
    *selection weights and pacing bounds*, which are inputs to planning -- a
    plan that has already been cut cannot be restyled without re-cutting it, and
    re-cutting it is the regeneration this phase exists to avoid. So this
    operation stores the intent where the planner reads it, and the diff says so
    plainly rather than implying the existing cuts changed.
    """

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_STYLE_STRENGTH
    value: StyleStrength

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "value": self.value.value}

    def describe(self) -> str:
        return f"Reference style strength to {self.value.value}%"


@dataclass(frozen=True, slots=True)
class ChangeTransition:
    """How a clip enters, and for how long.

    ``duration_ms`` of ``None`` lets the patcher choose one that fits the two
    clips being joined, which is what a user means by "crossfade these" far more
    often than any particular number of milliseconds.
    """

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_TRANSITION
    segment: int
    transition: TransitionKind
    duration_ms: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "transition": self.transition.value,
            "duration_ms": self.duration_ms,
        }

    def describe(self) -> str:
        return f"{self.transition.value.replace('_', ' ').title()} into clip {self.segment + 1}"


@dataclass(frozen=True, slots=True)
class AddEffect:
    """Put one effect on one clip, or on every clip.

    ``segment: None`` means all of them, which is what "make the whole edit
    warmer" decomposes to. It is a convenience, not a new capability: the result
    is the same per-segment effects the schema already allows, and each one is
    checked against ``EFFECT_BOUNDS`` for its kind.
    """

    KIND: ClassVar[OperationKind] = OperationKind.ADD_EFFECT
    effect: EffectKind
    amount: float
    segment: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "effect": self.effect.value,
            "amount": round(self.amount, 4),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    def describe(self) -> str:
        where = "every clip" if self.segment is None else f"clip {self.segment + 1}"
        return f"Add {self.effect.value.replace('_', ' ')} to {where}"


@dataclass(frozen=True, slots=True)
class RemoveEffect:
    """Take one effect kind off one clip, or off every clip."""

    KIND: ClassVar[OperationKind] = OperationKind.REMOVE_EFFECT
    effect: EffectKind
    segment: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "effect": self.effect.value,
        }

    def describe(self) -> str:
        where = "every clip" if self.segment is None else f"clip {self.segment + 1}"
        return f"Remove {self.effect.value.replace('_', ' ')} from {where}"


@dataclass(frozen=True, slots=True)
class ModifyEffect:
    """Change the amount of an effect a clip already carries.

    Distinct from ``ADD_EFFECT`` on purpose. Adding is idempotent and always
    succeeds; modifying asserts that the effect is there, and a request to turn
    down a zoom that was never applied is a misunderstanding worth reporting
    rather than silently turning into an addition.
    """

    KIND: ClassVar[OperationKind] = OperationKind.MODIFY_EFFECT
    effect: EffectKind
    amount: float
    segment: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "segment": self.segment,
            "effect": self.effect.value,
            "amount": round(self.amount, 4),
        }

    def describe(self) -> str:
        where = "every clip" if self.segment is None else f"clip {self.segment + 1}"
        return f"Adjust {self.effect.value.replace('_', ' ')} on {where}"


@dataclass(frozen=True, slots=True)
class AddSubtitle:
    """One new cue, in timeline coordinates.

    ``text`` is the only free string in the whole vocabulary. It is cleaned by
    the same function a hand-typed cue goes through, capped at the same length,
    and ends up in an ASS document that libass parses as text -- never in a
    filter expression. See ``domain.subtitles`` for why that is structural.
    """

    KIND: ClassVar[OperationKind] = OperationKind.ADD_SUBTITLE
    start_ms: int
    end_ms: int
    text: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }

    def describe(self) -> str:
        return f"Add a subtitle at {self.start_ms / 1000:.1f}s"


@dataclass(frozen=True, slots=True)
class ModifySubtitle:
    """Change one cue, or the look of the whole track.

    ``cue: None`` addresses the track, and then only ``style`` and ``position``
    may be set -- there is no cue whose text it could mean. ``cue`` set
    addresses one line, and then only its text and timing may change, because
    per-cue styling is not something ``SubtitleTrack`` can express.

    Both halves live in one operation rather than two because the vocabulary is
    fixed at sixteen kinds; the parser enforces the split, so neither half can
    reach fields belonging to the other.
    """

    KIND: ClassVar[OperationKind] = OperationKind.MODIFY_SUBTITLE
    cue: int | None = None
    text: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    style: SubtitleStyle | None = None
    position: SubtitlePosition | None = None

    @property
    def is_track_level(self) -> bool:
        return self.cue is None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "cue": self.cue,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "style": self.style.value if self.style else None,
            "position": self.position.value if self.position else None,
        }

    def describe(self) -> str:
        if self.cue is None:
            parts = [
                value
                for value in (
                    f"{self.style.value} subtitles" if self.style else None,
                    f"text at {self.position.value.replace('_', ' ')}" if self.position else None,
                )
                if value
            ]
            return "Use " + " and ".join(parts) if parts else "Restyle the subtitles"
        return f"Edit subtitle {self.cue + 1}"


@dataclass(frozen=True, slots=True)
class RemoveSubtitle:
    """Delete one cue, or the whole track when ``cue`` is ``None``."""

    KIND: ClassVar[OperationKind] = OperationKind.REMOVE_SUBTITLE
    cue: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "cue": self.cue}

    def describe(self) -> str:
        return "Remove the subtitles" if self.cue is None else f"Remove subtitle {self.cue + 1}"


@dataclass(frozen=True, slots=True)
class ChangeMusicVolume:
    """Linear gain on the music bed. 1.0 is unity; the UI shows a percentage."""

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_MUSIC_VOLUME
    value: float

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "value": round(self.value, 4)}

    def describe(self) -> str:
        return f"Music volume to {round(self.value * 100)}%"


@dataclass(frozen=True, slots=True)
class ChangeMusicFade:
    """The bed's fades. Either end may be left alone with ``None``."""

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_MUSIC_FADE
    fade_in_ms: int | None = None
    fade_out_ms: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
        }

    def describe(self) -> str:
        parts = [
            text
            for text in (
                f"fade in {self.fade_in_ms} ms" if self.fade_in_ms is not None else None,
                f"fade out {self.fade_out_ms} ms" if self.fade_out_ms is not None else None,
            )
            if text
        ]
        return "Music " + " and ".join(parts) if parts else "Music fades"


@dataclass(frozen=True, slots=True)
class ChangeBeatSync:
    """Whether the *next* plan should cut to the track's beats.

    Like ``CHANGE_STYLE_STRENGTH``, this is an input to planning rather than a
    property of a finished cut: moving existing cuts onto a beat grid is a
    re-plan, not a patch. It is recorded where the planner reads it and the diff
    says which it is.
    """

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_BEAT_SYNC
    enabled: bool

    def as_payload(self) -> dict[str, Any]:
        return {"kind": self.KIND.value, "enabled": self.enabled}

    def describe(self) -> str:
        return f"Beat sync {'on' if self.enabled else 'off'}"


@dataclass(frozen=True, slots=True)
class ChangeOutputPreset:
    """Shape, frame rate, quality, audio mode and source gain.

    Every field is a member of an enum the server owns, or a bounded number.
    Width and height are absent: geometry is derived from the aspect ratio
    through the server's own preset table, which is the Phase 4 property that a
    delta must not be the thing to erode.
    """

    KIND: ClassVar[OperationKind] = OperationKind.CHANGE_OUTPUT_PRESET
    aspect_ratio: AspectRatio | None = None
    fps: int | None = None
    quality: QualityPreset | None = None
    audio: AudioMode | None = None
    source_gain: float | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value,
            "aspect_ratio": self.aspect_ratio.value if self.aspect_ratio else None,
            "fps": self.fps,
            "quality": self.quality.value if self.quality else None,
            "audio": self.audio.value if self.audio else None,
            "source_gain": None if self.source_gain is None else round(self.source_gain, 4),
        }

    def describe(self) -> str:
        parts = [
            text
            for text in (
                self.aspect_ratio.value if self.aspect_ratio else None,
                f"{self.fps} fps" if self.fps else None,
                self.quality.value if self.quality else None,
                f"source audio {self.audio.value}" if self.audio else None,
                (
                    f"clip volume {round(self.source_gain * 100)}%"
                    if self.source_gain is not None
                    else None
                ),
            )
            if text
        ]
        return "Output: " + ", ".join(parts) if parts else "Output settings"


#: Every operation, as one type. Exhaustive: a kind without a member here has no
#: way to be constructed, which is what keeps the vocabulary closed.
EditOperation: TypeAlias = (
    RemoveSegment
    | ReorderSegment
    | TrimSegment
    | ChangeDuration
    | ChangeStyleStrength
    | ChangeTransition
    | AddEffect
    | RemoveEffect
    | ModifyEffect
    | AddSubtitle
    | ModifySubtitle
    | RemoveSubtitle
    | ChangeMusicVolume
    | ChangeMusicFade
    | ChangeBeatSync
    | ChangeOutputPreset
)


@dataclass(frozen=True, slots=True)
class EditDelta:
    """A complete change request. Operations plus a sentence about them."""

    operations: tuple[EditOperation, ...]
    rationale: str = ""
    #: Where the operations came from: ``rules`` for the deterministic resolver,
    #: ``llm`` for a model, ``client`` for operations a user confirmed. Recorded
    #: on the version so a history entry can say what produced it.
    source: str = "rules"

    def as_payload(self) -> dict[str, Any]:
        return {
            "operations": [operation.as_payload() for operation in self.operations],
            "rationale": self.rationale,
            "source": self.source,
        }

    @property
    def summary(self) -> tuple[str, ...]:
        return tuple(operation.describe() for operation in self.operations)


# ------------------------------------------------------------------ violations
@dataclass(frozen=True, slots=True)
class DeltaViolation:
    """One reason an operation was rejected. Machine-readable, like the rest.

    ``index`` is the position in the operations array, so a UI can point at the
    operation that failed and a repair prompt can name it.
    """

    code: str
    message: str
    index: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "index": self.index}


class DeltaInvalidError(ValueError):
    """A delta could not be read, or could not be applied. Carries every reason."""

    def __init__(self, violations: list[DeltaViolation]) -> None:
        self.violations = violations
        super().__init__("; ".join(v.message for v in violations) or "unreadable delta")


class _Rejected(Exception):
    """Internal: one field was missing, mistyped or out of range."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


#: Any of the string enums the vocabulary draws on.
_E = TypeVar("_E", bound=StrEnum)


# --------------------------------------------------------------- field readers
#
# Every reader is total and refuses rather than coerces. ``bool`` is rejected
# where a number is wanted because ``True`` is an ``int`` in Python, and
# ``{"segment": true}`` must not quietly become clip 2.


def _require_int(entry: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Rejected("field_missing", f"{key} must be a number")
    number = int(value)
    if not minimum <= number <= maximum:
        raise _Rejected("field_range", f"{key} {number} is outside {minimum}-{maximum}")
    return number


def _optional_int(entry: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int | None:
    if entry.get(key) is None:
        return None
    return _require_int(entry, key, minimum=minimum, maximum=maximum)


def _require_float(entry: dict[str, Any], key: str, *, minimum: float, maximum: float) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Rejected("field_missing", f"{key} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise _Rejected("field_range", f"{key} {number} is outside {minimum}-{maximum}")
    return number


def _optional_float(
    entry: dict[str, Any], key: str, *, minimum: float, maximum: float
) -> float | None:
    if entry.get(key) is None:
        return None
    return _require_float(entry, key, minimum=minimum, maximum=maximum)


def _require_bool(entry: dict[str, Any], key: str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise _Rejected("field_missing", f"{key} must be true or false")
    return value


def _require_enum(entry: dict[str, Any], key: str, enum: type[_E]) -> _E:
    value = entry.get(key)
    if not isinstance(value, str):
        raise _Rejected("field_missing", f"{key} must be a name")
    try:
        return enum(value.strip().lower())
    except ValueError as exc:
        names = ", ".join(member.value for member in enum)
        raise _Rejected("unknown_value", f"{key} {value!r} is not one of: {names}") from exc


def _optional_enum(entry: dict[str, Any], key: str, enum: type[_E]) -> _E | None:
    if entry.get(key) is None:
        return None
    return _require_enum(entry, key, enum)


def _require_text(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise _Rejected("field_missing", f"{key} must be text")
    cleaned = clean_text(value)[:MAX_CUE_CHARS]
    if not cleaned:
        raise _Rejected("field_empty", f"{key} has no displayable characters")
    return cleaned


def _segment_index(entry: dict[str, Any], *, required: bool) -> int | None:
    """Read a segment address. Bounds against the plan come later, in the patcher.

    ``MAX_SEGMENT_INDEX`` here is only a sanity ceiling -- the real check is
    "does this plan have that clip", which needs the plan and therefore belongs
    to validation rather than parsing.
    """
    if entry.get("segment") is None:
        if required:
            raise _Rejected("field_missing", "segment must be a clip index")
        return None
    return _require_int(entry, "segment", minimum=0, maximum=_MAX_INDEX)


#: Sanity ceiling on any index before the plan is consulted. A plan can hold at
#: most ``MAX_SEGMENTS`` clips and a track at most ``MAX_CUES`` cues; this is
#: comfortably above both and exists so a parser never allocates against a
#: number a model invented.
_MAX_INDEX = 999


# ------------------------------------------------------------------ per-kind parsers
def _parse_remove_segment(entry: dict[str, Any]) -> EditOperation:
    return RemoveSegment(segment=_require_int(entry, "segment", minimum=0, maximum=_MAX_INDEX))


def _parse_reorder_segment(entry: dict[str, Any]) -> EditOperation:
    return ReorderSegment(
        segment=_require_int(entry, "segment", minimum=0, maximum=_MAX_INDEX),
        to_index=_require_int(entry, "to_index", minimum=0, maximum=_MAX_INDEX),
    )


def _parse_trim_segment(entry: dict[str, Any]) -> EditOperation:
    source_in = _optional_int(entry, "source_in_ms", minimum=0, maximum=MAX_OUTPUT_MS)
    source_out = _optional_int(entry, "source_out_ms", minimum=1, maximum=MAX_OUTPUT_MS)
    if source_in is None and source_out is None:
        raise _Rejected("nothing_to_do", "a trim must move source_in_ms or source_out_ms")
    if source_in is not None and source_out is not None and source_out <= source_in:
        raise _Rejected("field_range", "source_out_ms must be after source_in_ms")
    return TrimSegment(
        segment=_require_int(entry, "segment", minimum=0, maximum=_MAX_INDEX),
        source_in_ms=source_in,
        source_out_ms=source_out,
    )


def _parse_change_duration(entry: dict[str, Any]) -> EditOperation:
    segment = _segment_index(entry, required=False)
    # A per-clip duration is bounded by what one segment may be; a whole-edit
    # duration by what one output may be. Reading the bound from which of the
    # two was addressed is what stops "make it 3 minutes" becoming a clip.
    ceiling = MAX_SEGMENT_MS if segment is not None else MAX_OUTPUT_MS
    return ChangeDuration(
        duration_ms=_require_int(entry, "duration_ms", minimum=MIN_SEGMENT_MS, maximum=ceiling),
        segment=segment,
    )


def _parse_change_style_strength(entry: dict[str, Any]) -> EditOperation:
    return ChangeStyleStrength(value=_require_enum(entry, "value", StyleStrength))


def _parse_change_transition(entry: dict[str, Any]) -> EditOperation:
    return ChangeTransition(
        segment=_require_int(entry, "segment", minimum=0, maximum=_MAX_INDEX),
        transition=_require_enum(entry, "transition", TransitionKind),
        duration_ms=_optional_int(entry, "duration_ms", minimum=0, maximum=MAX_TRANSITION_MS),
    )


def _effect_amount(entry: dict[str, Any], kind: EffectKind) -> float:
    """The amount, checked against this kind's own range.

    Rejected rather than clamped, unlike the Phase 5 directive. A planner's
    numbers are guesses about material it cannot see, so clamping them is
    reasonable; a co-edit operation is a statement about a specific clip, and
    quietly turning "slow it to 0.1x" into 0.25x would be the editor claiming to
    have done what it was asked.
    """
    low, high, _ = EFFECT_BOUNDS[kind]
    return _require_float(entry, "amount", minimum=low, maximum=high)


def _parse_add_effect(entry: dict[str, Any]) -> EditOperation:
    kind = _require_enum(entry, "effect", EffectKind)
    start = _optional_int(entry, "start_ms", minimum=0, maximum=MAX_SEGMENT_MS)
    end = _optional_int(entry, "end_ms", minimum=1, maximum=MAX_SEGMENT_MS)
    if (start is not None or end is not None) and kind.spans_whole_segment:
        raise _Rejected(
            "effect_cannot_be_ranged",
            f"{kind.value} applies to a whole clip, not part of one",
        )
    return AddEffect(
        effect=kind,
        amount=_effect_amount(entry, kind),
        segment=_segment_index(entry, required=False),
        start_ms=start,
        end_ms=end,
    )


def _parse_remove_effect(entry: dict[str, Any]) -> EditOperation:
    return RemoveEffect(
        effect=_require_enum(entry, "effect", EffectKind),
        segment=_segment_index(entry, required=False),
    )


def _parse_modify_effect(entry: dict[str, Any]) -> EditOperation:
    kind = _require_enum(entry, "effect", EffectKind)
    return ModifyEffect(
        effect=kind,
        amount=_effect_amount(entry, kind),
        segment=_segment_index(entry, required=False),
    )


def _parse_add_subtitle(entry: dict[str, Any]) -> EditOperation:
    start = _require_int(entry, "start_ms", minimum=0, maximum=MAX_OUTPUT_MS)
    end = _require_int(entry, "end_ms", minimum=1, maximum=MAX_OUTPUT_MS)
    if end <= start:
        raise _Rejected("field_range", "end_ms must be after start_ms")
    if end - start > MAX_CUE_MS:
        raise _Rejected("field_range", f"a cue may not run longer than {MAX_CUE_MS} ms")
    return AddSubtitle(start_ms=start, end_ms=end, text=_require_text(entry, "text"))


def _parse_modify_subtitle(entry: dict[str, Any]) -> EditOperation:
    cue = _optional_int(entry, "cue", minimum=0, maximum=_MAX_INDEX)
    style = _optional_enum(entry, "style", SubtitleStyle)
    position = _optional_enum(entry, "position", SubtitlePosition)

    if cue is None:
        # Addressing the track. There is no cue whose text this could mean, so
        # a text or timing field here is a confused operation, not a partial one.
        if any(entry.get(key) is not None for key in ("text", "start_ms", "end_ms")):
            raise _Rejected(
                "cue_required",
                "a subtitle's text or timing needs a cue index; without one only "
                "style and position may be set",
            )
        if style is None and position is None:
            raise _Rejected("nothing_to_do", "set a style, a position, or name a cue")
        return ModifySubtitle(cue=None, style=style, position=position)

    if style is not None or position is not None:
        # One track, one look. Per-cue styling is not something the stored
        # document can express, and accepting it here would mean silently
        # restyling every other cue as well.
        raise _Rejected(
            "style_is_track_level",
            "style and position apply to the whole track; omit the cue index",
        )

    text = _require_text(entry, "text") if entry.get("text") is not None else None
    start = _optional_int(entry, "start_ms", minimum=0, maximum=MAX_OUTPUT_MS)
    end = _optional_int(entry, "end_ms", minimum=1, maximum=MAX_OUTPUT_MS)
    if text is None and start is None and end is None:
        raise _Rejected("nothing_to_do", "a cue edit must change text or timing")
    if start is not None and end is not None and end <= start:
        raise _Rejected("field_range", "end_ms must be after start_ms")
    return ModifySubtitle(cue=cue, text=text, start_ms=start, end_ms=end)


def _parse_remove_subtitle(entry: dict[str, Any]) -> EditOperation:
    return RemoveSubtitle(cue=_optional_int(entry, "cue", minimum=0, maximum=_MAX_INDEX))


def _parse_change_music_volume(entry: dict[str, Any]) -> EditOperation:
    return ChangeMusicVolume(
        value=_require_float(entry, "value", minimum=MIN_GAIN, maximum=MAX_GAIN)
    )


def _parse_change_music_fade(entry: dict[str, Any]) -> EditOperation:
    fade_in = _optional_int(entry, "fade_in_ms", minimum=0, maximum=MAX_FADE_MS)
    fade_out = _optional_int(entry, "fade_out_ms", minimum=0, maximum=MAX_FADE_MS)
    if fade_in is None and fade_out is None:
        raise _Rejected("nothing_to_do", "set fade_in_ms, fade_out_ms, or both")
    return ChangeMusicFade(fade_in_ms=fade_in, fade_out_ms=fade_out)


def _parse_change_beat_sync(entry: dict[str, Any]) -> EditOperation:
    return ChangeBeatSync(enabled=_require_bool(entry, "enabled"))


def _parse_change_output_preset(entry: dict[str, Any]) -> EditOperation:
    preset = ChangeOutputPreset(
        aspect_ratio=_optional_enum(entry, "aspect_ratio", AspectRatio),
        # Bounded here and checked against the offered presets by the patcher,
        # which is the layer that knows what this deployment offers.
        fps=_optional_int(entry, "fps", minimum=1, maximum=120),
        quality=_optional_enum(entry, "quality", QualityPreset),
        audio=_optional_enum(entry, "audio", AudioMode),
        source_gain=_optional_float(entry, "source_gain", minimum=MIN_GAIN, maximum=MAX_GAIN),
    )
    if all(
        value is None
        for value in (
            preset.aspect_ratio,
            preset.fps,
            preset.quality,
            preset.audio,
            preset.source_gain,
        )
    ):
        raise _Rejected("nothing_to_do", "an output change must set at least one field")
    return preset


_PARSERS: dict[OperationKind, Callable[[dict[str, Any]], EditOperation]] = {
    OperationKind.REMOVE_SEGMENT: _parse_remove_segment,
    OperationKind.REORDER_SEGMENT: _parse_reorder_segment,
    OperationKind.TRIM_SEGMENT: _parse_trim_segment,
    OperationKind.CHANGE_DURATION: _parse_change_duration,
    OperationKind.CHANGE_STYLE_STRENGTH: _parse_change_style_strength,
    OperationKind.CHANGE_TRANSITION: _parse_change_transition,
    OperationKind.ADD_EFFECT: _parse_add_effect,
    OperationKind.REMOVE_EFFECT: _parse_remove_effect,
    OperationKind.MODIFY_EFFECT: _parse_modify_effect,
    OperationKind.ADD_SUBTITLE: _parse_add_subtitle,
    OperationKind.MODIFY_SUBTITLE: _parse_modify_subtitle,
    OperationKind.REMOVE_SUBTITLE: _parse_remove_subtitle,
    OperationKind.CHANGE_MUSIC_VOLUME: _parse_change_music_volume,
    OperationKind.CHANGE_MUSIC_FADE: _parse_change_music_fade,
    OperationKind.CHANGE_BEAT_SYNC: _parse_change_beat_sync,
    OperationKind.CHANGE_OUTPUT_PRESET: _parse_change_output_preset,
}


def parse_operations(raw: Any) -> tuple[tuple[EditOperation, ...], list[DeltaViolation]]:
    """Read an operations array. Returns what parsed and every reason for the rest.

    Strict about the vocabulary, forgiving about presentation -- the same
    balance ``parse_directive`` strikes. An unknown *key* inside an operation is
    ignored, because a model adding ``"confidence": 0.9`` has not done anything
    dangerous. An unknown *kind* is a hard failure, because that is where an
    invented capability would arrive; and a field outside its range is a hard
    failure, because that is where a number nobody meant would.

    Note what happens to an entry like
    ``{"kind": "ADD_EFFECT", "effect": "brightness", "amount": 0.2,
    "filter": "drawtext=...", "path": "/etc/passwd"}``: it yields a brightness
    effect, and the other two keys are not read, not stored and not seen again.
    """
    if not isinstance(raw, list):
        return (), [DeltaViolation("not_an_array", "operations must be an array")]
    if not raw:
        return (), [DeltaViolation("no_operations", "operations must not be empty")]

    operations: list[EditOperation] = []
    violations: list[DeltaViolation] = []

    for index, entry in enumerate(raw[:MAX_OPERATIONS]):
        if not isinstance(entry, dict):
            violations.append(
                DeltaViolation("not_an_object", f"operations[{index}] is not an object", index)
            )
            continue

        name = entry.get("kind") or entry.get("op") or entry.get("operation")
        if not isinstance(name, str):
            violations.append(DeltaViolation("no_kind", f"operations[{index}] has no kind", index))
            continue

        try:
            kind = OperationKind(name.strip().upper())
        except ValueError:
            violations.append(
                DeltaViolation(
                    "unknown_operation",
                    f"operations[{index}].kind {name!r} is not one of: "
                    + ", ".join(member.value for member in OperationKind),
                    index,
                )
            )
            continue

        try:
            operations.append(_PARSERS[kind](entry))
        except _Rejected as rejected:
            violations.append(
                DeltaViolation(
                    rejected.code, f"operations[{index}] ({kind.value}): {rejected.message}", index
                )
            )

    if len(raw) > MAX_OPERATIONS:
        violations.append(
            DeltaViolation(
                "too_many_operations",
                f"{len(raw)} operations exceeds the limit of {MAX_OPERATIONS}",
            )
        )

    return tuple(operations), violations


def parse_delta(raw: Any, *, source: str = "llm") -> EditDelta:
    """Read a whole delta document, or raise with every reason it failed.

    Raises rather than returning a partial delta. A change request that half
    parsed would be applied half-way, and a user who asked for two things and
    silently got one is worse served than one who is told what was not
    understood.
    """
    if not isinstance(raw, dict):
        raise DeltaInvalidError([DeltaViolation("not_an_object", "the delta is not an object")])

    operations, violations = parse_operations(raw.get("operations"))
    if violations:
        raise DeltaInvalidError(violations)
    if not operations:
        raise DeltaInvalidError(
            [DeltaViolation("no_operations", "no operation could be understood")]
        )

    return EditDelta(
        operations=operations,
        rationale=_clean_rationale(raw.get("rationale")),
        source=source,
    )


def _clean_rationale(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_RATIONALE_CHARS]


def vocabulary() -> list[dict[str, Any]]:
    """The operations this server accepts, for the UI and the prompt.

    Generated from the enum rather than retyped, so an operation added here
    appears in the capabilities endpoint and in the model's instructions without
    a second edit -- and one removed stops being offered in both.
    """
    return [{"kind": kind.value} for kind in OperationKind]


__all__ = [
    "MAX_OPERATIONS",
    "MAX_RATIONALE_CHARS",
    "AddEffect",
    "AddSubtitle",
    "ChangeBeatSync",
    "ChangeDuration",
    "ChangeMusicFade",
    "ChangeMusicVolume",
    "ChangeOutputPreset",
    "ChangeStyleStrength",
    "ChangeTransition",
    "DeltaInvalidError",
    "DeltaViolation",
    "EditDelta",
    "EditOperation",
    "ModifyEffect",
    "ModifySubtitle",
    "OperationKind",
    "RemoveEffect",
    "RemoveSegment",
    "RemoveSubtitle",
    "ReorderSegment",
    "TrimSegment",
    "parse_delta",
    "parse_operations",
    "vocabulary",
]
