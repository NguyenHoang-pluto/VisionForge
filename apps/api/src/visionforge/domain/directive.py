"""The EditDirective: the only thing a model is allowed to say.

An ``EditPlan`` is already a closed vocabulary, but it is not the right shape to
ask a model for. It carries UUIDs, and a model that emits UUIDs is a model that
can emit the *wrong* UUID -- one from another project, one that was deleted, one
it made up. So the model does not produce a plan. It produces a directive:

    EditDirective
      ├── style      one of a fixed enum
      ├── pacing     one of a fixed enum
      ├── clips      [ {ref: "c3", duration_ms: 1800}, ... ]   handles, not ids
      └── rationale  a short string, for the user, never interpreted

and deterministic code turns that into a plan:

    directive --resolve handles--> media ids
              --clamp durations--> style bounds, then plan bounds
              --renumber--------> contiguous orders from zero
              --snap geometry---> server-side preset map
              = EditPlan --> the existing Phase 4 validator

Three narrowings before the gate the rules engine already passes through. What
survives is a plan that is valid by construction, from a model that had no way
to name a file, a path, a command, or a pixel dimension.

``rationale`` is the single free-text field, and it is treated as display text:
bounded in length, stored, shown to the user, never parsed, never matched
against, never used to make a decision. It exists because an automatic edit that
cannot say why it cut the way it did is not reviewable, and reviewability is the
whole argument for this architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visionforge.domain.editbrief import EditBrief
from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MIN_OUTPUT_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    OutputSpec,
    Segment,
    TransitionKind,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.style import EditStyle, QualityPreset, StyleProfile

#: Longest rationale kept. Enough for two sentences of explanation; short enough
#: that the field cannot become a channel for bulk text.
MAX_RATIONALE_CHARS = 400


class Pacing(StrEnum):
    """How quickly the edit cuts. A coarse control, deliberately.

    The model picks a pacing; the style profile decides what that means in
    milliseconds. Asking a model for "how many ms per clip" invites confident
    numbers with no basis; asking it to choose among three words it understands,
    and mapping those words to bounds we chose, keeps the judgement where it
    belongs on each side.
    """

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"

    def clip_ms(self, profile: StyleProfile) -> int:
        """Where this pacing sits inside the style's bounds."""
        if self is Pacing.SLOW:
            return profile.max_clip_ms
        if self is Pacing.FAST:
            return profile.min_clip_ms
        return profile.target_clip_ms


@dataclass(frozen=True, slots=True)
class DirectiveClip:
    """One clip the model wants, by handle."""

    ref: str
    #: What the model asked for. Advisory: it is clamped to the style bounds and
    #: then to the plan bounds, and it cannot exceed the source's real length.
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class EditDirective:
    """A model's complete answer. Closed vocabulary throughout."""

    clips: tuple[DirectiveClip, ...]
    style: EditStyle
    pacing: Pacing = Pacing.MEDIUM
    rationale: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "style": self.style.value,
            "pacing": self.pacing.value,
            "clips": [{"ref": clip.ref, "duration_ms": clip.duration_ms} for clip in self.clips],
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DirectiveViolation:
    """One reason a model's output was rejected.

    Machine-readable, and fed back to the model verbatim on the repair attempt:
    a model told "clips[2].ref 'c99' is not in the brief" fixes that specific
    error far more reliably than one told "invalid output".
    """

    code: str
    message: str

    def as_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


class DirectiveInvalidError(ValueError):
    """The model's output could not be read as a directive."""

    def __init__(self, violations: list[DirectiveViolation]) -> None:
        self.violations = violations
        super().__init__("; ".join(v.message for v in violations) or "unparseable directive")


# ------------------------------------------------------------------ extraction
def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first complete JSON object out of a completion.

    Models wrap JSON in prose and fences however they like, and the instruction
    not to is followed most but not all of the time. Rather than fail a
    perfectly good directive over a `````json`` fence, scan for the
    first balanced ``{...}`` and parse that.

    Balanced-brace scanning rather than a regex because JSON nests, and
    string-aware because a brace inside the rationale must not end the object.
    Anything that does not parse returns ``None``; nothing here executes or
    evaluates the text.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_directive(
    text: str, brief: EditBrief, *, requested_style: EditStyle | None = None
) -> EditDirective:
    """Read a completion into a directive, or raise with every reason it failed.

    Strict about structure, forgiving about presentation. An unknown key is
    ignored rather than rejected -- a model that adds ``"confidence": 0.8`` has
    not done anything dangerous, and failing on it would make the feature
    brittle for no gain. An unknown *value* in a typed field is a hard failure,
    because that is where a made-up handle or an invented style would arrive.
    """
    violations: list[DirectiveViolation] = []

    payload = extract_json_object(text)
    if payload is None:
        raise DirectiveInvalidError(
            [DirectiveViolation("not_json", "the response contained no JSON object")]
        )

    # --- style ---
    style = requested_style or EditStyle.CUSTOM
    raw_style = payload.get("style")
    if isinstance(raw_style, str):
        try:
            style = EditStyle(raw_style.strip().lower())
        except ValueError:
            violations.append(
                DirectiveViolation(
                    "unknown_style",
                    f"style {raw_style!r} is not one of: "
                    + ", ".join(member.value for member in EditStyle),
                )
            )

    # --- pacing ---
    pacing = Pacing.MEDIUM
    raw_pacing = payload.get("pacing")
    if isinstance(raw_pacing, str):
        try:
            pacing = Pacing(raw_pacing.strip().lower())
        except ValueError:
            violations.append(
                DirectiveViolation(
                    "unknown_pacing",
                    f"pacing {raw_pacing!r} is not one of: slow, medium, fast",
                )
            )

    # --- clips ---
    raw_clips = payload.get("clips")
    clips: list[DirectiveClip] = []
    seen: set[str] = set()
    if not isinstance(raw_clips, list) or not raw_clips:
        violations.append(DirectiveViolation("no_clips", "clips must be a non-empty array"))
    else:
        for index, entry in enumerate(raw_clips[:MAX_SEGMENTS]):
            clip = _parse_clip(entry, index, brief, seen, violations)
            if clip is not None:
                clips.append(clip)
                seen.add(clip.ref)

    if violations:
        raise DirectiveInvalidError(violations)
    if not clips:
        raise DirectiveInvalidError(
            [DirectiveViolation("no_usable_clips", "no clip reference could be resolved")]
        )

    return EditDirective(
        clips=tuple(clips),
        style=style,
        pacing=pacing,
        rationale=_clean_rationale(payload.get("rationale")),
    )


def _parse_clip(
    entry: Any,
    index: int,
    brief: EditBrief,
    seen: set[str],
    violations: list[DirectiveViolation],
) -> DirectiveClip | None:
    # A bare string is accepted as a handle with no duration: models produce
    # ["c1", "c2"] often enough that rejecting it would be pedantry.
    if isinstance(entry, str):
        entry = {"ref": entry}
    if not isinstance(entry, dict):
        violations.append(DirectiveViolation("clip_not_object", f"clips[{index}] is not an object"))
        return None

    ref = entry.get("ref") or entry.get("id") or entry.get("clip")
    if not isinstance(ref, str):
        violations.append(DirectiveViolation("clip_no_ref", f"clips[{index}] has no ref"))
        return None
    ref = ref.strip()

    # The containment check. A handle the brief did not issue does not resolve,
    # and there is no fallback that guesses what the model meant.
    if brief.media_id_for(ref) is None:
        violations.append(
            DirectiveViolation(
                "unknown_ref",
                f"clips[{index}].ref {ref!r} is not one of the clips in the brief",
            )
        )
        return None
    if ref in seen:
        # Not an error worth failing over -- a repeated clip is a legitimate
        # edit -- but Phase 5 has no notion of reusing a source, so the repeat
        # is dropped and the edit proceeds.
        return None

    # ``bool`` is a subclass of ``int``, so a model answering
    # ``"duration_ms": true`` would otherwise become a 1 ms clip.
    duration = entry.get("duration_ms")
    if (
        duration is not None
        and not isinstance(duration, bool)
        and isinstance(duration, int | float)
    ):
        return DirectiveClip(ref=ref, duration_ms=int(duration))
    return DirectiveClip(ref=ref, duration_ms=None)


def _clean_rationale(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_RATIONALE_CHARS]


# ----------------------------------------------------------------- compilation
@dataclass(frozen=True, slots=True)
class CompileContext:
    """Everything the compiler needs that the model was never asked for.

    Geometry, codec, quality and project identity all come from the server. The
    model influences *which clips, in what order, for how long* -- and nothing
    else about the output file.
    """

    project_id: ProjectId
    aspect_ratio: AspectRatio
    width: int
    height: int
    fps: int
    fit: FitMode
    audio: AudioMode
    quality: QualityPreset
    profile: StyleProfile
    #: The handle table from the brief. The *only* way a directive's ``ref``
    #: becomes a media id, and the reason a model cannot name media it was not
    #: offered: an unknown handle resolves to nothing.
    handles: dict[str, MediaId]
    #: Real source durations by media id, so a requested trim can never run past
    #: the end of the file it trims.
    source_durations: dict[MediaId, int]
    target_duration_ms: int
    max_clips: int
    planner: str
    planner_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compile_directive(directive: EditDirective, context: CompileContext) -> EditPlan:
    """Turn a directive into a plan. Deterministic, total, and clamping.

    "Total" is the important word: this never raises on an in-range directive
    and never trusts a number. Every duration is clamped three times -- to the
    style's pacing bounds, to the plan's segment bounds, and to the source's own
    length -- and the total is clamped to the plan's output bounds. A model
    asking for a nine-hour edit of a four-second clip produces a short plan, not
    an error and not a nine-hour render.
    """
    profile = context.profile
    default_ms = directive.pacing.clip_ms(profile)

    segments: list[Segment] = []
    for clip in directive.clips[: context.max_clips]:
        media_id = _resolved(clip, context)
        if media_id is None:
            continue

        source_ms = context.source_durations.get(media_id, 0)
        if source_ms < MIN_SEGMENT_MS:
            continue

        wanted = clip.duration_ms if clip.duration_ms is not None else default_ms
        take = profile.clamp_clip_ms(wanted)
        take = max(MIN_SEGMENT_MS, min(MAX_SEGMENT_MS, take))
        take = min(take, source_ms)
        if take < MIN_SEGMENT_MS:
            continue

        # Centre trim, exactly as the rules engine does it and for the same
        # reason: the head of a handheld clip is where the focus hunt lives.
        start = max(0, (source_ms - take) // 2)
        segments.append(
            Segment(
                media_id=media_id,
                order=len(segments),
                source_in_ms=start,
                source_out_ms=start + take,
                transition_in=TransitionKind.CUT,
            )
        )

    segments = _fit_total_duration(segments)

    return EditPlan(
        project_id=context.project_id,
        segments=tuple(segments),
        output=OutputSpec(
            aspect_ratio=context.aspect_ratio,
            width=context.width,
            height=context.height,
            fps=context.fps,
            fit=context.fit,
            audio=context.audio,
            quality=context.quality,
        ),
        planner=context.planner,
        planner_version=context.planner_version,
        metadata={
            **context.metadata,
            "style": directive.style.value,
            "pacing": directive.pacing.value,
            "rationale": directive.rationale,
            "requested_clips": len(directive.clips),
            "compiled_segments": len(segments),
            "target_duration_ms": context.target_duration_ms,
        },
    )


def _resolved(clip: DirectiveClip, context: CompileContext) -> MediaId | None:
    """Handle to media id, through the brief's table and nothing else.

    Checked again here even though ``parse_directive`` already rejected unknown
    handles, because the two run against separately-supplied tables and this one
    is the last word before a media id enters a plan. A handle with no entry is
    dropped, never guessed at.
    """
    return context.handles.get(clip.ref)


def _fit_total_duration(segments: list[Segment]) -> list[Segment]:
    """Trim the tail until the plan fits the maximum output length.

    Dropping whole clips from the end rather than shortening every clip: a plan
    that silently rescales every segment to fit a limit produces an edit nobody
    asked for, while a plan with two fewer clips is the same edit, shorter.
    """
    if not segments:
        return segments

    kept: list[Segment] = []
    total = 0
    for segment in segments:
        if total + segment.duration_ms > MAX_OUTPUT_MS:
            break
        kept.append(segment)
        total += segment.duration_ms

    if not kept:
        # A single segment longer than the whole output budget: keep it, capped.
        first = segments[0]
        take = min(first.duration_ms, MAX_OUTPUT_MS)
        kept = [
            Segment(
                media_id=first.media_id,
                order=0,
                source_in_ms=first.source_in_ms,
                source_out_ms=first.source_in_ms + take,
                transition_in=first.transition_in,
            )
        ]
        total = take

    if total < MIN_OUTPUT_MS:
        # Too short to be a legal plan. The caller (the planner) treats an empty
        # result as "the model produced nothing usable" and falls back.
        return []

    return [
        Segment(
            media_id=segment.media_id,
            order=index,
            source_in_ms=segment.source_in_ms,
            source_out_ms=segment.source_out_ms,
            transition_in=segment.transition_in,
        )
        for index, segment in enumerate(kept)
    ]


__all__ = [
    "MAX_RATIONALE_CHARS",
    "CompileContext",
    "DirectiveClip",
    "DirectiveInvalidError",
    "DirectiveViolation",
    "EditDirective",
    "Pacing",
    "compile_directive",
    "extract_json_object",
    "parse_directive",
]
