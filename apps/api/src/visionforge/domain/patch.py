"""Applying a delta to a plan: deterministic, total, and all-or-nothing.

    current EditPlan + validated EditDelta  -->  new EditPlan
                                            -->  or violations, and nothing else

Three properties, and the third is the one the phase turns on.

**Deterministic.** No model runs here and no randomness enters. The same plan
and the same operations produce byte-identical output every time, which is what
makes a version history meaningful and a patch reviewable.

**Addressed against the plan the user saw.** ``segment: 2`` means the third clip
of the plan being edited, not the third clip of whatever the previous operation
left behind. Every slot carries the index it had in the original plan and
operations resolve through that, so "remove clips 3 and 4" removes clips 3 and 4
rather than 3 and then whatever slid into 4's place. A clip an earlier operation
removed is reported as gone, never silently re-pointed at its neighbour.

**Atomic.** Operations are applied to a working copy. If any one of them fails,
or if the finished plan fails the same ``validate_plan`` gate every plan passes,
the result carries violations and the *original plan object is returned
untouched* -- there is no code path that writes a partially patched plan,
because the patched plan does not exist until every operation has succeeded.

Two normalisations run after the operations and before validation, and both are
reported in the diff rather than applied quietly:

- orders are renumbered contiguously from zero, and a first clip left carrying a
  crossfade -- which has nothing to dissolve from -- is refitted;
- a transition that no longer fits the clips it joins is shortened to what they
  can spare, or dropped to a cut when they can spare nothing.

Doing neither would mean rejecting "remove the first clip" because the second
one happened to arrive on a dissolve, which is a correct-but-useless editor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from visionforge.domain.editdelta import (
    AddEffect,
    AddSubtitle,
    ChangeBeatSync,
    ChangeDuration,
    ChangeMusicFade,
    ChangeMusicVolume,
    ChangeOutputPreset,
    ChangeStyleStrength,
    ChangeTransition,
    DeltaViolation,
    EditDelta,
    EditOperation,
    ModifyEffect,
    ModifySubtitle,
    RemoveEffect,
    RemoveSegment,
    RemoveSubtitle,
    ReorderSegment,
    TrimSegment,
)
from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_SEGMENT_MS,
    MIN_TRANSITION_MS,
    AspectRatio,
    EditPlan,
    MediaFact,
    MusicCue,
    OutputSpec,
    PlanViolation,
    Segment,
    TransitionKind,
    validate_plan,
)
from visionforge.domain.effects import MAX_EFFECTS_PER_SEGMENT, Effect, EffectKind
from visionforge.domain.ids import MediaId
from visionforge.domain.subtitles import (
    MIN_CUE_GAP_MS,
    MIN_CUE_MS,
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
)

#: Bumped when patching semantics change in a way that could produce a different
#: plan from the same delta. Stored on every patched plan, for the same reason
#: the prompt version is stored on every planned one.
PATCH_VERSION = "1"

#: The planner name a patched plan carries. Not the name of the planner that
#: produced the plan it was patched from -- that one is kept in metadata -- so a
#: listing can tell a co-edited plan from a generated one at a glance.
PATCH_PLANNER = "co-editor"

#: The transition length chosen when an operation asks for one without saying
#: how long. Four frames at 50 fps: visible as a dissolve, short enough that it
#: reads as an edit rather than as an effect.
DEFAULT_TRANSITION_MS = MIN_TRANSITION_MS * 4


@dataclass(frozen=True, slots=True)
class PatchContext:
    """What the patcher needs that the delta could not carry.

    All of it is server-side fact: how long each source really is, what geometry
    a shape maps to, which frame rates this deployment offers, and the media
    facts the plan validator checks against. None of it is client-supplied,
    which is what keeps a delta from being able to assert a source is longer
    than it is.
    """

    #: Real source durations by media id, so a trim can never run past the end
    #: of the file it trims.
    source_durations: dict[MediaId, int]
    #: The server's shape-to-pixels table. A delta names a shape; only this
    #: decides what that is in pixels.
    dimensions: dict[AspectRatio, tuple[int, int]]
    #: Frame rates a caller may ask for.
    fps_presets: tuple[int, ...]
    #: What the plan validator is allowed to know about this project's media.
    media_facts: dict[MediaId, MediaFact]


# ------------------------------------------------------------------- the diff
@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One visible difference between the current plan and the proposed one.

    ``before``/``after`` are already-formatted display strings rather than raw
    numbers. The formatting rule ("5.0s", "40%", "Crossfade") belongs with the
    comparison that produced it -- a client reformatting a bare float would have
    to re-derive which unit each field is in, and would get one of them wrong.
    """

    #: What changed, as a stable key the UI may translate: ``total_duration``,
    #: ``segment_duration``, ``music_gain``, ``subtitle_style``, ...
    field: str
    label: str
    before: str
    after: str
    #: The clip this is about, 1-based, when it is about one.
    clip: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "label": self.label,
            "before": self.before,
            "after": self.after,
            "clip": self.clip,
        }


@dataclass(frozen=True, slots=True)
class PlanDiff:
    """Everything that changed, plus what the operations said they would do.

    Two lists because they answer different questions. ``applied`` is the
    request read back -- "Music volume to 40%" -- and comes from the operations.
    ``entries`` is what actually became different, computed by comparing the two
    plans, and it is the one that catches an operation which turned out to be a
    no-op or which moved something the user did not mention.
    """

    entries: tuple[DiffEntry, ...] = ()
    applied: tuple[str, ...] = ()
    #: Adjustments the patcher made on its own to keep the plan renderable -- a
    #: dissolve shortened, a cue clipped to a shorter programme. Surfaced rather
    #: than silent: the user asked for one thing and got that thing plus these.
    adjustments: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def as_payload(self) -> dict[str, Any]:
        return {
            "entries": [entry.as_payload() for entry in self.entries],
            "applied": list(self.applied),
            "adjustments": list(self.adjustments),
        }


@dataclass(frozen=True, slots=True)
class PatchOutcome:
    """The result of applying a delta. Either a new plan, or every reason not.

    ``plan`` is the *patched* plan when ``ok``; when not ok it is the original,
    unchanged object, so a caller that ignores ``ok`` still cannot accidentally
    persist a half-applied edit.
    """

    ok: bool
    plan: EditPlan
    diff: PlanDiff = field(default_factory=PlanDiff)
    violations: tuple[DeltaViolation, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "diff": self.diff.as_payload(),
            "violations": [violation.as_payload() for violation in self.violations],
        }


# ------------------------------------------------------------- working copy
@dataclass
class _Slot:
    """One clip, with the address it had in the plan the user was shown."""

    origin: int
    segment: Segment


@dataclass
class _Cue:
    origin: int | None
    cue: SubtitleCue


@dataclass
class _Working:
    """A mutable plan under construction. Never observed outside this module."""

    slots: list[_Slot]
    output: OutputSpec
    music: MusicCue | None
    cues: list[_Cue]
    subtitle_style: SubtitleStyle
    subtitle_position: SubtitlePosition
    has_subtitles: bool
    metadata: dict[str, Any]
    violations: list[DeltaViolation] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)

    def reject(self, index: int, code: str, message: str) -> None:
        self.violations.append(DeltaViolation(code, message, index))

    def slot_for(self, index: int, address: int) -> _Slot | None:
        """The clip an operation addressed, or a violation explaining why not."""
        for slot in self.slots:
            if slot.origin == address:
                return slot
        if any(slot.origin > address for slot in self.slots) or not self.slots:
            self.reject(
                index,
                "segment_removed",
                f"clip {address + 1} was removed by an earlier operation in this change",
            )
        else:
            self.reject(index, "no_such_segment", f"this edit has no clip {address + 1}")
        return None

    def targets(self, index: int, address: int | None) -> list[_Slot]:
        """The clips an operation applies to. ``None`` means every clip."""
        if address is None:
            return list(self.slots)
        slot = self.slot_for(index, address)
        return [slot] if slot is not None else []


def apply_delta(plan: EditPlan, delta: EditDelta, context: PatchContext) -> PatchOutcome:
    """Apply every operation, or none of them.

    Returns rather than raises, because "this change cannot be made" is an
    ordinary answer the co-editor shows the user, not an exceptional condition.
    The caller checks ``ok``; the plan it gets back when ``ok`` is false is the
    one it passed in.
    """
    working = _start(plan)

    for index, operation in enumerate(delta.operations):
        _apply(working, index, operation, plan, context)

    if working.violations:
        return PatchOutcome(ok=False, plan=plan, violations=tuple(working.violations))

    patched = _finish(working, plan, delta, context)
    if working.violations:
        return PatchOutcome(ok=False, plan=plan, violations=tuple(working.violations))

    plan_violations = validate_plan(patched, context.media_facts)
    if plan_violations:
        return PatchOutcome(
            ok=False,
            plan=plan,
            violations=tuple(_as_delta_violations(plan_violations)),
        )

    return PatchOutcome(
        ok=True,
        plan=patched,
        diff=_diff(plan, patched, working, delta),
    )


def _as_delta_violations(violations: list[PlanViolation]) -> list[DeltaViolation]:
    """Plan violations, reported in the co-editor's own vocabulary.

    The codes are kept verbatim: a rejected patch and a rejected hand-cut
    timeline fail for the same reasons and should be diagnosable the same way.
    """
    return [
        DeltaViolation(
            violation.code,
            (
                f"clip {violation.segment_order + 1}: {violation.message}"
                if violation.segment_order is not None
                else violation.message
            ),
        )
        for violation in violations
    ]


def _start(plan: EditPlan) -> _Working:
    segments = plan.ordered_segments
    track = plan.subtitles
    return _Working(
        slots=[_Slot(origin=index, segment=segment) for index, segment in enumerate(segments)],
        output=plan.output,
        music=plan.music,
        cues=(
            [_Cue(origin=index, cue=cue) for index, cue in enumerate(track.ordered)]
            if track
            else []
        ),
        subtitle_style=track.style if track else SubtitleStyle.CLEAN,
        subtitle_position=track.position if track else SubtitlePosition.BOTTOM,
        has_subtitles=track is not None,
        metadata=dict(plan.metadata),
    )


# --------------------------------------------------------------- the operations
def _apply(
    working: _Working,
    index: int,
    operation: EditOperation,
    plan: EditPlan,
    context: PatchContext,
) -> None:
    """Dispatch one operation.

    An if-chain over the union rather than a table of handlers: mypy narrows
    each branch to the concrete dataclass, so a field that does not exist on an
    operation is a type error here rather than an AttributeError at runtime.
    """
    if isinstance(operation, RemoveSegment):
        _remove_segment(working, index, operation)
    elif isinstance(operation, ReorderSegment):
        _reorder_segment(working, index, operation)
    elif isinstance(operation, TrimSegment):
        _trim_segment(working, index, operation, context)
    elif isinstance(operation, ChangeDuration):
        _change_duration(working, index, operation, context)
    elif isinstance(operation, ChangeStyleStrength):
        working.metadata["style_strength"] = operation.value.value
    elif isinstance(operation, ChangeTransition):
        _change_transition(working, index, operation)
    elif isinstance(operation, AddEffect):
        _add_effect(working, index, operation)
    elif isinstance(operation, RemoveEffect):
        _remove_effect(working, index, operation)
    elif isinstance(operation, ModifyEffect):
        _modify_effect(working, index, operation)
    elif isinstance(operation, AddSubtitle):
        _add_subtitle(working, index, operation)
    elif isinstance(operation, ModifySubtitle):
        _modify_subtitle(working, index, operation)
    elif isinstance(operation, RemoveSubtitle):
        _remove_subtitle(working, index, operation)
    elif isinstance(operation, ChangeMusicVolume):
        _change_music_volume(working, index, operation)
    elif isinstance(operation, ChangeMusicFade):
        _change_music_fade(working, index, operation)
    elif isinstance(operation, ChangeBeatSync):
        working.metadata["beat_sync"] = operation.enabled
    elif isinstance(operation, ChangeOutputPreset):
        _change_output(working, index, operation, plan, context)


def _remove_segment(working: _Working, index: int, operation: RemoveSegment) -> None:
    slot = working.slot_for(index, operation.segment)
    if slot is None:
        return
    if len(working.slots) == 1:
        working.reject(
            index,
            "last_segment",
            "an edit needs at least one clip; removing this one would empty it",
        )
        return
    working.slots.remove(slot)


def _reorder_segment(working: _Working, index: int, operation: ReorderSegment) -> None:
    slot = working.slot_for(index, operation.segment)
    if slot is None:
        return
    if not 0 <= operation.to_index < len(working.slots):
        working.reject(
            index,
            "no_such_position",
            f"position {operation.to_index + 1} is outside this edit's "
            f"{len(working.slots)} clip(s)",
        )
        return
    working.slots.remove(slot)
    working.slots.insert(operation.to_index, slot)


def _source_length(working: _Working, slot: _Slot, context: PatchContext) -> int | None:
    """How long the clip's source really is, as the server measured it.

    ``None`` when the media is not in the facts at all -- which the plan
    validator will reject anyway, but saying so here produces a message about
    the clip the user named rather than about a media id they never saw.
    """
    return context.source_durations.get(slot.segment.media_id)


def _trim_segment(
    working: _Working, index: int, operation: TrimSegment, context: PatchContext
) -> None:
    slot = working.slot_for(index, operation.segment)
    if slot is None:
        return

    segment = slot.segment
    source_in = (
        operation.source_in_ms if operation.source_in_ms is not None else segment.source_in_ms
    )
    source_out = (
        operation.source_out_ms if operation.source_out_ms is not None else segment.source_out_ms
    )

    if source_out <= source_in:
        working.reject(
            index,
            "inverted_trim",
            f"clip {operation.segment + 1}: the end would land at or before the start",
        )
        return

    length = source_out - source_in
    if not MIN_SEGMENT_MS <= length <= MAX_SEGMENT_MS:
        working.reject(
            index,
            "trim_length",
            f"clip {operation.segment + 1}: {length} ms is outside "
            f"{MIN_SEGMENT_MS}-{MAX_SEGMENT_MS} ms",
        )
        return

    available = _source_length(working, slot, context)
    if available is not None and source_out > available:
        working.reject(
            index,
            "trim_past_end",
            f"clip {operation.segment + 1}: the source is only {available} ms long",
        )
        return

    slot.segment = replace(segment, source_in_ms=source_in, source_out_ms=source_out)


def _hold_for(
    working: _Working,
    index: int,
    slot: _Slot,
    wanted_ms: int,
    context: PatchContext,
    *,
    label: str,
) -> bool:
    """Make one clip hold for ``wanted_ms`` of source, moving its out point.

    The in point moves only when the tail cannot give enough -- a clip trimmed
    from 5 s to 7 s of an 8 s source can be extended to 4 s by starting at 4 s,
    and refusing to do that would mean refusing a request the material supports.
    """
    segment = slot.segment
    available = _source_length(working, slot, context)
    limit = available if available is not None else segment.source_out_ms

    if wanted_ms > limit:
        working.reject(
            index,
            "source_too_short",
            f"{label}: its source is {limit} ms, shorter than the {wanted_ms} ms asked for",
        )
        return False

    source_in = segment.source_in_ms
    if source_in + wanted_ms > limit:
        source_in = max(0, limit - wanted_ms)
    slot.segment = replace(segment, source_in_ms=source_in, source_out_ms=source_in + wanted_ms)
    return True


def _change_duration(
    working: _Working, index: int, operation: ChangeDuration, context: PatchContext
) -> None:
    if operation.segment is not None:
        slot = working.slot_for(index, operation.segment)
        if slot is not None:
            _hold_for(
                working,
                index,
                slot,
                operation.duration_ms,
                context,
                label=f"clip {operation.segment + 1}",
            )
        return

    # The whole programme. Scaled rather than truncated: "keep this edit but
    # make it twenty seconds" means the same shots, tighter -- not the same
    # shots with the last two missing.
    current = sum(slot.segment.duration_ms for slot in working.slots)
    if current <= 0:
        working.reject(index, "empty_edit", "this edit has no clips to shorten")
        return

    factor = operation.duration_ms / current
    for slot in working.slots:
        wanted = max(MIN_SEGMENT_MS, min(MAX_SEGMENT_MS, round(slot.segment.duration_ms * factor)))
        available = _source_length(working, slot, context)
        if available is not None:
            wanted = min(wanted, available)
        if wanted < MIN_SEGMENT_MS:
            continue
        _hold_for(working, index, slot, wanted, context, label="the edit")


def _transition_ceiling(working: _Working, slot: _Slot) -> int:
    """The longest transition this clip may carry, given its neighbours.

    The same rule the plan validator enforces, computed here so an operation can
    be refused with a useful message instead of producing a plan that fails the
    gate with a number the user never chose.
    """
    position = working.slots.index(slot)
    own = slot.segment.output_duration_ms
    previous = working.slots[position - 1].segment.output_duration_ms if position > 0 else own
    return min(MAX_TRANSITION_MS, int(min(own, previous) * MAX_TRANSITION_SHARE))


def _change_transition(working: _Working, index: int, operation: ChangeTransition) -> None:
    slot = working.slot_for(index, operation.segment)
    if slot is None:
        return

    kind = operation.transition
    if kind is TransitionKind.CUT:
        slot.segment = replace(slot.segment, transition_in=kind, transition_ms=0)
        return

    if kind.needs_previous and working.slots.index(slot) == 0:
        working.reject(
            index,
            "no_previous_clip",
            f"clip {operation.segment + 1} is first; a {kind.value} needs a clip to come from",
        )
        return

    ceiling = _transition_ceiling(working, slot)
    if ceiling < MIN_TRANSITION_MS:
        working.reject(
            index,
            "clips_too_short",
            f"clip {operation.segment + 1}: the clips are too short to carry a {kind.value}",
        )
        return

    wanted = operation.duration_ms if operation.duration_ms is not None else DEFAULT_TRANSITION_MS
    if operation.duration_ms is not None and not MIN_TRANSITION_MS <= wanted <= ceiling:
        working.reject(
            index,
            "transition_duration",
            f"clip {operation.segment + 1}: a {kind.value} here must be between "
            f"{MIN_TRANSITION_MS} and {ceiling} ms",
        )
        return

    slot.segment = replace(
        slot.segment,
        transition_in=kind,
        transition_ms=max(MIN_TRANSITION_MS, min(wanted, ceiling)),
    )


def _without(effects: tuple[Effect, ...], kind: EffectKind) -> tuple[Effect, ...]:
    return tuple(effect for effect in effects if effect.kind is not kind)


def _add_effect(working: _Working, index: int, operation: AddEffect) -> None:
    targets = working.targets(index, operation.segment)
    if not targets:
        return

    for slot in targets:
        effects = _without(slot.segment.effects, operation.effect)
        if operation.effect.changes_duration:
            # One rate per clip. Two would multiply into a third nobody asked
            # for, and the plan validator refuses the pair regardless.
            effects = tuple(effect for effect in effects if not effect.kind.changes_duration)

        if len(effects) >= MAX_EFFECTS_PER_SEGMENT:
            working.reject(
                index,
                "too_many_effects",
                f"clip {slot.origin + 1} already carries {MAX_EFFECTS_PER_SEGMENT} effects",
            )
            return

        if operation.end_ms is not None and operation.end_ms > slot.segment.duration_ms:
            working.reject(
                index,
                "effect_window_past_end",
                f"clip {slot.origin + 1} is {slot.segment.duration_ms} ms long, "
                f"shorter than the {operation.end_ms} ms window asked for",
            )
            return

        slot.segment = replace(
            slot.segment,
            effects=(
                *effects,
                Effect(
                    kind=operation.effect,
                    amount=operation.amount,
                    start_ms=operation.start_ms,
                    end_ms=operation.end_ms,
                ),
            ),
        )


def _remove_effect(working: _Working, index: int, operation: RemoveEffect) -> None:
    targets = working.targets(index, operation.segment)
    if not targets:
        return

    touched = 0
    for slot in targets:
        if any(effect.kind is operation.effect for effect in slot.segment.effects):
            touched += 1
            slot.segment = replace(
                slot.segment, effects=_without(slot.segment.effects, operation.effect)
            )

    if touched == 0:
        where = "this edit" if operation.segment is None else f"clip {operation.segment + 1}"
        working.reject(
            index,
            "effect_not_present",
            f"{where} has no {operation.effect.value.replace('_', ' ')} to remove",
        )


def _modify_effect(working: _Working, index: int, operation: ModifyEffect) -> None:
    targets = working.targets(index, operation.segment)
    if not targets:
        return

    touched = 0
    for slot in targets:
        existing = next(
            (effect for effect in slot.segment.effects if effect.kind is operation.effect), None
        )
        if existing is None:
            continue
        touched += 1
        slot.segment = replace(
            slot.segment,
            effects=(
                *_without(slot.segment.effects, operation.effect),
                replace(existing, amount=operation.amount),
            ),
        )

    if touched == 0:
        where = "this edit" if operation.segment is None else f"clip {operation.segment + 1}"
        name = operation.effect.value.replace("_", " ")
        working.reject(
            index,
            "effect_not_present",
            f"{where} has no {name} to adjust; add one instead",
        )


def _cue_fits(working: _Working, start_ms: int, end_ms: int, *, ignoring: _Cue | None) -> bool:
    """Whether a cue can sit at these times without touching another.

    Overlap is refused rather than layered, exactly as the plan validator
    refuses it: two cues on screen at once is a feature the single-track
    document cannot express, and drawing them on top of each other is not it.
    """
    for existing in working.cues:
        if existing is ignoring:
            continue
        if start_ms < existing.cue.end_ms + MIN_CUE_GAP_MS and existing.cue.start_ms < end_ms:
            return False
    return True


def _add_subtitle(working: _Working, index: int, operation: AddSubtitle) -> None:
    if not _cue_fits(working, operation.start_ms, operation.end_ms, ignoring=None):
        working.reject(
            index,
            "cue_overlap",
            f"a subtitle already runs across {operation.start_ms}-{operation.end_ms} ms",
        )
        return

    working.has_subtitles = True
    working.cues.append(
        _Cue(
            origin=None,
            cue=SubtitleCue(
                start_ms=operation.start_ms, end_ms=operation.end_ms, text=operation.text
            ),
        )
    )


def _cue_for(working: _Working, index: int, address: int) -> _Cue | None:
    for cue in working.cues:
        if cue.origin == address:
            return cue
    working.reject(index, "no_such_cue", f"this edit has no subtitle {address + 1}")
    return None


def _modify_subtitle(working: _Working, index: int, operation: ModifySubtitle) -> None:
    if operation.is_track_level:
        if not working.has_subtitles:
            working.reject(
                index,
                "no_subtitles",
                "this edit has no subtitles to restyle; add some first",
            )
            return
        if operation.style is not None:
            working.subtitle_style = operation.style
        if operation.position is not None:
            working.subtitle_position = operation.position
        return

    assert operation.cue is not None  # is_track_level is exactly `cue is None`
    entry = _cue_for(working, index, operation.cue)
    if entry is None:
        return

    start = operation.start_ms if operation.start_ms is not None else entry.cue.start_ms
    end = operation.end_ms if operation.end_ms is not None else entry.cue.end_ms
    if end - start < MIN_CUE_MS:
        working.reject(
            index,
            "cue_too_short",
            f"subtitle {operation.cue + 1} would be {end - start} ms; "
            f"the minimum is {MIN_CUE_MS} ms",
        )
        return
    if not _cue_fits(working, start, end, ignoring=entry):
        working.reject(
            index,
            "cue_overlap",
            f"subtitle {operation.cue + 1} would overlap another",
        )
        return

    entry.cue = SubtitleCue(
        start_ms=start,
        end_ms=end,
        text=operation.text if operation.text is not None else entry.cue.text,
    )


def _remove_subtitle(working: _Working, index: int, operation: RemoveSubtitle) -> None:
    if not working.has_subtitles:
        working.reject(index, "no_subtitles", "this edit has no subtitles to remove")
        return

    if operation.cue is None:
        working.cues.clear()
        working.has_subtitles = False
        return

    entry = _cue_for(working, index, operation.cue)
    if entry is None:
        return
    working.cues.remove(entry)
    if not working.cues:
        # An empty track is not a track. Dropping it entirely is what makes
        # "remove the last cue" send no subtitles rather than an empty document.
        working.has_subtitles = False


def _change_music_volume(working: _Working, index: int, operation: ChangeMusicVolume) -> None:
    if working.music is None:
        working.reject(index, "no_music", "this edit has no music to adjust")
        return
    working.music = replace(working.music, gain=operation.value)


def _change_music_fade(working: _Working, index: int, operation: ChangeMusicFade) -> None:
    cue = working.music
    if cue is None:
        working.reject(index, "no_music", "this edit has no music to fade")
        return

    fade_in = operation.fade_in_ms if operation.fade_in_ms is not None else cue.fade_in_ms
    fade_out = operation.fade_out_ms if operation.fade_out_ms is not None else cue.fade_out_ms
    if fade_in + fade_out > cue.duration_ms:
        working.reject(
            index,
            "fades_overlap",
            f"the fades total {fade_in + fade_out} ms, longer than the "
            f"{cue.duration_ms} ms the music plays for",
        )
        return
    working.music = replace(cue, fade_in_ms=fade_in, fade_out_ms=fade_out)


def _change_output(
    working: _Working,
    index: int,
    operation: ChangeOutputPreset,
    plan: EditPlan,
    context: PatchContext,
) -> None:
    output = working.output

    if operation.fps is not None and operation.fps not in context.fps_presets:
        offered = ", ".join(str(value) for value in context.fps_presets)
        working.reject(index, "fps_not_offered", f"this server renders at {offered} fps")
        return

    width, height = output.width, output.height
    if operation.aspect_ratio is not None:
        preset = context.dimensions.get(operation.aspect_ratio)
        if preset is None:
            working.reject(
                index,
                "aspect_not_offered",
                f"this server does not render {operation.aspect_ratio.value}",
            )
            return
        width, height = preset

    working.output = OutputSpec(
        aspect_ratio=operation.aspect_ratio or output.aspect_ratio,
        width=width,
        height=height,
        fps=operation.fps or output.fps,
        fit=output.fit,
        audio=operation.audio or output.audio,
        quality=operation.quality or output.quality,
        source_gain=(
            operation.source_gain if operation.source_gain is not None else output.source_gain
        ),
    )


# ------------------------------------------------------------- normalisation
def _finish(working: _Working, plan: EditPlan, delta: EditDelta, context: PatchContext) -> EditPlan:
    """Renumber, refit and assemble. The only place a patched plan is built."""
    _refit_transitions(working)

    segments = tuple(
        replace(slot.segment, order=position) for position, slot in enumerate(working.slots)
    )

    candidate = EditPlan(
        project_id=plan.project_id,
        segments=segments,
        output=working.output,
        music=working.music,
        subtitles=None,
        planner=PATCH_PLANNER,
        planner_version=PATCH_VERSION,
        metadata=working.metadata,
    )

    subtitles = _fit_cues(working, candidate.total_duration_ms)
    return replace(
        candidate,
        subtitles=subtitles,
        metadata={
            **working.metadata,
            "patched_from_planner": plan.planner,
            "patch_version": PATCH_VERSION,
            "co_edit": {
                "source": delta.source,
                "rationale": delta.rationale,
                "operations": [operation.as_payload() for operation in delta.operations],
            },
        },
    )


def _refit_transitions(working: _Working) -> None:
    """Make every transition fit the clips it now joins.

    Runs after the operations because the operations are what changed the
    lengths. A dissolve that no longer fits is shortened to what the clips can
    spare, and one whose clips can spare nothing becomes a cut -- both recorded
    as adjustments, because the user did not ask for either and should be able
    to see that it happened.
    """
    for position, slot in enumerate(working.slots):
        segment = slot.segment
        kind = segment.transition_in

        if kind is TransitionKind.CUT:
            if segment.transition_ms:
                slot.segment = replace(segment, transition_ms=0)
            continue

        if position == 0 and kind.needs_previous:
            slot.segment = replace(
                segment, transition_in=TransitionKind.FADE_IN, transition_ms=segment.transition_ms
            )
            working.adjustments.append(
                f"Clip {position + 1} is now first, so its dissolve became a fade in"
            )
            segment = slot.segment

        ceiling = _transition_ceiling(working, slot)
        if ceiling < MIN_TRANSITION_MS:
            slot.segment = replace(segment, transition_in=TransitionKind.CUT, transition_ms=0)
            working.adjustments.append(
                f"Clip {position + 1} is now too short for a transition, so it cuts"
            )
            continue

        if segment.transition_ms > ceiling:
            slot.segment = replace(segment, transition_ms=ceiling)
            working.adjustments.append(
                f"Clip {position + 1}'s transition was shortened to {ceiling} ms to fit"
            )
        elif segment.transition_ms < MIN_TRANSITION_MS:
            slot.segment = replace(segment, transition_ms=MIN_TRANSITION_MS)


def _fit_cues(working: _Working, total_ms: int) -> SubtitleTrack | None:
    """Keep the cues inside a programme whose length may have changed.

    A cue past the end of the edit is one the validator refuses and one nobody
    would ever see. Rather than reject the whole change -- which would mean
    "you cannot shorten a subtitled edit" -- the tail is clipped and any cue
    with no room left is dropped, and both are reported as adjustments.
    """
    if not working.has_subtitles or not working.cues:
        return None

    kept: list[SubtitleCue] = []
    dropped = 0
    clipped = 0

    for entry in sorted(working.cues, key=lambda item: (item.cue.start_ms, item.cue.end_ms)):
        cue = entry.cue
        if cue.start_ms + MIN_CUE_MS > total_ms:
            dropped += 1
            continue
        if cue.end_ms > total_ms:
            cue = SubtitleCue(start_ms=cue.start_ms, end_ms=total_ms, text=cue.text)
            clipped += 1
        kept.append(cue)

    if dropped:
        working.adjustments.append(
            f"{dropped} subtitle(s) fell outside the shorter edit and were removed"
        )
    if clipped:
        working.adjustments.append(f"{clipped} subtitle(s) were trimmed to the new length")

    if not kept:
        return None
    return SubtitleTrack(
        cues=tuple(kept), style=working.subtitle_style, position=working.subtitle_position
    )


# --------------------------------------------------------------------- the diff
def _seconds(ms: int) -> str:
    return f"{ms / 1000:.1f}s"


def _percent(gain: float) -> str:
    return f"{round(gain * 100)}%"


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def _effects_label(effects: tuple[Effect, ...]) -> str:
    if not effects:
        return "None"
    return ", ".join(
        f"{_title(effect.kind.value)} {effect.amount:g}"
        for effect in sorted(effects, key=lambda item: item.kind.value)
    )


def _diff(before: EditPlan, after: EditPlan, working: _Working, delta: EditDelta) -> PlanDiff:
    """Compare the two plans and describe every visible difference.

    Computed from the plans rather than from the operations, so an operation
    that turned out to change nothing shows up as nothing -- which is the point
    of showing a diff before committing rather than a list of intentions.
    """
    entries: list[DiffEntry] = []
    originals = before.ordered_segments
    patched = after.ordered_segments

    if before.total_duration_ms != after.total_duration_ms:
        entries.append(
            DiffEntry(
                "total_duration",
                "Total length",
                _seconds(before.total_duration_ms),
                _seconds(after.total_duration_ms),
            )
        )
    if len(originals) != len(patched):
        entries.append(DiffEntry("segment_count", "Clips", str(len(originals)), str(len(patched))))

    # Slots carry the address each clip had in the original plan, so a clip that
    # moved is compared with itself rather than with whoever took its place.
    for position, slot in enumerate(working.slots):
        original = originals[slot.origin]
        current = patched[position]
        label = f"Clip {slot.origin + 1:02d}"

        if position != slot.origin:
            entries.append(
                DiffEntry(
                    "segment_position",
                    f"{label} position",
                    str(slot.origin + 1),
                    str(position + 1),
                    clip=slot.origin + 1,
                )
            )
        if original.duration_ms != current.duration_ms:
            entries.append(
                DiffEntry(
                    "segment_duration",
                    label,
                    _seconds(original.duration_ms),
                    _seconds(current.duration_ms),
                    clip=slot.origin + 1,
                )
            )
        if (original.transition_in, original.transition_ms) != (
            current.transition_in,
            current.transition_ms,
        ):
            entries.append(
                DiffEntry(
                    "segment_transition",
                    f"{label} transition",
                    _transition_label(original.transition_in, original.transition_ms),
                    _transition_label(current.transition_in, current.transition_ms),
                    clip=slot.origin + 1,
                )
            )
        if original.effects != current.effects:
            entries.append(
                DiffEntry(
                    "segment_effects",
                    f"{label} effects",
                    _effects_label(original.effects),
                    _effects_label(current.effects),
                    clip=slot.origin + 1,
                )
            )

    entries.extend(_music_diff(before.music, after.music))
    entries.extend(_subtitle_diff(before.subtitles, after.subtitles))
    entries.extend(_output_diff(before.output, after.output))
    entries.extend(_metadata_diff(before.metadata, after.metadata))

    return PlanDiff(
        entries=tuple(entries),
        applied=delta.summary,
        adjustments=tuple(dict.fromkeys(working.adjustments)),
    )


def _transition_label(kind: TransitionKind, ms: int) -> str:
    return _title(kind.value) if kind is TransitionKind.CUT else f"{_title(kind.value)} {ms} ms"


def _music_diff(before: MusicCue | None, after: MusicCue | None) -> list[DiffEntry]:
    if before is None and after is None:
        return []
    if before is None or after is None:
        return [
            DiffEntry(
                "music",
                "Music",
                "None" if before is None else _percent(before.gain),
                "None" if after is None else _percent(after.gain),
            )
        ]

    entries: list[DiffEntry] = []
    if before.gain != after.gain:
        entries.append(
            DiffEntry("music_gain", "Music", _percent(before.gain), _percent(after.gain))
        )
    if (before.fade_in_ms, before.fade_out_ms) != (after.fade_in_ms, after.fade_out_ms):
        entries.append(
            DiffEntry(
                "music_fade",
                "Music fades",
                f"{before.fade_in_ms} / {before.fade_out_ms} ms",
                f"{after.fade_in_ms} / {after.fade_out_ms} ms",
            )
        )
    return entries


def _subtitle_diff(before: SubtitleTrack | None, after: SubtitleTrack | None) -> list[DiffEntry]:
    if before is None and after is None:
        return []

    entries: list[DiffEntry] = []
    before_style = _title(before.style.value) if before else "None"
    after_style = _title(after.style.value) if after else "None"
    if before_style != after_style:
        entries.append(DiffEntry("subtitle_style", "Subtitles", before_style, after_style))

    before_position = before.position.value if before else None
    after_position = after.position.value if after else None
    if before_position != after_position:
        entries.append(
            DiffEntry(
                "subtitle_position",
                "Subtitle position",
                _title(before_position) if before_position else "None",
                _title(after_position) if after_position else "None",
            )
        )

    before_count = len(before.cues) if before else 0
    after_count = len(after.cues) if after else 0
    if before_count != after_count:
        entries.append(
            DiffEntry("subtitle_count", "Subtitle cues", str(before_count), str(after_count))
        )
    elif before and after and before.ordered != after.ordered:
        entries.append(
            DiffEntry("subtitle_cues", "Subtitle text", f"{before_count} cues", "edited")
        )
    return entries


def _output_diff(before: OutputSpec, after: OutputSpec) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    if before.aspect_ratio is not after.aspect_ratio:
        entries.append(
            DiffEntry(
                "output_aspect",
                "Shape",
                f"{before.aspect_ratio.value} · {before.width}x{before.height}",
                f"{after.aspect_ratio.value} · {after.width}x{after.height}",
            )
        )
    if before.fps != after.fps:
        entries.append(
            DiffEntry("output_fps", "Frame rate", f"{before.fps} fps", f"{after.fps} fps")
        )
    if before.quality is not after.quality:
        entries.append(
            DiffEntry(
                "output_quality",
                "Quality",
                _title(before.quality.value),
                _title(after.quality.value),
            )
        )
    if before.audio is not after.audio:
        entries.append(
            DiffEntry(
                "output_audio", "Clip audio", _title(before.audio.value), _title(after.audio.value)
            )
        )
    if before.source_gain != after.source_gain:
        entries.append(
            DiffEntry(
                "output_source_gain",
                "Clip volume",
                _percent(before.source_gain),
                _percent(after.source_gain),
            )
        )
    return entries


def _metadata_diff(before: dict[str, Any], after: dict[str, Any]) -> list[DiffEntry]:
    """The two settings a patch records for the *next* plan rather than this one.

    Labelled so the UI can say what they are. A user who asks for less reference
    influence and is shown "Reference style 100% → 50%" with no further change
    to the cuts has been told the truth about what happened.
    """
    entries: list[DiffEntry] = []

    if before.get("style_strength") != after.get("style_strength"):
        entries.append(
            DiffEntry(
                "style_strength",
                "Reference style (next plan)",
                f"{before.get('style_strength', '0')}%",
                f"{after.get('style_strength', '0')}%",
            )
        )
    if bool(before.get("beat_sync")) != bool(after.get("beat_sync")):
        entries.append(
            DiffEntry(
                "beat_sync",
                "Beat sync (next plan)",
                "On" if before.get("beat_sync") else "Off",
                "On" if after.get("beat_sync") else "Off",
            )
        )
    return entries


__all__ = [
    "DEFAULT_TRANSITION_MS",
    "PATCH_PLANNER",
    "PATCH_VERSION",
    "DiffEntry",
    "PatchContext",
    "PatchOutcome",
    "PlanDiff",
    "apply_delta",
]
