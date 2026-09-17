"""The editorial decision engine.

    readings + policy + pacing + intent
        -> creative selection   which clips, and why not the others
        -> story assignment     what each one is for
        -> trim windows         which part of each clip, and why that part
        -> treatment            speed, transitions, effects, beat placement
        = EditorialPlan         structured decisions, with reasons

**This module produces planning data and nothing else.** It has no idea what
FFmpeg is, cannot name a filter, and never sees a path or a storage key. What it
emits is a list of typed decisions over media ids and integers, which
``domain.editorial_planner`` then compiles into the existing ``EditPlan`` -- the
same boundary Phase 5 drew between a model and the renderer, applied here to the
deterministic engine so that the two paths are structurally identical.

Two properties are the point of the whole phase.

**Selection is not ranking.** Taking the top *n* by score is what produced the
slideshow this phase exists to replace: five excellent shots of the same wall
score five times and become five sixths of the edit. Here a clip competes for a
*role*, against the clips already chosen, on six weighted axes of which quality
is one -- so five near-identical clips yield one or two, and a weak clip that is
the only record of the moment that matters can still be the peak.

**Every decision carries its reasons.** Not prose, and not a model's reasoning:
a tuple of short codes drawn from a closed enum, which the UI renders as a list,
tests assert on, and a person can argue with. An automatic edit that cannot say
why it cut the way it did is not reviewable, and reviewability is the entire
argument for doing this deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

from visionforge.domain.editorial import (
    SEMANTIC_RELATED_SIMILARITY,
    ClipReading,
    DetectedEvent,
    EditorialEvent,
    SignalBoard,
    cosine,
)
from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_SEGMENT_MS,
    MIN_TRANSITION_MS,
    TransitionKind,
)
from visionforge.domain.effects import EFFECT_BOUNDS, Effect, EffectKind
from visionforge.domain.ids import MediaId
from visionforge.domain.pacing import PacingPlan, PacingSlot
from visionforge.domain.story import EditorialPolicy, StoryRole

#: Bumped when the engine's behaviour changes in a way that could alter an edit.
DECISION_ENGINE_VERSION = "1"


class DecisionKind(StrEnum):
    """What the engine decided to do. A closed vocabulary.

    Eleven members, and every one of them is *planning data*: a decision to
    slow a clip is a rate and a segment, not an ``atempo`` filter. The compiler
    downstream knows how to render each; this module knows only that it wants
    them.
    """

    KEEP = "KEEP"
    REJECT = "REJECT"
    TRIM = "TRIM"
    REORDER = "REORDER"
    EMPHASIZE = "EMPHASIZE"
    SLOW = "SLOW"
    SPEED_UP = "SPEED_UP"
    PLACE_ON_BEAT = "PLACE_ON_BEAT"
    TRANSITION = "TRANSITION"
    ADD_EFFECT = "ADD_EFFECT"
    ADD_SUBTITLE = "ADD_SUBTITLE"


class ReasonCode(StrEnum):
    """Why a decision was made. Short, structured, closed.

    Codes rather than sentences, for three consumers a sentence would serve
    badly: the UI renders them as a list in the user's own language, tests
    assert on them exactly, and the co-editor can address them. They are also
    deliberately *not* a model's explanation -- no chain of thought reaches this
    enum, because no model produces these decisions.
    """

    # --- why it was kept ---
    HIGH_QUALITY = "high_quality"
    HIGH_MOTION = "high_motion"
    PEAK_MOMENT = "peak_moment"
    UNIQUE_CONTENT = "unique_content"
    ROLE_FIT = "role_fit"
    ENERGY_FIT = "energy_fit"
    STYLE_MATCH = "style_match"
    SEMANTIC_RELEVANCE = "semantic_relevance"
    FACES_PRESENT = "faces_present"
    ONLY_CANDIDATE = "only_candidate"

    # --- why it was not ---
    WEAK_QUALITY = "weak_quality"
    DUPLICATE_CONTENT = "duplicate_content"
    TOO_SIMILAR = "too_similar"
    NOT_NEEDED = "not_needed"
    TOO_SHORT = "too_short"
    WEAK_ROLE_FIT = "weak_role_fit"

    # --- why it looks the way it does ---
    PACING_SHORTENS = "pacing_shortens"
    PACING_LENGTHENS = "pacing_lengthens"
    POLICY_EMPHASIS = "policy_emphasis"
    BEAT_ALIGNED = "beat_aligned"
    OFF_BEAT = "off_beat"
    AVOIDS_INTERNAL_CUT = "avoids_internal_cut"
    CENTRED_ON_ACTION = "centred_on_action"
    CENTRE_TRIM = "centre_trim"
    SOURCE_TOO_SHORT = "source_too_short"
    CHRONOLOGY = "chronology"
    NARRATIVE_ORDER = "narrative_order"
    STYLE_POLICY = "style_policy"
    LIKELY_SPEECH = "likely_speech"


@dataclass(frozen=True, slots=True)
class EditorialDecision:
    """One decision, with everything needed to act on it and to explain it.

    Every field is an integer, a float, a media id or a member of an enum this
    codebase already validates. There is no options dictionary and no string
    field, which is the same structural guarantee ``Effect`` and ``MusicCue``
    give: there is nowhere here to put a filter expression, so one cannot
    arrive.
    """

    kind: DecisionKind
    media_id: MediaId | None = None
    #: Position in the finished edit, when the decision is about one.
    slot: int | None = None
    role: StoryRole | None = None

    # --- TRIM ---
    source_in_ms: int | None = None
    source_out_ms: int | None = None

    # --- SLOW / SPEED_UP / ADD_EFFECT ---
    effect: EffectKind | None = None
    amount: float | None = None

    # --- TRANSITION ---
    transition: TransitionKind | None = None
    transition_ms: int | None = None

    # --- PLACE_ON_BEAT ---
    beats: int | None = None
    start_ms: int | None = None

    # --- REORDER ---
    from_index: int | None = None
    to_index: int | None = None

    reasons: tuple[ReasonCode, ...] = ()
    #: How strongly the evidence supports this decision, 0..1. A trim centred on
    #: a measured motion peak is more confident than one centred by default, and
    #: the UI is entitled to say so.
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("decision confidence must be between 0 and 1")

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "reasons": [reason.value for reason in self.reasons],
            "confidence": round(self.confidence, 3),
        }
        if self.media_id is not None:
            payload["media_id"] = str(self.media_id)
        for name, value in (
            ("slot", self.slot),
            ("source_in_ms", self.source_in_ms),
            ("source_out_ms", self.source_out_ms),
            ("amount", self.amount),
            ("transition_ms", self.transition_ms),
            ("beats", self.beats),
            ("start_ms", self.start_ms),
            ("from_index", self.from_index),
            ("to_index", self.to_index),
        ):
            if value is not None:
                payload[name] = value
        if self.role is not None:
            payload["role"] = self.role.value
        if self.effect is not None:
            payload["effect"] = self.effect.value
        if self.transition is not None:
            payload["transition"] = self.transition.value
        return payload


@dataclass(frozen=True, slots=True)
class EditorialSegment:
    """One position in the finished edit, fully decided and fully explained."""

    slot: int
    role: StoryRole
    media_id: MediaId
    source_in_ms: int
    source_out_ms: int
    #: How long this occupies the timeline once its speed effect is applied.
    output_ms: int
    #: The clip's own measured energy, 0..1.
    energy: float
    #: The energy the pacing curve wanted here.
    target_energy: float
    events: tuple[DetectedEvent, ...]
    reasons: tuple[ReasonCode, ...]
    #: The creative score that won this slot, and its components.
    score: float
    components: dict[str, float] = field(default_factory=dict)
    beats: int | None = None
    transition: TransitionKind = TransitionKind.CUT
    transition_ms: int = 0
    effects: tuple[Effect, ...] = ()
    #: Where this clip sat in the upload order, 1-based, so the UI can say the
    #: edit reordered it.
    sequence: int = 0

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms

    @property
    def on_beat(self) -> bool:
        return self.beats is not None

    def as_payload(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "role": self.role.value,
            "media_id": str(self.media_id),
            "sequence": self.sequence,
            "source_in_ms": self.source_in_ms,
            "source_out_ms": self.source_out_ms,
            "duration_ms": self.duration_ms,
            "output_ms": self.output_ms,
            "energy": round(self.energy, 4),
            "target_energy": round(self.target_energy, 4),
            "events": [event.as_payload() for event in self.events],
            "reasons": [reason.value for reason in self.reasons],
            "score": round(self.score, 4),
            "components": {
                name: round(value, 4) for name, value in sorted(self.components.items())
            },
            "beats": self.beats,
            "on_beat": self.on_beat,
            "transition": self.transition.value,
            "transition_ms": self.transition_ms,
            "effects": [effect.as_payload() for effect in self.effects],
        }


@dataclass(frozen=True, slots=True)
class EditorialRejection:
    """A clip the engine did not use, and the reason it did not."""

    media_id: MediaId
    reasons: tuple[ReasonCode, ...]
    #: The clip it duplicates, when that is why. A media id, because the caller
    #: resolves ids to names; this module never sees a filename.
    similar_to: MediaId | None = None
    score: float | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "media_id": str(self.media_id),
            "reasons": [reason.value for reason in self.reasons],
        }
        if self.similar_to is not None:
            payload["similar_to"] = str(self.similar_to)
        if self.score is not None:
            payload["score"] = round(self.score, 4)
        return payload


@dataclass(frozen=True, slots=True)
class EditorialPlan:
    """The complete editorial decision, before it becomes an ``EditPlan``."""

    version: str
    policy_id: str
    arc: str
    segments: tuple[EditorialSegment, ...]
    decisions: tuple[EditorialDecision, ...]
    rejected: tuple[EditorialRejection, ...]
    pacing: PacingPlan
    #: Which variant produced this, when one did. ``None`` is the plain plan.
    variant: str | None = None
    #: What a model contributed, if anything. Numbers and enum names only.
    intent: dict[str, Any] | None = None

    @property
    def total_output_ms(self) -> int:
        """How long the programme runs, accounting for speed and overlap."""
        if not self.segments:
            return 0
        total = sum(segment.output_ms for segment in self.segments)
        total -= sum(
            segment.transition_ms
            for segment in self.segments[1:]
            if segment.transition.consumes_time
        )
        return total

    @property
    def roles(self) -> tuple[StoryRole, ...]:
        return tuple(segment.role for segment in self.segments)

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy": self.policy_id,
            "arc": self.arc,
            "variant": self.variant,
            "intent": self.intent,
            "segments": [segment.as_payload() for segment in self.segments],
            "decisions": [decision.as_payload() for decision in self.decisions],
            "rejected": [rejection.as_payload() for rejection in self.rejected],
            "pacing": self.pacing.as_payload(),
            "total_output_ms": self.total_output_ms,
        }


# ------------------------------------------------------------------ the engine
#: How far below the leader a candidate may score and still be considered for a
#: slot on diversity grounds. Below this the gap is not "a close call decided by
#: uniqueness", it is a worse clip.
CONTENDER_MARGIN = 0.25

#: The rate a slowed peak plays at. 0.5 is the strongest slow motion this
#: system does without frame interpolation; 0.6 keeps the motion readable on
#: 30 fps source, which most phone footage is.
PEAK_SLOW_RATE = 0.6

#: How much of a clip a push-in moves, when a policy asks for one.
GENTLE_ZOOM = 0.08


@dataclass(frozen=True, slots=True)
class EngineInput:
    """Everything the engine decides from. Assembled by the planner.

    A single frozen object rather than fourteen keyword arguments, because the
    engine is called three times per variant request and the call site should be
    readable. Nothing here is mutable and nothing here is a handle to the
    outside world -- it is readings, a policy and a rhythm.
    """

    readings: tuple[ClipReading, ...]
    board: SignalBoard
    policy: EditorialPolicy
    pacing: PacingPlan
    roles: tuple[StoryRole, ...]
    variant: str | None = None
    intent: dict[str, Any] | None = None
    #: Whether to let the policy place dissolves and effects. Off for a caller
    #: that wants structure alone -- the acceptance scripts use it to prove the
    #: structure is doing the work rather than the decoration.
    treatments: bool = True


def decide(inputs: EngineInput) -> EditorialPlan:
    """Run the whole engine. Deterministic, total, and explainable.

    Order of operations, and every step depends on the one before it:

    1. **Select**, slot by slot in narrative order, each slot competing on the
       policy's six weighted axes against the clips already taken.
    2. **Trim**, choosing *where* in each source to cut from based on what the
       role wants and what the clip's own signals say is in it.
    3. **Treat**: speed on the peak, transitions at the seams, an effect where
       the policy asks for one.
    4. **Explain**, throughout: every step appends its reasons rather than
       reconstructing them afterwards, because a reason reconstructed from an
       outcome is a rationalisation.
    """
    chosen, rejected, decisions = _select(inputs)

    segments: list[EditorialSegment] = []
    for index, (reading, role, pick) in enumerate(chosen):
        slot = inputs.pacing.slots[index] if index < len(inputs.pacing.slots) else None
        segment, trim_decisions = _cut(
            reading=reading,
            role=role,
            pick=pick,
            slot=slot,
            index=index,
            policy=inputs.policy,
            treatments=inputs.treatments,
        )
        segments.append(segment)
        decisions.extend(trim_decisions)

    if inputs.treatments:
        segments, seam_decisions = _seams(segments, inputs.policy)
        decisions.extend(seam_decisions)

    decisions.extend(_ordering_decisions(segments))

    return EditorialPlan(
        version=DECISION_ENGINE_VERSION,
        policy_id=inputs.policy.id.value,
        arc=inputs.policy.arc.name,
        segments=tuple(segments),
        decisions=tuple(decisions),
        rejected=tuple(rejected),
        pacing=inputs.pacing,
        variant=inputs.variant,
        intent=inputs.intent,
    )


# ------------------------------------------------------------------ selection
@dataclass(frozen=True, slots=True)
class _Pick:
    """A scored candidacy for one slot."""

    score: float
    components: dict[str, float]
    reasons: tuple[ReasonCode, ...]


def _select(
    inputs: EngineInput,
) -> tuple[
    list[tuple[ClipReading, StoryRole, _Pick]],
    list[EditorialRejection],
    list[EditorialDecision],
]:
    """Fill each role with the clip that best argues for it.

    **In priority order, not narrative order.** The peak picks first, then the
    ending, then the hook, and the setup picks last from what is left. That is
    how the arc's ``priority`` field earns its place, and it is the difference
    between an edit whose peak is the strongest moment in the project and one
    whose peak is whatever the setup did not want -- filling left to right gives
    the least important role first refusal on the best footage.

    Greedy within that order, and greedy is right for a specific reason: the
    diversity term is defined *against what has already been chosen*, so the
    problem is genuinely sequential. A global optimisation over all assignments
    would be exponential, would not be more correct, and -- worse -- would not be
    explainable, because no single clip's presence would have a reason that did
    not depend on every other clip's.

    The diversity gate is applied before scoring, not inside it. A clip 95%
    semantically identical to one already in the edit is not "a slightly worse
    choice", it is a repeat, and letting quality outvote that is precisely the
    failure this phase exists to fix.
    """
    policy = inputs.policy
    roles = inputs.roles
    available = list(inputs.readings)
    filled: list[tuple[ClipReading, StoryRole, _Pick] | None] = [None] * len(roles)
    taken: list[MediaId] = []
    rejected: list[EditorialRejection] = []
    suppressed: dict[MediaId, MediaId] = {}

    relevance = _relevance_map(inputs.board)

    # Ties broken by narrative position, so two equal-priority roles fill left
    # to right and the order two runs produce is identical.
    order = sorted(range(len(roles)), key=lambda i: (-policy.priority_for(roles[i]), i))

    for position in order:
        if not available:
            break
        role = roles[position]

        scored: list[tuple[ClipReading, _Pick, float]] = []
        for reading in available:
            distinct = inputs.board.distinctiveness(reading.media_id, tuple(taken))
            pick = _score(reading, role, policy, distinct, relevance.get(reading.media_id))
            scored.append((reading, pick, distinct))

        scored.sort(key=lambda item: (-item[1].score, item[0].signals.sequence))

        leader = scored[0][1].score
        winner: tuple[ClipReading, _Pick, float] | None = None
        for reading, pick, distinct in scored:
            if distinct >= policy.min_diversity:
                winner = (reading, pick, distinct)
                break
            # Too similar to something already in the edit. Recorded against the
            # nearest already-chosen clip, so the UI can say which.
            nearest = _nearest(inputs.board, reading.media_id, tuple(taken))
            if nearest is not None:
                suppressed[reading.media_id] = nearest

        if winner is None:
            # Every remaining clip repeats something already chosen. Taking the
            # best of them anyway would fill the arc with duplicates; leaving the
            # slot empty is the honest outcome and the arc simply ends shorter.
            continue

        reading, pick, _distinct = winner
        if pick.score < leader - CONTENDER_MARGIN:
            # The leader was gated out and the survivor is much weaker. Still
            # taken -- a role wants filling -- but the reason says so.
            pick = _Pick(pick.score, pick.components, (*pick.reasons, ReasonCode.NOT_NEEDED))

        filled[position] = (reading, role, pick)
        taken.append(reading.media_id)
        available.remove(reading)

    chosen = [entry for entry in filled if entry is not None]
    if policy.prefer_chronological:
        chosen = _chronological_within_roles(chosen)

    decisions: list[EditorialDecision] = [
        EditorialDecision(
            kind=DecisionKind.KEEP,
            media_id=reading.media_id,
            slot=index,
            role=role,
            reasons=pick.reasons,
            confidence=round(min(1.0, 0.5 + pick.score / 2), 3),
        )
        for index, (reading, role, pick) in enumerate(chosen)
    ]

    # --- everything not used ---
    #
    # A clip's nearest chosen neighbour is computed here even when the gate
    # never fired on it. The gate only runs while a slot is looking for a
    # winner, so whether a duplicate was *encountered* depends on evaluation
    # order -- and "this clip is not in the edit because it repeats that one" is
    # true regardless of when anybody noticed.
    for reading in available:
        similar_to = suppressed.get(reading.media_id) or _nearest(
            inputs.board, reading.media_id, tuple(taken)
        )
        reasons: list[ReasonCode] = []
        if similar_to is not None:
            reasons.append(ReasonCode.TOO_SIMILAR)
        quality = reading.signals.quality
        if quality is not None and quality < 0.25:
            reasons.append(ReasonCode.WEAK_QUALITY)
        if not reasons:
            reasons.append(ReasonCode.NOT_NEEDED)
        rejected.append(
            EditorialRejection(
                media_id=reading.media_id,
                reasons=tuple(reasons),
                similar_to=similar_to,
                score=quality,
            )
        )
        decisions.append(
            EditorialDecision(
                kind=DecisionKind.REJECT,
                media_id=reading.media_id,
                reasons=tuple(reasons),
            )
        )

    return chosen, rejected, decisions


def _chronological_within_roles(
    chosen: list[tuple[ClipReading, StoryRole, _Pick]],
) -> list[tuple[ClipReading, StoryRole, _Pick]]:
    """Put each role's clips back into the order they were shot in.

    Within a role, not across the whole edit, and the distinction is the point.
    A match highlight that reorders the match is not a highlight of the match --
    so the three build shots run in the order they happened. But the arc still
    decides that the build comes before the peak, because a *chronological* edit
    that opens on forty seconds of empty pitch is a recording, not a highlight.

    The roles themselves are untouched: a clip keeps the slot it won, it just
    may swap position with another clip holding the same role.
    """
    result: list[tuple[ClipReading, StoryRole, _Pick]] = []
    index = 0
    while index < len(chosen):
        end = index
        while end < len(chosen) and chosen[end][1] is chosen[index][1]:
            end += 1
        run = chosen[index:end]
        run.sort(key=lambda item: item[0].signals.sequence)
        result.extend(run)
        index = end
    return result


def _score(
    reading: ClipReading,
    role: StoryRole,
    policy: EditorialPolicy,
    distinctiveness: float,
    relevance: float | None,
) -> _Pick:
    """A clip's case for one role, on six weighted axes.

    Every component is 0..1 and the weights sum to one, so the result is 0..1
    and two clips are comparable -- the same discipline ``score_candidate``
    keeps, extended to the four axes a technical ranker cannot see.

    A component that cannot be measured keeps its share rather than forfeiting
    it, for the reason Phase 8 established: silently ranking unanalysed footage
    last means an analyzer that has not finished changes the edit.
    """
    weights = policy.weights
    signals = reading.signals
    events = {detected.event: detected.confidence for detected in reading.events}

    quality = signals.quality if signals.quality is not None else 0.5
    role_fit = policy.fit_for(role, events)
    wanted = policy.energy_for(role)
    energy_fit = 1.0 - abs(signals.energy - wanted)
    style = signals.style_match if signals.style_match is not None else 1.0
    relevant = relevance if relevance is not None else 1.0

    components = {
        "quality": round(quality, 4),
        "energy_fit": round(max(0.0, energy_fit), 4),
        "diversity": round(distinctiveness, 4),
        "role_fit": round(role_fit, 4),
        "style_match": round(style, 4),
        "relevance": round(relevant, 4),
    }
    score = (
        components["quality"] * weights.quality
        + components["energy_fit"] * weights.energy_fit
        + components["diversity"] * weights.diversity
        + components["role_fit"] * weights.role_fit
        + components["style_match"] * weights.style_match
        + components["relevance"] * weights.relevance
    )

    return _Pick(
        score=round(score, 6),
        components=components,
        reasons=_reasons_for(reading, role, components, events),
    )


def _reasons_for(
    reading: ClipReading,
    role: StoryRole,
    components: dict[str, float],
    events: dict[EditorialEvent, float],
) -> tuple[ReasonCode, ...]:
    """The short, structured account of why this clip won this slot.

    Thresholds rather than "the top three components", and the difference
    matters: a clip whose best component is 0.4 has no strong reason, and
    reporting its least-bad axis as a reason would be inventing one. A clip with
    nothing above the bar gets the one reason that is always true -- it was the
    best available -- which is honest and occasionally the whole story.
    """
    reasons: list[ReasonCode] = []

    if components["quality"] >= 0.55:
        reasons.append(ReasonCode.HIGH_QUALITY)
    if components["diversity"] >= 0.6:
        reasons.append(ReasonCode.UNIQUE_CONTENT)
    if components["role_fit"] >= 0.5:
        reasons.append(ReasonCode.ROLE_FIT)
    if components["energy_fit"] >= 0.75:
        reasons.append(ReasonCode.ENERGY_FIT)
    if components["relevance"] >= 0.7:
        reasons.append(ReasonCode.SEMANTIC_RELEVANCE)
    if reading.signals.style_match is not None and reading.signals.style_match >= 0.7:
        reasons.append(ReasonCode.STYLE_MATCH)

    if events.get(EditorialEvent.PEAK_MOTION, 0.0) > 0:
        reasons.append(ReasonCode.PEAK_MOMENT)
    elif (reading.signals.motion or 0.0) >= 0.55:
        reasons.append(ReasonCode.HIGH_MOTION)
    if (reading.signals.face_presence or 0.0) >= 0.5 and role in (
        StoryRole.REACTION,
        StoryRole.PEAK,
        StoryRole.HOOK,
    ):
        reasons.append(ReasonCode.FACES_PRESENT)
    if events.get(EditorialEvent.DIALOGUE, 0.0) > 0:
        reasons.append(ReasonCode.LIKELY_SPEECH)

    if not reasons:
        reasons.append(ReasonCode.ONLY_CANDIDATE)
    # Deduplicated, order preserved: a reason repeated is not a stronger reason.
    return tuple(dict.fromkeys(reasons))


def _relevance_map(board: SignalBoard) -> dict[MediaId, float]:
    """How close each clip is to what this project is mostly about, 0..1.

    The cosine to the *centroid* of every embedding in the project. It answers a
    question no per-clip signal can: a project of twelve football clips and one
    shot of a car park contains an outlier, and the outlier is almost certainly
    not what the user wants in their highlight.

    It is a mild term by design -- five percent of the score in most policies --
    because "unlike the rest of the project" is also what a genuinely
    interesting cutaway looks like, and a strong relevance weight would delete
    exactly the shots that make an edit worth watching.

    Empty when nothing is embedded: absent evidence contributes nothing rather
    than contributing a default.
    """
    vectors = [signal.embedding for signal in board.signals if signal.embedding]
    if len(vectors) < 2:
        return {}

    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return {}
    centroid = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(width))

    relevance: dict[MediaId, float] = {}
    for signal in board.signals:
        score = cosine(signal.embedding, centroid)
        if score is not None:
            # Cosine over CLIP vectors of real footage lives in roughly 0.5-1.0;
            # rescaling that band onto 0..1 is what makes the term able to
            # discriminate at all rather than reporting 0.8 for everything.
            relevance[signal.media_id] = round(max(0.0, min(1.0, (score - 0.5) / 0.5)), 4)
    return relevance


def _nearest(board: SignalBoard, media_id: MediaId, against: tuple[MediaId, ...]) -> MediaId | None:
    """The chosen clip this one most resembles, if it resembles any of them.

    ``None`` below the *related* threshold rather than the duplicate one: a
    clip gated out on diversity is by definition close to something, and the
    looser bound is what lets the rejection name it. Above nothing at all, the
    answer is that there is no such clip, which the caller reports as a plain
    "not needed" instead of inventing a resemblance.
    """
    best: MediaId | None = None
    best_score = 0.0
    for other in against:
        score = board.similarity(media_id, other)
        if score is not None and score > best_score:
            best, best_score = other, score
    return best if best_score >= SEMANTIC_RELATED_SIMILARITY else None


# ----------------------------------------------------------------- the trim
def _cut(
    *,
    reading: ClipReading,
    role: StoryRole,
    pick: _Pick,
    slot: PacingSlot | None,
    index: int,
    policy: EditorialPolicy,
    treatments: bool,
) -> tuple[EditorialSegment, list[EditorialDecision]]:
    """Decide which part of one clip is used, how long, and at what rate.

    This is where the phase's claim about *trimming* is cashed. A centre trim is
    the right default for an unexamined clip and the wrong answer for one whose
    signals say where the action is: the engine centres an action role on the
    busiest measured moment, keeps a calm role away from the clip's own internal
    cuts, and falls back to the centre only when it has nothing better -- which
    it reports rather than hiding.
    """
    signals = reading.signals
    source_ms = signals.duration_ms
    decisions: list[EditorialDecision] = []

    wanted = slot.duration_ms if slot is not None else policy.target_clip_ms
    target_energy = slot.energy if slot is not None else policy.energy_for(role)
    emphasis = slot.emphasis if slot is not None else 1.0

    # --- speed -------------------------------------------------------------
    # Decided before the trim, because a slowed clip needs *more* source to
    # occupy the same screen time. Getting this order wrong is how a peak given
    # slow motion silently becomes twice as long as the pacing asked for.
    effects: list[Effect] = []
    rate = 1.0
    if treatments and role is StoryRole.PEAK and policy.slow_motion_peak:
        low, _high, _neutral = EFFECT_BOUNDS[EffectKind.SLOW_MOTION]
        rate = max(low, PEAK_SLOW_RATE)
        take_needed = int(round(wanted * rate))
        if take_needed <= source_ms and take_needed >= MIN_SEGMENT_MS:
            effects.append(Effect(kind=EffectKind.SLOW_MOTION, amount=rate))
            decisions.append(
                EditorialDecision(
                    kind=DecisionKind.SLOW,
                    media_id=reading.media_id,
                    slot=index,
                    role=role,
                    effect=EffectKind.SLOW_MOTION,
                    amount=rate,
                    reasons=(ReasonCode.PEAK_MOMENT, ReasonCode.POLICY_EMPHASIS),
                    confidence=reading.confidence_for(EditorialEvent.PEAK_MOTION) or 0.6,
                )
            )
        else:
            # Not enough footage to slow it and still fill the slot. No effect,
            # and no silent half-application.
            rate = 1.0

    take = max(MIN_SEGMENT_MS, min(MAX_SEGMENT_MS, int(round(wanted * rate))))
    take = min(take, max(source_ms, MIN_SEGMENT_MS))

    # --- where in the source ----------------------------------------------
    start, trim_reasons = _window(signals, role, take, source_ms)
    end = start + min(take, max(source_ms - start, MIN_SEGMENT_MS))
    if end > source_ms:
        start, end = max(0, source_ms - take), source_ms

    decisions.append(
        EditorialDecision(
            kind=DecisionKind.TRIM,
            media_id=reading.media_id,
            slot=index,
            role=role,
            source_in_ms=start,
            source_out_ms=end,
            reasons=trim_reasons,
            confidence=0.8 if ReasonCode.CENTRED_ON_ACTION in trim_reasons else 0.6,
        )
    )

    output_ms = int(round((end - start) / rate))

    # --- emphasis, as a decision in its own right --------------------------
    if emphasis > 1.01:
        decisions.append(
            EditorialDecision(
                kind=DecisionKind.EMPHASIZE,
                media_id=reading.media_id,
                slot=index,
                role=role,
                amount=round(emphasis, 3),
                reasons=(ReasonCode.POLICY_EMPHASIS,),
            )
        )

    # --- pacing, as a reason ----------------------------------------------
    pacing_reasons: list[ReasonCode] = []
    if wanted < policy.target_clip_ms * 0.9:
        pacing_reasons.append(ReasonCode.PACING_SHORTENS)
    elif wanted > policy.target_clip_ms * 1.1:
        pacing_reasons.append(ReasonCode.PACING_LENGTHENS)

    if slot is not None and slot.on_beat:
        decisions.append(
            EditorialDecision(
                kind=DecisionKind.PLACE_ON_BEAT,
                media_id=reading.media_id,
                slot=index,
                role=role,
                beats=slot.beats,
                start_ms=slot.start_ms,
                reasons=(ReasonCode.BEAT_ALIGNED,),
            )
        )
        pacing_reasons.append(ReasonCode.BEAT_ALIGNED)
    elif slot is not None and slot.beats is None and slot.index > 0:
        pacing_reasons.append(ReasonCode.OFF_BEAT)

    # --- a look, where the policy asks for one -----------------------------
    if treatments and rate == 1.0:
        extra = _look(reading, role, policy, hold_ms=end - start)
        if extra is not None:
            effects.append(extra)
            decisions.append(
                EditorialDecision(
                    kind=DecisionKind.ADD_EFFECT,
                    media_id=reading.media_id,
                    slot=index,
                    role=role,
                    effect=extra.kind,
                    amount=extra.amount,
                    reasons=(ReasonCode.STYLE_POLICY,),
                    confidence=0.5,
                )
            )

    # --- a caption, as a recommendation only -------------------------------
    #
    # Nothing here transcribes anything. What the engine can say is that this
    # clip looks like one somebody is speaking in, which is a reason to *offer*
    # subtitles -- the Phase 9 suggestion flow writes the words, and the user
    # keeps or rewrites them. A decision carrying invented text would be the
    # system telling the user what they said.
    if treatments and reading.has(EditorialEvent.DIALOGUE):
        decisions.append(
            EditorialDecision(
                kind=DecisionKind.ADD_SUBTITLE,
                media_id=reading.media_id,
                slot=index,
                role=role,
                start_ms=slot.start_ms if slot is not None else None,
                reasons=(ReasonCode.LIKELY_SPEECH,),
                confidence=reading.confidence_for(EditorialEvent.DIALOGUE),
            )
        )

    segment = EditorialSegment(
        slot=index,
        role=role,
        media_id=reading.media_id,
        source_in_ms=start,
        source_out_ms=end,
        output_ms=output_ms,
        energy=signals.energy,
        target_energy=target_energy,
        events=reading.events,
        reasons=tuple(dict.fromkeys([*pick.reasons, *pacing_reasons])),
        score=pick.score,
        components=pick.components,
        beats=slot.beats if slot is not None else None,
        effects=tuple(effects),
        sequence=signals.sequence + 1,
    )
    return segment, decisions


def _window(
    signals: Any, role: StoryRole, take: int, source_ms: int
) -> tuple[int, tuple[ReasonCode, ...]]:
    """Where inside a source clip to cut from.

    Three strategies, tried in order, and each one reports itself:

    **Centre on the action.** For the roles that want movement, when the
    dynamics analyzer recorded which of its sample points was busiest. It names
    a fifth of the clip rather than a frame, which is coarse -- and still the
    difference between cutting the goal and cutting the run-up to it.

    **Avoid an internal cut.** A source that already contains a scene boundary
    will produce a jump in the middle of what the plan calls one shot. Where a
    boundary-free stretch long enough for the take exists, the window goes
    there.

    **Centre.** The Phase 4 default, and still right for an unexamined clip: the
    head of handheld footage is where the focus hunt and the camera being raised
    live.
    """
    if source_ms <= take:
        return 0, (ReasonCode.SOURCE_TOO_SHORT,)

    boundaries = tuple(signals.scene_boundaries_ms or ())
    wants_action = role in (StoryRole.PEAK, StoryRole.BUILD, StoryRole.HOOK)
    peak_ms = signals.motion_peak_ms

    if wants_action and peak_ms is not None:
        start = max(0, min(source_ms - take, peak_ms - take // 2))
        if not _crosses(start, start + take, boundaries):
            return start, (ReasonCode.CENTRED_ON_ACTION,)
        shifted = _avoid(start, take, source_ms, boundaries)
        if shifted is not None:
            return shifted, (ReasonCode.CENTRED_ON_ACTION, ReasonCode.AVOIDS_INTERNAL_CUT)
        return start, (ReasonCode.CENTRED_ON_ACTION,)

    centre = max(0, (source_ms - take) // 2)
    if not _crosses(centre, centre + take, boundaries):
        return centre, (ReasonCode.CENTRE_TRIM,)

    shifted = _avoid(centre, take, source_ms, boundaries)
    if shifted is not None:
        return shifted, (ReasonCode.AVOIDS_INTERNAL_CUT,)
    return centre, (ReasonCode.CENTRE_TRIM,)


def _crosses(start: int, end: int, boundaries: tuple[int, ...]) -> bool:
    return any(start < boundary < end for boundary in boundaries)


def _avoid(preferred: int, take: int, source_ms: int, boundaries: tuple[int, ...]) -> int | None:
    """The boundary-free window closest to where the caller wanted one.

    Walks the intervals between cuts, keeps the ones long enough to hold the
    take, and returns the start nearest the preferred position. ``None`` when no
    interval is long enough, which is the common case for heavily-cut source and
    is why the caller keeps its original window rather than refusing the clip.
    """
    edges = [0, *sorted(boundaries), source_ms]
    best: int | None = None
    best_distance = source_ms + 1

    for low, high in pairwise(edges):
        if high - low < take:
            continue
        start = max(low, min(high - take, preferred))
        distance = abs(start - preferred)
        if distance < best_distance:
            best, best_distance = start, distance

    return best


#: Shortest shot a push-in belongs on. Below about two and a half seconds the
#: movement has no room to read as a movement; it reads as a wobble.
MIN_ZOOM_HOLD_MS = 2_500


def _look(
    reading: ClipReading, role: StoryRole, policy: EditorialPolicy, *, hold_ms: int
) -> Effect | None:
    """A single optional effect, when the policy's appetite justifies one.

    At most one, and only where it means something. A push-in on a held
    establishing shot is a real cinematic device; a push-in on every clip is a
    tic, and the appetite dial is what keeps the difference visible in the
    policy table rather than buried here.

    Two gates beyond the appetite, and both are craft rather than configuration.
    The shot has to be long enough for the move to read. And a policy that
    slows its peak is an action policy: its idea of emphasis is speed, so
    layering a push-in on top of it would be two devices arguing.
    """
    if policy.effect_appetite < 0.3 or hold_ms < MIN_ZOOM_HOLD_MS:
        return None

    if role in (StoryRole.SETUP, StoryRole.ENDING) and reading.has(EditorialEvent.ESTABLISHING):
        return Effect(kind=EffectKind.ZOOM_IN, amount=GENTLE_ZOOM)
    if role is StoryRole.HOOK and policy.effect_appetite >= 0.4 and not policy.slow_motion_peak:
        return Effect(kind=EffectKind.ZOOM_IN, amount=GENTLE_ZOOM)
    return None


# ------------------------------------------------------------------- the seams
def _seams(
    segments: list[EditorialSegment], policy: EditorialPolicy
) -> tuple[list[EditorialSegment], list[EditorialDecision]]:
    """Decide how each clip enters the one before it.

    The rule is about *energy*, not about position: a dissolve belongs where the
    edit is settling -- into a reaction, into an ending, between two calm shots
    -- and a cut belongs where it is not. That is a real editorial convention
    rather than a decoration, and it is why a policy's transition appetite is
    one number instead of a list of which segments get dissolves.
    """
    appetite = policy.transition_appetite
    if appetite <= 0.0 or not segments:
        return segments, []

    decisions: list[EditorialDecision] = []
    result: list[EditorialSegment] = []

    for index, segment in enumerate(segments):
        kind = TransitionKind.CUT
        length = 0
        reasons: tuple[ReasonCode, ...] = ()

        if index == 0 and appetite >= 0.3:
            kind = TransitionKind.FADE_IN
            length = _transition_ms(appetite, segment.output_ms, segment.output_ms)
            reasons = (ReasonCode.STYLE_POLICY,)
        elif index > 0:
            previous = result[index - 1]
            settling = segment.energy < previous.energy - 0.15
            calm = max(segment.energy, previous.energy) < 0.4
            if appetite >= 0.3 and (settling or calm):
                kind = TransitionKind.CROSSFADE
                length = _transition_ms(appetite, previous.output_ms, segment.output_ms)
                reasons = (ReasonCode.ENERGY_FIT, ReasonCode.STYLE_POLICY)

        if index == len(segments) - 1 and appetite >= 0.5 and kind is TransitionKind.CUT:
            kind = TransitionKind.FADE_TO_BLACK
            length = _transition_ms(appetite, segment.output_ms, segment.output_ms)
            reasons = (ReasonCode.STYLE_POLICY,)

        if length < MIN_TRANSITION_MS:
            kind, length, reasons = TransitionKind.CUT, 0, ()

        if kind is not TransitionKind.CUT:
            decisions.append(
                EditorialDecision(
                    kind=DecisionKind.TRANSITION,
                    media_id=segment.media_id,
                    slot=index,
                    role=segment.role,
                    transition=kind,
                    transition_ms=length,
                    reasons=reasons,
                    confidence=0.6,
                )
            )

        result.append(
            EditorialSegment(
                slot=segment.slot,
                role=segment.role,
                media_id=segment.media_id,
                source_in_ms=segment.source_in_ms,
                source_out_ms=segment.source_out_ms,
                output_ms=segment.output_ms,
                energy=segment.energy,
                target_energy=segment.target_energy,
                events=segment.events,
                reasons=segment.reasons,
                score=segment.score,
                components=segment.components,
                beats=segment.beats,
                transition=kind,
                transition_ms=length,
                effects=segment.effects,
                sequence=segment.sequence,
            )
        )

    return result, decisions


def _transition_ms(appetite: float, left_ms: int, right_ms: int) -> int:
    """How long a dissolve runs, given how much this policy likes them.

    Clamped to half of the shorter neighbour, which is the plan validator's own
    rule applied here so a transition is never proposed that the gate will then
    refuse -- a plan rejected over its decoration is a worse outcome than a
    straight cut.
    """
    wanted = int(MIN_TRANSITION_MS + appetite * 700)
    ceiling = min(MAX_TRANSITION_MS, int(min(left_ms, right_ms) * MAX_TRANSITION_SHARE))
    if ceiling < MIN_TRANSITION_MS:
        return 0
    return max(MIN_TRANSITION_MS, min(wanted, ceiling))


def _ordering_decisions(segments: list[EditorialSegment]) -> list[EditorialDecision]:
    """Record where the edit departs from the order the clips were shot in.

    Not a change to anything -- the order is already decided by the arc. This
    exists so the departure is *visible*: an edit that silently reorders a
    match is indistinguishable from one that got the chronology wrong, and the
    only difference a user can see is whether the system said it meant to.
    """
    decisions: list[EditorialDecision] = []
    chronological = sorted(range(len(segments)), key=lambda i: segments[i].sequence)
    for position, source_index in enumerate(chronological):
        if position != source_index:
            segment = segments[source_index]
            decisions.append(
                EditorialDecision(
                    kind=DecisionKind.REORDER,
                    media_id=segment.media_id,
                    slot=segment.slot,
                    role=segment.role,
                    from_index=position,
                    to_index=source_index,
                    reasons=(ReasonCode.NARRATIVE_ORDER,),
                    confidence=0.7,
                )
            )
    return decisions


__all__ = [
    "DECISION_ENGINE_VERSION",
    "DecisionKind",
    "EditorialDecision",
    "EditorialPlan",
    "EditorialRejection",
    "EditorialSegment",
    "EngineInput",
    "ReasonCode",
    "decide",
]
