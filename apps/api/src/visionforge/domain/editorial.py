"""Editorial signals and the closed vocabulary of editorial events.

    analysis rows  ->  EditorialSignals   (normalised, versioned, per clip)
                   ->  SignalBoard        (the same clips, compared to each other)
                   ->  DetectedEvent[]    (a closed vocabulary, with confidence)

This is the layer that turns *measurements* into *observations*. Phase 3 can say
a clip's median inter-frame difference is 0.31; nothing before Phase 11 could say
"something happens in this clip". The distance between those two statements is
the whole of this module, and it is walked deliberately and narrowly.

Three rules govern it, and they are the same three that govern
``domain.reference`` -- because they are the rules that stop a measurement system
from becoming a guessing system.

**Every signal is derived, never invented.** A signal whose analyzer did not run
is ``None``. It is not zero, it is not the population mean, and it is not filled
in from the style the user picked. Downstream code decides what to do without it.

**Every event carries its own confidence, and the evidence that produced it.**
``CLOSE_UP`` from a face box covering a third of the frame is a strong claim;
``DIALOGUE`` from "there are faces and there is an audio stream" is a weak one,
and it says so -- there is no speech detection in this product, so the strongest
honest claim is "this looks like a clip where someone might be talking". An
observation that cannot clear the floor becomes ``UNKNOWN`` rather than becoming
the most plausible guess.

**Nothing here knows what an edit is.** Signals and events describe footage.
Which of them matters, and what to do about it, is decided in ``domain.story``
and ``domain.decisions`` -- so a policy change is a table edit there rather than
a rewrite of the perception here.

Pure domain: no OpenCV, no numpy, no SQL, no I/O. It consumes ``Candidate``
objects the application layer already builds from analysis rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.domain.selection import Candidate, ScoredCandidate
from visionforge.domain.stills import hold_span_ms, is_still

#: Bumped whenever the meaning of a signal changes. Recorded on every plan, for
#: the same reason analyzers are versioned: an edit that looks different next
#: month must be attributable to a change somebody made rather than to drift.
SIGNAL_VERSION = "2"

#: Bumped whenever an event is added, removed, or its evidence rule changes.
#: Separate from ``SIGNAL_VERSION`` because the two move independently -- a
#: better motion measurement does not change what "action" means.
EVENT_VOCABULARY_VERSION = "1"


# --------------------------------------------------------------- normalisation
#
# Every reference below is a *fixed* ceiling, never the maximum of the batch.
# Normalising within a batch would make a clip's signals depend on what it was
# uploaded alongside, which is the same hidden coupling ``SelectionWeights``
# refuses for exactly the same reason: it makes a result irreproducible.

#: Luma standard deviation that reads as full contrast. Matches the value
#: ``selection.contrast_component`` normalises against, so the two agree.
CONTRAST_REFERENCE = 80.0

#: Motion above this is "as energetic as this measurement can usefully say".
#: The dynamics analyzer's own scale already saturates well below 1.0 on real
#: footage -- a value of 0.35 is a fast pan -- so normalising against 1.0 would
#: compress every real clip into the bottom third of the range.
MOTION_REFERENCE = 0.35

#: Motion standard deviation that reads as "this clip changes character".
MOTION_SPREAD_REFERENCE = 0.12

#: Face box area, as a fraction of the frame, at which a shot is unambiguously
#: a close-up. A head filling an eighth of a 16:9 frame is a medium close-up by
#: any framing convention; a quarter is a portrait.
CLOSE_UP_FACE_AREA = 0.08

#: Shot length, in ms, at which a clip stops reading as a held shot. Used only
#: to normalise the "duration" signal onto 0..1.
DURATION_REFERENCE_MS = 10_000

#: Cuts per minute inside a single source clip at which it reads as already-cut
#: material rather than as one take.
SHOT_DENSITY_REFERENCE = 30.0


# -------------------------------------------------------------------- signals
@dataclass(frozen=True, slots=True)
class EditorialSignals:
    """One clip's footage, described on comparable 0..1 axes.

    Everything is normalised so that two clips can be compared without the
    reader remembering which scale each number is on -- the mistake
    ``ReferenceProfile`` also refuses to make. ``None`` means "the analyzer that
    would have measured this did not run", which is distinguishable from zero
    everywhere it matters.
    """

    version: str
    media_id: MediaId
    sequence: int
    #: How much of the source the engine may take. For a still (Phase 12) that
    #: is the longest permitted hold rather than a length the photo has.
    duration_ms: int
    #: A photo rather than a video (Phase 12). Held from zero, never slowed,
    #: and given a drift so it does not read as a freeze.
    is_still: bool = False
    #: Taller than wide, when the size is known. Decides which way a still
    #: drifts.
    is_portrait: bool = False

    # --- technical, from the quality analyzer ---
    sharpness: float | None = None
    exposure: float | None = None
    contrast: float | None = None
    brightness: float | None = None
    #: The selection score this clip already earned, 0..1. Carried rather than
    #: recomputed so the editorial layer and the ranker cannot disagree about
    #: which clip is technically stronger.
    quality: float | None = None

    # --- movement, from the dynamics analyzer ---
    motion: float | None = None
    motion_variation: float | None = None
    saturation: float | None = None

    # --- structure, from the scene analyzer ---
    #: Cuts per minute inside this clip's own source, normalised.
    shot_density: float | None = None
    #: Cut positions inside the clip, ms from its start. Not normalised: a
    #: trimmer needs the real number.
    scene_boundaries_ms: tuple[int, ...] = ()
    #: Where the busiest sampled moment sits, ms from the clip's start. Not
    #: normalised for the same reason: a trimmer needs a position, not a score.
    motion_peak_ms: int | None = None

    # --- people, from the face analyzer ---
    #: 1.0 when someone is on screen in every sampled frame, 0.0 when nobody
    #: ever is, and ``None`` when detection did not run.
    face_presence: float | None = None
    #: Largest face box as a fraction of frame area, normalised against
    #: ``CLOSE_UP_FACE_AREA``.
    face_scale: float | None = None
    #: Raw per-sample presence, in time order. Kept because entry and exit are
    #: about the *shape* of this list and averaging it away loses them.
    face_frames: tuple[bool, ...] = ()

    # --- the rest ---
    has_audio: bool = False
    #: How close this clip's look is to the project's reference video, 0..1.
    #: ``None`` when there is no reference or it measured nothing comparable.
    style_match: float | None = None
    #: The semantic vector, carried so the board can compare clips. Not a
    #: signal in its own right -- 512 numbers are not an observation.
    embedding: tuple[float, ...] | None = None

    @property
    def energy(self) -> float:
        """How much is going on, 0..1.

        Motion first, because it is the only direct measurement of activity;
        variation and saturation are minority contributors because each on its
        own is a poor proxy -- a saturated sunset is not an energetic shot, and
        a clip with a cut in it varies wildly while being perfectly calm.

        Total, never ``None``: a clip with no dynamics row gets the energy its
        other signals justify, and with none of them it gets 0.0. That is the
        one place a default is allowed here, and only because the alternative --
        an optional energy that every caller then has to special-case -- pushes
        the same decision outward to five places instead of stating it once.
        """
        parts: list[tuple[float, float]] = []
        if self.motion is not None:
            parts.append((self.motion, 0.70))
        if self.motion_variation is not None:
            parts.append((self.motion_variation, 0.20))
        if self.saturation is not None:
            parts.append((self.saturation, 0.10))
        if not parts:
            return 0.0
        weight = sum(w for _, w in parts)
        return round(sum(value * w for value, w in parts) / weight, 4)

    @property
    def is_measured(self) -> bool:
        """Whether anything beyond the quality analyzer looked at this clip."""
        return self.motion is not None or self.face_presence is not None

    def as_payload(self) -> dict[str, Any]:
        """The wire form. Numbers only -- no vector, no id-bearing structure."""
        payload: dict[str, Any] = {
            "version": self.version,
            "duration_ms": self.duration_ms,
            "energy": self.energy,
        }
        for name, value in (
            ("sharpness", self.sharpness),
            ("exposure", self.exposure),
            ("contrast", self.contrast),
            ("brightness", self.brightness),
            ("quality", self.quality),
            ("motion", self.motion),
            ("motion_variation", self.motion_variation),
            ("saturation", self.saturation),
            ("shot_density", self.shot_density),
            ("face_presence", self.face_presence),
            ("face_scale", self.face_scale),
            ("style_match", self.style_match),
        ):
            if value is not None:
                payload[name] = round(value, 4)
        payload["has_audio"] = self.has_audio
        payload["cuts_inside"] = len(self.scene_boundaries_ms)
        if self.is_still:
            payload["still"] = True
        return payload


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ratio(value: float | None, reference: float) -> float | None:
    if value is None:
        return None
    return round(_clamp(value / reference), 4)


def signals_for(
    candidate: Candidate,
    *,
    scored: ScoredCandidate | None = None,
    style_match: float | None = None,
) -> EditorialSignals:
    """Reduce one candidate to normalised editorial signals. Deterministic.

    ``scored`` is optional and supplies the technical components the ranker has
    already computed. Passing it is strongly preferred: it means the editorial
    layer's notion of "sharp" is the ranker's notion of "sharp", rather than a
    second normalisation that drifts from the first the next time a reference
    constant is tuned.
    """
    components = scored.components if scored is not None else {}

    duration_ms = hold_span_ms(candidate)
    density: float | None = None
    if candidate.mean_scene_ms and candidate.mean_scene_ms > 0:
        cuts_per_minute = 60_000.0 / float(candidate.mean_scene_ms)
        density = round(_clamp(cuts_per_minute / SHOT_DENSITY_REFERENCE), 4)

    presence: float | None = None
    if candidate.face_frames:
        seen = sum(1 for present in candidate.face_frames if present)
        presence = round(seen / len(candidate.face_frames), 4)
    elif candidate.has_faces is not None:
        # Detection ran but the per-frame detail is not available (an older
        # analysis row). "Someone was on screen" is still a fact, so it is
        # reported as a fact at the coarsest resolution the evidence supports.
        presence = 1.0 if candidate.has_faces else 0.0

    return EditorialSignals(
        version=SIGNAL_VERSION,
        media_id=candidate.media_id,
        sequence=candidate.sequence,
        duration_ms=duration_ms,
        is_still=is_still(candidate),
        is_portrait=(candidate.height or 0) > (candidate.width or 0),
        sharpness=_as_float(components.get("sharpness")),
        exposure=_as_float(components.get("exposure")),
        contrast=(
            _as_float(components.get("contrast"))
            if "contrast" in components
            else _ratio(candidate.contrast, CONTRAST_REFERENCE)
        ),
        brightness=(
            round(_clamp(candidate.mean_luminance / 255.0), 4)
            if candidate.mean_luminance is not None
            else None
        ),
        quality=round(scored.score, 4) if scored is not None else None,
        motion=_ratio(candidate.motion, MOTION_REFERENCE),
        motion_variation=_ratio(candidate.motion_spread, MOTION_SPREAD_REFERENCE),
        saturation=(
            None if candidate.saturation is None else round(_clamp(candidate.saturation), 4)
        ),
        shot_density=density,
        scene_boundaries_ms=candidate.scene_boundaries_ms,
        motion_peak_ms=candidate.motion_peak_ms,
        face_presence=presence,
        face_scale=_ratio(candidate.face_area, CLOSE_UP_FACE_AREA),
        face_frames=candidate.face_frames,
        has_audio=candidate.has_audio,
        style_match=(None if style_match is None else round(_clamp(style_match), 4)),
        embedding=candidate.embedding,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- the board
#
# Some editorial questions are not about a clip; they are about a clip *among
# these other clips*. "Is this the strongest action in the project" has no
# answer for one clip in isolation, and answering it against a fixed threshold
# would mean a calm nature project has no peak at all. Those questions live
# here, on a board that holds every clip at once.

#: Cosine similarity at or above which two clips are treated as showing the same
#: thing. Not a duplicate threshold -- pHash already handles literal duplicates
#: at ``PHASH_DUPLICATE_MAX_DISTANCE``. This is the looser, more useful notion:
#: two angles on one goal, three takes of the same doorway. CLIP puts genuinely
#: unrelated footage well below 0.8 and the same subject well above it.
SEMANTIC_DUPLICATE_SIMILARITY = 0.90

#: The similarity above which two clips are merely *related* -- same location,
#: same subject matter. Used to spread a selection across content rather than
#: to reject.
SEMANTIC_RELATED_SIMILARITY = 0.78


@dataclass(frozen=True, slots=True)
class SignalBoard:
    """Every clip's signals, and the comparisons that need all of them.

    Frozen and built once per plan. Every method is a pure function of the
    signals it was built from, so two runs over the same footage produce the
    same board and therefore the same edit.
    """

    version: str
    signals: tuple[EditorialSignals, ...]
    #: The semantic vocabulary version, so a plan records which notion of
    #: "the same thing" produced it.
    events_version: str = EVENT_VOCABULARY_VERSION

    @property
    def by_id(self) -> dict[MediaId, EditorialSignals]:
        return {signal.media_id: signal for signal in self.signals}

    def get(self, media_id: MediaId) -> EditorialSignals | None:
        return self.by_id.get(media_id)

    # ------------------------------------------------------------- ranking
    def rank(self, media_id: MediaId, attribute: str) -> float | None:
        """Where a clip sits in this project on one axis, 0..1.

        A *percentile within the project*, and that is the point. "High motion"
        is not a property footage has in the absolute; a nature project's most
        active shot is a bird taking off and a football project's is a sprint,
        and an editorial layer that used one threshold for both would find no
        peak in the first and eleven in the second.

        ``None`` when the clip was not measured on that axis, or when fewer than
        two clips were -- a percentile over one sample is not a percentile.
        """
        values = [(signal.media_id, _attribute(signal, attribute)) for signal in self.signals]
        measured = [(mid, value) for mid, value in values if value is not None]
        if len(measured) < 2:
            return None
        own = dict(measured).get(media_id)
        if own is None:
            return None
        below = sum(1 for _, value in measured if value < own)
        equal = sum(1 for _, value in measured if value == own)
        # Midpoint of the tied band, so three identical clips each rank 0.5
        # rather than one of them arbitrarily ranking top.
        return round((below + (equal - 1) / 2) / (len(measured) - 1), 4)

    # ------------------------------------------------------------ semantics
    def similarity(self, left: MediaId, right: MediaId) -> float | None:
        """Cosine similarity between two clips' CLIP vectors.

        ``None`` when either has no embedding. Absent is not "dissimilar": a
        clip that was never embedded must not be treated as guaranteed-unique,
        which is exactly the bug that would put five un-embedded copies of the
        same shot in one edit.
        """
        first, second = self.get(left), self.get(right)
        if first is None or second is None:
            return None
        return cosine(first.embedding, second.embedding)

    def semantic_groups(
        self, *, threshold: float = SEMANTIC_DUPLICATE_SIMILARITY
    ) -> tuple[tuple[MediaId, ...], ...]:
        """Clusters of clips showing the same thing.

        Single-link union-find over the pairwise comparison, the same shape as
        ``selection.group_duplicates`` and for the same reason: it is O(n^2) on
        at most forty clips, and the clarity is worth more than the bisect.
        Groups come back in stable input order so the survivor chosen downstream
        is deterministic.
        """
        embedded = [signal for signal in self.signals if signal.embedding]
        parent: dict[MediaId, MediaId] = {s.media_id: s.media_id for s in embedded}

        def find(node: MediaId) -> MediaId:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index, left in enumerate(embedded):
            for right in embedded[index + 1 :]:
                score = cosine(left.embedding, right.embedding)
                if score is not None and score >= threshold:
                    root_left, root_right = find(left.media_id), find(right.media_id)
                    if root_left != root_right:
                        parent[root_right] = root_left

        order = {signal.media_id: i for i, signal in enumerate(self.signals)}
        groups: dict[MediaId, list[MediaId]] = {}
        for signal in embedded:
            groups.setdefault(find(signal.media_id), []).append(signal.media_id)
        return tuple(
            tuple(sorted(group, key=lambda m: order[m]))
            for group in groups.values()
            if len(group) > 1
        )

    def distinctiveness(self, media_id: MediaId, against: tuple[MediaId, ...]) -> float:
        """How different a clip is from the ones already chosen, 0..1.

        One minus the closest similarity to anything in ``against``. With
        nothing to compare to -- an empty selection, or no embedding anywhere --
        it is 1.0, which means "nothing here argues this is a repeat" rather
        than "this is proven unique". The difference matters in the reasons the
        engine reports: absence of evidence is never reported as evidence.
        """
        closest = 0.0
        for other in against:
            if other == media_id:
                continue
            score = self.similarity(media_id, other)
            if score is not None:
                closest = max(closest, score)
        return round(_clamp(1.0 - closest), 4)

    def as_payload(self) -> dict[str, Any]:
        return {
            "signal_version": self.version,
            "events_version": self.events_version,
            "clip_count": len(self.signals),
            "embedded": sum(1 for s in self.signals if s.embedding),
            "semantic_groups": len(self.semantic_groups()),
        }


def _attribute(signal: EditorialSignals, name: str) -> float | None:
    if name == "energy":
        return signal.energy
    value = getattr(signal, name, None)
    return value if isinstance(value, float) else None


def cosine(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float | None:
    """Cosine similarity, or ``None`` when either vector is missing or degenerate.

    Written out rather than imported from numpy because this is domain code and
    the ``api-never-touches-media`` contract forbids numpy here. Over 512 floats
    and at most forty clips this is microseconds, and the explicitness is worth
    more than the vectorisation.
    """
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left <= 0 or norm_right <= 0:
        return None
    return round(_clamp(dot / (norm_left * norm_right), -1.0, 1.0), 6)


def board_from(
    candidates: list[Candidate],
    *,
    scores: dict[MediaId, ScoredCandidate] | None = None,
    style_matches: dict[MediaId, float] | None = None,
) -> SignalBoard:
    """Build the board for one project's usable footage.

    Images are included: a still has no motion and says so, and excluding it
    here would make a photo essay unplannable rather than calm.
    """
    scores = scores or {}
    style_matches = style_matches or {}
    return SignalBoard(
        version=SIGNAL_VERSION,
        signals=tuple(
            signals_for(
                candidate,
                scored=scores.get(candidate.media_id),
                style_match=style_matches.get(candidate.media_id),
            )
            for candidate in candidates
            if candidate.kind is not MediaKind.AUDIO
        ),
    )


# --------------------------------------------------------------------- events
class EditorialEvent(StrEnum):
    """What is happening in a clip. A closed, versioned vocabulary.

    Twelve members and an escape hatch, and the restraint is the same one
    ``EffectKind`` exercises: each of these needs evidence, a rule, a confidence
    model and a test, and shipping thirty of them on guesswork would make every
    one of them worthless.

    ``UNKNOWN`` is not a failure. It is the correct answer for a clip whose
    signals do not support any of the others, and it is deliberately usable --
    an unknown clip can still be selected on quality and given a role, it simply
    contributes no event-based argument for where it belongs.
    """

    #: A wide, held, low-movement shot. What an edit opens on.
    ESTABLISHING = "establishing"
    #: A face filling a meaningful share of the frame.
    CLOSE_UP = "close_up"
    #: Nobody on screen at the start, somebody by the end.
    SUBJECT_ENTRY = "subject_entry"
    #: Somebody at the start, nobody by the end.
    SUBJECT_EXIT = "subject_exit"
    #: Sustained movement.
    ACTION = "action"
    #: The most movement in this project.
    PEAK_MOTION = "peak_motion"
    #: Movement that changes sharply inside the clip -- a hit, a landing, a
    #: sudden stop. Measured as variation, not as a collision.
    IMPACT = "impact"
    #: A person, framed close, not moving much, after something that was.
    REACTION = "reaction"
    #: People, movement and colour together.
    CELEBRATION = "celebration"
    #: Framing and audio consistent with somebody talking. The weakest claim in
    #: this vocabulary, because nothing here listens to the audio.
    DIALOGUE = "dialogue"
    #: The clip contains a cut of its own.
    TRANSITION_MOMENT = "transition_moment"
    #: Late, calm, and darkening.
    ENDING = "ending"
    #: The signals support nothing above.
    UNKNOWN = "unknown"


#: Below this an observation is not reported at all, and a clip with no
#: observation above it is ``UNKNOWN``. Chosen so that a rule firing on one weak
#: signal does not produce a claim: every event in the table below needs either
#: one strong signal or two agreeing ones to clear it.
MIN_EVENT_CONFIDENCE = 0.35

#: How many events one clip may carry. A clip is allowed to be a close-up *and*
#: a reaction; it is not allowed to be eight things, which is what an unbounded
#: list would degenerate into and is indistinguishable from saying nothing.
MAX_EVENTS_PER_CLIP = 3


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    """One observation about a clip, with its confidence and its evidence.

    ``evidence`` is a tuple of short machine-readable tokens, not prose. They
    are shown to the user as the reason a clip was chosen, sent to the model as
    context, and asserted on in tests -- three consumers that a sentence would
    serve badly and a token serves exactly.
    """

    event: EditorialEvent
    confidence: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be between 0 and 1")

    def as_payload(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
        }


UNKNOWN_EVENT = DetectedEvent(EditorialEvent.UNKNOWN, confidence=1.0, evidence=("no_evidence",))


@dataclass(frozen=True, slots=True)
class EventContext:
    """What a clip's neighbours and its project contribute to reading it.

    Some events are not visible from inside one clip. "Reaction" means calm
    people *after* something loud; "peak motion" means the most movement *in
    this project*. Passing the context explicitly, rather than letting the
    detector reach for a global, is what keeps event detection a pure function
    and therefore testable with six lines of fixture.
    """

    motion_rank: float | None = None
    energy_rank: float | None = None
    duration_rank: float | None = None
    #: Energy of the clip before this one in upload order, when there is one.
    previous_energy: float | None = None
    #: Position in the upload order, 0..1. The only chronology available.
    position: float = 0.0
    #: Whether this clip is the last in upload order.
    is_last: bool = False


def detect_events(
    signal: EditorialSignals, context: EventContext | None = None
) -> tuple[DetectedEvent, ...]:
    """Read a clip's signals into observations. Deterministic and total.

    Returns at least one event, always: a clip with nothing to say about it gets
    ``UNKNOWN`` rather than an empty tuple, so no caller has to handle "no
    events" as a separate case.

    Every rule below states its evidence tokens, and the confidences are
    deliberately unequal. ``CLOSE_UP`` rests on a measured box area and reaches
    0.9; ``DIALOGUE`` rests on framing plus the presence of an audio stream and
    is capped at 0.45, because this system does not listen to audio and
    pretending otherwise would put words in its mouth.
    """
    context = context or EventContext()
    found: list[DetectedEvent] = []

    for rule in _RULES:
        detected = rule(signal, context)
        if detected is not None and detected.confidence >= MIN_EVENT_CONFIDENCE:
            found.append(detected)

    if not found:
        return (UNKNOWN_EVENT,)

    # Strongest first, then alphabetically, so the order two runs produce is
    # identical and the first event is always the best-supported claim.
    found.sort(key=lambda d: (-d.confidence, d.event.value))
    return tuple(found[:MAX_EVENTS_PER_CLIP])


# ----------------------------------------------------------------- the rules
#
# Each rule takes a clip and its context and returns one observation or nothing.
# They are written as separate functions rather than as a chain of ifs so that
# each one can be tested in isolation and so that adding a thirteenth event is a
# function plus a table entry rather than a surgery on a hundred-line branch.


def _establishing(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Wide, held and calm. The shot an edit opens on.

    Requires motion to have been *measured*: a clip with no dynamics row is not
    calm, it is unexamined, and reporting "establishing" for it would be the
    exact failure this module is written to avoid.
    """
    if signal.motion is None:
        return None
    if signal.motion > 0.35:
        return None

    evidence = ["low_motion"]
    confidence = 0.5

    # A long shot argues for it; a short one does not argue against, because a
    # user may have trimmed the clip before uploading it.
    if signal.duration_ms >= 3_000:
        evidence.append("held_shot")
        confidence += 0.15

    # Nobody framed close. Only checked when detection ran -- an unanalysed
    # clip neither gains nor loses here.
    if signal.face_scale is not None and signal.face_scale < 0.4:
        evidence.append("wide_framing")
        confidence += 0.15
    elif signal.face_scale is not None:
        # A face filling the frame is the opposite of an establishing shot.
        return None

    if signal.contrast is not None and signal.contrast >= 0.4:
        evidence.append("legible_detail")
        confidence += 0.1

    return DetectedEvent(EditorialEvent.ESTABLISHING, min(confidence, 0.85), tuple(evidence))


def _close_up(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """A face occupying a real share of the frame. The strongest claim here."""
    if signal.face_scale is None or signal.face_scale < 1.0:
        return None
    # ``face_scale`` is already normalised against CLOSE_UP_FACE_AREA, so 1.0 is
    # the threshold and anything above is more certain.
    confidence = min(0.9, 0.6 + 0.3 * min(signal.face_scale - 1.0, 1.0))
    return DetectedEvent(
        EditorialEvent.CLOSE_UP, round(confidence, 3), ("large_face_area", "face_detected")
    )


def _subject_entry(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Nobody at the head of the clip, somebody at the tail.

    Five sample points is thin evidence for a movement through the frame, and
    the confidence says so. It is still worth having: "the subject arrives" is
    the difference between a clip that belongs at the start of a sequence and
    one that belongs at the end of it.
    """
    frames = signal.face_frames
    if len(frames) < 3:
        return None
    head, tail = frames[0], frames[-1]
    if head or not tail:
        return None
    return DetectedEvent(
        EditorialEvent.SUBJECT_ENTRY, 0.45, ("no_face_at_start", "face_at_end", "sparse_sampling")
    )


def _subject_exit(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    frames = signal.face_frames
    if len(frames) < 3:
        return None
    head, tail = frames[0], frames[-1]
    if not head or tail:
        return None
    return DetectedEvent(
        EditorialEvent.SUBJECT_EXIT, 0.45, ("face_at_start", "no_face_at_end", "sparse_sampling")
    )


def _action(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Sustained movement. The most reliable non-face observation available."""
    if signal.motion is None or signal.motion < 0.5:
        return None
    confidence = min(0.85, 0.45 + 0.4 * (signal.motion - 0.5) / 0.5)
    evidence = ["high_motion"]
    if context.motion_rank is not None and context.motion_rank >= 0.6:
        evidence.append("above_project_median")
        confidence = min(0.9, confidence + 0.05)
    return DetectedEvent(EditorialEvent.ACTION, round(confidence, 3), tuple(evidence))


def _peak_motion(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """The most movement in *this* project.

    Relative, not absolute, and that is the whole reason ``EventContext``
    exists. A fixed threshold finds eleven peaks in a football project and none
    in a nature one; a percentile finds the strongest moment in either.
    """
    if signal.motion is None or context.motion_rank is None:
        return None
    if context.motion_rank < 0.85:
        return None
    confidence = min(0.85, 0.55 + 0.3 * (context.motion_rank - 0.85) / 0.15)
    return DetectedEvent(
        EditorialEvent.PEAK_MOTION, round(confidence, 3), ("top_of_project_motion", "high_motion")
    )


def _impact(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Movement that changes sharply within the clip.

    Note what this does *not* claim. It is not a collision detector and there is
    no object tracking here; what is measured is that the inter-frame difference
    at one sample point is very unlike the others. A whip pan qualifies, and so
    does a cut the scene detector missed. The evidence token says as much.
    """
    if signal.motion_variation is None or signal.motion_variation < 0.6:
        return None
    if signal.motion is None or signal.motion < 0.25:
        # Variation around nothing is sensor noise, not an impact.
        return None
    confidence = min(0.7, 0.4 + 0.3 * (signal.motion_variation - 0.6) / 0.4)
    return DetectedEvent(
        EditorialEvent.IMPACT, round(confidence, 3), ("motion_spike", "no_object_tracking")
    )


def _reaction(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """People, framed close, calm, after something that was not.

    The only rule that depends on a neighbour, and it needs to: a calm close-up
    at the head of an edit is a portrait, and the same shot after a sprint is a
    reaction. Nothing in the clip itself distinguishes them.
    """
    if signal.face_presence is None or signal.face_presence < 0.5:
        return None
    if signal.motion is not None and signal.motion > 0.45:
        return None
    if context.previous_energy is None or context.previous_energy < 0.45:
        return None
    confidence = 0.45 + 0.2 * min(context.previous_energy, 1.0)
    return DetectedEvent(
        EditorialEvent.REACTION,
        round(min(confidence, 0.7), 3),
        ("faces_present", "calm_after_energy"),
    )


def _celebration(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """People, movement and colour at once."""
    if signal.face_presence is None or signal.face_presence < 0.4:
        return None
    if signal.motion is None or signal.motion < 0.45:
        return None
    confidence = 0.45
    evidence = ["faces_present", "high_motion"]
    if signal.saturation is not None and signal.saturation >= 0.4:
        evidence.append("saturated_colour")
        confidence += 0.15
    if context.energy_rank is not None and context.energy_rank >= 0.7:
        evidence.append("above_project_energy")
        confidence += 0.1
    return DetectedEvent(
        EditorialEvent.CELEBRATION, round(min(confidence, 0.7), 3), tuple(evidence)
    )


def _dialogue(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Framing and an audio stream consistent with somebody talking.

    Capped at 0.45 by construction, and the cap is the honest part. There is no
    speech detection in this product, no transcription and no voice activity
    measurement: what is actually known is that faces are on screen, the camera
    is still, and the file has an audio track. That is a hypothesis, not an
    observation, and the confidence ceiling is what stops the rest of the
    pipeline from treating it as one.
    """
    if not signal.has_audio:
        return None
    if signal.face_presence is None or signal.face_presence < 0.6:
        return None
    if signal.motion is None or signal.motion > 0.3:
        return None
    return DetectedEvent(
        EditorialEvent.DIALOGUE,
        0.40,
        ("faces_present", "still_framing", "audio_stream", "no_speech_detection"),
    )


def _transition_moment(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """The clip contains a cut of its own.

    Worth knowing for a reason that is purely practical: a centre trim across
    an internal boundary produces a jump in the middle of what the plan calls
    one shot, and the trimmer can avoid it once somebody tells it the boundary
    is there.
    """
    if not signal.scene_boundaries_ms:
        return None
    confidence = min(0.8, 0.5 + 0.1 * len(signal.scene_boundaries_ms))
    return DetectedEvent(
        EditorialEvent.TRANSITION_MOMENT,
        round(confidence, 3),
        ("internal_cut", f"cuts_{len(signal.scene_boundaries_ms)}"),
    )


def _ending(signal: EditorialSignals, context: EventContext) -> DetectedEvent | None:
    """Late, calm, and darker than the project's middle.

    The weakest positional claim here, and it is offered only for the clip that
    is actually last in the upload order. Every other clip gets no ending
    hypothesis at all, because "this could be an ending" is true of almost any
    calm shot and is therefore worth nothing.
    """
    if not context.is_last:
        return None
    # Motion must have been *measured*, for the same reason ``_establishing``
    # insists on it: being last in the upload order is a fact about a filename's
    # position, and on its own it is not evidence that a clip ends anything.
    if signal.motion is None or signal.motion > 0.4:
        return None
    confidence = 0.5
    evidence = ["last_in_sequence", "low_motion"]
    if signal.brightness is not None and signal.brightness < 0.4:
        evidence.append("darkening")
        confidence += 0.1
    return DetectedEvent(EditorialEvent.ENDING, round(min(confidence, 0.6), 3), tuple(evidence))


_RULES = (
    _establishing,
    _close_up,
    _subject_entry,
    _subject_exit,
    _action,
    _peak_motion,
    _impact,
    _reaction,
    _celebration,
    _dialogue,
    _transition_moment,
    _ending,
)


@dataclass(frozen=True, slots=True)
class ClipReading:
    """One clip's signals and everything observed about it.

    The unit the rest of the editorial pipeline passes around: the selection
    engine ranks these, the story builder assigns roles to these, and the
    decision engine explains itself in terms of these.
    """

    signals: EditorialSignals
    events: tuple[DetectedEvent, ...] = field(default_factory=lambda: (UNKNOWN_EVENT,))

    @property
    def media_id(self) -> MediaId:
        return self.signals.media_id

    @property
    def energy(self) -> float:
        return self.signals.energy

    def confidence_for(self, event: EditorialEvent) -> float:
        """How strongly this clip is that kind of thing. 0.0 when it is not."""
        for detected in self.events:
            if detected.event is event:
                return detected.confidence
        return 0.0

    def has(self, event: EditorialEvent) -> bool:
        return self.confidence_for(event) > 0.0

    @property
    def primary(self) -> DetectedEvent:
        """The best-supported observation. Always present."""
        return self.events[0] if self.events else UNKNOWN_EVENT

    def as_payload(self) -> dict[str, Any]:
        return {
            "signals": self.signals.as_payload(),
            "events": [detected.as_payload() for detected in self.events],
        }


def read_board(board: SignalBoard) -> tuple[ClipReading, ...]:
    """Detect events for every clip on a board, with each one's context.

    The context for a clip is built from the board, so "the previous clip" means
    the previous clip *in upload order* rather than in whatever order the caller
    happened to hold them. Chronology is the one narrative fact the system has
    and it is not thrown away by an iteration accident.
    """
    ordered = sorted(board.signals, key=lambda s: s.sequence)
    readings: list[ClipReading] = []

    for index, signal in enumerate(ordered):
        previous = ordered[index - 1] if index > 0 else None
        context = EventContext(
            motion_rank=board.rank(signal.media_id, "motion"),
            energy_rank=board.rank(signal.media_id, "energy"),
            duration_rank=board.rank(signal.media_id, "duration_ms"),
            previous_energy=previous.energy if previous is not None else None,
            position=index / max(len(ordered) - 1, 1),
            is_last=index == len(ordered) - 1,
        )
        readings.append(ClipReading(signals=signal, events=detect_events(signal, context)))

    return tuple(readings)


__all__ = [
    "CLOSE_UP_FACE_AREA",
    "EVENT_VOCABULARY_VERSION",
    "MAX_EVENTS_PER_CLIP",
    "MIN_EVENT_CONFIDENCE",
    "SEMANTIC_DUPLICATE_SIMILARITY",
    "SEMANTIC_RELATED_SIMILARITY",
    "SIGNAL_VERSION",
    "UNKNOWN_EVENT",
    "ClipReading",
    "DetectedEvent",
    "EditorialEvent",
    "EditorialSignals",
    "EventContext",
    "SignalBoard",
    "board_from",
    "cosine",
    "detect_events",
    "read_board",
    "signals_for",
]
