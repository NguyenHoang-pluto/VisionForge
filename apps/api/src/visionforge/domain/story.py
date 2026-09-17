"""Story arcs, narrative roles, and the editorial policies that configure them.

    StoryRole    what a clip is *for* in the edit
    StoryArc     the sequence of roles an edit is built from
    EditorialPolicy   one genre's answers: which arc, which pacing, how much
                      diversity, how long to hold, how willingly to cut

The central decision in this module is that **there is one pipeline and many
policies.** A football highlight and a nature sequence are not different
programs; they are the same program reading different numbers. Writing a second
pipeline for the second genre is how a codebase acquires ten subtly different
selection algorithms, nine of which are never fixed when the tenth is.

So a policy is a *table*, in the same spirit as ``STYLE_PROFILES``: a set of
numbers with a recognisable name, reviewable as a diff and testable as data.
Adding "wedding" later is an entry here, not a module.

**A role is not an event.** ``EditorialEvent`` says what a clip *is*
(``peak_motion``, ``close_up``); ``StoryRole`` says what the edit is *using it
for* (``PEAK``, ``REACTION``). The mapping between them is a policy's opinion --
football wants its peak to be the strongest motion, a product sequence wants its
peak to be the clearest close-up -- and keeping the two vocabularies separate is
what lets that opinion be configured instead of hard-coded.

Pure domain: tables, dataclasses and arithmetic. No I/O, no model, no FFmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from visionforge.domain.editorial import EditorialEvent
from visionforge.domain.pacing import DEFAULT_PACING_SPREAD, PacingShape
from visionforge.domain.style import EditStyle

#: Bumped when a policy's numbers or an arc's shape changes in a way that could
#: alter output. Recorded on every plan, for the same reason the prompt and the
#: analyzers are versioned.
POLICY_VERSION = "1"


class StoryRole(StrEnum):
    """What a clip is doing in the edit. A closed, ordered vocabulary.

    Six, in narrative order. The order is load-bearing: ``StoryArc`` lays roles
    out in the sequence declared here, and an arc that wanted ``PEAK`` before
    ``SETUP`` would be describing a different story rather than the same one
    shuffled.
    """

    #: The first thing the viewer sees, chosen to earn the second shot.
    HOOK = "hook"
    #: Context. Where we are, what we are looking at.
    SETUP = "setup"
    #: The middle, where energy accumulates.
    BUILD = "build"
    #: The moment the edit exists for.
    PEAK = "peak"
    #: The consequence -- a face, a celebration, a settling.
    REACTION = "reaction"
    #: How it stops.
    ENDING = "ending"


#: The order roles appear in an edit. Derived from the enum's declaration order
#: rather than retyped, so a seventh role cannot be added in the wrong place.
ROLE_ORDER: tuple[StoryRole, ...] = tuple(StoryRole)


@dataclass(frozen=True, slots=True)
class ArcSlot:
    """One role's place in an arc: how many clips, and what belongs there.

    ``affinity`` is the policy's opinion about which observations argue a clip
    belongs in this role, and by how much. It is a table rather than a rule
    because the same observation means different things to different genres: a
    ``close_up`` is a reaction in football, a peak in fashion, and a setup in a
    product sequence.
    """

    role: StoryRole
    #: Share of the clip budget this role wants, relative to the other slots.
    weight: float
    min_clips: int = 1
    max_clips: int = 3
    #: Which events argue for this role, 0..1 each.
    affinity: dict[EditorialEvent, float] = field(default_factory=dict)
    #: The energy this role wants from its footage, 0..1. Used to *choose*
    #: clips; the pacing curve independently decides how long they are held.
    energy: float = 0.5
    #: How much longer than the curve implies a clip here should be held. 1.0
    #: is "whatever pacing says"; above that is emphasis.
    emphasis: float = 1.0
    #: Higher survives when there are not enough clips for every slot.
    priority: int = 5

    def __post_init__(self) -> None:
        if self.min_clips < 0 or self.max_clips < self.min_clips:
            raise ValueError(f"{self.role.value}: max_clips must be at least min_clips")
        if self.weight <= 0:
            raise ValueError(f"{self.role.value}: weight must be positive")
        if not 0.0 <= self.energy <= 1.0:
            raise ValueError(f"{self.role.value}: energy must be between 0 and 1")

    def fit(self, events: dict[EditorialEvent, float]) -> float:
        """How well a clip's observations argue for this role, 0..1.

        The best single argument, not the sum. Summing would make a clip that
        weakly matches four criteria outrank one that strongly matches the
        criterion the role is actually about -- and "it is a bit like four
        things" is not a reason to make something the peak of an edit.
        """
        if not self.affinity or not events:
            return 0.0
        best = 0.0
        for event, weight in self.affinity.items():
            confidence = events.get(event, 0.0)
            best = max(best, confidence * weight)
        return round(min(best, 1.0), 4)


@dataclass(frozen=True, slots=True)
class StoryArc:
    """An ordered sequence of roles, with how many clips each wants."""

    name: str
    slots: tuple[ArcSlot, ...]

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("an arc needs at least one slot")
        seen = [slot.role for slot in self.slots]
        if len(set(seen)) != len(seen):
            raise ValueError("an arc cannot use the same role twice")
        if seen != sorted(seen, key=ROLE_ORDER.index):
            raise ValueError("arc slots must be in narrative order")

    @property
    def roles(self) -> tuple[StoryRole, ...]:
        return tuple(slot.role for slot in self.slots)

    def slot_for(self, role: StoryRole) -> ArcSlot | None:
        return next((slot for slot in self.slots if slot.role is role), None)

    @property
    def min_clips(self) -> int:
        return sum(slot.min_clips for slot in self.slots)

    @property
    def max_clips(self) -> int:
        """The most shots this arc can carry.

        A real ceiling, not a formality: an arc allows at most one hook and at
        most two peaks because an edit with four peaks has none. A caller asking
        for more shots than this gets this many, so the planner clamps its clip
        budget here rather than discovering the shortfall as a pacing plan with
        slots nothing was ever assigned to.
        """
        return sum(slot.max_clips for slot in self.slots)

    def expand(self, clip_count: int) -> tuple[StoryRole, ...]:
        """Which role each of ``clip_count`` positions carries, in order.

        Degrades rather than failing. With fewer clips than the arc has slots,
        the lowest-priority slots are dropped -- so a three-clip football edit
        is hook, peak, ending rather than a truncated six-part structure or an
        error. With more clips than the slots want, the surplus goes to the
        roles that scale: ``BUILD`` before ``SETUP``, never ``PEAK``, because an
        edit with four peaks has none.

        Deterministic: same count in, same roles out, every time.
        """
        if clip_count <= 0:
            return ()

        # --- who survives ---
        ordered = sorted(self.slots, key=lambda s: (-s.priority, ROLE_ORDER.index(s.role)))
        kept: list[ArcSlot] = []
        budget = clip_count
        for slot in ordered:
            need = max(slot.min_clips, 1)
            if need <= budget:
                kept.append(slot)
                budget -= need
        if not kept:
            # Fewer clips than even the top-priority slot wants. The single most
            # important role takes everything: a one-clip edit is its peak.
            kept = [ordered[0]]
            budget = max(0, clip_count - 1)

        allocation = {slot.role: max(slot.min_clips, 1) for slot in kept}

        # --- who grows ---
        # Largest-remainder over the slots' weights, so surplus distribution is
        # deterministic rather than dependent on float ordering.
        while budget > 0:
            candidates = [slot for slot in kept if allocation[slot.role] < slot.max_clips]
            if not candidates:
                break
            # Weight per clip already allocated: the slot furthest behind its
            # share gets the next clip.
            candidates.sort(
                key=lambda s: (-(s.weight / allocation[s.role]), ROLE_ORDER.index(s.role))
            )
            allocation[candidates[0].role] += 1
            budget -= 1

        roles: list[StoryRole] = []
        for slot in sorted(kept, key=lambda s: ROLE_ORDER.index(s.role)):
            roles.extend([slot.role] * allocation[slot.role])
        return tuple(roles[:clip_count])

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "roles": [slot.role.value for slot in self.slots],
            "slots": [
                {
                    "role": slot.role.value,
                    "weight": round(slot.weight, 3),
                    "min_clips": slot.min_clips,
                    "max_clips": slot.max_clips,
                    "energy": round(slot.energy, 3),
                    "emphasis": round(slot.emphasis, 3),
                    "prefers": sorted(event.value for event in slot.affinity),
                }
                for slot in self.slots
            ],
        }


# ------------------------------------------------------- creative selection
@dataclass(frozen=True, slots=True)
class CreativeWeights:
    """How a policy trades off the things selection has to balance.

    A convex combination, like ``SelectionWeights``, and for the same reason: a
    creative score is then always 0..1, two clips are comparable, and a typo
    fails at import rather than producing a quietly skewed edit.

    Note what ``quality`` is *not*. It is not the whole of selection -- it is
    one term among six, which is the entire point of this phase. Five
    technically excellent shots of the same wall must not become five sixths of
    an edit, and the only way that stops happening is for diversity to be able
    to outvote quality.
    """

    quality: float = 0.30
    #: How close the clip's own energy is to what its role wants.
    energy_fit: float = 0.15
    #: How unlike the already-chosen clips this one is.
    diversity: float = 0.25
    #: How strongly the clip's observed events argue for the role.
    role_fit: float = 0.20
    #: Resemblance to the project's reference video, when there is one.
    style_match: float = 0.05
    #: Semantic closeness to what the project is mostly about.
    relevance: float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.quality
            + self.energy_fit
            + self.diversity
            + self.role_fit
            + self.style_match
            + self.relevance
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"creative weights must sum to 1.0, got {total}")

    def as_payload(self) -> dict[str, float]:
        return {
            "quality": round(self.quality, 4),
            "energy_fit": round(self.energy_fit, 4),
            "diversity": round(self.diversity, 4),
            "role_fit": round(self.role_fit, 4),
            "style_match": round(self.style_match, 4),
            "relevance": round(self.relevance, 4),
        }


class PolicyId(StrEnum):
    """The editorial policies this build ships. A closed set."""

    FOOTBALL = "football"
    GAMING = "gaming"
    ANIME = "anime"
    CINEMATIC_TRAVEL = "cinematic_travel"
    NATURE = "nature"
    VLOG = "vlog"
    SOCIAL = "social"
    PRODUCT = "product"
    FASHION = "fashion"
    AUTOMOTIVE = "automotive"
    #: No genre opinion. The arc is still built and the pacing still curves --
    #: "neutral" means unopinionated, not "behave like Phase 4".
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class EditorialPolicy:
    """One genre's editorial opinions, as numbers."""

    id: PolicyId
    label: str
    #: One line, shown in the UI and given to the model as this policy's
    #: definition. Never parsed.
    description: str

    arc: StoryArc
    pacing: PacingShape
    pacing_spread: float = DEFAULT_PACING_SPREAD

    # --- how long a shot is held ---
    min_clip_ms: int = 800
    target_clip_ms: int = 2_500
    max_clip_ms: int = 8_000

    # --- selection ---
    weights: CreativeWeights = field(default_factory=CreativeWeights)
    #: The least distinct a clip may be from what is already chosen and still be
    #: taken. At 0.10 a clip 90% semantically identical to one already in the
    #: edit is refused -- which is exactly the "five near-identical clips"
    #: failure this phase exists to fix.
    min_diversity: float = 0.10
    #: Whether the finished edit should respect the order the clips were shot
    #: in. A match highlight that reorders the match is not a highlight of the
    #: match; a fashion sequence has no chronology to respect.
    prefer_chronological: bool = False

    # --- treatment ---
    #: How willingly this policy reaches for a dissolve rather than a cut, 0..1.
    transition_appetite: float = 0.2
    #: How willingly it puts an effect on a clip, 0..1.
    effect_appetite: float = 0.2
    #: Whether the peak should be slowed. A judgement per genre, not a global.
    slow_motion_peak: bool = False
    #: How much this policy wants cuts on the beat, 0..1. Advisory: it raises
    #: the suggestion, never switches beat sync on behind the user, for the same
    #: reason Phase 8 refused to -- adding music and re-timing an edit are
    #: separate decisions.
    beat_sync_preference: float = 0.3

    # --- defaults ---
    default_duration_ms: int = 25_000

    def __post_init__(self) -> None:
        if not self.min_clip_ms <= self.target_clip_ms <= self.max_clip_ms:
            raise ValueError(f"{self.id.value}: clip bounds must be ordered")
        if not 0.0 <= self.min_diversity <= 1.0:
            raise ValueError(f"{self.id.value}: min_diversity must be between 0 and 1")

    def emphasis_for(self, role: StoryRole) -> float:
        slot = self.arc.slot_for(role)
        return slot.emphasis if slot is not None else 1.0

    def priority_for(self, role: StoryRole) -> int:
        """How early this role gets to pick from the footage.

        Read by the decision engine, which fills slots in priority order rather
        than in narrative order. That is the difference between an edit whose
        peak is the best moment in the project and one whose peak is whatever
        was left after the setup had chosen: a human editor finds the money shot
        first and builds toward it, and filling the arc left to right does the
        opposite.
        """
        slot = self.arc.slot_for(role)
        return slot.priority if slot is not None else 0

    def energy_for(self, role: StoryRole) -> float:
        slot = self.arc.slot_for(role)
        return slot.energy if slot is not None else 0.5

    def fit_for(self, role: StoryRole, events: dict[EditorialEvent, float]) -> float:
        slot = self.arc.slot_for(role)
        return slot.fit(events) if slot is not None else 0.0

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": POLICY_VERSION,
            "id": self.id.value,
            "label": self.label,
            "description": self.description,
            "arc": self.arc.as_payload(),
            "pacing": self.pacing.value,
            "pacing_spread": round(self.pacing_spread, 3),
            "min_clip_ms": self.min_clip_ms,
            "target_clip_ms": self.target_clip_ms,
            "max_clip_ms": self.max_clip_ms,
            "weights": self.weights.as_payload(),
            "min_diversity": round(self.min_diversity, 3),
            "prefer_chronological": self.prefer_chronological,
            "transition_appetite": round(self.transition_appetite, 3),
            "effect_appetite": round(self.effect_appetite, 3),
            "slow_motion_peak": self.slow_motion_peak,
            "beat_sync_preference": round(self.beat_sync_preference, 3),
            "default_duration_ms": self.default_duration_ms,
        }


# ------------------------------------------------------------------- the arcs
#
# Arcs are shared between policies where the structure genuinely is the same.
# Three of them cover everything shipped here, which is itself evidence that the
# per-genre differences live in pacing, holding and selection rather than in
# structure -- and therefore that a per-genre pipeline would have been nine
# copies of one algorithm.

E = EditorialEvent

#: Something happens, and then somebody reacts to it. Sport, gaming, anything
#: with a moment the edit exists for.
EVENT_ARC = StoryArc(
    name="event",
    slots=(
        ArcSlot(
            role=StoryRole.HOOK,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.7,
            priority=8,
            affinity={E.ACTION: 0.9, E.PEAK_MOTION: 0.8, E.IMPACT: 0.8, E.CLOSE_UP: 0.4},
        ),
        ArcSlot(
            role=StoryRole.SETUP,
            weight=1.2,
            min_clips=1,
            max_clips=2,
            energy=0.3,
            priority=4,
            affinity={E.ESTABLISHING: 0.95, E.SUBJECT_ENTRY: 0.6, E.DIALOGUE: 0.4},
        ),
        ArcSlot(
            role=StoryRole.BUILD,
            weight=2.5,
            min_clips=1,
            max_clips=6,
            energy=0.65,
            priority=6,
            affinity={E.ACTION: 0.9, E.IMPACT: 0.7, E.TRANSITION_MOMENT: 0.3},
        ),
        ArcSlot(
            role=StoryRole.PEAK,
            weight=1.0,
            min_clips=1,
            max_clips=2,
            energy=0.95,
            emphasis=1.5,
            priority=10,
            affinity={E.PEAK_MOTION: 1.0, E.IMPACT: 0.85, E.ACTION: 0.6, E.CELEBRATION: 0.6},
        ),
        ArcSlot(
            role=StoryRole.REACTION,
            weight=1.0,
            min_clips=1,
            max_clips=2,
            energy=0.35,
            emphasis=1.2,
            priority=7,
            affinity={E.REACTION: 1.0, E.CELEBRATION: 0.9, E.CLOSE_UP: 0.7},
        ),
        ArcSlot(
            role=StoryRole.ENDING,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.25,
            priority=9,
            affinity={E.ENDING: 1.0, E.SUBJECT_EXIT: 0.7, E.ESTABLISHING: 0.5, E.CLOSE_UP: 0.4},
        ),
    ),
)

#: A place, observed. No single moment the edit exists for; the shape is a
#: breath rather than a spike.
OBSERVATIONAL_ARC = StoryArc(
    name="observational",
    slots=(
        ArcSlot(
            role=StoryRole.HOOK,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.45,
            priority=8,
            affinity={E.ESTABLISHING: 0.8, E.CLOSE_UP: 0.6, E.ACTION: 0.5},
        ),
        ArcSlot(
            role=StoryRole.SETUP,
            weight=1.8,
            min_clips=1,
            max_clips=3,
            energy=0.25,
            priority=7,
            affinity={E.ESTABLISHING: 1.0, E.SUBJECT_ENTRY: 0.5},
        ),
        ArcSlot(
            role=StoryRole.BUILD,
            weight=2.5,
            min_clips=1,
            max_clips=6,
            energy=0.5,
            priority=6,
            affinity={E.ACTION: 0.7, E.CLOSE_UP: 0.6, E.TRANSITION_MOMENT: 0.3},
        ),
        ArcSlot(
            role=StoryRole.PEAK,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.75,
            emphasis=1.35,
            priority=9,
            affinity={E.PEAK_MOTION: 0.9, E.CLOSE_UP: 0.7, E.ACTION: 0.7, E.CELEBRATION: 0.5},
        ),
        ArcSlot(
            role=StoryRole.ENDING,
            weight=1.2,
            min_clips=1,
            max_clips=2,
            energy=0.2,
            emphasis=1.25,
            priority=10,
            affinity={E.ENDING: 1.0, E.ESTABLISHING: 0.8, E.SUBJECT_EXIT: 0.6},
        ),
    ),
)

#: One subject, shown. No setup and no ending in the narrative sense: the first
#: frame is the pitch and the last is the last. Product, fashion, automotive.
SHOWCASE_ARC = StoryArc(
    name="showcase",
    slots=(
        ArcSlot(
            role=StoryRole.HOOK,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.6,
            priority=10,
            affinity={E.CLOSE_UP: 0.9, E.ESTABLISHING: 0.6, E.ACTION: 0.5},
        ),
        ArcSlot(
            role=StoryRole.BUILD,
            weight=3.0,
            min_clips=1,
            max_clips=8,
            energy=0.55,
            priority=8,
            affinity={E.CLOSE_UP: 0.8, E.ACTION: 0.6, E.ESTABLISHING: 0.5},
        ),
        ArcSlot(
            role=StoryRole.PEAK,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.8,
            emphasis=1.4,
            priority=9,
            affinity={E.CLOSE_UP: 1.0, E.PEAK_MOTION: 0.7, E.ACTION: 0.6},
        ),
        ArcSlot(
            role=StoryRole.ENDING,
            weight=1.0,
            min_clips=1,
            max_clips=1,
            energy=0.3,
            priority=7,
            affinity={E.ENDING: 0.9, E.ESTABLISHING: 0.7, E.CLOSE_UP: 0.6},
        ),
    ),
)


# --------------------------------------------------------------- the policies
EDITORIAL_POLICIES: dict[PolicyId, EditorialPolicy] = {
    PolicyId.FOOTBALL: EditorialPolicy(
        id=PolicyId.FOOTBALL,
        label="Football",
        description=(
            "Context, then a build of action into the strongest moment, then the "
            "reaction. Cut in the order it happened."
        ),
        arc=EVENT_ARC,
        pacing=PacingShape.RAMP,
        pacing_spread=0.65,
        min_clip_ms=1_000,
        target_clip_ms=2_200,
        max_clip_ms=6_000,
        # Diversity high: ten angles on one goal is the characteristic failure
        # of sports footage, and quality alone would select all ten.
        weights=CreativeWeights(
            quality=0.22,
            energy_fit=0.18,
            diversity=0.28,
            role_fit=0.24,
            style_match=0.04,
            relevance=0.04,
        ),
        min_diversity=0.12,
        prefer_chronological=True,
        transition_appetite=0.1,
        effect_appetite=0.5,
        slow_motion_peak=True,
        beat_sync_preference=0.5,
        default_duration_ms=30_000,
    ),
    PolicyId.GAMING: EditorialPolicy(
        id=PolicyId.GAMING,
        label="Gaming",
        description="Short shots, dense cutting, action emphasised, energy always rising.",
        arc=EVENT_ARC,
        pacing=PacingShape.BUILD,
        pacing_spread=0.55,
        min_clip_ms=700,
        target_clip_ms=1_400,
        max_clip_ms=3_500,
        weights=CreativeWeights(
            quality=0.24,
            energy_fit=0.26,
            diversity=0.22,
            role_fit=0.22,
            style_match=0.03,
            relevance=0.03,
        ),
        min_diversity=0.10,
        prefer_chronological=False,
        transition_appetite=0.05,
        effect_appetite=0.45,
        slow_motion_peak=True,
        beat_sync_preference=0.8,
        default_duration_ms=25_000,
    ),
    PolicyId.ANIME: EditorialPolicy(
        id=PolicyId.ANIME,
        label="Anime / AMV",
        description="Rhythmic mid-length cuts locked to the music, colour over sharpness.",
        arc=EVENT_ARC,
        pacing=PacingShape.RAMP,
        pacing_spread=0.5,
        min_clip_ms=800,
        target_clip_ms=1_700,
        max_clip_ms=3_500,
        weights=CreativeWeights(
            quality=0.20,
            energy_fit=0.22,
            diversity=0.26,
            role_fit=0.24,
            style_match=0.04,
            relevance=0.04,
        ),
        min_diversity=0.12,
        transition_appetite=0.25,
        effect_appetite=0.3,
        slow_motion_peak=False,
        beat_sync_preference=0.9,
        default_duration_ms=30_000,
    ),
    PolicyId.CINEMATIC_TRAVEL: EditorialPolicy(
        id=PolicyId.CINEMATIC_TRAVEL,
        label="Cinematic travel",
        description=(
            "Long composed takes, a calm opening, one moment that lifts, and a "
            "held ending. Dissolves where they help."
        ),
        arc=OBSERVATIONAL_ARC,
        pacing=PacingShape.WAVE,
        pacing_spread=0.5,
        min_clip_ms=2_200,
        target_clip_ms=4_200,
        max_clip_ms=9_000,
        weights=CreativeWeights(
            quality=0.30,
            energy_fit=0.12,
            diversity=0.26,
            role_fit=0.20,
            style_match=0.07,
            relevance=0.05,
        ),
        min_diversity=0.15,
        prefer_chronological=True,
        transition_appetite=0.6,
        effect_appetite=0.35,
        slow_motion_peak=False,
        beat_sync_preference=0.15,
        default_duration_ms=45_000,
    ),
    PolicyId.NATURE: EditorialPolicy(
        id=PolicyId.NATURE,
        label="Nature",
        description="Long opening shots, slow pacing, few transitions, a calm energy curve.",
        arc=OBSERVATIONAL_ARC,
        pacing=PacingShape.WAVE,
        pacing_spread=0.40,
        min_clip_ms=2_800,
        target_clip_ms=5_200,
        max_clip_ms=11_000,
        weights=CreativeWeights(
            quality=0.32,
            energy_fit=0.10,
            diversity=0.28,
            role_fit=0.20,
            style_match=0.05,
            relevance=0.05,
        ),
        min_diversity=0.18,
        prefer_chronological=True,
        transition_appetite=0.35,
        effect_appetite=0.15,
        slow_motion_peak=False,
        beat_sync_preference=0.05,
        default_duration_ms=50_000,
    ),
    PolicyId.VLOG: EditorialPolicy(
        id=PolicyId.VLOG,
        label="Vlog",
        description="Chronological, people-led, conversational pacing with room to breathe.",
        arc=OBSERVATIONAL_ARC,
        pacing=PacingShape.STEADY,
        pacing_spread=0.35,
        min_clip_ms=1_500,
        target_clip_ms=3_000,
        max_clip_ms=7_000,
        weights=CreativeWeights(
            quality=0.26,
            energy_fit=0.12,
            diversity=0.24,
            role_fit=0.28,
            style_match=0.05,
            relevance=0.05,
        ),
        min_diversity=0.12,
        prefer_chronological=True,
        transition_appetite=0.2,
        effect_appetite=0.1,
        slow_motion_peak=False,
        beat_sync_preference=0.1,
        default_duration_ms=40_000,
    ),
    PolicyId.SOCIAL: EditorialPolicy(
        id=PolicyId.SOCIAL,
        label="Social",
        description="The strongest shot first, very short holds, and no preamble at all.",
        arc=EVENT_ARC,
        pacing=PacingShape.DECAY,
        pacing_spread=0.5,
        min_clip_ms=600,
        target_clip_ms=1_200,
        max_clip_ms=2_800,
        weights=CreativeWeights(
            quality=0.26,
            energy_fit=0.24,
            diversity=0.24,
            role_fit=0.20,
            style_match=0.03,
            relevance=0.03,
        ),
        min_diversity=0.10,
        prefer_chronological=False,
        transition_appetite=0.05,
        effect_appetite=0.3,
        slow_motion_peak=False,
        beat_sync_preference=0.7,
        default_duration_ms=15_000,
    ),
    PolicyId.PRODUCT: EditorialPolicy(
        id=PolicyId.PRODUCT,
        label="Product",
        description="Even, deliberate shots of one subject. Detail over movement.",
        arc=SHOWCASE_ARC,
        pacing=PacingShape.STEADY,
        pacing_spread=0.25,
        min_clip_ms=1_400,
        target_clip_ms=2_600,
        max_clip_ms=5_000,
        weights=CreativeWeights(
            quality=0.38,
            energy_fit=0.08,
            diversity=0.26,
            role_fit=0.18,
            style_match=0.05,
            relevance=0.05,
        ),
        min_diversity=0.14,
        transition_appetite=0.3,
        effect_appetite=0.35,
        slow_motion_peak=False,
        beat_sync_preference=0.25,
        default_duration_ms=20_000,
    ),
    PolicyId.FASHION: EditorialPolicy(
        id=PolicyId.FASHION,
        label="Fashion",
        description="Rhythmic, people-led, locked to the music, close framing favoured.",
        arc=SHOWCASE_ARC,
        pacing=PacingShape.BUILD,
        pacing_spread=0.55,
        min_clip_ms=700,
        target_clip_ms=1_500,
        max_clip_ms=3_200,
        weights=CreativeWeights(
            quality=0.26,
            energy_fit=0.18,
            diversity=0.24,
            role_fit=0.24,
            style_match=0.05,
            relevance=0.03,
        ),
        min_diversity=0.12,
        transition_appetite=0.15,
        effect_appetite=0.3,
        slow_motion_peak=False,
        beat_sync_preference=0.85,
        default_duration_ms=20_000,
    ),
    PolicyId.AUTOMOTIVE: EditorialPolicy(
        id=PolicyId.AUTOMOTIVE,
        label="Automotive",
        description="Detail, then movement, then one shot that shows the whole thing moving.",
        arc=SHOWCASE_ARC,
        pacing=PacingShape.RAMP,
        pacing_spread=0.5,
        min_clip_ms=1_200,
        target_clip_ms=2_400,
        max_clip_ms=5_500,
        weights=CreativeWeights(
            quality=0.30,
            energy_fit=0.18,
            diversity=0.24,
            role_fit=0.20,
            style_match=0.05,
            relevance=0.03,
        ),
        min_diversity=0.13,
        transition_appetite=0.2,
        effect_appetite=0.4,
        slow_motion_peak=True,
        beat_sync_preference=0.5,
        default_duration_ms=25_000,
    ),
    PolicyId.NEUTRAL: EditorialPolicy(
        id=PolicyId.NEUTRAL,
        label="Neutral",
        description="No genre opinion: an arc and a curve, with nothing weighted toward a style.",
        arc=OBSERVATIONAL_ARC,
        pacing=PacingShape.RAMP,
        pacing_spread=DEFAULT_PACING_SPREAD,
        min_clip_ms=1_200,
        target_clip_ms=3_000,
        max_clip_ms=7_000,
        weights=CreativeWeights(),
        min_diversity=0.12,
        transition_appetite=0.2,
        effect_appetite=0.2,
        slow_motion_peak=False,
        beat_sync_preference=0.3,
        default_duration_ms=25_000,
    ),
}


#: Which policy a named edit style means.
#:
#: A table rather than a rename, because the two vocabularies are not the same
#: shape and pretending otherwise would force one of them to grow members it
#: does not want. ``EditStyle`` is what the user picks in a chip row; a policy
#: is how the editorial engine behaves. A style maps onto a policy; a policy can
#: exist that no style names, which is how "product" and "automotive" ship
#: without adding two more chips nobody asked for.
STYLE_POLICIES: dict[EditStyle, PolicyId] = {
    EditStyle.CINEMATIC: PolicyId.CINEMATIC_TRAVEL,
    EditStyle.FAST_MONTAGE: PolicyId.SOCIAL,
    EditStyle.SPORTS_HIGHLIGHT: PolicyId.FOOTBALL,
    EditStyle.GAMING: PolicyId.GAMING,
    EditStyle.ANIME: PolicyId.ANIME,
    EditStyle.NATURE: PolicyId.NATURE,
    EditStyle.SOCIAL: PolicyId.SOCIAL,
    EditStyle.CUSTOM: PolicyId.NEUTRAL,
}


def policy_for(policy_id: PolicyId | None, style: EditStyle | None = None) -> EditorialPolicy:
    """The policy for a request.

    An explicit policy wins; otherwise the style names one; otherwise neutral.
    In that order, and deliberately: a user who chose a policy chose it, and a
    style is a default they may not have thought about.
    """
    if policy_id is not None:
        return EDITORIAL_POLICIES[policy_id]
    if style is not None:
        return EDITORIAL_POLICIES[STYLE_POLICIES[style]]
    return EDITORIAL_POLICIES[PolicyId.NEUTRAL]


def with_bounds(
    policy: EditorialPolicy, *, min_clip_ms: int, target_clip_ms: int, max_clip_ms: int
) -> EditorialPolicy:
    """A copy of a policy whose pacing bounds come from somewhere else.

    Exists for one caller: Phase 8's ``StylePolicy`` already blends a named
    style with a measured reference video, and when a user has turned that dial
    up those bounds are the ones the edit should respect. Overriding here rather
    than inside the engine keeps "where do the bounds come from" a single
    visible decision instead of a branch buried in the pacing call.
    """
    low = max(1, min(min_clip_ms, max_clip_ms))
    high = max(low, max(min_clip_ms, max_clip_ms))
    target = max(low, min(high, target_clip_ms))
    return replace(policy, min_clip_ms=low, target_clip_ms=target, max_clip_ms=high)


__all__ = [
    "EDITORIAL_POLICIES",
    "EVENT_ARC",
    "OBSERVATIONAL_ARC",
    "POLICY_VERSION",
    "ROLE_ORDER",
    "SHOWCASE_ARC",
    "STYLE_POLICIES",
    "ArcSlot",
    "CreativeWeights",
    "EditorialPolicy",
    "PolicyId",
    "StoryArc",
    "StoryRole",
    "policy_for",
    "with_bounds",
]
