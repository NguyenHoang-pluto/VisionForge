"""The Planner port, and the deterministic rules engine that implements it.

    Planner (Protocol)
    ├── RulesEnginePlanner    <- Phase 4, here
    └── LLMPlanner            <- Phase 5, same interface

The port exists now, before there is a second implementation, for a specific
reason: it forces the rules engine to produce the same artefact an LLM will have
to produce. When the LLM arrives it inherits the validation gate, the timeline
compiler and the renderer unchanged, and it has a baseline whose output can be
compared against its own.

The rules engine is not a placeholder to be thrown away. It is the fallback for
when a model is unavailable, rate-limited, or fails validation twice — so it has
to be good enough to ship on its own.

**No creativity is claimed here.** This picks technically stronger clips, drops
near-duplicates, trims to a budget and orders the result by a stated rule. That
is a defensible automatic first cut, not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    OutputSpec,
    QualityPreset,
    Segment,
    TransitionKind,
)
from visionforge.domain.ids import ProjectId
from visionforge.domain.selection import (
    DEFAULT_SELECTION_WEIGHTS,
    Candidate,
    ScoredCandidate,
    SelectionResult,
    SelectionWeights,
    select,
    weights_payload,
)
from visionforge.domain.style import EditStyle, StyleProfile, profile_for


class ClipOrder(StrEnum):
    """How selected clips are sequenced.

    Ordering is a stated rule rather than an emergent property, so that two runs
    over the same input produce the same edit and a change can be attributed.
    """

    #: Strongest first. Front-loads the best material, which is what a preview
    #: wants; it can read as front-heavy over a long sequence.
    SCORE_DESC = "score_desc"
    #: Upload order. Preserves whatever narrative the source folder had -- for a
    #: day's shooting, that is usually chronological and usually right.
    SEQUENCE = "sequence"


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """What the caller wants, independent of how a planner achieves it."""

    project_id: ProjectId
    target_duration_ms: int = 25_000
    max_clips: int = 8
    min_clips: int = 2
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    width: int = 1280
    height: int = 720
    fps: int = 30
    fit: FitMode = FitMode.COVER
    audio: AudioMode = AudioMode.NONE
    order: ClipOrder = ClipOrder.SCORE_DESC

    #: Encoder effort. Reaches the renderer through the plan's ``OutputSpec``;
    #: no planner ever sees a CRF.
    quality: QualityPreset = QualityPreset.BALANCED

    #: The style asked for, if any. ``None`` means "no stylistic bias", which is
    #: the Phase 4 behaviour and remains the default -- a style is something a
    #: caller opts into, never something inferred behind their back.
    style: EditStyle | None = None

    #: What the user wrote, if they wrote anything. Only the LLM planner reads
    #: it; the rules engine ignores it entirely rather than pattern-matching
    #: prose, which it would do badly.
    request_text: str | None = None

    def __post_init__(self) -> None:
        if self.max_clips < self.min_clips:
            raise ValueError("max_clips must be at least min_clips")
        if self.min_clips < 1:
            raise ValueError("min_clips must be positive")
        if self.target_duration_ms <= 0:
            raise ValueError("target_duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    """A plan plus the selection that produced it.

    The selection travels with the plan so the UI can show *why* a clip was
    dropped. An automatic edit that cannot explain its omissions is not
    reviewable.
    """

    plan: EditPlan
    selection: SelectionResult


class Planner(Protocol):
    """Turns analysed media into a declarative edit.

    An implementation may return only an ``EditPlan``. It has no way to express
    a command, a path, or anything executable -- the boundary is structural, not
    a matter of trust.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome: ...


class NoUsableMediaError(ValueError):
    """Nothing in the project survived selection."""


class RulesEnginePlanner:
    """Deterministic planner. No model, no network, no randomness.

    The strategy, in full:

    1. score and rank candidates, rejecting the unusable and the duplicated;
    2. take as many as the clip budget allows;
    3. divide the target duration evenly among them;
    4. trim each clip from its centre, where the usable material usually is;
    5. order the result by the requested rule.

    Even division rather than score-weighted division: a clip that scores 0.7
    is not obviously worth twice the screen time of one at 0.35, and pretending
    the scores carry that meaning would be dressing a heuristic up as judgement.

    **Styles apply here too.** When a style is requested, its profile supplies
    the ranking weights and the pacing bounds, so "Cinematic" produces long
    takes ranked on exposure whether or not a model was involved. That is what
    makes this a fallback rather than a downgrade: the user loses the
    interpretation of their sentence, not the style they chose.
    """

    name = "rules-engine"
    version = "1"

    def __init__(self, weights: SelectionWeights = DEFAULT_SELECTION_WEIGHTS) -> None:
        self._weights = weights

    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        profile = profile_for(request.style)
        # A requested style overrides the injected weights; with no style the
        # weights given at construction win, so Phase 4 behaviour is untouched
        # and an explicitly-weighted planner still means what it says.
        weights = profile.weights if request.style is not None else self._weights
        selection = select(candidates, limit=request.max_clips, weights=weights)

        if len(selection.selected) < request.min_clips:
            raise NoUsableMediaError(
                f"only {len(selection.selected)} usable clip(s) after selection; "
                f"at least {request.min_clips} required"
            )

        chosen = self._order(list(selection.selected), request.order)
        per_clip_ms = self._per_clip_duration(request, len(chosen), profile)

        segments: list[Segment] = []
        for index, scored in enumerate(chosen):
            window = self._trim_window(scored.candidate, per_clip_ms)
            if window is None:
                continue
            source_in, source_out = window
            segments.append(
                Segment(
                    media_id=scored.candidate.media_id,
                    order=index,
                    source_in_ms=source_in,
                    source_out_ms=source_out,
                    transition_in=TransitionKind.CUT,
                )
            )

        if len(segments) < request.min_clips:
            raise NoUsableMediaError(
                f"only {len(segments)} clip(s) yielded a usable trim window; "
                f"at least {request.min_clips} required"
            )

        # Renumber after any drop, so orders stay contiguous from zero.
        segments = [
            Segment(
                media_id=s.media_id,
                order=i,
                source_in_ms=s.source_in_ms,
                source_out_ms=s.source_out_ms,
                transition_in=s.transition_in,
            )
            for i, s in enumerate(segments)
        ]

        plan = EditPlan(
            project_id=request.project_id,
            segments=tuple(segments),
            output=OutputSpec(
                aspect_ratio=request.aspect_ratio,
                width=request.width,
                height=request.height,
                fps=request.fps,
                fit=request.fit,
                audio=request.audio,
                quality=request.quality,
            ),
            planner=self.name,
            planner_version=self.version,
            metadata={
                "strategy": "even-split centre trim",
                "style": request.style.value if request.style else None,
                "order": request.order.value,
                "target_duration_ms": request.target_duration_ms,
                "per_clip_ms": per_clip_ms,
                "considered": len(candidates),
                "selected": len(segments),
                "rejected": len(selection.rejected),
                "duplicate_groups": len(selection.duplicate_groups),
                "weights": weights_payload(self._weights),
            },
        )
        return PlanOutcome(plan=plan, selection=selection)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _order(chosen: list[ScoredCandidate], order: ClipOrder) -> list[ScoredCandidate]:
        if order is ClipOrder.SEQUENCE:
            return sorted(chosen, key=lambda s: s.candidate.sequence)
        # Sequence is the tie-break, so equal scores never reorder run to run.
        return sorted(chosen, key=lambda s: (-s.score, s.candidate.sequence))

    @staticmethod
    def _per_clip_duration(request: PlanRequest, clip_count: int, profile: StyleProfile) -> int:
        """Target duration split evenly, clamped to the style then the plan.

        Clamping can make the total miss the target -- with two clips and a
        60-second target, ``MAX_SEGMENT_MS`` binds first. Honouring the per-clip
        bounds matters more: they are what keep a single clip from becoming the
        entire edit.

        Two clamps, in order. The style bounds express what the pacing should
        be; the plan bounds express what is renderable at all, and they are
        applied last so no style can widen them.
        """
        raw = request.target_duration_ms // max(clip_count, 1)
        if request.style is not None:
            raw = profile.clamp_clip_ms(raw)
        return max(MIN_SEGMENT_MS, min(MAX_SEGMENT_MS, raw))

    @staticmethod
    def _trim_window(candidate: Candidate, per_clip_ms: int) -> tuple[int, int] | None:
        """Centre-weighted trim.

        Taking from the middle rather than the head: the opening of a handheld
        clip is disproportionately likely to contain the camera being raised,
        a focus hunt, or a fade. The centre is where the usable material is.

        Returns ``None`` when the source is too short to yield a legal segment,
        so the caller can drop it rather than emit something the validator would
        reject.
        """
        duration = candidate.duration_ms or 0
        if duration < MIN_SEGMENT_MS:
            return None

        take = min(per_clip_ms, duration)
        if take < MIN_SEGMENT_MS:
            return None

        start = max(0, (duration - take) // 2)
        end = start + take
        if end > duration:
            start, end = max(0, duration - take), duration
        return start, end


__all__ = [
    "ClipOrder",
    "EditStyle",
    "NoUsableMediaError",
    "PlanOutcome",
    "PlanRequest",
    "Planner",
    "RulesEnginePlanner",
]
