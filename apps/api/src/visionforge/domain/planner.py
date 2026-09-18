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
from typing import Any, Protocol

from visionforge.domain.beats import BeatGrid
from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MIN_MUSIC_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    EditPlan,
    Encoder,
    FitMode,
    MusicCue,
    OutputSpec,
    QualityPreset,
    Segment,
    TransitionKind,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.policy import StylePolicy, policy_for
from visionforge.domain.selection import (
    DEFAULT_SELECTION_WEIGHTS,
    Candidate,
    ScoredCandidate,
    SelectionResult,
    SelectionWeights,
    select,
    weights_payload,
)
from visionforge.domain.stills import hold_span_ms, hold_start_ms, with_still_motion
from visionforge.domain.story import PolicyId
from visionforge.domain.style import EditStyle
from visionforge.domain.template import EditTemplate
from visionforge.domain.variants import VariantId


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
    encoder: Encoder = Encoder.CPU

    #: The style asked for, if any. ``None`` means "no stylistic bias", which is
    #: the Phase 4 behaviour and remains the default -- a style is something a
    #: caller opts into, never something inferred behind their back.
    style: EditStyle | None = None

    #: What the user wrote, if they wrote anything. Only the LLM planner reads
    #: it; the rules engine ignores it entirely rather than pattern-matching
    #: prose, which it would do badly.
    request_text: str | None = None

    # --- reference style (Phase 8) ---
    #: The named style blended with whatever a reference video measured, at the
    #: strength the user chose. ``None`` means "build one from the style alone",
    #: which is what every caller before Phase 8 effectively asked for and what
    #: keeps their plans identical.
    style_policy: StylePolicy | None = None

    #: The reference this project is styled after, recorded so the plan can say
    #: so. The planner never resolves it and never selects it -- the candidate
    #: list it is handed has already had it removed (see
    #: ``domain.reference.without_reference``). It is here for provenance, and
    #: because a request that mentions a reference should be able to prove which
    #: one without the plan carrying a media id into the renderer.
    reference_media_id: MediaId | None = None

    # --- music (Phase 7) ---
    #: The track to lay under the edit, if the caller chose one. The planner
    #: needs the id to write the cue; it never sees a path or a storage key.
    music_media_id: MediaId | None = None
    #: How long that track is, so the cue can be trimmed to something real.
    music_duration_ms: int | None = None
    #: Its beat grid, when one has been analysed. ``None`` means "plan as Phase
    #: 4 did", and so does a grid the domain considers unreliable -- the planner
    #: asks, it does not assume.
    beats: BeatGrid | None = None
    #: Whether to let those beats move the cuts. Off by default: adding music
    #: and re-timing the edit are separate decisions, and a user who wanted a
    #: bed under an edit they already liked should get that edit.
    beat_sync: bool = False
    music_gain: float = 0.7
    music_fade_in_ms: int = 0
    music_fade_out_ms: int = 1_500

    # --- editorial engine (Phase 11) ---
    #: Which editorial policy to plan under. ``None`` means "derive it from the
    #: style", which is what every caller before Phase 11 effectively asked for.
    #: Only ``EditorialPlanner`` reads it; the rules engine ignores it entirely,
    #: so a request carrying one still plans identically through the old path.
    editorial_policy: PolicyId | None = None

    #: Which named alternative to produce. ``None`` is the plain edit. A variant
    #: modifies the *policy*, so it changes which clips are chosen as well as how
    #: they are cut -- see ``domain.variants``.
    variant: VariantId | None = None

    # --- templates (Phase 12) ---
    #: The template to fill. Only ``EditorialPlanner`` reads it: the template
    #: decides how many slots there are, how long each is and how they join, and
    #: the engine decides which photo or video goes in each.
    template: EditTemplate | None = None

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
        policy = request.style_policy or policy_for(request.style)
        # A requested style, or a reference that actually moved something,
        # overrides the injected weights; with neither, the weights given at
        # construction win, so Phase 4 behaviour is untouched and an
        # explicitly-weighted planner still means what it says.
        styled = request.style is not None or policy.is_styled
        weights = policy.weights if styled else self._weights
        selection = select(
            candidates,
            limit=request.max_clips,
            weights=weights,
            # The look vector only matters when it was given weight to act
            # through, and passing it otherwise would compute a component
            # nothing multiplies.
            target=policy.target if weights.affinity > 0 else None,
        )

        if len(selection.selected) < request.min_clips:
            raise NoUsableMediaError(
                f"only {len(selection.selected)} usable clip(s) after selection; "
                f"at least {request.min_clips} required"
            )

        chosen = self._order(list(selection.selected), request.order)
        per_clip_ms = self._per_clip_duration(request, len(chosen), policy, styled)
        grid, beats_per_clip = self._beat_sync(request, per_clip_ms, policy, styled)

        segments: list[Segment] = []
        # Beats placed so far, and the timeline position that corresponds to.
        # Both are carried because the position is derived from the *total*
        # beat count: see ``BeatGrid.span_for_beats`` for why a running sum of
        # per-clip durations would drift off the grid.
        placed_beats = 0
        placed_ms = 0
        window: tuple[int, int] | None

        for index, scored in enumerate(chosen):
            if grid is not None and beats_per_clip is not None:
                fitted = self._beat_window(
                    scored.candidate,
                    grid,
                    placed_beats=placed_beats,
                    placed_ms=placed_ms,
                    wanted_beats=beats_per_clip,
                )
                # Advancing the running totals only on a clip that was actually
                # placed is what keeps the grid aligned across a dropped source.
                window = None if fitted is None else fitted[0]
                if fitted is not None:
                    placed_beats, placed_ms = fitted[1], fitted[2]
            else:
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
                    # A still is given a drift so it does not read as a freeze.
                    effects=with_still_motion(scored.candidate, (), index),
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
                effects=s.effects,
            )
            for i, s in enumerate(segments)
        ]

        timeline_ms = sum(segment.duration_ms for segment in segments)
        music = self._music_cue(request, grid, timeline_ms)

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
                encoder=request.encoder,
            ),
            music=music,
            planner=self.name,
            planner_version=self.version,
            metadata={
                "strategy": (
                    "beat-quantised even split" if grid is not None else "even-split centre trim"
                ),
                "style": request.style.value if request.style else None,
                "order": request.order.value,
                "target_duration_ms": request.target_duration_ms,
                "per_clip_ms": per_clip_ms,
                "considered": len(candidates),
                "selected": len(segments),
                "rejected": len(selection.rejected),
                "duplicate_groups": len(selection.duplicate_groups),
                "weights": weights_payload(weights),
                # What the reference was allowed to do, recorded on the plan.
                # An edit that cannot say which measurements moved its pacing is
                # not reviewable, and "style strength 75" alone does not say it.
                "style_policy": policy.as_payload() if policy.is_styled else None,
                # Why the cuts fall where they do, recorded rather than left to
                # be inferred. A beat-synced edit that cannot say which tempo it
                # was synced to is not reviewable.
                "beat_sync": self._beat_sync_payload(request, grid, beats_per_clip),
            },
        )
        return PlanOutcome(plan=plan, selection=selection)

    # ---------------------------------------------------------- beat sync
    @staticmethod
    def _beat_sync(
        request: PlanRequest, per_clip_ms: int, policy: StylePolicy, styled: bool
    ) -> tuple[BeatGrid | None, int | None]:
        """The grid to cut against, and how many beats each clip gets.

        Returns the integer beat count alongside the grid, because that integer
        is the thing the rest of planning needs: it is what the running total is
        kept in, and it is what the plan records. Deriving it back out of a
        duration is what made a plan claim 10.999 beats per clip.

        Four ways to get ``(None, None)``, deliberately indistinguishable
        downstream: the caller did not ask, nothing was analysed, the analysis
        is not trusted, or no whole number of beats fits between the style's
        pacing and what is renderable. All four mean "plan the way Phase 4
        planned", which is the graceful degradation this feature is required to
        have -- a spoken-word bed must not silently re-time an edit, and neither
        must a tempo so slow that one beat exceeds the segment ceiling.
        """
        if not request.beat_sync or request.beats is None:
            return None, None
        grid = request.beats
        if not grid.is_reliable():
            return None, None

        floor = max(MIN_SEGMENT_MS, policy.min_clip_ms if styled else MIN_SEGMENT_MS)
        ceiling = min(MAX_SEGMENT_MS, policy.max_clip_ms if styled else MAX_SEGMENT_MS)
        beats = grid.snap_beats(per_clip_ms, min_ms=floor, max_ms=ceiling)
        if beats is None:
            return None, None
        return grid, beats

    @staticmethod
    def _beat_window(
        candidate: Candidate,
        grid: BeatGrid,
        *,
        placed_beats: int,
        placed_ms: int,
        wanted_beats: int,
    ) -> tuple[tuple[int, int], int, int] | None:
        """A centre trim whose length is a whole number of beats.

        Returns the window plus the new running totals, or ``None`` when the
        source cannot supply even one beat's worth of legal footage.

        The length is measured as the *difference between two cumulative
        positions* rather than as one beat count times a period. That is what
        keeps cut number forty as close to its beat as cut number one: rounding
        `placed + n` beats and subtracting the rounding of `placed` beats can
        never drift, while adding up separately-rounded clip lengths drifts by
        up to half a millisecond every time.

        A source too short for the full allocation is given fewer whole beats
        rather than a truncated one. Truncating is what Phase 6's
        ``_trim_window`` does and it is right when nothing is quantised; here it
        would put the clip -- and therefore every cut after it -- off the grid.
        """
        source = hold_span_ms(candidate)
        if source < MIN_SEGMENT_MS:
            return None

        period = grid.period_ms
        if period <= 0:
            return None

        # The most whole beats this source could supply, so a short clip is
        # shortened *to a beat* instead of to an arbitrary length.
        affordable = int(source // period)
        beats = min(wanted_beats, affordable)

        while beats >= 1:
            end_ms = grid.span_for_beats(placed_beats + beats)
            duration = end_ms - placed_ms
            if MIN_SEGMENT_MS <= duration <= MAX_SEGMENT_MS and duration <= source:
                start = hold_start_ms(candidate, source, duration)
                return (start, start + duration), placed_beats + beats, end_ms
            beats -= 1
        return None

    # --------------------------------------------------------------- music

    @staticmethod
    def _music_cue(
        request: PlanRequest, grid: BeatGrid | None, timeline_ms: int
    ) -> MusicCue | None:
        return build_music_cue(request, grid, timeline_ms)

    @staticmethod
    def _beat_sync_payload(
        request: PlanRequest, grid: BeatGrid | None, beats_per_clip: int | None
    ) -> dict[str, Any]:
        """Why the cuts fall where they do.

        ``beats_per_clip`` is the integer the planner chose, carried here rather
        than recomputed from a duration. A plan that says "10.999 beats" is a
        plan whose own account of itself cannot be checked.
        """
        if grid is None or beats_per_clip is None:
            return {
                "applied": False,
                "reason": (
                    "not_requested"
                    if not request.beat_sync
                    else "no_beats"
                    if request.beats is None
                    else "low_confidence"
                    if not request.beats.is_reliable()
                    else "no_fitting_span"
                ),
                "confidence": request.beats.confidence if request.beats else None,
            }
        return {
            "applied": True,
            "bpm": round(grid.bpm, 2),
            "confidence": round(grid.confidence, 4),
            "first_beat_ms": grid.first_beat_ms,
            "beats_per_clip": beats_per_clip,
            "beat_period_ms": round(grid.period_ms, 4),
        }

    # ------------------------------------------------------------- internals
    @staticmethod
    def _order(chosen: list[ScoredCandidate], order: ClipOrder) -> list[ScoredCandidate]:
        if order is ClipOrder.SEQUENCE:
            return sorted(chosen, key=lambda s: s.candidate.sequence)
        # Sequence is the tie-break, so equal scores never reorder run to run.
        return sorted(chosen, key=lambda s: (-s.score, s.candidate.sequence))

    @staticmethod
    def _per_clip_duration(
        request: PlanRequest, clip_count: int, policy: StylePolicy, styled: bool
    ) -> int:
        """Target duration split evenly, clamped to the style then the plan.

        Clamping can make the total miss the target -- with two clips and a
        60-second target, ``MAX_SEGMENT_MS`` binds first. Honouring the per-clip
        bounds matters more: they are what keep a single clip from becoming the
        entire edit.

        Two clamps, in order. The style bounds express what the pacing should
        be -- as of Phase 8 they are the style blended with whatever a reference
        video measured -- and the plan bounds express what is renderable at all,
        applied last so neither a style nor a reference can widen them.

        A reference also *pulls* the duration before either clamp, in proportion
        to the strength the user chose and the confidence of the measurement.
        Clamping alone made the dial inert until a bound bit; see
        ``StylePolicy.pace``.

        Unchanged by Phase 7, and deliberately so: this is the unquantised
        length, which is both the answer when there is no music and the target
        that ``_beat_sync`` rounds to a whole number of beats.
        """
        raw = request.target_duration_ms // max(clip_count, 1)
        if styled:
            # Toward the reference's own shot length first -- by however much
            # the dial and that measurement's confidence allow -- and only then
            # clamped. Without the pull, a reference did nothing until the
            # bounds narrowed past the requested pacing and then did all of it
            # at once, which is a switch wearing a dial's clothes.
            raw = policy.clamp_clip_ms(policy.pace(raw))
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
        duration = hold_span_ms(candidate)
        if duration < MIN_SEGMENT_MS:
            return None

        take = min(per_clip_ms, duration)
        if take < MIN_SEGMENT_MS:
            return None

        start = hold_start_ms(candidate, duration, take)
        end = start + take
        if end > duration:
            start, end = max(0, duration - take), duration
        return start, end


def build_music_cue(
    request: PlanRequest, grid: BeatGrid | None, timeline_ms: int
) -> MusicCue | None:
    """The bed, trimmed to the cut it will play under.

    When there is a usable grid the cue starts at its first beat rather than at
    the head of the file, so the downbeat coincides with the first cut. Without
    one it starts at zero, which is the only defensible guess.

    A cue the validator would reject is worse than no cue: it would fail the
    whole plan over the bed rather than dropping it.

    Module-level since Phase 11, because the editorial planner needs exactly
    this and a second copy of it is a guarantee that the two planners will one
    day disagree about where the music starts.
    """
    if request.music_media_id is None:
        return None

    start = grid.first_beat_ms if grid is not None else 0
    available = request.music_duration_ms
    if available is not None:
        start = min(start, max(0, available - 1))
        end = min(available, start + timeline_ms)
    else:
        end = start + timeline_ms

    if end - start < MIN_MUSIC_MS:
        return None

    fade_out = min(request.music_fade_out_ms, max(0, (end - start) - request.music_fade_in_ms))
    return MusicCue(
        media_id=request.music_media_id,
        source_in_ms=start,
        source_out_ms=end,
        timeline_start_ms=0,
        gain=request.music_gain,
        fade_in_ms=request.music_fade_in_ms,
        fade_out_ms=fade_out,
    )


__all__ = [
    "ClipOrder",
    "EditStyle",
    "NoUsableMediaError",
    "PlanOutcome",
    "PlanRequest",
    "Planner",
    "RulesEnginePlanner",
    "build_music_cue",
]
