"""The editorial planner: the whole chain, behind the Phase 4 ``Planner`` port.

    candidates
      -> select()            the unchanged usability floor and pHash dedup
      -> SignalBoard         normalised signals, compared across the project
      -> read_board()        events, with confidence and evidence
      -> policy              genre table, blended with a reference, a variant,
                             and any intent a model contributed
      -> PacingPlan          an energy curve turned into per-slot lengths
      -> decide()            creative selection, roles, trims, treatment
      -> EditPlan            the unchanged Phase 4 document
      -> the unchanged validator, timeline compiler and renderer

    EditorialPlanner  implements Planner
    IntentPlanner     implements Planner, with a model deciding direction only

Two things are worth stating plainly about the shape of this.

**It is the same port.** ``RulesEnginePlanner`` is untouched and still the
fallback; ``LlmPlanner`` is untouched and still the directive path. This is a
third implementation of an interface that already existed, so the validator, the
timeline compiler, the render worker, the version history and the co-editor all
receive exactly what they received before. Phase 11 changes how an edit is
*decided*, and changes nothing about how one is executed.

**The decisions are upstream of the plan, not derived from it.** The
``EditorialPlan`` -- roles, energies, reasons, metrics -- is built first and then
compiled into segments. It is recorded on the plan's metadata so the editor can
show why each clip is there, but the plan document itself remains the closed
Phase 4 vocabulary with nowhere to put a path or a command.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from visionforge.domain.decisions import (
    DECISION_ENGINE_VERSION,
    EditorialPlan,
    EngineInput,
    decide,
)
from visionforge.domain.editorial import (
    EVENT_VOCABULARY_VERSION,
    SIGNAL_VERSION,
    ClipReading,
    SignalBoard,
    board_from,
    read_board,
)
from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MIN_OUTPUT_MS,
    MIN_SEGMENT_MS,
    EditPlan,
    OutputSpec,
    Segment,
)
from visionforge.domain.ids import MediaId
from visionforge.domain.intent import (
    INTENT_PROMPT_VERSION,
    EditorialIntent,
    FootageSummary,
    IntentInvalidError,
    apply_intent,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
    parse_intent,
    summarise,
)
from visionforge.domain.llm import (
    LlmProvider,
    LlmRequest,
    ProviderPermanentError,
    ProviderTransientError,
    ProviderUnavailableError,
)
from visionforge.domain.llm_planner import FallbackReason, LlmPlanRejectedError, LlmRunRecord
from visionforge.domain.metrics import EditMetrics, evaluate
from visionforge.domain.pacing import PacingPlan, PacingShape, curve_for, plan_pacing
from visionforge.domain.planner import (
    NoUsableMediaError,
    Planner,
    PlanOutcome,
    PlanRequest,
    build_music_cue,
)
from visionforge.domain.policy import StylePolicy
from visionforge.domain.policy import policy_for as style_policy_for
from visionforge.domain.selection import (
    Candidate,
    Rejection,
    RejectionReason,
    ScoredCandidate,
    SelectionResult,
    SelectionWeights,
    affinity_component,
    select,
    weights_payload,
)
from visionforge.domain.story import (
    POLICY_VERSION,
    EditorialPolicy,
    policy_for,
    with_bounds,
)
from visionforge.domain.variants import apply_variant, variant_for

logger = logging.getLogger(__name__)

#: How many clips are considered by the editorial engine at most. The usability
#: gate runs over everything; this caps what reaches the O(n^2) semantic
#: comparisons, which at 60 clips is 1770 dot products over 512 floats -- a few
#: milliseconds, and the cap exists so that number cannot become 200,000 on a
#: library somebody bulk-imported.
MAX_CONSIDERED = 60

#: How many times the arc may be re-fitted to a smaller clip count before the
#: engine stops trying. Three: each pass asks for strictly fewer slots, so the
#: sequence terminates on its own, and the bound is there so a pathological
#: project cannot turn "try again smaller" into a hang.
_MAX_ARC_FITS = 3


@dataclass(frozen=True, slots=True)
class EditorialOutcome:
    """An editorial plan and its metrics, before either becomes an ``EditPlan``.

    The two are separate objects because ``metrics`` measures ``plan`` -- putting
    the measurement inside the thing measured would make the module that
    evaluates a plan a dependency of the module that defines one, which is a
    cycle and, worse, an invitation to compute a metric during construction.
    """

    plan: EditorialPlan
    metrics: EditMetrics
    board: SignalBoard
    readings: tuple[ClipReading, ...]
    policy: EditorialPolicy
    #: How the arc and the clock were reconciled: how many shots were asked for,
    #: how many the footage supported, whether the policy's hold had to be
    #: widened to reach the requested length, and what is still missing. An edit
    #: that came in short should say so rather than leaving the user to notice.
    fit: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "fit": self.fit,
            "signal_version": SIGNAL_VERSION,
            "events_version": EVENT_VOCABULARY_VERSION,
            "policy_version": POLICY_VERSION,
            "engine_version": DECISION_ENGINE_VERSION,
            **self.plan.as_payload(),
            "policy_detail": self.policy.as_payload(),
            "board": self.board.as_payload(),
            "metrics": self.metrics.as_payload(),
        }


class EditorialPlanner:
    """The deterministic editorial engine. No model, no network, no randomness.

    Same input, same edit, forever -- which is what makes a variant safe to
    preview without persisting it, and what makes a regression attributable to a
    table change rather than to chance.
    """

    name = "editorial"

    def __init__(self, weights: SelectionWeights | None = None) -> None:
        self._weights = weights
        #: The four versions that together determine this planner's output. A
        #: single string, because "which build made this edit" is one question
        #: and answering it from four fields is how two of them drift.
        self.version = (
            f"{DECISION_ENGINE_VERSION}.{SIGNAL_VERSION}."
            f"{EVENT_VOCABULARY_VERSION}.{POLICY_VERSION}"
        )

    # ------------------------------------------------------------------ plan
    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        outcome, selection = self.compose(request, candidates)
        return PlanOutcome(
            plan=self.compile(request, outcome, selection),
            selection=selection,
        )

    # --------------------------------------------------------------- compose
    def compose(
        self,
        request: PlanRequest,
        candidates: list[Candidate],
        *,
        intent: EditorialIntent | None = None,
    ) -> tuple[EditorialOutcome, SelectionResult]:
        """Everything up to, but not including, the ``EditPlan``.

        Split out from ``plan`` because two callers want it without the
        compilation: the variants endpoint, which previews three editorial plans
        without writing any of them, and ``IntentPlanner``, which needs the
        footage summary before it can ask a model anything.
        """
        style_policy = request.style_policy or style_policy_for(request.style)
        selection = self._usable(request, candidates, style_policy)

        scores = {item.candidate.media_id: item for item in selection.selected}
        usable = [item.candidate for item in selection.selected][:MAX_CONSIDERED]
        if len(usable) < request.min_clips:
            raise NoUsableMediaError(
                f"only {len(usable)} usable clip(s) after selection; "
                f"at least {request.min_clips} required"
            )

        board = board_from(
            usable,
            scores=scores,
            style_matches=self._style_matches(usable, style_policy),
        )
        readings = read_board(board)

        policy = self._policy(request, style_policy, intent)
        intent_payload = self._intent_payload(request, intent, style_policy)
        count = self._clip_count(request, policy, len(readings))

        def run(wanted: int, limits: tuple[int, ...] = ()) -> EditorialPlan:
            return decide(
                EngineInput(
                    readings=readings,
                    board=board,
                    policy=policy,
                    pacing=self._pacing(request, policy, wanted, source_limits_ms=limits),
                    roles=policy.arc.expand(wanted),
                    variant=request.variant.value if request.variant else None,
                    intent=intent_payload,
                )
            )

        # Fit the arc to the footage, then fit the pacing to the arc.
        #
        # The clip count starts as an estimate from the target duration, and the
        # engine can fill fewer slots than that -- every remaining candidate may
        # repeat something already chosen. Two things go wrong if that is left
        # alone: the roles that were dropped are the *last* ones, so an edit
        # loses its ending rather than one of its three build shots; and the
        # pacing plan is still laid out for the larger count, so the edit comes
        # in short of the duration the user asked for.
        #
        # So the arc is re-expanded for the count actually achievable and the
        # engine run again. This converges -- each pass either matches or asks
        # for fewer -- and the loop is bounded anyway, because an unbounded
        # "try again smaller" is a hang waiting for a pathological project.
        requested = count
        editorial = run(count)
        for _ in range(_MAX_ARC_FITS):
            achieved = len(editorial.segments)
            if achieved >= count or achieved == 0:
                break
            count = achieved
            editorial = run(count)

        # Fewer shots than planned means each one has to be held longer to reach
        # the length the user asked for -- and a policy's ceiling can be lower
        # than that arithmetic needs. A fast-cut policy holding at most three
        # seconds cannot make thirty seconds out of five shots.
        #
        # The ceiling is widened rather than the target quietly abandoned,
        # because the target is what the user stated and the ceiling is what a
        # table preferred. It is recorded, so the plan says which of its own
        # numbers it had to move; and it is still bounded by the renderable
        # maximum, which nothing may widen.
        relaxed: int | None = None
        achieved = len(editorial.segments)
        if achieved:
            needed = -(-request.target_duration_ms // achieved)
            if needed > policy.max_clip_ms:
                # Widened past the bare arithmetic, by the curve's own spread.
                # Relaxing to exactly ``target / count`` would put every slot on
                # the new ceiling and flatten the pacing into the even split
                # this phase exists to replace -- the curve needs headroom above
                # the mean to put a long shot anywhere.
                relaxed = min(MAX_SEGMENT_MS, int(round(needed * (1.0 + policy.pacing_spread))))
                policy = with_bounds(
                    policy,
                    min_clip_ms=policy.min_clip_ms,
                    target_clip_ms=policy.target_clip_ms,
                    max_clip_ms=relaxed,
                )
                editorial = run(achieved)

        # Now the same arc, paced against the real source lengths. Selection is
        # identical -- it never reads a slot's duration -- so this pass is
        # strictly better timing for the same edit, never a different one.
        limits = tuple(board.by_id[segment.media_id].duration_ms for segment in editorial.segments)
        if limits:
            editorial = run(len(limits), limits)

        fit = {
            "requested_clips": requested,
            "achieved_clips": len(editorial.segments),
            "relaxed_max_clip_ms": relaxed,
            "target_ms": request.target_duration_ms,
            "achieved_ms": editorial.total_output_ms,
            "shortfall_ms": max(0, request.target_duration_ms - editorial.total_output_ms),
        }

        return (
            EditorialOutcome(
                plan=editorial,
                metrics=evaluate(editorial, board, policy),
                board=board,
                readings=readings,
                policy=policy,
                fit=fit,
            ),
            self._explained(selection, editorial),
        )

    # --------------------------------------------------------------- compile
    def compile(
        self, request: PlanRequest, outcome: EditorialOutcome, selection: SelectionResult
    ) -> EditPlan:
        """Turn editorial decisions into the Phase 4 plan document.

        Nothing creative happens here. Every number was decided upstream; this
        is the translation into the closed vocabulary the validator, the
        timeline compiler and the renderer already speak -- and the only place
        in the editorial chain that knows those types exist.
        """
        editorial = outcome.plan
        segments = [
            Segment(
                media_id=segment.media_id,
                order=index,
                source_in_ms=segment.source_in_ms,
                source_out_ms=segment.source_out_ms,
                transition_in=segment.transition,
                transition_ms=segment.transition_ms,
                effects=segment.effects,
            )
            for index, segment in enumerate(editorial.segments[:MAX_SEGMENTS])
        ]
        if len(segments) < request.min_clips:
            raise NoUsableMediaError(
                f"the editorial engine produced {len(segments)} segment(s); "
                f"at least {request.min_clips} required"
            )

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
        )

        total = plan.total_duration_ms
        if total < MIN_OUTPUT_MS or total > MAX_OUTPUT_MS:
            raise NoUsableMediaError(
                f"the editorial engine produced a {total} ms edit, outside "
                f"{MIN_OUTPUT_MS}-{MAX_OUTPUT_MS} ms"
            )

        grid = request.beats if request.beat_sync else None
        music = build_music_cue(request, grid, total)

        return EditPlan(
            project_id=plan.project_id,
            segments=plan.segments,
            output=plan.output,
            music=music,
            planner=self.name,
            planner_version=self.version,
            metadata={
                "strategy": "editorial decision engine",
                "style": request.style.value if request.style else None,
                "target_duration_ms": request.target_duration_ms,
                "considered": len(selection.selected) + len(selection.rejected),
                "selected": len(segments),
                "rejected": len(selection.rejected),
                "weights": weights_payload(self._selection_weights(request)),
                "style_policy": (
                    request.style_policy.as_payload()
                    if request.style_policy is not None and request.style_policy.is_styled
                    else None
                ),
                "beat_sync": editorial.pacing.beat_sync
                or {"applied": False, "reason": "not_requested", "confidence": None},
                # The whole editorial account of the edit, in one key. The plan
                # document above stays the Phase 4 vocabulary; this is the
                # explanation beside it, which is what the editor renders and
                # what the co-editor addresses roles through.
                "editorial": outcome.as_payload(),
            },
        )

    # ------------------------------------------------------------- internals
    def _selection_weights(self, request: PlanRequest) -> SelectionWeights:
        policy = request.style_policy or style_policy_for(request.style)
        if request.style is not None or policy.is_styled:
            return policy.weights
        return self._weights or SelectionWeights()

    def _usable(
        self, request: PlanRequest, candidates: list[Candidate], style_policy: StylePolicy
    ) -> SelectionResult:
        """The unchanged Phase 4 usability floor, over everything available.

        Run with a limit of *all* candidates rather than the clip budget, and
        the difference is the whole reason the editorial engine can do anything:
        the ranker's job here is to say which clips are watchable, not which
        clips are in the edit. Handing it the budget as a limit would mean
        creative selection got the top eight by technical score and could only
        reorder them -- which is exactly the failure this phase removes.
        """
        weights = self._selection_weights(request)
        return select(
            candidates,
            limit=max(1, len(candidates)),
            weights=weights,
            target=style_policy.target if weights.affinity > 0 else None,
        )

    @staticmethod
    def _style_matches(
        candidates: list[Candidate], style_policy: StylePolicy
    ) -> dict[MediaId, float]:
        """How close each clip looks to the reference, when there is one.

        Computed whether or not the affinity weight is non-zero, because the
        editorial engine has its own ``style_match`` term and a user who
        attached a reference should see it reported even at a strength where it
        does not move the technical ranking.
        """
        if not style_policy.target:
            return {}
        matches: dict[MediaId, float] = {}
        for candidate in candidates:
            value = affinity_component(candidate, style_policy.target)
            if value is not None:
                matches[candidate.media_id] = value
        return matches

    @staticmethod
    def _policy(
        request: PlanRequest, style_policy: StylePolicy, intent: EditorialIntent | None
    ) -> EditorialPolicy:
        """The policy this plan runs under, after everything has had its say.

        Order matters and is stated rather than emergent: the genre table first,
        then the reference video's measured pacing, then the variant, then the
        model's intent. Each later step is more specific to *this request* than
        the one before, and a model asked to interpret a reference should be
        able to override what the reference measured rather than the other way
        round.
        """
        policy = policy_for(request.editorial_policy, request.style)

        if style_policy.is_styled:
            policy = with_bounds(
                policy,
                min_clip_ms=style_policy.min_clip_ms,
                target_clip_ms=style_policy.target_clip_ms,
                max_clip_ms=style_policy.max_clip_ms,
            )

        variant = variant_for(request.variant)
        if variant is not None:
            policy = apply_variant(policy, variant)

        if intent is not None:
            policy = apply_intent(policy, intent)

        return policy

    @staticmethod
    def _clip_count(request: PlanRequest, policy: EditorialPolicy, available: int) -> int:
        """How many shots this edit gets.

        Derived from the *pacing*, not from the clip budget: a 30-second edit at
        a policy that holds 2.2 seconds a shot wants about fourteen shots, and
        the budget is a ceiling on that rather than a target. Before Phase 11
        the count was whatever ``max_clips`` said and the duration was divided
        among them, which is why an eight-clip default produced eight clips
        whether the style was "social" or "nature".
        """
        variant = variant_for(request.variant)
        budget = request.max_clips
        if variant is not None:
            budget = max(1, int(round(budget * variant.clip_budget_scale)))
        # The arc's own ceiling is part of the budget. An arc that allows one
        # hook and two peaks cannot carry twenty shots however many the caller
        # offers, and asking for them would lay out a pacing plan with slots
        # nothing is ever assigned to -- which comes back as an edit shorter
        # than the target for no reason the user can see.
        budget = min(budget, MAX_SEGMENTS, available, policy.arc.max_clips)

        wanted = max(1, round(request.target_duration_ms / max(policy.target_clip_ms, 1)))
        count = min(budget, wanted)
        return max(min(request.min_clips, budget), count)

    @staticmethod
    def _pacing(
        request: PlanRequest,
        policy: EditorialPolicy,
        count: int,
        *,
        source_limits_ms: tuple[int, ...] = (),
    ) -> PacingPlan:
        roles = policy.arc.expand(count)
        return plan_pacing(
            count=count,
            target_ms=request.target_duration_ms,
            curve=curve_for(policy.pacing, spread=policy.pacing_spread),
            min_clip_ms=policy.min_clip_ms,
            max_clip_ms=policy.max_clip_ms,
            emphasis=tuple(policy.emphasis_for(role) for role in roles),
            grid=request.beats,
            beat_sync=request.beat_sync,
            source_limits_ms=source_limits_ms,
        )

    @staticmethod
    def _intent_payload(
        request: PlanRequest, intent: EditorialIntent | None, style_policy: StylePolicy
    ) -> dict[str, Any]:
        """What was in play beyond the policy table, recorded on the plan.

        Always present, even with no model: ``reference_style`` is what
        ``metrics._style_adherence`` reads to decide whether style adherence is
        a measurement or an absence, and a key that appears only sometimes is a
        key every reader has to defend against.
        """
        payload: dict[str, Any] = {
            "reference_style": style_policy.is_styled,
            "style_strength": style_policy.strength.value,
            "beat_sync_requested": request.beat_sync,
        }
        if intent is not None:
            payload["model"] = intent.as_payload()
        return payload

    @staticmethod
    def _explained(selection: SelectionResult, editorial: EditorialPlan) -> SelectionResult:
        """The selection as the editor should see it.

        The technical selector's own list said every usable clip was "selected",
        which was true of *it* and is not true of the edit. This narrows the
        selected list to what the engine actually used and folds the editorial
        rejections in beside the usability ones, so a user reading "why is this
        clip not in my edit" gets one list with one answer per clip.
        """
        used = {segment.media_id for segment in editorial.segments}
        scored = {item.candidate.media_id: item for item in selection.selected}

        chosen: list[ScoredCandidate] = [
            scored[segment.media_id] for segment in editorial.segments if segment.media_id in scored
        ]

        editorial_rejections = [
            Rejection(
                media_id=rejection.media_id,
                reason=_REJECTION_REASONS.get(
                    rejection.reasons[0].value if rejection.reasons else "",
                    RejectionReason.NOT_NEEDED,
                ),
                detail=(
                    f"semantically close to {rejection.similar_to}"
                    if rejection.similar_to is not None
                    else "the story was already told"
                ),
            )
            for rejection in editorial.rejected
            if rejection.media_id not in used
        ]

        return SelectionResult(
            selected=tuple(chosen),
            rejected=(*selection.rejected, *editorial_rejections),
            duplicate_groups=selection.duplicate_groups,
        )


#: Editorial reason code -> the selection vocabulary the editor already renders.
#: A table rather than a cast, because the two enums answer different questions
#: and a silent string match between them would break the first time either
#: gained a member.
_REJECTION_REASONS = {
    "too_similar": RejectionReason.SEMANTIC_DUPLICATE,
    "duplicate_content": RejectionReason.SEMANTIC_DUPLICATE,
    "weak_quality": RejectionReason.WEAK_FOR_ROLE,
    "weak_role_fit": RejectionReason.WEAK_FOR_ROLE,
    "not_needed": RejectionReason.NOT_NEEDED,
}


# --------------------------------------------------------------- the AI path
#: Provider calls per plan, before the repair attempt. Two, matching the Phase 5
#: planner: one real attempt and one retry for a blip.
MAX_PROVIDER_ATTEMPTS = 2


@dataclass
class IntentRun(LlmRunRecord):
    """An ``LlmRunRecord`` that also remembers what direction was accepted.

    Subclassed rather than replaced so that ``EditService`` persists it, the
    ``llm_runs`` table receives it and the UI renders it with no change at all
    -- the existing provenance path is already the right shape for this and a
    second one would be a second thing to keep in sync.
    """

    intent: dict[str, Any] | None = field(default=None)

    def as_payload(self) -> dict[str, Any]:
        return {**super().as_payload(), "intent": self.intent}


class IntentPlanner:
    """Plans editorially, with a model deciding *direction* only.

    Implements ``Planner``. The difference from ``EditorialPlanner`` is one
    call and one object: a model is shown an aggregate summary of the footage
    and the user's request, and returns an ``EditorialIntent`` -- a policy name
    and six clamped numbers. Everything after that is the deterministic engine.

    Raises rather than falling back, exactly as ``LlmPlanner`` does. Falling
    back is ``FallbackPlanner``'s job, and keeping that decision out of here
    means this class has one behaviour and the fallback policy can change
    without touching it.
    """

    name = "editorial-ai"

    def __init__(
        self,
        provider: LlmProvider,
        *,
        engine: EditorialPlanner | None = None,
        timeout_s: float = 30.0,
        max_output_tokens: int = 1_000,
    ) -> None:
        self._provider = provider
        self._engine = engine or EditorialPlanner()
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self.version = f"{self._engine.version}+intent{INTENT_PROMPT_VERSION}"
        self.last_run: LlmRunRecord = IntentRun(
            provider=provider.name, model=provider.model, prompt_version=INTENT_PROMPT_VERSION
        )

    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        run = IntentRun(
            provider=self._provider.name,
            model=self._provider.model,
            prompt_version=INTENT_PROMPT_VERSION,
        )
        self.last_run = run

        # The footage is read *before* the model is asked, because the summary
        # is what the model is asked about. It also means a project with nothing
        # usable in it fails here, cheaply, rather than after a round trip.
        base, _selection = self._engine.compose(request, candidates)
        summary = self._summarise(request, base.readings)

        intent = self._ask(request, summary, base.policy, run)
        run.intent = intent.as_payload()

        outcome, selection = self._engine.compose(request, candidates, intent=intent)
        plan = self._engine.compile(request, outcome, selection)
        plan.metadata["planner"] = self.name
        plan.metadata["rationale"] = intent.rationale
        run.status = "ok"
        return PlanOutcome(plan=plan, selection=selection)

    # -------------------------------------------------------------- internals
    @staticmethod
    def _summarise(request: PlanRequest, readings: tuple[ClipReading, ...]) -> FootageSummary:
        grid = request.beats
        return summarise(
            readings,
            has_music=request.music_media_id is not None,
            bpm=round(grid.bpm, 2) if grid is not None else None,
            beat_confidence=round(grid.confidence, 3) if grid is not None else None,
            has_reference_style=(
                request.style_policy is not None and request.style_policy.is_styled
            ),
        )

    def _ask(
        self,
        request: PlanRequest,
        summary: FootageSummary,
        policy: EditorialPolicy,
        run: IntentRun,
    ) -> EditorialIntent:
        """One completion, one repair, then give up and let the caller fall back."""
        user_prompt = build_user_prompt(
            summary,
            policy,
            request_text=request.request_text,
            target_duration_ms=request.target_duration_ms,
            max_clips=request.max_clips,
        )
        run.request_digest = _digest(request.request_text)
        run.request_chars = len(request.request_text or "")

        text = self._call(user_prompt, run)
        try:
            return parse_intent(text)
        except IntentInvalidError as first:
            problems = [violation.message for violation in first.violations]
            run.violations = [violation.as_payload() for violation in first.violations]
            logger.warning(
                "editorial intent rejected; attempting one repair",
                extra={
                    "provider": self._provider.name,
                    "violations": [violation.code for violation in first.violations],
                },
            )

        text = self._call(build_repair_prompt(user_prompt, problems), run)
        try:
            intent = parse_intent(text)
        except IntentInvalidError as second:
            run.violations = [violation.as_payload() for violation in second.violations]
            raise LlmPlanRejectedError(
                FallbackReason.INVALID_OUTPUT,
                "; ".join(violation.message for violation in second.violations),
                violations=run.violations,
            ) from second

        run.violations = []
        return intent

    def _call(self, user_prompt: str, run: IntentRun) -> str:
        last: Exception | None = None
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            run.attempts = attempt
            started = time.perf_counter()
            try:
                response = self._provider.complete(
                    LlmRequest(
                        system=build_system_prompt(),
                        user=user_prompt,
                        max_output_tokens=self._max_output_tokens,
                        timeout_s=self._timeout_s,
                    )
                )
            except ProviderTransientError as exc:
                last = exc
                run.latency_ms += (time.perf_counter() - started) * 1000
                logger.warning(
                    "editorial intent provider transient failure",
                    extra={"provider": self._provider.name, "attempt": attempt},
                )
                continue
            except ProviderUnavailableError as exc:
                run.latency_ms += (time.perf_counter() - started) * 1000
                raise LlmPlanRejectedError(FallbackReason.PROVIDER_UNAVAILABLE, str(exc)) from exc
            except ProviderPermanentError as exc:
                run.latency_ms += (time.perf_counter() - started) * 1000
                raise LlmPlanRejectedError(FallbackReason.PROVIDER_ERROR, str(exc)) from exc

            run.latency_ms += response.latency_ms
            run.request_id = response.request_id
            run.input_tokens = _add(run.input_tokens, response.usage.input_tokens)
            run.output_tokens = _add(run.output_tokens, response.usage.output_tokens)
            return response.text

        raise LlmPlanRejectedError(
            FallbackReason.PROVIDER_ERROR,
            f"provider failed {MAX_PROVIDER_ATTEMPTS} time(s): {last}",
        )


def _add(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _digest(text: str | None) -> str | None:
    """A short, stable fingerprint of the request text. The text is not stored."""
    if not text:
        return None
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def editorial_of(plan: EditPlan) -> dict[str, Any] | None:
    """The editorial account recorded on a plan, when there is one.

    One accessor rather than four callers each reaching into ``metadata`` with a
    string key: the co-editor, the API serializer, the metrics endpoint and the
    tests all want this, and four spellings of ``metadata["editorial"]`` is four
    places to get the key wrong.
    """
    editorial = plan.metadata.get("editorial")
    return editorial if isinstance(editorial, dict) else None


def roles_of(plan: EditPlan) -> tuple[str, ...]:
    """The narrative role of each segment, in order. Empty for a Phase 4 plan.

    Read from the recorded editorial payload rather than recomputed, because the
    roles are a property of the decision that produced the plan and recomputing
    them would answer "what would we decide now", which is a different question
    and occasionally a different answer.
    """
    editorial = editorial_of(plan)
    if not editorial:
        return ()
    segments = editorial.get("segments")
    if not isinstance(segments, list):
        return ()
    roles: list[str] = []
    for segment in segments:
        role = segment.get("role") if isinstance(segment, dict) else None
        roles.append(str(role) if isinstance(role, str) else "")
    return tuple(roles)


__all__ = [
    "MAX_CONSIDERED",
    "MAX_PROVIDER_ATTEMPTS",
    "MIN_SEGMENT_MS",
    "EditorialOutcome",
    "EditorialPlanner",
    "IntentPlanner",
    "IntentRun",
    "PacingShape",
    "Planner",
    "editorial_of",
    "roles_of",
]
