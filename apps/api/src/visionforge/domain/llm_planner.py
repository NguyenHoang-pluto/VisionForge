"""The LLM planner, and the fallback that wraps it.

    Planner (Protocol)
    ├── RulesEnginePlanner   deterministic, Phase 4, unchanged
    ├── LlmPlanner           this module
    └── FallbackPlanner      primary, then the rules engine, recording why

``LlmPlanner`` does the same job as the rules engine and returns the same type.
The difference is only in step 2 below:

1. run the deterministic selector -- unusable clips never reach the model;
2. ask a model which of the survivors to use, in what order, held how long;
3. resolve its handles, clamp its numbers, compile a plan;
4. validate that plan against the same gate every plan passes;
5. if any of 2-4 fails, hand over to the rules engine and record the reason.

Step 1 matters more than it looks. The model never sees a black frame, a blurred
frame, or a duplicate, because the deterministic gates already removed them. The
model is not being asked to be a quality checker; it is being asked for
judgement about material that is already known to be usable.

Step 4 is the same ``validate_plan`` the rules engine passes through, run here
rather than only in the service, so that a bad plan is caught while the fallback
is still available. By the time the service sees an outcome, the decision about
which planner produced it has already been made and recorded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from visionforge.domain.directive import (
    CompileContext,
    DirectiveInvalidError,
    EditDirective,
    compile_directive,
    parse_directive,
)
from visionforge.domain.editbrief import EditBrief, build_brief, clean_request_text
from visionforge.domain.editplan import MediaFact, validate_plan
from visionforge.domain.llm import (
    LlmProvider,
    LlmRequest,
    LlmResponse,
    ProviderPermanentError,
    ProviderTransientError,
    ProviderUnavailableError,
)
from visionforge.domain.planner import (
    ClipOrder,
    NoUsableMediaError,
    Planner,
    PlanOutcome,
    PlanRequest,
)
from visionforge.domain.policy import policy_for
from visionforge.domain.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_user_prompt,
)
from visionforge.domain.selection import Candidate, select
from visionforge.domain.style import infer_style, profile_for

logger = logging.getLogger(__name__)

#: Provider calls per plan, before the repair attempt. Two: one real attempt and
#: one retry for a blip. A third would mostly buy latency, and the rules engine
#: is a better answer than a slower one.
MAX_PROVIDER_ATTEMPTS = 2


class PlannerMode(StrEnum):
    """What the caller asked for.

    ``AUTOMATIC`` is not "always AI". It is "decide, and say what you decided" --
    resolved by :func:`resolve_mode`, which is a pure function precisely so the
    decision can be tested and shown rather than inferred from behaviour.
    """

    AUTOMATIC = "automatic"
    RULES = "rules"
    AI = "ai"


class PlannerEngine(StrEnum):
    """Which deterministic engine plans the edit.

    Two generations exist, and which one runs is a choice rather than a
    migration. ``EDITORIAL`` is Phase 11's five-stage engine and the default,
    because it is what automatic editing now means. ``RULES`` is Phase 4's even
    split with Phase 5's directive planner in front of it, kept reachable so the
    two can be compared against the same footage -- and so that a deployment
    that wants the older, blunter behaviour can still ask for it by name rather
    than by pinning an old build.

    The engine is orthogonal to :class:`PlannerMode`: the mode says whether a
    model is consulted, the engine says who turns the answer into clips.
    """

    EDITORIAL = "editorial"
    RULES = "rules"


class FallbackReason(StrEnum):
    """Why an AI plan became a rules plan. Always recorded, never swallowed."""

    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    INVALID_PLAN = "invalid_plan"
    NO_USABLE_MEDIA = "no_usable_media"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass(frozen=True, slots=True)
class ModeDecision:
    """The resolved mode, and the reason in words the UI can show."""

    mode: PlannerMode
    reason: str
    #: Present only when automatic mode guessed a style by keyword match, so the
    #: UI can say "matched: cinematic" rather than implying comprehension.
    inferred_style: Any = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "inferred_style": self.inferred_style.value if self.inferred_style else None,
        }


def resolve_mode(
    requested: PlannerMode,
    *,
    provider_available: bool,
    request_text: str | None,
    style: Any = None,
) -> ModeDecision:
    """Decide which planner runs. Pure, deterministic, and explainable.

    Automatic mode's rule, in order:

    - no provider configured -> rules, because there is no choice to make;
    - the user wrote a request -> AI, because interpreting prose is the one
      thing the rules engine genuinely cannot do;
    - a style was picked but nothing was written -> rules, because a style is
      already a complete instruction and the profile behind it is deterministic.
      Paying a model to re-derive numbers we already wrote down would be slower,
      costlier and less predictable for no gain;
    - nothing stated at all -> rules.

    The third case is the interesting one. It is tempting to route everything to
    the model, but "Cinematic, 30 seconds" contains no ambiguity for a model to
    resolve.
    """
    if requested is PlannerMode.RULES:
        return ModeDecision(PlannerMode.RULES, "rules engine requested")

    if requested is PlannerMode.AI:
        if not provider_available:
            return ModeDecision(
                PlannerMode.RULES,
                "AI requested but no provider is configured; using the rules engine",
            )
        return ModeDecision(PlannerMode.AI, "AI planner requested")

    # --- automatic ---
    cleaned = clean_request_text(request_text)
    if not provider_available:
        inferred = infer_style(cleaned) if cleaned else None
        if inferred is not None and style is None:
            return ModeDecision(
                PlannerMode.RULES,
                f"no AI provider configured; matched the style {inferred.value!r} by keyword",
                inferred_style=inferred,
            )
        return ModeDecision(PlannerMode.RULES, "no AI provider configured")

    if cleaned:
        return ModeDecision(PlannerMode.AI, "a written request needs interpreting")

    if style is not None:
        return ModeDecision(
            PlannerMode.RULES,
            "a chosen style is already a complete instruction; the rules engine is "
            "deterministic and immediate",
        )

    return ModeDecision(PlannerMode.RULES, "nothing to interpret")


@dataclass
class LlmRunRecord:
    """Everything worth knowing about one planning attempt.

    Observability is a first-class output here, not a log line. The record is
    returned alongside the plan, persisted, and shown in the UI, because an
    automatic edit whose provenance is invisible cannot be trusted or debugged.

    Note what is *not* here: the prompt, the completion, and the user's request
    text. Only a digest of the request is kept. Retaining user content to debug
    a token count is not a trade worth making, and a digest is enough to tell
    two runs apart or to correlate a repeat.
    """

    provider: str | None = None
    model: str | None = None
    prompt_version: str = PROMPT_VERSION
    status: str = "ok"
    attempts: int = 0
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    request_digest: str | None = None
    request_chars: int = 0
    fallback_reason: FallbackReason | None = None
    fallback_detail: str | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    directive: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "status": self.status,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_id": self.request_id,
            # The fingerprint travels; the text it fingerprints does not. Both
            # the plan metadata and the llm_runs row are built from this
            # payload, so anything omitted here is simply never persisted.
            "request_digest": self.request_digest,
            "request_chars": self.request_chars,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "fallback_detail": self.fallback_detail,
            "violations": self.violations,
        }


class LlmPlanRejectedError(Exception):
    """The model's output could not be turned into a valid plan.

    Carries the reason so the fallback can record *which* failure happened, not
    merely that one did. "The provider timed out" and "the model referenced a
    clip that does not exist" call for different responses from whoever reads
    the dashboard.
    """

    def __init__(
        self,
        reason: FallbackReason,
        detail: str,
        *,
        violations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.violations = violations or []
        super().__init__(f"{reason.value}: {detail}")


class LlmPlanner:
    """Plans by asking a model which clips to use. Implements ``Planner``.

    Holds no session, no repository and no HTTP client: a provider, some
    weights, and the prompt. That is what lets the whole class be tested against
    a five-line fake provider with no network and no database.
    """

    name = "llm"

    def __init__(
        self,
        provider: LlmProvider,
        *,
        timeout_s: float = 30.0,
        max_output_tokens: int = 2_000,
    ) -> None:
        self._provider = provider
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        #: The prompt version, not a model version. What changes the output of
        #: this planner is the prompt and the parsing; the model name is
        #: recorded separately on every run.
        self.version = PROMPT_VERSION
        self.last_run = LlmRunRecord(provider=provider.name, model=provider.model)

    # ------------------------------------------------------------------- plan
    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        """Select, ask, compile, validate. Raises rather than falling back.

        Falling back is ``FallbackPlanner``'s job. Keeping that decision out of
        here means this class has exactly one behaviour -- produce an LLM plan or
        say why it could not -- and the fallback policy can change without
        touching the planning logic.
        """
        run = LlmRunRecord(provider=self._provider.name, model=self._provider.model)
        self.last_run = run

        profile = profile_for(request.style)
        # The same policy the rules engine plans against, so a fallback produces
        # an edit with the same pacing rather than a differently-styled one.
        policy = request.style_policy or policy_for(request.style)
        selection = select(
            candidates,
            limit=request.max_clips,
            weights=policy.weights,
            target=policy.target if policy.weights.affinity > 0 else None,
        )
        if len(selection.selected) < request.min_clips:
            raise LlmPlanRejectedError(
                FallbackReason.NO_USABLE_MEDIA,
                f"only {len(selection.selected)} usable clip(s) after selection",
            )

        brief = build_brief(selection)
        cleaned = clean_request_text(request.request_text)
        run.request_digest = _digest(cleaned)
        run.request_chars = len(cleaned or "")

        user_prompt = build_user_prompt(
            brief,
            profile,
            request_text=cleaned,
            style=request.style,
            target_duration_ms=request.target_duration_ms,
            max_clips=request.max_clips,
            policy=policy,
        )

        directive, response = self._ask(user_prompt, brief, request, run)
        run.directive = directive.as_payload()

        plan = compile_directive(
            directive,
            CompileContext(
                project_id=request.project_id,
                aspect_ratio=request.aspect_ratio,
                width=request.width,
                height=request.height,
                fps=request.fps,
                fit=request.fit,
                audio=request.audio,
                quality=request.quality,
                profile=profile,
                handles=brief.handles,
                source_durations={
                    candidate.media_id: candidate.duration_ms or 0 for candidate in candidates
                },
                target_duration_ms=request.target_duration_ms,
                max_clips=request.max_clips,
                planner=self.name,
                planner_version=self.version,
                metadata={
                    "provider": response.provider,
                    "model": response.model,
                    "prompt_version": self.version,
                    "attempts": run.attempts,
                    "considered": len(candidates),
                    "offered": len(brief.clips),
                    "rejected": len(selection.rejected),
                    "requested_style": request.style.value if request.style else None,
                },
            ),
        )

        if len(plan.segments) < request.min_clips:
            raise LlmPlanRejectedError(
                FallbackReason.INVALID_PLAN,
                f"compiled to {len(plan.segments)} segment(s); at least "
                f"{request.min_clips} required",
            )

        # The same gate the service will run, run here while falling back is
        # still possible. Facts come from the candidates, which is the same
        # database read the selection was built from.
        facts = {
            candidate.media_id: MediaFact(
                media_id=candidate.media_id,
                project_id=request.project_id,
                is_renderable=candidate.is_ready,
                duration_ms=candidate.duration_ms,
                width=candidate.width,
                height=candidate.height,
            )
            for candidate in candidates
        }
        violations = validate_plan(plan, facts)
        if violations:
            raise LlmPlanRejectedError(
                FallbackReason.INVALID_PLAN,
                "; ".join(v.message for v in violations),
                violations=[v.as_payload() for v in violations],
            )

        run.status = "ok"
        return PlanOutcome(plan=plan, selection=selection)

    # -------------------------------------------------------------- internals
    def _ask(
        self,
        user_prompt: str,
        brief: EditBrief,
        request: PlanRequest,
        run: LlmRunRecord,
    ) -> tuple[EditDirective, LlmResponse]:
        """Call the provider, parse, and repair once if the parse failed."""
        response = self._call(user_prompt, run)

        try:
            directive = parse_directive(response.text, brief, requested_style=request.style)
            return directive, response
        except DirectiveInvalidError as first:
            # Bound outside the handler: Python clears the name at the end of an
            # except block, and the repair prompt needs the messages.
            problems = [v.message for v in first.violations]
            run.violations = [v.as_payload() for v in first.violations]
            logger.warning(
                "llm directive rejected; attempting one repair",
                extra={
                    "provider": self._provider.name,
                    "violations": [v.code for v in first.violations],
                },
            )

        repair = build_repair_prompt(user_prompt, problems)
        response = self._call(repair, run)
        try:
            directive = parse_directive(response.text, brief, requested_style=request.style)
        except DirectiveInvalidError as second:
            run.violations = [v.as_payload() for v in second.violations]
            raise LlmPlanRejectedError(
                FallbackReason.INVALID_OUTPUT,
                "; ".join(v.message for v in second.violations),
                violations=run.violations,
            ) from second

        run.violations = []
        return directive, response

    def _call(self, user_prompt: str, run: LlmRunRecord) -> LlmResponse:
        """One completion, with a bounded retry on transient provider failures.

        No backoff sleep. The caller is a user waiting on an HTTP response, and
        the alternative to waiting two seconds for a retry is a rules plan
        delivered immediately -- so a failed retry should cost as little as
        possible before handing over.
        """
        last: Exception | None = None
        for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            run.attempts = attempt
            started = time.perf_counter()
            try:
                response = self._provider.complete(
                    LlmRequest(
                        system=SYSTEM_PROMPT,
                        user=user_prompt,
                        max_output_tokens=self._max_output_tokens,
                        timeout_s=self._timeout_s,
                    )
                )
            except ProviderTransientError as exc:
                last = exc
                run.latency_ms += (time.perf_counter() - started) * 1000
                logger.warning(
                    "llm provider transient failure",
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
            return response

        raise LlmPlanRejectedError(
            FallbackReason.PROVIDER_ERROR,
            f"provider failed {MAX_PROVIDER_ATTEMPTS} time(s): {last}",
        )


@runtime_checkable
class RecordingPlanner(Protocol):
    """A planner that keeps a provenance record of its last attempt.

    Stated as a protocol rather than as ``LlmPlanner`` so that
    ``FallbackPlanner`` can wrap any planner that talks to a model and records
    what happened -- which since Phase 11 means the editorial intent planner as
    well. The fallback policy is about *a model failing*, not about which
    question the model was asked, and typing it to one implementation would have
    meant a second, identical fallback class for the second one.
    """

    name: str
    version: str
    last_run: LlmRunRecord

    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome: ...


class FallbackPlanner:
    """Runs a primary planner, and the rules engine when it fails.

    Implements ``Planner``, so the service cannot tell it apart from either of
    the planners inside it -- which is the point. The fallback is not an error
    path bolted on; it is the planner the system actually uses when AI is
    requested, and the AI is the optimisation inside it.

    The reason for every fallback is recorded on ``last_run`` and ends up in the
    plan metadata, the ``llm_runs`` table and the UI. A fallback that nobody can
    see is indistinguishable from an AI planner that quietly does nothing.
    """

    def __init__(self, primary: RecordingPlanner, fallback: Planner) -> None:
        self._primary = primary
        self._fallback = fallback
        # Named after whoever is actually in front, from construction rather
        # than from the first successful plan: a fallback that reports "llm"
        # while wrapping the editorial intent planner would make every plan's
        # provenance wrong until the first one succeeded.
        self.name = primary.name
        self.version = primary.version
        self.last_run: LlmRunRecord = primary.last_run

    def plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        try:
            outcome = self._primary.plan(request, candidates)
            self.last_run = self._primary.last_run
            self.name = self._primary.name
            self.version = self._primary.version
            return outcome
        except LlmPlanRejectedError as exc:
            run = self._primary.last_run
            run.status = "fallback"
            run.fallback_reason = exc.reason
            run.fallback_detail = exc.detail
            if exc.violations:
                run.violations = exc.violations
            self.last_run = run
            logger.warning(
                "llm planner fell back to the rules engine",
                extra={"reason": exc.reason.value, "detail": exc.detail},
            )
        except Exception as exc:
            run = self._primary.last_run
            run.status = "fallback"
            run.fallback_reason = FallbackReason.UNEXPECTED_ERROR
            run.fallback_detail = f"{type(exc).__name__}: {exc}"
            self.last_run = run
            logger.exception("llm planner raised unexpectedly; falling back")

        return self._fallback_plan(request, candidates)

    def _fallback_plan(self, request: PlanRequest, candidates: list[Candidate]) -> PlanOutcome:
        """The rules engine, with the style the user asked for still applied.

        This is why styles are deterministic profiles rather than prompt text: a
        fallback edit is still a *cinematic* edit, not a generic one. The user
        loses the interpretation of their sentence, not the style they picked.
        """
        outcome = self._fallback.plan(request, candidates)
        self.name = self._fallback.name
        self.version = self._fallback.version
        plan = outcome.plan
        plan.metadata["fallback_from"] = "llm"
        plan.metadata["fallback_reason"] = (
            self.last_run.fallback_reason.value if self.last_run.fallback_reason else None
        )
        plan.metadata["fallback_detail"] = self.last_run.fallback_detail
        return outcome


def _add(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _digest(text: str | None) -> str | None:
    """A short, stable fingerprint of the request text.

    Stored instead of the text itself. Enough to notice that the same request
    was made twice or to correlate a report with a run; useless for recovering
    what the user wrote, which is the intent.
    """
    if not text:
        return None
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "MAX_PROVIDER_ATTEMPTS",
    "ClipOrder",
    "FallbackPlanner",
    "FallbackReason",
    "LlmPlanRejectedError",
    "LlmPlanner",
    "LlmRunRecord",
    "ModeDecision",
    "NoUsableMediaError",
    "PlannerMode",
    "RecordingPlanner",
    "resolve_mode",
]
