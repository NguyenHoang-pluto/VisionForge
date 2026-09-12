"""Choosing which planner runs, and building it.

    mode + provider availability + what the user said
        -> ModeDecision   (pure, in the domain, testable)
        -> Planner        (built here, with the provider wired in)

The decision and the construction are split on purpose. ``resolve_mode`` is a
pure function in the domain with no knowledge of settings or HTTP, so "what
would automatic mode do given no provider and a written request?" is a unit test
with no fixtures. This module is the thin part that knows about configuration.

The rules engine is always constructed, whatever the mode, because it is always
the fallback. There is no configuration in which VisionForge cannot produce an
edit.
"""

from __future__ import annotations

import logging

from visionforge.core.config import Settings, get_settings
from visionforge.domain.llm import LlmProvider
from visionforge.domain.llm_planner import (
    FallbackPlanner,
    LlmPlanner,
    ModeDecision,
    PlannerMode,
    resolve_mode,
)
from visionforge.domain.planner import Planner, RulesEnginePlanner
from visionforge.domain.style import EditStyle

logger = logging.getLogger(__name__)


class PlannerSelection:
    """A planner, plus the decision that produced it.

    Both are needed downstream: the planner to run, and the decision to store on
    the plan and show to the user. Returning only the planner would leave the UI
    unable to explain why an AI request produced a rules plan.
    """

    def __init__(self, planner: Planner, decision: ModeDecision) -> None:
        self.planner = planner
        self.decision = decision

    @property
    def is_llm(self) -> bool:
        return isinstance(self.planner, FallbackPlanner)


def build_planner(
    *,
    mode: PlannerMode,
    style: EditStyle | None,
    request_text: str | None,
    provider: LlmProvider | None,
    settings: Settings | None = None,
) -> PlannerSelection:
    """Resolve the mode and construct the planner it calls for.

    An AI plan is always built as ``FallbackPlanner(LlmPlanner, rules)``, never
    as a bare ``LlmPlanner``. The fallback is not an error path that might be
    reached; it is part of what "AI planning" means here, and wiring it
    unconditionally means there is no configuration in which a provider failure
    becomes a failed request.
    """
    settings = settings or get_settings()
    rules = RulesEnginePlanner()

    decision = resolve_mode(
        mode,
        provider_available=provider is not None,
        request_text=request_text,
        style=style,
    )

    if decision.mode is PlannerMode.RULES or provider is None:
        return PlannerSelection(rules, decision)

    llm = LlmPlanner(
        provider,
        timeout_s=settings.llm_timeout_s,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    return PlannerSelection(FallbackPlanner(llm, rules), decision)


__all__ = ["PlannerMode", "PlannerSelection", "build_planner"]
