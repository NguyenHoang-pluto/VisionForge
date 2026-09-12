"""Provider construction from configuration.

The single place that reads LLM settings and turns them into an ``LlmProvider``.
Adding a provider means adding a branch here and a module beside it; nothing in
``api``, ``application`` or ``domain`` changes, which is the test of whether the
abstraction was worth having.

**Keys are read here and nowhere else.** They come from settings, which come
from the server environment. They are never returned by an endpoint, never
logged, never placed in plan metadata or the ``llm_runs`` table, and never sent
to the browser. :func:`describe_capabilities` is what the frontend gets: whether
a provider is configured, and its name and model. A model name is not a secret;
a key is, and there is no code path that serialises one.
"""

from __future__ import annotations

import logging

from visionforge.core.config import Settings, get_settings
from visionforge.domain.llm import LlmProvider
from visionforge.infra.llm.anthropic_provider import AnthropicProvider
from visionforge.infra.llm.openai_provider import OpenAiProvider
from visionforge.infra.llm.stub_provider import PROVIDER_NAME as STUB_NAME
from visionforge.infra.llm.stub_provider import StubProvider

logger = logging.getLogger(__name__)

#: Providers this build can construct.
SUPPORTED_PROVIDERS = ("anthropic", "openai", STUB_NAME)


class ProviderMisconfiguredError(RuntimeError):
    """Configuration asks for something this build cannot or must not do."""


def build_provider(settings: Settings | None = None) -> LlmProvider | None:
    """The configured provider, or ``None`` when AI planning is off.

    ``None`` is a supported, first-class state, not a failure: it is what every
    developer without an API key sees, and every code path above treats it as
    "plan with the rules engine" rather than as an error. That is why the
    feature degrades instead of breaking.
    """
    settings = settings or get_settings()

    if not settings.llm_enabled:
        return None

    provider = settings.llm_provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderMisconfiguredError(
            f"unknown LLM provider {provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if provider == STUB_NAME:
        # The stub answers without a model. Reaching production means real users
        # would be told an edit was AI-planned when it was arithmetic, so this
        # is a hard failure at construction rather than a warning nobody reads.
        if settings.environment == "production":
            raise ProviderMisconfiguredError(
                "the stub LLM provider must never be enabled in production"
            )
        logger.info("LLM planning enabled with the deterministic stub provider (no model)")
        return StubProvider()

    key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    if not key:
        # Enabled but unkeyed is a misconfiguration, not an outage. Logging it
        # and returning None keeps the app up and planning deterministically,
        # which is the right behaviour for a missing optional credential.
        logger.warning(
            "LLM planning is enabled but no API key is configured; "
            "falling back to the rules engine"
        )
        return None

    if provider == "anthropic":
        return AnthropicProvider(
            key,
            model=settings.llm_model or "claude-sonnet-5",
            base_url=settings.llm_base_url or "https://api.anthropic.com",
        )

    return OpenAiProvider(
        key,
        model=settings.llm_model or "gpt-4o-mini",
        base_url=settings.llm_base_url or "https://api.openai.com",
    )


def describe_capabilities(
    provider: LlmProvider | None = None, *, settings: Settings | None = None
) -> dict[str, object]:
    """What the frontend is allowed to know about the AI configuration.

    Takes the provider the caller will actually plan with, rather than building
    a fresh one from settings. Those two can differ -- under a dependency
    override in tests, and in any future deployment that resolves a provider
    per request -- and a capabilities endpoint that disagrees with the planner
    is worse than none: the UI would offer AI that does not run, or hide AI that
    does.

    Three booleans and two names. Enough for the UI to disable the AI option
    honestly and to label a plan with what produced it; not enough to learn
    anything that is not already public. **No key, by construction** -- there is
    no branch here that reads one.
    """
    if provider is None:
        settings = settings or get_settings()
        try:
            provider = build_provider(settings)
        except ProviderMisconfiguredError as exc:
            return {
                "ai_available": False,
                "provider": None,
                "model": None,
                "is_stub": False,
                "error": str(exc),
            }

    return {
        "ai_available": provider is not None,
        "provider": provider.name if provider else None,
        "model": provider.model if provider else None,
        "is_stub": bool(provider and provider.name == STUB_NAME),
        "error": None,
    }


__all__ = [
    "SUPPORTED_PROVIDERS",
    "ProviderMisconfiguredError",
    "build_provider",
    "describe_capabilities",
]
