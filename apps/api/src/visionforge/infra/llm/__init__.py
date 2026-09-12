"""LLM provider adapters.

    LlmProvider (domain port)
    ├── AnthropicProvider
    ├── OpenAiProvider
    └── StubProvider          deterministic, no network, never in production

Adapters only. The prompt, the output schema and the interpretation of a
response are domain policy and live in ``visionforge.domain``.
"""

from visionforge.infra.llm.anthropic_provider import AnthropicProvider
from visionforge.infra.llm.factory import (
    SUPPORTED_PROVIDERS,
    ProviderMisconfiguredError,
    build_provider,
    describe_capabilities,
)
from visionforge.infra.llm.openai_provider import OpenAiProvider
from visionforge.infra.llm.stub_provider import StubProvider

__all__ = [
    "SUPPORTED_PROVIDERS",
    "AnthropicProvider",
    "OpenAiProvider",
    "ProviderMisconfiguredError",
    "StubProvider",
    "build_provider",
    "describe_capabilities",
]
