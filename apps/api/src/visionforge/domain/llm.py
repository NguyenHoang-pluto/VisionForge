"""The LLM provider port.

    LlmProvider (Protocol)
    ├── AnthropicProvider    infra/llm
    ├── OpenAiProvider       infra/llm
    └── StubProvider         infra/llm -- deterministic, no network, for tests

The port is deliberately the narrowest thing that can express "send text, get
text back". It has no notion of tools, functions, streaming, images, or
conversation history, because the planner needs none of those and every one of
them is a capability an attacker would rather the model had.

**The model cannot act.** It receives a string and returns a string. Everything
that happens next -- parsing, resolving handles to media ids, clamping numbers,
validating the plan -- is ordinary deterministic code that does not consult the
model again. There is no path from a completion to a subprocess, a file, or a
request.

Providers live in ``infra`` because talking HTTP is infrastructure. The prompt,
the schema and the interpretation of the response are domain policy, so they
live here and in the modules alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from visionforge.domain.errors import PermanentError, TransientError


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """One completion request.

    ``max_output_tokens`` is a cost ceiling as much as a correctness one: a
    directive for forty clips fits comfortably in a couple of thousand tokens,
    so a response that wants more than that has gone wrong and should be cut off
    rather than paid for.
    """

    system: str
    user: str
    max_output_tokens: int = 2_000
    #: Zero, always, in this codebase. An edit plan is not a creative writing
    #: task at the token level -- the creativity is in which clips and what
    #: pacing, and sampling noise there just makes two identical requests
    #: produce two different edits for no benefit.
    temperature: float = 0.0
    timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """Token accounting, when the provider reports it.

    Every field is optional: providers differ in what they return, and a missing
    token count must not fail a request that otherwise succeeded.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def as_payload(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """What came back, plus everything worth recording about how it came back."""

    text: str
    provider: str
    model: str
    latency_ms: float
    usage: LlmUsage = field(default_factory=LlmUsage)
    #: The provider's own id for the call, when it gives one. Worth keeping: it
    #: is the only handle that lets a support conversation with a vendor refer
    #: to a specific request.
    request_id: str | None = None
    finish_reason: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            **self.usage.as_payload(),
        }


class LlmProvider(Protocol):
    """Sends a prompt and returns the completion. Nothing else.

    Synchronous on purpose. The call is blocking I/O with a hard timeout, and
    the application layer runs it on a worker thread rather than the event loop
    -- which keeps this interface implementable by anything, including a local
    function with no I/O at all.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LlmRequest) -> LlmResponse: ...


# ---------------------------------------------------------------------- errors
class ProviderUnavailableError(TransientError):
    """No provider is configured, or it cannot be reached at all.

    Transient because the cure is usually time or configuration, not a different
    request -- but the planner does not retry on it either way. It falls back,
    which is faster and always available.
    """

    code = "llm_provider_unavailable"


class ProviderTransientError(TransientError):
    """Rate limited, overloaded, timed out, or a 5xx. Worth one more attempt."""

    code = "llm_provider_transient"


class ProviderPermanentError(PermanentError):
    """Rejected the request: bad credentials, unknown model, malformed call.

    Retrying sends the identical request to the identical endpoint and gets the
    identical refusal, so the planner falls back immediately.
    """

    code = "llm_provider_permanent"


__all__ = [
    "LlmProvider",
    "LlmRequest",
    "LlmResponse",
    "LlmUsage",
    "ProviderPermanentError",
    "ProviderTransientError",
    "ProviderUnavailableError",
]
