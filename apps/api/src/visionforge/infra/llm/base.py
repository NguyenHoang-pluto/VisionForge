"""Shared HTTP plumbing for LLM providers.

One place decides how a network failure is classified, because that decision is
what the planner's retry and fallback logic is built on and it must not drift
between providers. A 429 is transient at Anthropic and transient at OpenAI; a
401 is permanent at both.

Everything here is deliberately small. Providers are thin because the interface
they implement is thin: text in, text out, no tools, no streaming, no
conversation state.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from visionforge.domain.llm import (
    LlmRequest,
    LlmResponse,
    ProviderPermanentError,
    ProviderTransientError,
)

logger = logging.getLogger(__name__)

#: Status codes worth another attempt. 408 and 409 are included because some
#: gateways use them for "busy, come back"; 5xx is handled by range below.
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})


def classify_status(status: int, body: str) -> Exception:
    """Turn an HTTP failure into the right domain error.

    The distinction drives everything downstream: a transient error is retried
    once and then falls back, a permanent one falls back immediately. Getting it
    backwards means either burning latency on a hopeless retry or giving up on a
    request that would have succeeded.

    The body is truncated hard. Provider error bodies can echo request content,
    and this string ends up in logs.
    """
    detail = body.strip()[:200]
    if status in _TRANSIENT_STATUSES or status >= 500:
        return ProviderTransientError(
            f"provider returned {status}",
            hint="The provider is busy or unavailable; the rules engine will plan instead.",
        )
    return ProviderPermanentError(
        f"provider rejected the request ({status}): {detail}",
        hint="Check the configured model name and API key.",
    )


def post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[dict[str, Any], float, str | None]:
    """POST JSON, return the parsed body, the latency and the request id.

    ``follow_redirects`` is off and no proxy is configured: this call goes to the
    endpoint the server was configured with and nowhere else. A provider that
    starts answering with a 302 is a provider that has gone wrong, not one to
    follow somewhere new.
    """
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderTransientError(
            f"provider timed out after {timeout_s:.0f}s",
            hint="The rules engine will plan instead.",
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderTransientError(f"provider unreachable: {exc}") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    request_id = response.headers.get("request-id") or response.headers.get("x-request-id")

    if response.status_code >= 400:
        raise classify_status(response.status_code, response.text)

    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderTransientError("provider returned a non-JSON body") from exc
    if not isinstance(body, dict):
        raise ProviderTransientError("provider returned an unexpected body shape")

    return body, latency_ms, request_id


def redact(headers: dict[str, str]) -> dict[str, str]:
    """Headers with credentials removed, for logging.

    Never log a header map directly. An API key in a log file is an API key in
    every backup, every log shipper and every screenshot of a terminal.
    """
    hidden = {"authorization", "x-api-key", "api-key"}
    return {k: ("<redacted>" if k.lower() in hidden else v) for k, v in headers.items()}


def describe(request: LlmRequest) -> str:
    """A short, content-free description of a request, safe to log."""
    return json.dumps(
        {
            "system_chars": len(request.system),
            "user_chars": len(request.user),
            "max_output_tokens": request.max_output_tokens,
            "timeout_s": request.timeout_s,
        },
        separators=(",", ":"),
    )


__all__ = ["LlmRequest", "LlmResponse", "classify_status", "describe", "post_json", "redact"]
