"""Anthropic Messages API provider.

Raw HTTP rather than the vendor SDK. Three reasons, in order of weight:

- the SDK pulls its own dependency tree into a process that also runs FFmpeg and
  PyTorch, and the surface it adds is a retry policy and a streaming client that
  this code deliberately does not use;
- the port is text in, text out, and an SDK is a poor fit for an interface
  narrower than the one it offers;
- writing the call out makes it obvious what is sent -- which matters when the
  claim being made is that the model receives no files, no tools and no paths.

Tools are not requested, so the API cannot return a tool call. That is not a
policy setting that could be flipped by a config change; the field is simply not
in the request body.
"""

from __future__ import annotations

from typing import Any

from visionforge.domain.llm import LlmRequest, LlmResponse, LlmUsage
from visionforge.infra.llm.base import post_json

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider:
    """Implements ``LlmProvider`` against the Messages API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LlmRequest) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            # The system prompt is a separate field here, not a message. Keeping
            # the user's content out of the instruction channel is the one piece
            # of prompt-injection hygiene the API itself can help with.
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }

        body, latency_ms, request_id = post_json(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            payload=payload,
            timeout_s=request.timeout_s,
        )

        return LlmResponse(
            text=_text_from(body),
            provider=self.name,
            model=str(body.get("model") or self._model),
            latency_ms=latency_ms,
            usage=LlmUsage(
                input_tokens=_int(body.get("usage", {}).get("input_tokens")),
                output_tokens=_int(body.get("usage", {}).get("output_tokens")),
            ),
            request_id=str(body.get("id") or request_id or "") or None,
            finish_reason=body.get("stop_reason"),
        )


def _text_from(body: dict[str, Any]) -> str:
    """Concatenate the text blocks of a response.

    A content array can hold several blocks; only text blocks are read, and
    anything else is ignored rather than interpreted. There is nothing else this
    code knows how to act on, and that is the intent.
    """
    content = body.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts)


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


__all__ = ["DEFAULT_MODEL", "AnthropicProvider"]
