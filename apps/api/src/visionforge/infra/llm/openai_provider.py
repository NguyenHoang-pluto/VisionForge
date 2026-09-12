"""OpenAI-compatible Chat Completions provider.

Its real job is to prove the abstraction. A port with one implementation is a
guess about what varies; a port with two is a boundary that has been tested
against an actual difference -- and the differences here are real: a separate
``system`` field versus a system message, ``max_tokens`` versus
``max_completion_tokens``, ``usage.input_tokens`` versus
``usage.prompt_tokens``.

The ``/v1/chat/completions`` shape is also what most self-hosted and gateway
services speak, so a base-URL change points this at a local model without new
code. That, not vendor choice, is the argument for having it.
"""

from __future__ import annotations

from typing import Any

from visionforge.domain.llm import LlmRequest, LlmResponse, LlmUsage
from visionforge.infra.llm.base import post_json

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com"


class OpenAiProvider:
    """Implements ``LlmProvider`` against a Chat Completions endpoint."""

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
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LlmRequest) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            # Ask for JSON at the API level as well as in the prompt. It does not
            # replace parsing -- the parser still treats the response as untrusted
            # text -- but it removes the most common failure, a fenced code block.
            "response_format": {"type": "json_object"},
        }

        body, latency_ms, request_id = post_json(
            f"{self._base_url}/v1/chat/completions",
            headers={
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            payload=payload,
            timeout_s=request.timeout_s,
        )

        choice = _first_choice(body)
        usage = body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}

        return LlmResponse(
            text=_text_from(choice),
            provider=self.name,
            model=str(body.get("model") or self._model),
            latency_ms=latency_ms,
            usage=LlmUsage(
                input_tokens=_int(usage.get("prompt_tokens")),
                output_tokens=_int(usage.get("completion_tokens")),
            ),
            request_id=str(body.get("id") or request_id or "") or None,
            finish_reason=choice.get("finish_reason") if choice else None,
        )


def _first_choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _text_from(choice: dict[str, Any]) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


__all__ = ["DEFAULT_MODEL", "OpenAiProvider"]
