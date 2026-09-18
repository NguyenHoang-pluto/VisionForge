"""A deterministic local provider. **Not a language model.**

It exists so the entire LLM path -- brief, prompt, provider call, parse, handle
resolution, clamping, compilation, validation -- can be exercised in CI and on a
laptop with no API key, no network and no cost. Every layer above it runs
exactly as it would in production; only the thing producing the JSON is
different.

It is named ``stub`` everywhere it surfaces: in configuration, in the plan
metadata, in the ``llm_runs`` table and in the UI, which reports the provider by
name. Nothing in this system will tell a user that a stub-planned edit was made
by AI, and selecting it in production raises rather than degrades quietly --
a test double that can reach real users is not a test double.

What it actually does is read the brief back out of the prompt it was handed and
apply three rules: take the strongest clips, hold each for the style's typical
duration, stop at the target length. That is roughly what the rules engine does,
which is the point -- the stub tests the *plumbing*, and makes no claim about
the quality of the judgement.

Phase 10 gave it a second prompt to answer. A co-edit request asks for
operations rather than clips, and what the stub returns is one operation chosen
by arithmetic from the shape it was shown -- **not** an interpretation of the
user's sentence, which it cannot read and does not try to. It exists so that the
co-editor's parse, patch, validate and version path can be exercised end to end
with no key; its rationale says so in the words the user would see.
"""

from __future__ import annotations

import json
from typing import Any

from visionforge.domain.directive import extract_json_object
from visionforge.domain.llm import LlmRequest, LlmResponse, LlmUsage

PROVIDER_NAME = "stub"
MODEL_NAME = "deterministic-stub-1"


class StubProvider:
    """Implements ``LlmProvider`` with arithmetic instead of inference."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return MODEL_NAME

    def complete(self, request: LlmRequest) -> LlmResponse:
        if _is_co_edit(request):
            return self._co_edit(request)
        return self._plan(request)

    def _co_edit(self, request: LlmRequest) -> LlmResponse:
        """Answer a co-edit prompt with one operation derived from the shape.

        Deliberately a *small* change, and one the patcher will accept on any
        plan: a dissolve into the second clip when there is a second clip, and
        otherwise a nudge to the encoder preset. The stub is not pretending to
        have understood anything -- see the rationale it returns.
        """
        shape = extract_json_object(request.user) or {}
        raw_clips = shape.get("clips")
        clips: list[Any] = raw_clips if isinstance(raw_clips, list) else []

        operations: list[dict[str, Any]]
        if len(clips) >= 2:
            operations = [{"kind": "CHANGE_TRANSITION", "segment": 1, "transition": "crossfade"}]
        else:
            operations = [{"kind": "CHANGE_OUTPUT_PRESET", "quality": "balanced"}]

        return self._respond(
            {
                "operations": operations,
                "rationale": (
                    "Stub co-editor: this change was chosen by arithmetic from the "
                    "edit's shape. No language model read the request."
                ),
            }
        )

    def _plan(self, request: LlmRequest) -> LlmResponse:
        brief = extract_json_object(request.user) or {}
        raw_clips = brief.get("clips")
        clips: list[Any] = raw_clips if isinstance(raw_clips, list) else []

        hold_ms, max_clips, target_ms = _constraints(request.user)

        chosen: list[dict[str, Any]] = []
        total = 0
        for entry in clips:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            if len(chosen) >= max_clips or total >= target_ms:
                break
            # Never ask to hold a clip longer than it is. The compiler would
            # clamp it anyway; emitting a legal number keeps the stub's output
            # a realistic example of what a model should produce.
            take = min(hold_ms, int(entry.get("duration_ms") or hold_ms))
            chosen.append({"ref": entry["id"], "duration_ms": take})
            total += take

        directive = {
            "style": _style_from(request.user),
            "pacing": "medium",
            "clips": chosen,
            "rationale": (
                f"Stub planner: took the {len(chosen)} highest-scoring clip(s) and held "
                f"each for about {hold_ms} ms. No language model was involved."
            ),
        }

        return self._respond(directive)

    def _respond(self, payload: dict[str, Any]) -> LlmResponse:
        return LlmResponse(
            text=json.dumps(payload),
            provider=self.name,
            model=self.model,
            latency_ms=0.0,
            # No tokens were spent, and reporting a plausible-looking count
            # would put fiction into the usage table.
            usage=LlmUsage(),
            request_id=None,
            finish_reason="stub",
        )


def _is_co_edit(request: LlmRequest) -> bool:
    """Whether this is a change request rather than a planning one.

    Matched on the system prompt, which this codebase wrote, rather than on the
    user's text, which it did not. A request whose *content* decided which
    branch ran would be a request that could choose its own handler.
    """
    return "REMOVE_SEGMENT" in request.system and "operations" in request.system


def _constraints(prompt: str) -> tuple[int, int, int]:
    """Recover the numbers the prompt stated: hold time, clip cap, target length.

    String scanning, which would be fragile in anything that mattered. Here it
    is contained: this class exists only to answer prompts this codebase built,
    and its defaults are sane when a line is missing.
    """
    hold_ms = _int_after(prompt, "typical for this style:", 2_000)
    max_clips = _int_after(prompt, "Use at most", 8)
    target_ms = _int_after(prompt, "Aim for a total of about", 25_000)
    return hold_ms, max(1, max_clips), max(1_000, target_ms)


def _int_after(text: str, marker: str, default: int) -> int:
    index = text.find(marker)
    if index < 0:
        return default
    digits = ""
    for char in text[index + len(marker) :]:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else default


def _style_from(prompt: str) -> str:
    """Echo back the style the prompt named, so the directive is consistent."""
    marker = "Style: "
    index = prompt.find(marker)
    if index < 0:
        return "custom"
    tail = prompt[index + len(marker) :]
    return tail.split(" --", 1)[0].strip().split("\n", 1)[0].strip() or "custom"


__all__ = ["MODEL_NAME", "PROVIDER_NAME", "StubProvider"]
