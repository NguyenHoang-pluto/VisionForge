"""Reading a change request: the rules, the model, and the line between them.

The rules are tested hardest, because they are the path most requests take and
the one that must never half-understand a sentence. The property that matters:
**every clause resolves, or the whole request goes to the model.** A resolver
that quietly applied the half it understood would be worse than no resolver.

The model is tested for what it is allowed to say and what happens when it says
something else -- and for the fact that the prompt it is handed contains no
media id, no filename and no project identity.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from visionforge.domain.coeditor import (
    DEFAULT_SLOW_MOTION,
    DEFAULT_SPEED_UP,
    SYSTEM_PROMPT,
    CoEditCommand,
    CoEditFailure,
    CoEditSource,
    build_user_prompt,
    parse_llm_delta,
    propose,
    recognise_command,
    resolve_deterministic,
    shape_of,
)
from visionforge.domain.editdelta import (
    AddEffect,
    ChangeBeatSync,
    ChangeDuration,
    ChangeMusicFade,
    ChangeMusicVolume,
    ChangeOutputPreset,
    ChangeStyleStrength,
    ChangeTransition,
    ModifySubtitle,
    OperationKind,
    RemoveSegment,
    RemoveSubtitle,
    ReorderSegment,
)
from visionforge.domain.editplan import (
    AspectRatio,
    EditPlan,
    MusicCue,
    OutputSpec,
    QualityPreset,
    Segment,
    TransitionKind,
)
from visionforge.domain.effects import EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.llm import (
    LlmProvider,
    LlmRequest,
    LlmResponse,
    LlmUsage,
    ProviderTransientError,
    ProviderUnavailableError,
)
from visionforge.domain.policy import StyleStrength
from visionforge.domain.subtitles import (
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
)

PROJECT = ProjectId(uuid.uuid4())
MEDIA = [MediaId(uuid.uuid4()) for _ in range(4)]
MUSIC = MediaId(uuid.uuid4())


def plan(
    *,
    clips: int = 3,
    music: bool = True,
    subtitles: bool = True,
    metadata: dict[str, Any] | None = None,
) -> EditPlan:
    return EditPlan(
        project_id=PROJECT,
        segments=tuple(
            Segment(media_id=MEDIA[i], order=i, source_in_ms=0, source_out_ms=4_000)
            for i in range(clips)
        ),
        output=OutputSpec(),
        music=(
            MusicCue(media_id=MUSIC, source_in_ms=0, source_out_ms=12_000, gain=0.7)
            if music
            else None
        ),
        subtitles=(
            SubtitleTrack(
                cues=(SubtitleCue(start_ms=0, end_ms=2_000, text="hello"),),
                style=SubtitleStyle.CLEAN,
                position=SubtitlePosition.BOTTOM,
            )
            if subtitles
            else None
        ),
        metadata=dict(metadata or {}),
    )


def shape(**kwargs: Any) -> Any:
    return shape_of(plan(**kwargs))


def resolve(text: str, **kwargs: Any) -> Any:
    return resolve_deterministic(text, shape(**kwargs))


def operations(text: str, **kwargs: Any) -> list[Any]:
    delta = resolve(text, **kwargs)
    assert delta is not None, f"the rules did not resolve {text!r}"
    return list(delta.operations)


# ------------------------------------------------------------------ the shape
def test_the_shape_names_nothing() -> None:
    """No media id, no project id, no filename, no storage key. By construction."""
    payload = shape().as_payload()
    serialised = json.dumps(payload)

    for media_id in [*MEDIA, MUSIC]:
        assert str(media_id) not in serialised
    assert str(PROJECT) not in serialised
    for forbidden in ("media_id", "project_id", "storage_key", "filename", "path"):
        assert forbidden not in serialised


def test_the_shape_reports_played_length_not_trim() -> None:
    """A clip at half speed occupies twice its trim, and that is what is shown."""
    base = plan(clips=1)
    from visionforge.domain.effects import Effect

    slowed = base.segments[0]
    base = EditPlan(
        project_id=PROJECT,
        segments=(
            Segment(
                media_id=slowed.media_id,
                order=0,
                source_in_ms=0,
                source_out_ms=4_000,
                effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),),
            ),
        ),
    )
    assert shape_of(base).clips[0].duration_ms == 8_000


def test_the_prompt_carries_the_shape_and_nothing_else() -> None:
    prompt = build_user_prompt(shape(), request_text="lower the music")
    for media_id in [*MEDIA, MUSIC]:
        assert str(media_id) not in prompt
    assert str(PROJECT) not in prompt
    assert "USER_REQUEST" in prompt


def test_the_system_prompt_lists_exactly_the_vocabulary() -> None:
    """A prompt advertising an operation the parser lacks -- or omitting one it
    has -- is how a model gets blamed for a capability nobody told it about."""
    for kind in OperationKind:
        assert kind.value in SYSTEM_PROMPT


# ------------------------------------------------------------- history commands
@pytest.mark.parametrize("text", ["Undo", "undo that", "Undo.", "  go back  ", "revert"])
def test_undo_is_recognised(text: str) -> None:
    assert recognise_command(text) is CoEditCommand.UNDO


@pytest.mark.parametrize("text", ["redo", "Redo that", "put it back"])
def test_redo_is_recognised(text: str) -> None:
    assert recognise_command(text) is CoEditCommand.REDO


def test_a_change_request_mentioning_undo_is_not_an_undo() -> None:
    assert recognise_command("remove clip 2 and undo the crossfade") is None
    assert recognise_command(None) is None


# --------------------------------------------------------------- the rules: music
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Set music volume to 30%.", 0.3),
        ("lower the music to 40%", 0.4),
        ("Set the music to 100 percent", 1.0),
        ("mute the music", 0.0),
    ],
)
def test_music_volume_resolves(text: str, expected: float) -> None:
    assert operations(text) == [ChangeMusicVolume(value=expected)]


def test_music_volume_without_a_number_goes_to_the_model() -> None:
    """ "Lower the music" names no amount, and this resolver invents none."""
    assert resolve("lower the music") is None
    assert resolve("make the music quieter") is None


def test_music_fades_need_a_number() -> None:
    assert operations("fade the music out over 2 seconds") == [ChangeMusicFade(fade_out_ms=2_000)]
    assert operations("fade the music in over 800ms") == [ChangeMusicFade(fade_in_ms=800)]
    assert resolve("fade the music out") is None


# ------------------------------------------------------------ the rules: text
def test_subtitle_style_resolves() -> None:
    assert operations("Use bold subtitles.") == [ModifySubtitle(cue=None, style=SubtitleStyle.BOLD)]
    assert operations("change the subtitle style to cinematic") == [
        ModifySubtitle(cue=None, style=SubtitleStyle.CINEMATIC)
    ]


def test_subtitle_position_resolves() -> None:
    assert operations("put the subtitles at the top") == [
        ModifySubtitle(cue=None, position=SubtitlePosition.TOP)
    ]


def test_removing_subtitles_resolves() -> None:
    assert operations("remove the subtitles") == [RemoveSubtitle(cue=None)]


def test_an_unknown_subtitle_style_goes_to_the_model() -> None:
    assert resolve("use neon sparkle subtitles") is None


# ------------------------------------------------------- the rules: structure
def test_remove_clip_by_number_and_by_ordinal() -> None:
    assert operations("Remove the third clip.") == [RemoveSegment(segment=2)]
    assert operations("remove clip 3") == [RemoveSegment(segment=2)]
    assert operations("delete the last clip") == [RemoveSegment(segment=2)]


def test_removing_a_clip_that_does_not_exist_goes_to_the_model() -> None:
    """The rules decline rather than guess; the model sees the real clip count."""
    assert resolve("remove clip 9") is None
    assert resolve("remove the seventh clip") is None


def test_move_the_last_clip_to_the_beginning() -> None:
    assert operations("Move the last clip to the beginning.") == [
        ReorderSegment(segment=2, to_index=0)
    ]


def test_move_to_a_numbered_position() -> None:
    assert operations("move clip 1 to position 3") == [ReorderSegment(segment=0, to_index=2)]


def test_whole_edit_duration() -> None:
    assert operations("make the edit 20 seconds") == [
        ChangeDuration(duration_ms=20_000, segment=None)
    ]


def test_one_clip_duration() -> None:
    assert operations("hold clip 2 for 3 seconds") == [ChangeDuration(duration_ms=3_000, segment=1)]


def test_a_number_written_as_a_word_goes_to_the_model() -> None:
    """The rules read digits, not English numerals. Declining is the honest
    answer; guessing at "three" is the start of a very deep rabbit hole."""
    assert resolve("hold clip 2 for three seconds") is None


# ----------------------------------------------------------- the rules: look
def test_speed_uses_a_named_default_when_no_rate_is_given() -> None:
    assert operations("Make the second clip slow motion.") == [
        AddEffect(effect=EffectKind.SLOW_MOTION, amount=DEFAULT_SLOW_MOTION, segment=1)
    ]
    assert operations("make the opening faster") == [
        AddEffect(effect=EffectKind.SPEED_UP, amount=DEFAULT_SPEED_UP, segment=0)
    ]


def test_speed_obeys_a_stated_rate() -> None:
    assert operations("make clip 2 2x faster") == [
        AddEffect(effect=EffectKind.SPEED_UP, amount=2.0, segment=1)
    ]
    assert operations("play the last clip at half speed") == [
        AddEffect(effect=EffectKind.SLOW_MOTION, amount=0.5, segment=2)
    ]


def test_a_rate_outside_the_effect_bounds_goes_to_the_model() -> None:
    assert resolve("make clip 2 10x faster") is None


def test_crossfade_the_first_two_clips_lands_on_the_second() -> None:
    """A transition belongs to the clip it brings in."""
    assert operations("Crossfade the first two clips.") == [
        ChangeTransition(segment=1, transition=TransitionKind.CROSSFADE, duration_ms=None)
    ]


def test_crossfade_with_a_stated_length() -> None:
    assert operations("crossfade into clip 2 over 500ms") == [
        ChangeTransition(segment=1, transition=TransitionKind.CROSSFADE, duration_ms=500)
    ]


def test_a_crossfade_on_the_first_clip_goes_to_the_model() -> None:
    """The rules will not turn an impossible request into a different one."""
    assert resolve("crossfade into the first clip") is None


def test_style_strength_only_accepts_the_dial_stops() -> None:
    assert operations("Reduce the reference style strength to 50%.") == [
        ChangeStyleStrength(value=StyleStrength.HALF)
    ]
    # 60% is not a stop on the dial, and snapping it to 50 would be the editor
    # quietly doing something else.
    assert resolve("set the reference style strength to 60%") is None


def test_beat_sync_toggles() -> None:
    assert operations("Turn beat sync off.") == [ChangeBeatSync(enabled=False)]
    assert operations("turn beat sync on") == [ChangeBeatSync(enabled=True)]


def test_output_preset_resolves() -> None:
    assert operations("make it vertical") == [
        ChangeOutputPreset(aspect_ratio=AspectRatio.PORTRAIT_9_16)
    ]
    assert operations("render at high quality") == [ChangeOutputPreset(quality=QualityPreset.HIGH)]
    assert operations("use 60 fps") == [ChangeOutputPreset(fps=60)]


# ------------------------------------------------------- all-or-nothing clauses
def test_two_clauses_both_resolve() -> None:
    assert operations("Remove the third clip and use bold subtitles.") == [
        RemoveSegment(segment=2),
        ModifySubtitle(cue=None, style=SubtitleStyle.BOLD),
    ]


def test_the_acceptance_request_resolves_without_a_model() -> None:
    assert operations("Make the opening faster and lower the music to 40%.") == [
        AddEffect(effect=EffectKind.SPEED_UP, amount=DEFAULT_SPEED_UP, segment=0),
        ChangeMusicVolume(value=0.4),
    ]


def test_one_unresolvable_clause_sends_the_whole_request_to_the_model() -> None:
    """The safety property. Half a change request is never applied."""
    assert resolve("remove clip 2 and make it feel more like a music video") is None
    assert resolve("make it more cinematic and lower the music to 40%") is None


def test_a_purely_creative_request_is_not_resolved() -> None:
    for text in (
        "Make the whole edit more cinematic.",
        "give it more energy",
        "make it feel like a trailer",
        "tighten it up a bit",
    ):
        assert resolve(text) is None, text


def test_nothing_resolves_against_an_empty_plan() -> None:
    assert resolve_deterministic("remove clip 1", shape_of(EditPlan(PROJECT, segments=()))) is None


# ----------------------------------------------------------------- the model
class FakeProvider:
    """A provider that answers with whatever it was told to, for the parse path."""

    name = "fake"
    model = "fake-1"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.requests: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LlmResponse(
            text=self.text,
            provider=self.name,
            model=self.model,
            latency_ms=12.0,
            usage=LlmUsage(input_tokens=100, output_tokens=20),
        )


def provider_for(payload: dict[str, Any] | str) -> LlmProvider:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return FakeProvider(text=text)


def test_a_creative_request_reaches_the_model_and_parses() -> None:
    provider = FakeProvider(
        text=json.dumps(
            {
                "operations": [
                    {"kind": "ADD_EFFECT", "effect": "contrast", "amount": 1.2},
                    {"kind": "CHANGE_TRANSITION", "segment": 1, "transition": "crossfade"},
                ],
                "rationale": "Cooler grade and a softer join.",
            }
        )
    )
    outcome = propose(provider, shape(), request_text="make the whole edit more cinematic")

    assert outcome.ok
    assert outcome.source is CoEditSource.LLM
    assert outcome.delta is not None
    assert len(outcome.delta.operations) == 2
    assert outcome.delta.rationale == "Cooler grade and a softer join."
    assert outcome.provider == "fake"


def test_a_deterministic_request_never_reaches_the_model() -> None:
    provider = FakeProvider(text="{}")
    outcome = propose(provider, shape(), request_text="set the music to 30%")

    assert outcome.ok
    assert outcome.source is CoEditSource.RULES
    assert provider.requests == [], "the rules resolved it; no call should have been made"


def test_an_unknown_operation_from_the_model_is_refused() -> None:
    provider = provider_for({"operations": [{"kind": "RUN_COMMAND", "argv": ["rm", "-rf", "/"]}]})
    outcome = propose(provider, shape(), request_text="make it cinematic")

    assert not outcome.ok
    assert outcome.failure is CoEditFailure.NO_USABLE_OPERATIONS
    assert outcome.delta is None
    assert [violation.code for violation in outcome.violations] == ["unknown_operation"]


def test_an_out_of_range_parameter_from_the_model_is_refused() -> None:
    provider = provider_for({"operations": [{"kind": "CHANGE_MUSIC_VOLUME", "value": 99}]})
    outcome = propose(provider, shape(), request_text="make it cinematic")

    assert not outcome.ok
    assert outcome.delta is None
    assert [violation.code for violation in outcome.violations] == ["field_range"]


def test_prose_instead_of_json_is_unreadable() -> None:
    outcome = propose(
        provider_for("I would suggest shortening the opening."),
        shape(),
        request_text="make it cinematic",
    )
    assert not outcome.ok
    assert outcome.failure is CoEditFailure.UNREADABLE


def test_a_model_that_declines_produces_a_failure_not_an_empty_delta() -> None:
    provider = provider_for({"operations": [], "rationale": "I cannot do that with these tools."})
    outcome = propose(provider, shape(), request_text="add a voiceover")

    assert not outcome.ok
    assert outcome.delta is None
    assert outcome.failure is CoEditFailure.UNREADABLE


def test_json_wrapped_in_a_fence_is_still_read() -> None:
    fenced = (
        "Here you go:\n```json\n"
        + json.dumps({"operations": [{"kind": "CHANGE_MUSIC_VOLUME", "value": 0.25}]})
        + "\n```"
    )
    outcome = propose(provider_for(fenced), shape(), request_text="make it cinematic")
    assert outcome.ok
    assert outcome.delta is not None
    assert outcome.delta.operations == (ChangeMusicVolume(value=0.25),)


def test_extra_keys_in_a_model_operation_are_not_read() -> None:
    provider = provider_for(
        {
            "operations": [
                {
                    "kind": "ADD_EFFECT",
                    "segment": 0,
                    "effect": "brightness",
                    "amount": 0.2,
                    "filter": "drawtext=...",
                    "output_path": "/tmp/out.mp4",
                }
            ]
        }
    )
    outcome = propose(provider, shape(), request_text="brighten the opening a touch")

    assert outcome.ok
    assert outcome.delta is not None
    assert outcome.delta.operations == (
        AddEffect(effect=EffectKind.BRIGHTNESS, amount=0.2, segment=0),
    )
    assert "drawtext" not in json.dumps(outcome.delta.as_payload())


# -------------------------------------------------------------- no provider
def test_no_provider_and_an_unreadable_request_fails_cleanly() -> None:
    outcome = propose(None, shape(), request_text="make it feel like a trailer")

    assert not outcome.ok
    assert outcome.failure is CoEditFailure.PROVIDER_DISABLED
    assert outcome.delta is None
    # The message tells the user what the rules *can* do rather than "error".
    assert "remove clip" in outcome.detail


def test_no_provider_still_resolves_what_the_rules_understand() -> None:
    outcome = propose(None, shape(), request_text="remove the third clip")
    assert outcome.ok
    assert outcome.source is CoEditSource.RULES


def test_a_provider_outage_is_a_named_failure() -> None:
    outcome = propose(
        FakeProvider(error=ProviderUnavailableError("no route to host")),
        shape(),
        request_text="make it cinematic",
    )
    assert not outcome.ok
    assert outcome.failure is CoEditFailure.PROVIDER_UNAVAILABLE
    assert outcome.delta is None


def test_a_transient_provider_error_is_a_named_failure() -> None:
    outcome = propose(
        FakeProvider(error=ProviderTransientError("429")),
        shape(),
        request_text="make it cinematic",
    )
    assert not outcome.ok
    assert outcome.failure is CoEditFailure.PROVIDER_ERROR


def test_an_empty_request_is_refused_before_anything_runs() -> None:
    provider = FakeProvider(text="{}")
    outcome = propose(provider, shape(), request_text="   ")
    assert not outcome.ok
    assert outcome.failure is CoEditFailure.EMPTY_REQUEST
    assert provider.requests == []


def test_the_llm_can_be_switched_off_per_request() -> None:
    provider = FakeProvider(text="{}")
    outcome = propose(provider, shape(), request_text="make it cinematic", allow_llm=False)
    assert not outcome.ok
    assert outcome.failure is CoEditFailure.NOT_UNDERSTOOD
    assert provider.requests == []


def test_parse_llm_delta_directly() -> None:
    delta, violations = parse_llm_delta(json.dumps({"operations": [{"kind": "REMOVE_SUBTITLE"}]}))
    assert violations == ()
    assert delta is not None
    assert delta.source == CoEditSource.LLM.value
