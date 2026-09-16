"""The delta vocabulary, and what it refuses to read.

Two questions under test, and the second one is the security property.

**Does a well-formed operation parse into exactly what it said?** Closed enums,
bounded numbers, no coercion -- ``"segment": true`` is not clip 2.

**Is there any input that produces something outside the vocabulary?** The
answer has to be no by construction rather than by filtering, so the tests here
throw paths, filter strings, commands and unknown kinds at the parser and assert
that what comes back is either a violation or an operation carrying none of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from visionforge.domain.editdelta import (
    MAX_OPERATIONS,
    AddEffect,
    AddSubtitle,
    ChangeBeatSync,
    ChangeDuration,
    ChangeMusicFade,
    ChangeMusicVolume,
    ChangeOutputPreset,
    ChangeStyleStrength,
    ChangeTransition,
    DeltaInvalidError,
    ModifyEffect,
    ModifySubtitle,
    OperationKind,
    RemoveEffect,
    RemoveSegment,
    RemoveSubtitle,
    ReorderSegment,
    TrimSegment,
    parse_delta,
    parse_operations,
    vocabulary,
)
from visionforge.domain.editplan import AspectRatio, QualityPreset, TransitionKind
from visionforge.domain.effects import EffectKind
from visionforge.domain.policy import StyleStrength
from visionforge.domain.subtitles import SubtitlePosition, SubtitleStyle


def parse_one(entry: dict[str, Any]) -> Any:
    operations, violations = parse_operations([entry])
    assert not violations, violations
    assert len(operations) == 1
    return operations[0]


def reject(entry: dict[str, Any]) -> str:
    """Parse one operation expecting failure, and return the violation code."""
    operations, violations = parse_operations([entry])
    assert not operations, f"expected a rejection, got {operations}"
    assert violations
    return violations[0].code


# --------------------------------------------------------------- the vocabulary
def test_every_kind_has_a_parser() -> None:
    """A kind the enum declares but nothing can build is a hole in the schema."""
    for kind in OperationKind:
        operations, violations = parse_operations([{"kind": kind.value}])
        # Either it parses with no required fields, or it complains about the
        # fields it needs -- never "unknown_operation".
        codes = {violation.code for violation in violations}
        assert "unknown_operation" not in codes, kind


def test_vocabulary_is_generated_from_the_enum() -> None:
    assert [item["kind"] for item in vocabulary()] == [kind.value for kind in OperationKind]


def test_unknown_operation_is_rejected() -> None:
    assert reject({"kind": "DELETE_PROJECT"}) == "unknown_operation"
    assert reject({"kind": "RUN_FFMPEG", "args": "-i /etc/passwd"}) == "unknown_operation"


def test_kind_matching_is_case_insensitive_but_exact() -> None:
    assert parse_one({"kind": "change_beat_sync", "enabled": True}) == ChangeBeatSync(enabled=True)
    assert reject({"kind": "change beat sync", "enabled": True}) == "unknown_operation"


# ------------------------------------------------------------------- structure
def test_remove_segment() -> None:
    assert parse_one({"kind": "REMOVE_SEGMENT", "segment": 2}) == RemoveSegment(segment=2)


def test_reorder_segment() -> None:
    operation = parse_one({"kind": "REORDER_SEGMENT", "segment": 3, "to_index": 0})
    assert operation == ReorderSegment(segment=3, to_index=0)


def test_trim_segment_accepts_one_edge() -> None:
    assert parse_one({"kind": "TRIM_SEGMENT", "segment": 0, "source_out_ms": 3200}) == TrimSegment(
        segment=0, source_in_ms=None, source_out_ms=3200
    )


def test_trim_segment_needs_an_edge_to_move() -> None:
    assert reject({"kind": "TRIM_SEGMENT", "segment": 0}) == "nothing_to_do"


def test_trim_segment_refuses_an_inverted_range() -> None:
    entry = {"kind": "TRIM_SEGMENT", "segment": 0, "source_in_ms": 5000, "source_out_ms": 1000}
    assert reject(entry) == "field_range"


def test_change_duration_bounds_depend_on_what_was_addressed() -> None:
    """A clip may not be three minutes long; the whole edit may."""
    whole = parse_one({"kind": "CHANGE_DURATION", "duration_ms": 180_000})
    assert whole == ChangeDuration(duration_ms=180_000, segment=None)
    assert (
        reject({"kind": "CHANGE_DURATION", "segment": 0, "duration_ms": 180_000}) == "field_range"
    )


# ------------------------------------------------------------------------ look
def test_change_style_strength_is_a_closed_dial() -> None:
    operation = parse_one({"kind": "CHANGE_STYLE_STRENGTH", "value": "50"})
    assert operation == ChangeStyleStrength(value=StyleStrength.HALF)
    assert reject({"kind": "CHANGE_STYLE_STRENGTH", "value": "62"}) == "unknown_value"


def test_change_transition() -> None:
    operation = parse_one(
        {"kind": "CHANGE_TRANSITION", "segment": 1, "transition": "crossfade", "duration_ms": 500}
    )
    assert operation == ChangeTransition(
        segment=1, transition=TransitionKind.CROSSFADE, duration_ms=500
    )


def test_unknown_transition_is_rejected_rather_than_downgraded() -> None:
    entry = {"kind": "CHANGE_TRANSITION", "segment": 1, "transition": "whip_pan"}
    assert reject(entry) == "unknown_value"


def test_add_effect_checks_the_amount_against_its_own_kind() -> None:
    assert parse_one(
        {"kind": "ADD_EFFECT", "segment": 0, "effect": "slow_motion", "amount": 0.5}
    ) == AddEffect(effect=EffectKind.SLOW_MOTION, amount=0.5, segment=0)
    # 0.1x is outside slow motion's range, and is refused rather than clamped:
    # a co-edit operation is a statement about a specific clip.
    assert reject({"kind": "ADD_EFFECT", "segment": 0, "effect": "slow_motion", "amount": 0.1})


def test_effect_window_is_refused_on_a_whole_clip_effect() -> None:
    entry = {
        "kind": "ADD_EFFECT",
        "segment": 0,
        "effect": "zoom_in",
        "amount": 0.2,
        "start_ms": 100,
        "end_ms": 900,
    }
    assert reject(entry) == "effect_cannot_be_ranged"


def test_effect_applies_to_every_clip_when_no_segment_is_given() -> None:
    operation = parse_one({"kind": "ADD_EFFECT", "effect": "contrast", "amount": 1.1})
    assert operation == AddEffect(effect=EffectKind.CONTRAST, amount=1.1, segment=None)


def test_remove_and_modify_effect() -> None:
    assert parse_one({"kind": "REMOVE_EFFECT", "segment": 1, "effect": "zoom_in"}) == RemoveEffect(
        effect=EffectKind.ZOOM_IN, segment=1
    )
    assert parse_one(
        {"kind": "MODIFY_EFFECT", "segment": 1, "effect": "brightness", "amount": 0.2}
    ) == ModifyEffect(effect=EffectKind.BRIGHTNESS, amount=0.2, segment=1)


# ------------------------------------------------------------------------ text
def test_add_subtitle_cleans_its_text() -> None:
    operation = parse_one(
        {"kind": "ADD_SUBTITLE", "start_ms": 0, "end_ms": 2000, "text": "line\nbreak{tag}"}
    )
    assert isinstance(operation, AddSubtitle)
    # The same cleaner a hand-typed cue goes through: the document-format
    # characters are removed, not escaped.
    assert "\n" not in operation.text
    assert "{" not in operation.text and "}" not in operation.text


def test_add_subtitle_disarms_an_override_tag_rather_than_escaping_it() -> None:
    """An ASS override tag loses the characters that make it one.

    What survives is inert display text, which is the Phase 9 rule applied to a
    model's output: the braces and the backslash carry meaning to libass, so
    they are removed rather than escaped.
    """
    operation = parse_one(
        {"kind": "ADD_SUBTITLE", "start_ms": 0, "end_ms": 2000, "text": "{\\pos(0,0)}Hello"}
    )
    assert isinstance(operation, AddSubtitle)
    assert operation.text == "pos(0,0) Hello"
    assert not {"{", "}", "\\"} & set(operation.text)


def test_add_subtitle_refuses_text_with_nothing_displayable_left() -> None:
    entry = {"kind": "ADD_SUBTITLE", "start_ms": 0, "end_ms": 2000, "text": "{}\\\n\t"}
    assert reject(entry) == "field_empty"


def test_modify_subtitle_track_level_takes_style_and_position_only() -> None:
    operation = parse_one({"kind": "MODIFY_SUBTITLE", "style": "bold"})
    assert operation == ModifySubtitle(cue=None, style=SubtitleStyle.BOLD)

    operation = parse_one({"kind": "MODIFY_SUBTITLE", "position": "top"})
    assert operation == ModifySubtitle(cue=None, position=SubtitlePosition.TOP)

    assert reject({"kind": "MODIFY_SUBTITLE", "text": "hello"}) == "cue_required"


def test_modify_subtitle_cue_level_refuses_a_style() -> None:
    """One track, one look. Per-cue styling is not something the plan can store."""
    entry = {"kind": "MODIFY_SUBTITLE", "cue": 1, "style": "bold"}
    assert reject(entry) == "style_is_track_level"


def test_modify_subtitle_needs_something_to_change() -> None:
    assert reject({"kind": "MODIFY_SUBTITLE", "cue": 1}) == "nothing_to_do"
    assert reject({"kind": "MODIFY_SUBTITLE"}) == "nothing_to_do"


def test_remove_subtitle_addresses_a_cue_or_the_track() -> None:
    assert parse_one({"kind": "REMOVE_SUBTITLE", "cue": 2}) == RemoveSubtitle(cue=2)
    assert parse_one({"kind": "REMOVE_SUBTITLE"}) == RemoveSubtitle(cue=None)


# ----------------------------------------------------------------------- sound
def test_change_music_volume_is_bounded() -> None:
    assert parse_one({"kind": "CHANGE_MUSIC_VOLUME", "value": 0.4}) == ChangeMusicVolume(value=0.4)
    assert reject({"kind": "CHANGE_MUSIC_VOLUME", "value": 8.0}) == "field_range"
    assert reject({"kind": "CHANGE_MUSIC_VOLUME", "value": -1}) == "field_range"


def test_change_music_fade_accepts_either_end() -> None:
    assert parse_one({"kind": "CHANGE_MUSIC_FADE", "fade_out_ms": 2000}) == ChangeMusicFade(
        fade_in_ms=None, fade_out_ms=2000
    )
    assert reject({"kind": "CHANGE_MUSIC_FADE"}) == "nothing_to_do"


def test_change_beat_sync_wants_a_real_boolean() -> None:
    assert parse_one({"kind": "CHANGE_BEAT_SYNC", "enabled": False}) == ChangeBeatSync(
        enabled=False
    )
    assert reject({"kind": "CHANGE_BEAT_SYNC", "enabled": "yes"}) == "field_missing"


# ---------------------------------------------------------------------- output
def test_change_output_preset_names_a_shape_never_a_size() -> None:
    operation = parse_one({"kind": "CHANGE_OUTPUT_PRESET", "aspect_ratio": "9:16", "fps": 60})
    assert operation == ChangeOutputPreset(aspect_ratio=AspectRatio.PORTRAIT_9_16, fps=60)
    assert not hasattr(operation, "width")
    assert not hasattr(operation, "height")


def test_change_output_preset_ignores_a_smuggled_geometry() -> None:
    operation = parse_one(
        {
            "kind": "CHANGE_OUTPUT_PRESET",
            "quality": "high",
            "width": 4096,
            "height": 2160,
            "crf": 0,
        }
    )
    assert operation == ChangeOutputPreset(quality=QualityPreset.HIGH)
    assert operation.as_payload() == {
        "kind": "CHANGE_OUTPUT_PRESET",
        "aspect_ratio": None,
        "fps": None,
        "quality": "high",
        "audio": None,
        "source_gain": None,
    }


def test_change_output_preset_needs_a_field() -> None:
    assert reject({"kind": "CHANGE_OUTPUT_PRESET"}) == "nothing_to_do"


# ------------------------------------------------------------------- the holes
def test_booleans_are_not_numbers() -> None:
    """``True`` is an ``int`` in Python; clip ``True`` must not become clip 2."""
    assert reject({"kind": "REMOVE_SEGMENT", "segment": True}) == "field_missing"


def test_there_is_nowhere_to_put_a_path_or_a_command() -> None:
    """The keys an attacker would want are simply never read.

    Not filtered -- read into nothing. The operation that comes back carries the
    two fields the schema declares and no trace of the other five.
    """
    operation = parse_one(
        {
            "kind": "ADD_EFFECT",
            "segment": 0,
            "effect": "brightness",
            "amount": 0.2,
            "filter": "drawtext=text='x':fontfile=/windows/fonts/arial.ttf",
            "path": "/etc/passwd",
            "storage_key": "projects/other/media/secret.mp4",
            "command": "rm -rf /",
            "url": "https://example.invalid/payload",
        }
    )
    payload = operation.as_payload()
    serialised = repr(payload)
    for leaked in ("drawtext", "/etc/passwd", "storage_key", "rm -rf", "https://"):
        assert leaked not in serialised
    assert set(payload) == {"kind", "segment", "effect", "amount", "start_ms", "end_ms"}


def test_a_media_id_cannot_be_named() -> None:
    """No operation has a media field, so no delta can point at an asset."""
    operations, _ = parse_operations(
        [
            {
                "kind": "REMOVE_SEGMENT",
                "segment": 0,
                "media_id": "00000000-0000-0000-0000-000000000001",
            }
        ]
    )
    assert [operation.as_payload() for operation in operations] == [
        {"kind": "REMOVE_SEGMENT", "segment": 0}
    ]


# ------------------------------------------------------------------ the document
def test_parse_delta_reads_operations_and_a_rationale() -> None:
    delta = parse_delta(
        {
            "operations": [
                {"kind": "CHANGE_MUSIC_VOLUME", "value": 0.4},
                {"kind": "TRIM_SEGMENT", "segment": 0, "source_out_ms": 3200},
            ],
            "rationale": "  Tightened the   opening.  ",
        }
    )
    assert len(delta.operations) == 2
    assert delta.rationale == "Tightened the opening."
    assert delta.source == "llm"
    assert delta.summary == ("Music volume to 40%", "Retrim clip 1")


def test_parse_delta_is_all_or_nothing() -> None:
    """One bad operation fails the document. Half a change request is worse
    than none: the user asked for two things and would silently get one."""
    with pytest.raises(DeltaInvalidError) as caught:
        parse_delta(
            {
                "operations": [
                    {"kind": "CHANGE_MUSIC_VOLUME", "value": 0.4},
                    {"kind": "TELEPORT_CLIP", "segment": 1},
                ]
            }
        )
    assert [violation.code for violation in caught.value.violations] == ["unknown_operation"]


def test_parse_delta_refuses_an_empty_or_malformed_document() -> None:
    for payload in ({}, {"operations": []}, {"operations": "everything"}, []):
        with pytest.raises(DeltaInvalidError):
            parse_delta(payload)


def test_too_many_operations_is_refused() -> None:
    entries = [{"kind": "REMOVE_SEGMENT", "segment": index} for index in range(MAX_OPERATIONS + 5)]
    _, violations = parse_operations(entries)
    assert any(violation.code == "too_many_operations" for violation in violations)
