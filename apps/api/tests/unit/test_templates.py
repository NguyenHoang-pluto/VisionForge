"""Templates (Phase 12): the library, measuring one from a video, and filling one.

A template is an edit with the footage taken out. These tests pin the three
things that make that useful: every shipped template can become a valid plan,
a template measured from a video copies that video's cutting, and filling a
template keeps its structure -- slot count, lengths, joins -- while choosing
the photo or video that fits each slot.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from tests.unit.editorial_fixtures import PROJECT, clip
from tests.unit.test_mixed_media_planning import facts, photo
from visionforge.domain.beats import BeatGrid
from visionforge.domain.decisions import ReasonCode
from visionforge.domain.editorial_planner import EditorialPlanner, editorial_of
from visionforge.domain.editplan import (
    MAX_SEGMENTS,
    MIN_SEGMENT_MS,
    AspectRatio,
    TransitionKind,
    validate_plan,
)
from visionforge.domain.effects import EffectKind
from visionforge.domain.media import MediaKind
from visionforge.domain.pacing import PacingShape
from visionforge.domain.planner import PlanRequest
from visionforge.domain.story import StoryRole
from visionforge.domain.template import (
    EditTemplate,
    SlotPreference,
    TemplateInvalidError,
    TemplateSlot,
    TemplateSource,
    slot_motion_bounds_ok,
    template_from_payload,
    validate_template,
)
from visionforge.domain.template_extract import aspect_for, extract_template
from visionforge.domain.template_fill import template_pacing
from visionforge.domain.template_library import LIBRARY, LIBRARY_BY_ID, SLIDESHOW, TRAVEL


# ------------------------------------------------------------------- library
class TestLibrary:
    def test_every_library_template_is_valid(self) -> None:
        for template in LIBRARY:
            assert validate_template(template) == [], template.id

    def test_ids_are_unique(self) -> None:
        assert len(LIBRARY_BY_ID) == len(LIBRARY)

    def test_the_slot_motion_is_inside_every_motion_bound(self) -> None:
        assert slot_motion_bounds_ok()

    def test_a_template_round_trips_through_its_payload(self) -> None:
        for template in LIBRARY:
            assert template_from_payload(template.as_payload()) == template

    def test_a_malformed_payload_is_refused_not_trusted(self) -> None:
        payload = TRAVEL.as_payload()
        payload["slots"][0]["role"] = "climax"
        with pytest.raises(TemplateInvalidError):
            template_from_payload(payload)

    def test_a_dissolve_longer_than_half_a_slot_is_refused(self) -> None:
        bad = replace(
            TRAVEL,
            slots=(
                TemplateSlot(duration_ms=1_000, energy=0.5, role=StoryRole.HOOK),
                TemplateSlot(
                    duration_ms=1_000,
                    energy=0.5,
                    role=StoryRole.ENDING,
                    transition_in=TransitionKind.CROSSFADE,
                    transition_ms=800,
                ),
            ),
        )
        assert any("dissolve" in problem for problem in validate_template(bad))


# ---------------------------------------------------------------- extraction
def scenes(*lengths: int) -> dict:
    out, start = [], 0
    for index, length in enumerate(lengths):
        out.append(
            {
                "scene_id": index,
                "start_ms": start,
                "end_ms": start + length,
                "duration_ms": length,
            }
        )
        start += length
    return {"scenes": out, "scene_count": len(out)}


def dynamics(*per_shot_motion: tuple[int, int, float]) -> dict:
    """Motion samples: one every 250 ms across each ``(start, end, motion)``."""
    samples = [
        {"timestamp_ms": at, "motion": motion}
        for start, end, motion in per_shot_motion
        for at in range(start, end, 250)
    ]
    return {"samples": samples}


class TestExtraction:
    def test_each_shot_becomes_a_slot_of_its_length(self) -> None:
        result = extract_template(
            template_id="t1",
            name="Mine",
            width=1920,
            height=1080,
            scenes=scenes(1_000, 2_000, 1_500, 3_000),
        )
        assert [slot.duration_ms for slot in result.template.slots] == [1_000, 2_000, 1_500, 3_000]
        assert result.template.source is TemplateSource.USER
        assert result.template.aspect is AspectRatio.LANDSCAPE_16_9

    def test_the_busiest_middle_shot_is_the_peak(self) -> None:
        lengths = (1_000, 1_000, 1_000, 1_000, 1_000)
        motion = dynamics(
            (0, 1_000, 0.1), (1_000, 2_000, 0.1), (2_000, 3_000, 0.1),
            (3_000, 4_000, 0.3), (4_000, 5_000, 0.1),
        )  # fmt: skip
        slots = extract_template(
            template_id="t", name="x", width=1920, height=1080,
            scenes=scenes(*lengths), dynamics=motion,
        ).template.slots  # fmt: skip
        roles = [slot.role for slot in slots]
        assert roles[0] is StoryRole.HOOK and roles[-1] is StoryRole.ENDING
        assert roles[3] is StoryRole.PEAK
        assert slots[3].energy > slots[1].energy
        # A fast shot asks for a video; a still has no motion of its own.
        assert slots[3].prefer is SlotPreference.VIDEO

    def test_a_shot_too_short_to_be_a_clip_is_folded_into_its_neighbour(self) -> None:
        result = extract_template(
            template_id="t", name="x", width=1920, height=1080,
            scenes=scenes(1_000, 100, 1_000),
        )  # fmt: skip
        assert result.merged == 1
        assert all(slot.duration_ms >= MIN_SEGMENT_MS for slot in result.template.slots)
        assert sum(slot.duration_ms for slot in result.template.slots) == 2_100

    def test_a_video_with_more_shots_than_a_plan_holds_is_truncated_and_says_so(self) -> None:
        result = extract_template(
            template_id="t", name="x", width=1920, height=1080,
            scenes=scenes(*([500] * (MAX_SEGMENTS + 5))),
        )  # fmt: skip
        assert len(result.template.slots) == MAX_SEGMENTS
        assert result.dropped == 5

    def test_a_single_take_has_no_structure_to_copy(self) -> None:
        with pytest.raises(TemplateInvalidError):
            extract_template(
                template_id="t", name="x", width=1920, height=1080, scenes=scenes(12_000)
            )

    def test_shots_on_a_trusted_grid_are_counted_in_beats(self) -> None:
        beats = {"bpm": 120.0, "confidence": 0.9, "beats_ms": list(range(0, 20_000, 500))}
        result = extract_template(
            template_id="t", name="x", width=1920, height=1080,
            scenes=scenes(1_000, 2_000, 1_500), beats=beats,
        )  # fmt: skip
        assert [slot.beats for slot in result.template.slots] == [2, 4, 3]
        assert result.beat_synced and result.template.bpm == 120.0

    def test_joins_are_cuts_because_dissolves_are_not_measured(self) -> None:
        result = extract_template(
            template_id="t", name="x", width=1920, height=1080, scenes=scenes(1_000, 1_000)
        )
        assert {slot.transition_in for slot in result.template.slots} == {TransitionKind.CUT}
        assert result.as_payload()["transitions"] == "cuts_only"

    def test_a_vertical_video_makes_a_vertical_template(self) -> None:
        assert aspect_for(1080, 1920) is AspectRatio.PORTRAIT_9_16
        assert aspect_for(1080, 1080) is AspectRatio.SQUARE_1_1


# ------------------------------------------------------------------- filling
def request(template: EditTemplate, **overrides: object) -> PlanRequest:
    values: dict[str, object] = {
        "project_id": PROJECT,
        "target_duration_ms": 20_000,
        "max_clips": 20,
        "min_clips": 1,
        "template": template,
    }
    values.update(overrides)
    return PlanRequest(**values)  # type: ignore[arg-type]


def mixed() -> list:
    return [
        clip(0, motion=0.6),
        photo(1),
        clip(2, motion=0.2),
        photo(3),
        photo(4, portrait=True),
        clip(5, motion=0.5),
        photo(6),
        clip(7, motion=0.3),
    ]


class TestFilling:
    def test_every_slot_is_filled_and_the_plan_validates(self) -> None:
        candidates = mixed()
        outcome = EditorialPlanner().plan(request(TRAVEL), candidates)

        assert len(outcome.plan.segments) == len(TRAVEL.slots)
        assert validate_plan(outcome.plan, facts(candidates)) == []

    def test_slots_keep_the_template_lengths_when_the_footage_allows(self) -> None:
        candidates = mixed()
        outcome = EditorialPlanner().plan(request(TRAVEL), candidates)
        wanted = [slot.duration_ms for slot in TRAVEL.slots]
        got = [segment.output_duration_ms for segment in outcome.plan.segments]
        assert got == wanted

    def test_a_slot_asking_for_a_photo_gets_one_when_there_is_one(self) -> None:
        candidates = mixed()
        photos = {c.media_id for c in candidates if c.kind is MediaKind.IMAGE}
        outcome = EditorialPlanner().plan(request(SLIDESHOW), candidates)
        # Four photos for nine photo slots: the first slots filled get photos.
        placed = [segment.media_id in photos for segment in outcome.plan.segments]
        assert sum(placed) >= 4

    def test_the_template_joins_are_kept(self) -> None:
        candidates = mixed()
        outcome = EditorialPlanner().plan(request(TRAVEL), candidates)
        wanted = [slot.transition_in for slot in TRAVEL.slots]
        got = [segment.transition_in for segment in outcome.plan.segments]
        assert got == wanted

    def test_a_still_moves_the_way_its_slot_says(self) -> None:
        candidates = [photo(index) for index in range(9)]
        outcome = EditorialPlanner().plan(request(SLIDESHOW), candidates)
        for slot, segment in zip(SLIDESHOW.slots, outcome.plan.segments, strict=True):
            assert slot.motion in {effect.kind for effect in segment.effects}

    def test_clips_are_reused_when_the_template_has_more_slots(self) -> None:
        candidates = [clip(0, motion=0.5), photo(1), clip(2, motion=0.3)]
        outcome = EditorialPlanner().plan(request(TRAVEL), candidates)

        assert len(outcome.plan.segments) == len(TRAVEL.slots)
        record = editorial_of(outcome.plan)
        assert record is not None
        assert record["fit"]["template"]["reused"] == len(TRAVEL.slots) - len(candidates)
        reasons = {reason for segment in record["segments"] for reason in segment["reasons"]}
        assert ReasonCode.REUSED_FOR_TEMPLATE.value in reasons

    def test_the_plan_records_which_template_it_filled(self) -> None:
        outcome = EditorialPlanner().plan(request(TRAVEL), mixed())
        record = editorial_of(outcome.plan)
        assert record is not None
        assert record["fit"]["template"]["id"] == "travel"

    def test_filling_is_deterministic(self) -> None:
        candidates = mixed()
        first = EditorialPlanner().plan(request(TRAVEL), candidates).plan
        second = EditorialPlanner().plan(request(TRAVEL), candidates).plan
        assert first.segments == second.segments


class TestRetiming:
    def test_slots_follow_the_music_beats(self) -> None:
        # 100 BPM: 600 ms a beat, so a two-beat slot is 1200 ms.
        grid = BeatGrid(bpm=100.0, confidence=0.9, beats_ms=tuple(range(0, 60_000, 600)))
        pacing = template_pacing(TRAVEL, grid=grid, beat_sync=True, shape=PacingShape.RAMP)
        assert [slot.duration_ms for slot in pacing.slots] == [
            slot.beats * 600 for slot in TRAVEL.slots if slot.beats
        ]
        assert pacing.beat_sync is not None and pacing.beat_sync["applied"] is True

    def test_without_music_the_template_keeps_its_own_timing(self) -> None:
        pacing = template_pacing(TRAVEL, grid=None, beat_sync=True, shape=PacingShape.RAMP)
        assert [slot.duration_ms for slot in pacing.slots] == [
            slot.duration_ms for slot in TRAVEL.slots
        ]

    def test_an_untrusted_grid_is_ignored(self) -> None:
        grid = BeatGrid(bpm=100.0, confidence=0.1, beats_ms=tuple(range(0, 60_000, 600)))
        pacing = template_pacing(TRAVEL, grid=grid, beat_sync=True, shape=PacingShape.RAMP)
        assert pacing.slots[0].duration_ms == TRAVEL.slots[0].duration_ms


def test_uuid_ids_are_accepted_for_user_templates() -> None:
    template = replace(TRAVEL, id=str(uuid.uuid4()), source=TemplateSource.USER)
    assert validate_template(template) == []


def test_a_zoom_motion_is_offered_in_the_library() -> None:
    motions = {slot.motion for template in LIBRARY for slot in template.slots}
    assert EffectKind.ZOOM_IN in motions and EffectKind.PAN_LEFT in motions
