"""Turning media rows into the facts the plan validator is allowed to see.

This is the seam the music bed reaches the renderer through, and it is where
three separate guarantees have to hold at once:

- a cue may only name a *ready audio asset* in *its own project*;
- the validator sees kinds, statuses and durations -- never a storage key, so
  no future check can start depending on one;
- the worker resolves storage itself, from the database, so a plan cannot name
  a file.

``MediaFact.from_media`` is the single mapping both the planning service and
the render worker use. Before Phase 7 each wrote the rule out inline, which was
survivable with one derived flag and a guaranteed divergence with two -- and the
two callers sit on opposite sides of the validation gate, so a divergence means
a plan that validates at creation and fails at render.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.editplan import (
    AudioMode,
    EditPlan,
    MediaFact,
    MusicCue,
    OutputSpec,
    PlanInvalidError,
    Segment,
    assert_valid,
    validate_plan,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus

PROJECT = ProjectId(uuid.uuid4())
OTHER_PROJECT = ProjectId(uuid.uuid4())
CLIP = MediaId(uuid.uuid4())
TRACK = MediaId(uuid.uuid4())


def fact(
    media_id: MediaId = TRACK,
    *,
    project_id: ProjectId = PROJECT,
    kind: MediaKind = MediaKind.AUDIO,
    status: MediaStatus = MediaStatus.READY,
    duration_ms: int | None = 180_000,
) -> MediaFact:
    return MediaFact.from_media(
        media_id=media_id,
        project_id=project_id,
        kind=kind,
        status=status,
        duration_ms=duration_ms,
    )


def plan_with_music(cue: MusicCue | None) -> EditPlan:
    return EditPlan(
        project_id=PROJECT,
        segments=(
            Segment(media_id=CLIP, order=0, source_in_ms=0, source_out_ms=5_000),
            Segment(media_id=CLIP, order=1, source_in_ms=5_000, source_out_ms=10_000),
        ),
        output=OutputSpec(audio=AudioMode.SOURCE),
        music=cue,
    )


def cue(**overrides: object) -> MusicCue:
    values: dict = {
        "media_id": TRACK,
        "source_in_ms": 0,
        "source_out_ms": 10_000,
        "timeline_start_ms": 0,
        "gain": 0.7,
        "fade_in_ms": 0,
        "fade_out_ms": 0,
    }
    values.update(overrides)
    return MusicCue(**values)  # type: ignore[arg-type]


def codes(plan: EditPlan, facts: dict[MediaId, MediaFact]) -> set[str]:
    return {v.code for v in validate_plan(plan, facts)}


CLIP_FACT = MediaFact.from_media(
    media_id=CLIP,
    project_id=PROJECT,
    kind=MediaKind.VIDEO,
    status=MediaStatus.READY,
    duration_ms=30_000,
)


# ------------------------------------------------------- the shared mapping
class TestMediaFactFromMedia:
    def test_a_ready_audio_asset_is_a_music_source(self) -> None:
        assert fact().is_audio_asset is True

    def test_a_ready_audio_asset_is_not_renderable_video(self) -> None:
        """The two flags are independent. An audio file is not a clip."""
        assert fact().is_renderable is False

    def test_a_ready_video_is_renderable_and_not_a_music_source(self) -> None:
        """A video with a soundtrack is still not something a cue may name:
        Phase 7 beds are project-owned audio assets, and pointing a cue at a
        video's audio is a different feature, refused rather than half-working."""
        video = fact(kind=MediaKind.VIDEO)
        assert video.is_renderable is True
        assert video.is_audio_asset is False

    @pytest.mark.parametrize(
        "status",
        [
            MediaStatus.PENDING_UPLOAD,
            MediaStatus.UPLOADED,
            MediaStatus.PROCESSING,
            MediaStatus.FAILED,
        ],
    )
    def test_audio_that_is_not_ready_is_not_a_music_source(self, status: MediaStatus) -> None:
        """Ingest may not have probed it yet, so its duration is not to be
        trusted -- and a failed asset has no bytes worth reading."""
        assert fact(status=status).is_audio_asset is False

    def test_an_image_is_neither(self) -> None:
        image = fact(kind=MediaKind.IMAGE)
        assert image.is_renderable is False
        assert image.is_audio_asset is False

    def test_the_fact_carries_no_storage_key(self) -> None:
        """The property the whole split exists for: the validator cannot come to
        depend on a path, because there is no field holding one."""
        fields = set(MediaFact.__dataclass_fields__)
        assert fields == {
            "media_id",
            "project_id",
            "is_renderable",
            "duration_ms",
            "width",
            "height",
            "is_audio_asset",
        }
        assert not any("key" in f or "path" in f or "url" in f for f in fields)


# ------------------------------------------------------ what a cue may name
class TestWhatACueMayName:
    def test_valid_project_owned_audio_passes(self) -> None:
        facts = {CLIP: CLIP_FACT, TRACK: fact()}
        assert codes(plan_with_music(cue()), facts) == set()

    def test_missing_audio_is_reported_not_crashed(self) -> None:
        """The row is gone between planning and rendering. The render must be
        refused with a reason, not fail inside FFmpeg."""
        facts = {CLIP: CLIP_FACT}
        assert "music_unknown_media" in codes(plan_with_music(cue()), facts)

    def test_audio_from_another_project_is_refused(self) -> None:
        """Enforced in the domain, not only at the route. The worker fetches
        rows by id; this is what stops one project's plan reading another's
        bytes even if something upstream failed to notice."""
        facts = {CLIP: CLIP_FACT, TRACK: fact(project_id=OTHER_PROJECT)}
        assert "music_cross_project_media" in codes(plan_with_music(cue()), facts)

    def test_a_video_used_as_music_is_refused(self) -> None:
        facts = {CLIP: CLIP_FACT, TRACK: fact(kind=MediaKind.VIDEO)}
        assert "music_not_audio" in codes(plan_with_music(cue()), facts)

    def test_audio_still_ingesting_is_refused(self) -> None:
        facts = {CLIP: CLIP_FACT, TRACK: fact(status=MediaStatus.PROCESSING)}
        assert "music_not_audio" in codes(plan_with_music(cue()), facts)

    def test_an_invalid_source_range_is_refused(self) -> None:
        facts = {CLIP: CLIP_FACT, TRACK: fact()}
        found = codes(plan_with_music(cue(source_in_ms=9_000, source_out_ms=1_000)), facts)
        assert "music_non_positive_duration" in found

    def test_a_range_past_the_end_of_the_track_is_refused(self) -> None:
        facts = {CLIP: CLIP_FACT, TRACK: fact(duration_ms=20_000)}
        found = codes(plan_with_music(cue(source_out_ms=25_000)), facts)
        assert "music_trim_past_end" in found

    def test_the_gate_raises_before_anything_is_fetched(self) -> None:
        """``step_prepare`` calls ``assert_valid`` *before* downloading, so a
        cross-project cue never reaches object storage."""
        facts = {CLIP: CLIP_FACT, TRACK: fact(project_id=OTHER_PROJECT)}
        with pytest.raises(PlanInvalidError):
            assert_valid(plan_with_music(cue()), facts)


# ------------------------------------------------- what the worker resolves
class TestReferencedMedia:
    def test_the_music_asset_is_among_the_assets_to_fetch(self) -> None:
        """The bug this property exists to prevent: a bed that validates and is
        then never downloaded, so the render fails on a missing input."""
        assert TRACK in plan_with_music(cue()).referenced_media_ids

    def test_clips_come_first_and_in_order(self) -> None:
        referenced = plan_with_music(cue()).referenced_media_ids
        assert referenced[0] == CLIP
        assert referenced[-1] == TRACK

    def test_a_plan_without_music_references_only_its_clips(self) -> None:
        assert plan_with_music(None).referenced_media_ids == (CLIP,)

    def test_ids_are_deduplicated(self) -> None:
        """One file used by two segments is one download."""
        assert plan_with_music(None).source_media_ids == (CLIP, CLIP)
        assert plan_with_music(None).referenced_media_ids == (CLIP,)

    def test_a_track_also_used_as_a_clip_is_listed_once(self) -> None:
        plan = EditPlan(
            project_id=PROJECT,
            segments=(Segment(media_id=TRACK, order=0, source_in_ms=0, source_out_ms=5_000),),
            music=cue(),
        )
        assert plan.referenced_media_ids == (TRACK,)

    def test_source_media_ids_still_means_the_clips(self) -> None:
        """Phase 4-6 callers ask for the clips and must keep getting them."""
        assert TRACK not in plan_with_music(cue()).source_media_ids
