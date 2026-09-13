"""Hand-cut edits: the timeline's path into a stored, renderable plan.

Phase 6 gives the editor a timeline that can trim, delete and reorder. These
tests are the evidence that doing so buys no new authority. A plan assembled by
dragging clips is checked by the same validator, against the same media rows,
and is rejected in exactly the cases a planner's plan would be -- the only
difference being who chose the numbers.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from visionforge.application.edit_service import EditService
from visionforge.domain.editplan import (
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    Cut,
    FitMode,
    MediaFact,
    OutputSpec,
    PlanInvalidError,
    QualityPreset,
    TransitionKind,
    plan_from_cuts,
    validate_plan,
)
from visionforge.domain.errors import ValidationError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus

PROJECT = ProjectId(uuid.uuid4())
OTHER_PROJECT = ProjectId(uuid.uuid4())


def output(**overrides: Any) -> OutputSpec:
    spec: dict[str, Any] = {
        "aspect_ratio": AspectRatio.LANDSCAPE_16_9,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fit": FitMode.COVER,
        "audio": AudioMode.NONE,
        "quality": QualityPreset.BALANCED,
    }
    spec.update(overrides)
    return OutputSpec(**spec)


def fact(media_id: MediaId, **overrides: Any) -> MediaFact:
    defaults: dict[str, Any] = {
        "media_id": media_id,
        "project_id": PROJECT,
        "is_renderable": True,
        "duration_ms": 20_000,
        "width": 1920,
        "height": 1080,
    }
    defaults.update(overrides)
    return MediaFact(**defaults)


# ------------------------------------------------------------------- domain
class TestPlanFromCuts:
    def test_order_comes_from_sequence_position(self) -> None:
        """The editor sends a list; the list *is* the order."""
        ids = [MediaId(uuid.uuid4()) for _ in range(3)]
        plan = plan_from_cuts(
            project_id=PROJECT,
            cuts=[Cut(media_id=m, source_in_ms=0, source_out_ms=4000) for m in ids],
            output=output(),
        )

        assert [s.order for s in plan.ordered_segments] == [0, 1, 2]
        assert [s.media_id for s in plan.ordered_segments] == ids

    def test_reordering_the_list_reorders_the_edit(self) -> None:
        ids = [MediaId(uuid.uuid4()) for _ in range(3)]
        cuts = [Cut(media_id=m, source_in_ms=0, source_out_ms=4000) for m in ids]

        forwards = plan_from_cuts(project_id=PROJECT, cuts=cuts, output=output())
        backwards = plan_from_cuts(project_id=PROJECT, cuts=cuts[::-1], output=output())

        assert [s.media_id for s in backwards.ordered_segments] == ids[::-1]
        assert forwards.total_duration_ms == backwards.total_duration_ms

    def test_the_same_source_may_appear_twice(self) -> None:
        """Two cuts from one clip is an ordinary edit, not a duplicate."""
        media_id = MediaId(uuid.uuid4())
        plan = plan_from_cuts(
            project_id=PROJECT,
            cuts=[
                Cut(media_id=media_id, source_in_ms=0, source_out_ms=3000),
                Cut(media_id=media_id, source_in_ms=9000, source_out_ms=12_000),
            ],
            output=output(),
        )

        assert validate_plan(plan, {media_id: fact(media_id)}) == []
        assert plan.total_duration_ms == 6000

    def test_is_marked_as_manual(self) -> None:
        plan = plan_from_cuts(
            project_id=PROJECT,
            cuts=[Cut(media_id=MediaId(uuid.uuid4()), source_in_ms=0, source_out_ms=4000)],
            output=output(),
            metadata={"source": "manual"},
        )

        assert plan.planner == "manual"
        assert plan.as_payload()["metadata"]["source"] == "manual"

    def test_an_empty_sequence_produces_a_plan_the_validator_rejects(self) -> None:
        plan = plan_from_cuts(project_id=PROJECT, cuts=[], output=output())

        assert "no_segments" in {v.code for v in validate_plan(plan, {})}

    @pytest.mark.parametrize(
        ("cut", "expected"),
        [
            # A trim dragged past the end of the source.
            (Cut(MediaId(uuid.uuid4()), 18_000, 25_000), "trim_past_end"),
            # A clip squeezed below what the renderer can produce.
            (Cut(MediaId(uuid.uuid4()), 0, MIN_SEGMENT_MS - 1), "segment_too_short"),
            # Handles dragged through each other.
            (Cut(MediaId(uuid.uuid4()), 5000, 5000), "non_positive_duration"),
        ],
    )
    def test_bad_trims_are_caught(self, cut: Cut, expected: str) -> None:
        plan = plan_from_cuts(project_id=PROJECT, cuts=[cut], output=output())
        codes = {v.code for v in validate_plan(plan, {cut.media_id: fact(cut.media_id)})}

        assert expected in codes

    def test_cannot_reach_into_another_project(self) -> None:
        """The domain check, not just the route's."""
        media_id = MediaId(uuid.uuid4())
        plan = plan_from_cuts(
            project_id=PROJECT,
            cuts=[Cut(media_id=media_id, source_in_ms=0, source_out_ms=4000)],
            output=output(),
        )
        facts = {media_id: fact(media_id, project_id=OTHER_PROJECT)}

        assert "cross_project_media" in {v.code for v in validate_plan(plan, facts)}

    def test_transitions_survive_the_round_trip(self) -> None:
        media_id = MediaId(uuid.uuid4())
        plan = plan_from_cuts(
            project_id=PROJECT,
            cuts=[
                Cut(
                    media_id=media_id,
                    source_in_ms=0,
                    source_out_ms=4000,
                    transition_in=TransitionKind.CUT,
                )
            ],
            output=output(),
        )

        assert plan.as_payload()["segments"][0]["transition_in"] == "cut"


# ------------------------------------------------------------------ service
class FakeRecord:
    def __init__(self, media_id: MediaId, **overrides: Any) -> None:
        self.media_id = media_id
        self.kind = overrides.get("kind", MediaKind.VIDEO)
        self.status = overrides.get("status", MediaStatus.READY)
        self.duration_ms = overrides.get("duration_ms", 20_000)
        self.width = 1920
        self.height = 1080
        self.created_order = 0
        self.channels = 2
        self.analysis: dict[Any, Any] = {}


class FakeMediaRepo:
    def __init__(self, records: list[FakeRecord]) -> None:
        self._records = records

    async def records_with_analysis(self, project_id: ProjectId) -> list[FakeRecord]:
        return self._records


class FakePlanRepo:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, *, project_id: Any, plan: Any, selection: Any) -> Any:
        self.created.append({"plan": plan, "selection": selection})

        class Row:
            id = uuid.uuid4()

        return Row()


class FakeEvents:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


def service_for(records: list[FakeRecord]) -> tuple[EditService, FakePlanRepo, FakeEvents]:
    plans, events, session = FakePlanRepo(), FakeEvents(), FakeSession()
    # ``planner`` is required by the constructor and deliberately never used by
    # this path: a manual plan runs no planner at all.
    service = EditService(session, FakeMediaRepo(records), plans, events, planner=None)
    return service, plans, events


class TestCreateManualPlan:
    @pytest.mark.asyncio
    async def test_stores_the_timeline_as_a_plan(self) -> None:
        media_id = MediaId(uuid.uuid4())
        service, plans, events = service_for([FakeRecord(media_id)])

        await service.create_manual_plan(
            project_id=PROJECT,
            cuts=[
                Cut(media_id=media_id, source_in_ms=0, source_out_ms=4000),
                Cut(media_id=media_id, source_in_ms=8000, source_out_ms=12_000),
            ],
            output=output(),
        )

        stored = plans.created[0]["plan"]
        assert stored.planner == "manual"
        assert len(stored.segments) == 2
        assert stored.total_duration_ms == 8000
        assert events.records[0]["kind"] == "editplan.created"

    @pytest.mark.asyncio
    async def test_records_the_plan_it_was_cut_from(self) -> None:
        media_id = MediaId(uuid.uuid4())
        origin = uuid.uuid4()
        service, plans, _ = service_for([FakeRecord(media_id)])

        await service.create_manual_plan(
            project_id=PROJECT,
            cuts=[Cut(media_id=media_id, source_in_ms=0, source_out_ms=4000)],
            output=output(),
            derived_from=origin,
        )

        assert plans.created[0]["plan"].metadata["derived_from_edit_plan_id"] == str(origin)

    @pytest.mark.asyncio
    async def test_an_empty_timeline_is_rejected_before_any_query(self) -> None:
        service, plans, _ = service_for([])

        with pytest.raises(ValidationError):
            await service.create_manual_plan(project_id=PROJECT, cuts=[], output=output())

        assert plans.created == []

    @pytest.mark.asyncio
    async def test_a_trim_past_the_end_is_rejected_and_nothing_is_stored(self) -> None:
        """An invalid plan never reaches the database, hand-cut or not."""
        media_id = MediaId(uuid.uuid4())
        service, plans, _ = service_for([FakeRecord(media_id, duration_ms=5000)])

        with pytest.raises(PlanInvalidError) as caught:
            await service.create_manual_plan(
                project_id=PROJECT,
                cuts=[Cut(media_id=media_id, source_in_ms=0, source_out_ms=9000)],
                output=output(),
            )

        assert "trim_past_end" in {v.code for v in caught.value.violations}
        assert plans.created == []

    @pytest.mark.asyncio
    async def test_unknown_media_is_rejected(self) -> None:
        """A clip id the project does not have cannot be rendered into an edit."""
        service, plans, _ = service_for([FakeRecord(MediaId(uuid.uuid4()))])

        with pytest.raises(PlanInvalidError) as caught:
            await service.create_manual_plan(
                project_id=PROJECT,
                cuts=[Cut(media_id=MediaId(uuid.uuid4()), source_in_ms=0, source_out_ms=4000)],
                output=output(),
            )

        assert "unknown_media" in {v.code for v in caught.value.violations}
        assert plans.created == []

    @pytest.mark.asyncio
    async def test_media_that_is_not_ready_is_not_renderable(self) -> None:
        media_id = MediaId(uuid.uuid4())
        service, _, _ = service_for([FakeRecord(media_id, status=MediaStatus.PROCESSING)])

        with pytest.raises(PlanInvalidError) as caught:
            await service.create_manual_plan(
                project_id=PROJECT,
                cuts=[Cut(media_id=media_id, source_in_ms=0, source_out_ms=4000)],
                output=output(),
            )

        assert "media_not_renderable" in {v.code for v in caught.value.violations}

    @pytest.mark.asyncio
    async def test_an_image_is_not_renderable_as_a_timeline_clip(self) -> None:
        media_id = MediaId(uuid.uuid4())
        service, _, _ = service_for([FakeRecord(media_id, kind=MediaKind.IMAGE)])

        with pytest.raises(PlanInvalidError):
            await service.create_manual_plan(
                project_id=PROJECT,
                cuts=[Cut(media_id=media_id, source_in_ms=0, source_out_ms=4000)],
                output=output(),
            )
