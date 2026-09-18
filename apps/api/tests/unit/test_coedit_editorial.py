"""The co-editor's editorial vocabulary.

Phase 10's rules could resolve "remove clip 2" because a clip number is a fact
about a plan. They could not resolve "give the climax more time", because before
Phase 11 nothing in a plan said which clip was the climax -- so the whole
sentence went to a model.

Now a plan records the narrative role of every segment, and these requests
resolve deterministically. Two properties matter and are tested here:

**Nothing bypasses ``EditDelta``.** Every one of these sentences resolves to
operations from the same closed vocabulary Phase 10 established, applied by the
same patcher, and recorded as the same kind of version. There is no editorial
side-channel into a stored plan.

**A plan without roles is not guessed at.** Every rule below checks first, and
declines to the model on a hand-cut plan or one the Phase 4 rules engine made.
"""

from __future__ import annotations

from tests.unit.editorial_fixtures import PROJECT, football_project, gaming_project
from visionforge.domain.coeditor import (
    EDITORIAL_STEP,
    resolve_deterministic,
    shape_of,
)
from visionforge.domain.editdelta import (
    ChangeDuration,
    EditOperation,
    OperationKind,
    RemoveSegment,
)
from visionforge.domain.editorial_planner import EditorialPlanner
from visionforge.domain.planner import PlanRequest, RulesEnginePlanner
from visionforge.domain.story import PolicyId


def editorial_plan(policy: PolicyId = PolicyId.FOOTBALL, clips: list | None = None):  # type: ignore[no-untyped-def]
    return (
        EditorialPlanner()
        .plan(
            PlanRequest(
                project_id=PROJECT,
                target_duration_ms=30_000,
                max_clips=10,
                min_clips=2,
                editorial_policy=policy,
            ),
            clips if clips is not None else football_project(),
        )
        .plan
    )


def rules_plan():  # type: ignore[no-untyped-def]
    """A Phase 4 plan: valid, renderable, and carrying no roles at all."""
    return (
        RulesEnginePlanner()
        .plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=30_000, max_clips=6),
            football_project(),
        )
        .plan
    )


def operations(text: str, plan: object | None = None) -> list[EditOperation]:
    delta = resolve_deterministic(text, shape_of(plan if plan is not None else editorial_plan()))
    return list(delta.operations) if delta else []


# ---------------------------------------------------------------- the shape
class TestShape:
    def test_an_editorial_plan_exposes_its_roles(self) -> None:
        shape = shape_of(editorial_plan())
        assert all(clip.role for clip in shape.clips)
        assert "peak" in {clip.role for clip in shape.clips}

    def test_a_rules_plan_exposes_none(self) -> None:
        shape = shape_of(rules_plan())
        assert all(clip.role == "" for clip in shape.clips)

    def test_the_shape_payload_still_carries_no_identifier(self) -> None:
        """A role name says what a clip is *for*, not which file it is."""
        plan = editorial_plan()
        payload = repr(shape_of(plan).as_payload())
        for segment in plan.segments:
            assert str(segment.media_id) not in payload
        assert str(PROJECT) not in payload

    def test_energy_reaches_the_prompt_rounded(self) -> None:
        """The model is told roughly how busy a shot is, not handed a
        measurement it might reason about the precision of."""
        payload = shape_of(editorial_plan()).as_payload()
        for clip in payload["clips"]:
            if "energy" in clip:
                assert clip["energy"] == round(clip["energy"], 2)


# ---------------------------------------------------------------- addressing
class TestRoleAddressing:
    def test_the_climax_resolves_to_the_peak(self) -> None:
        plan = editorial_plan()
        peak_index = next(
            index for index, clip in enumerate(shape_of(plan).clips) if clip.role == "peak"
        )
        resolved = operations("give the climax more time", plan)
        assert len(resolved) == 1
        assert isinstance(resolved[0], ChangeDuration)
        assert resolved[0].segment == peak_index

    def test_the_opening_resolves_to_the_first_clip(self) -> None:
        resolved = operations("make the opening more aggressive")
        assert len(resolved) == 1
        assert resolved[0].segment == 0  # type: ignore[union-attr]

    def test_a_role_this_edit_does_not_have_goes_to_the_model(self) -> None:
        """Acting on the nearest thing would be worse than declining: "give the
        peak more time" on an edit with no peak cannot be satisfied by
        shortening the setup."""
        plan = editorial_plan(PolicyId.PRODUCT, clips=gaming_project())
        assert resolve_deterministic("give the reaction more time", shape_of(plan)) is None

    def test_a_plan_with_no_roles_goes_to_the_model(self) -> None:
        assert resolve_deterministic("give the climax more time", shape_of(rules_plan())) is None


# ------------------------------------------------------------------- the step
class TestEditorialSteps:
    def test_more_time_lengthens_by_the_documented_step(self) -> None:
        plan = editorial_plan()
        shape = shape_of(plan)
        peak_index = next(i for i, clip in enumerate(shape.clips) if clip.role == "peak")
        before = shape.clips[peak_index].duration_ms
        resolved = operations("give the climax more time", plan)
        assert resolved[0].duration_ms == round(before * EDITORIAL_STEP)  # type: ignore[union-attr]

    def test_more_aggressive_shortens_by_the_same_step(self) -> None:
        plan = editorial_plan()
        before = shape_of(plan).clips[0].duration_ms
        resolved = operations("make the opening more aggressive", plan)
        assert resolved[0].duration_ms < before  # type: ignore[union-attr]

    def test_a_stated_duration_is_a_duration_not_a_step(self) -> None:
        """``_duration`` owns that sentence; two rules claiming it is how one
        quietly wins."""
        resolved = operations("make the climax 4 seconds")
        assert len(resolved) == 1
        assert resolved[0].duration_ms == 4_000  # type: ignore[union-attr]

    def test_a_music_request_is_never_read_as_a_clip_length(self) -> None:
        assert resolve_deterministic("make the music longer", shape_of(editorial_plan())) is None


class TestFewerShots:
    def test_fewer_shots_removes_the_droppable_roles(self) -> None:
        """The setup and the build go; the peak, the reaction and the ending
        are what make the edit the edit."""
        plan = editorial_plan()
        shape = shape_of(plan)
        resolved = operations("use fewer shots", plan)
        assert resolved
        assert all(isinstance(op, RemoveSegment) for op in resolved)
        for op in resolved:
            assert shape.clips[op.segment].role in {"setup", "build"}  # type: ignore[union-attr]

    def test_removals_are_emitted_latest_first(self) -> None:
        """Each removal renumbers what follows, so 5 then 2 is the pair the user
        meant and 2 then 5 is not."""
        indices = [op.segment for op in operations("use fewer shots")]  # type: ignore[union-attr]
        assert indices == sorted(indices, reverse=True)

    def test_a_short_edit_goes_to_the_model_instead(self) -> None:
        plan = editorial_plan()
        shape = shape_of(plan)
        assert len(shape.clips) >= 3
        two_clips = shape_of(
            EditorialPlanner()
            .plan(
                PlanRequest(
                    project_id=PROJECT,
                    target_duration_ms=8_000,
                    max_clips=2,
                    min_clips=2,
                    editorial_policy=PolicyId.FOOTBALL,
                ),
                football_project(),
            )
            .plan
        )
        assert resolve_deterministic("use fewer shots", two_clips) is None

    def test_a_plan_with_no_roles_goes_to_the_model(self) -> None:
        assert resolve_deterministic("use fewer shots", shape_of(rules_plan())) is None


class TestVocabularyIsUnchanged:
    def test_every_operation_is_from_the_phase_ten_vocabulary(self) -> None:
        """Nothing bypasses ``EditDelta``: the editorial rules emit the same
        closed set, so versioning, validation and undo are untouched."""
        sentences = (
            "give the climax more time",
            "make the opening more aggressive",
            "use fewer shots",
        )
        for sentence in sentences:
            for operation in operations(sentence):
                assert operation.KIND in set(OperationKind)

    def test_the_phase_ten_sentences_still_resolve(self) -> None:
        assert operations("remove clip 2")
        volume = operations("lower the music to 40%")
        assert len(volume) == 1
        assert volume[0].KIND is OperationKind.CHANGE_MUSIC_VOLUME
