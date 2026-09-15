"""Planning and rendering use cases.

Builds selection candidates from stored analysis, runs a planner, validates what
it produced, persists the plan, and dispatches renders.

The planner is injected as a ``Planner``, never constructed here. That is what
made adding the LLM planner in Phase 5 a wiring change: this service still
cannot tell the difference, because all it ever receives back is an
``EditPlan``, and the validation below runs identically either way.

The one concession Phase 5 required is that planning now runs on a worker
thread. A rules plan is microseconds of arithmetic, but an LLM plan is a network
round trip, and blocking the event loop for twenty seconds would stall every
other request in the process.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import anyio.to_thread

from visionforge.domain.analysis import AnalysisStatus, AnalyzerName
from visionforge.domain.editplan import (
    Cut,
    MediaFact,
    MusicCue,
    OutputSpec,
    PlanInvalidError,
    plan_from_cuts,
    validate_plan,
)
from visionforge.domain.errors import ConflictError, NotFoundError, ValidationError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.llm_planner import FallbackPlanner, LlmRunRecord
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.planner import (
    NoUsableMediaError,
    Planner,
    PlanOutcome,
    PlanRequest,
)
from visionforge.domain.reference import without_reference
from visionforge.domain.selection import Candidate, SelectionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """One asset with its analysis, flattened for candidate construction.

    Assembled by the repository so this service needs no ORM knowledge and can
    be tested with plain data.
    """

    media_id: MediaId
    kind: MediaKind
    status: MediaStatus
    duration_ms: int | None
    width: int | None
    height: int | None
    created_order: int
    analysis: dict[AnalyzerName, dict[str, Any]]
    #: Audio channel count from ffprobe at ingest. ``None`` means no audio
    #: stream, which is what decides whether asking for source audio in the
    #: output means anything.
    channels: int | None = None


def build_candidate(record: MediaRecord) -> Candidate:
    """Flatten one asset's analysis into the signals selection uses.

    Missing analysis produces ``None`` fields rather than defaults. A zero would
    be indistinguishable from a genuinely black frame, and the selector needs to
    reject "never analysed" for a different reason than "analysed and bad".
    """
    quality = record.analysis.get(AnalyzerName.QUALITY, {})
    phash = record.analysis.get(AnalyzerName.PHASH, {})
    scenes = record.analysis.get(AnalyzerName.SCENES, {})
    faces = record.analysis.get(AnalyzerName.FACES, {})

    clipped: float | None = None
    under = quality.get("frames", [{}])[0].get("underexposed_ratio") if quality else None
    over = quality.get("frames", [{}])[0].get("overexposed_ratio") if quality else None
    if under is not None or over is not None:
        clipped = float(under or 0.0) + float(over or 0.0)

    return Candidate(
        media_id=record.media_id,
        kind=record.kind,
        is_ready=record.status is MediaStatus.READY,
        duration_ms=record.duration_ms,
        width=record.width,
        height=record.height,
        blur_score=_as_float(quality.get("blur_score")),
        contrast=_as_float(quality.get("contrast")),
        mean_luminance=_as_float(quality.get("mean_luminance")),
        clipped_ratio=clipped,
        phash=phash.get("phash") if phash else None,
        scene_count=scenes.get("scene_count") if scenes else None,
        # A boolean, from a count, and no further. Which faces, or whose, is
        # never computed and never stored (Phase 3, ADR-0008); "someone is on
        # screen" is the strongest claim available and all a planner needs.
        # ``None`` when detection has not run, so "not analysed" stays
        # distinguishable from "analysed, nobody there".
        has_faces=(bool(faces.get("face_count", 0)) if faces else None),
        has_audio=bool(record.channels),
        sequence=record.created_order,
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selection_payload(selection: SelectionResult) -> dict[str, Any]:
    """Serialise a selection so a plan can explain itself.

    Both halves are stored. An automatic edit that shows what it chose but not
    what it discarded, and why, is not reviewable -- and being reviewable is the
    main thing a deterministic planner has over a model.
    """
    return {
        "selected": [
            {
                "media_id": str(item.candidate.media_id),
                "score": item.score,
                "components": item.components,
            }
            for item in selection.selected
        ],
        "rejected": [
            {
                "media_id": str(rejection.media_id),
                "reason": rejection.reason.value,
                "detail": rejection.detail,
            }
            for rejection in selection.rejected
        ],
        "duplicate_groups": [
            [str(media_id) for media_id in group] for group in selection.duplicate_groups
        ],
    }


class EditService:
    """Plan generation and render dispatch."""

    def __init__(
        self,
        session: Any,
        media_repo: Any,
        plan_repo: Any,
        events: Any,
        planner: Planner,
        llm_runs: Any = None,
        mode_decision: Any = None,
    ) -> None:
        self._session = session
        self._media = media_repo
        self._plans = plan_repo
        self._events = events
        self._planner = planner
        self._llm_runs = llm_runs
        self._mode_decision = mode_decision

    # ------------------------------------------------------------------ plan
    async def create_plan(
        self, *, project_id: ProjectId, request: PlanRequest
    ) -> tuple[Any, PlanOutcome]:
        """Select, plan, validate, persist.

        Validation runs against the same database rows the candidates came from.
        A plan that fails here is never stored: an invalid plan on disk is a
        trap for whatever reads it next.
        """
        records = await self._media.records_with_analysis(project_id)
        if not records:
            raise ValidationError(
                "project has no analysed media",
                hint="Upload media and run analysis before generating an edit.",
            )

        # The reference is the user's own media, and it is still not footage
        # for this edit: see ``without_reference``. Removed here, where
        # candidates are built, so every planner is covered by one line.
        candidates = without_reference(
            [build_candidate(record) for record in records],
            request.reference_media_id,
        )

        try:
            # On a worker thread: the rules engine does not need it, but an LLM
            # planner performs a network round trip, and one slow plan must not
            # stall every other request sharing this event loop.
            outcome = await anyio.to_thread.run_sync(self._planner.plan, request, candidates)
        except NoUsableMediaError as exc:
            raise ConflictError(
                str(exc),
                hint="Add more footage, or lower the clip requirement.",
            ) from exc

        facts = _media_facts(project_id, records)

        violations = validate_plan(outcome.plan, facts)
        if violations:
            # The planner produced something invalid. That is a bug in the
            # planner, not user error, so it is logged as one -- and it is
            # exactly the failure mode the gate exists to catch when an LLM
            # takes this seat in Phase 5.
            logger.error(
                "planner produced an invalid plan",
                extra={
                    "planner": self._planner.name,
                    "violations": [v.code for v in violations],
                },
            )
            raise PlanInvalidError(violations)

        # Stamp how the plan was chosen before it is stored, so a plan is
        # self-describing: which mode ran, why that mode, and whether it was
        # what the caller asked for.
        if self._mode_decision is not None:
            outcome.plan.metadata["mode"] = self._mode_decision.as_payload()

        run = _run_record(self._planner)
        if run is not None:
            outcome.plan.metadata["llm"] = run.as_payload()

        row = await self._plans.create(
            project_id=project_id,
            plan=outcome.plan,
            selection=selection_payload(outcome.selection),
        )

        # Recorded even when the model succeeded, and especially when it did
        # not: a table that only holds successes cannot answer "how often does
        # this fall back?", which is the question worth asking of the feature.
        if run is not None and self._llm_runs is not None:
            await self._llm_runs.record(
                project_id=project_id, edit_plan_id=row.id, run=run.as_payload()
            )
        await self._events.record(
            kind="editplan.created",
            actor="dev-user",
            project_id=project_id,
            payload={
                "edit_plan_id": str(row.id),
                "planner": self._planner.name,
                "segments": len(outcome.plan.segments),
                "duration_ms": outcome.plan.total_duration_ms,
                "fallback": bool(run and run.fallback_reason),
            },
        )
        await self._session.commit()
        return row, outcome

    # ---------------------------------------------------------- manual plan
    async def create_manual_plan(
        self,
        *,
        project_id: ProjectId,
        cuts: Sequence[Cut],
        output: OutputSpec,
        music: MusicCue | None = None,
        derived_from: UUID | None = None,
    ) -> Any:
        """Persist an edit the user assembled themselves.

        No planner runs. The caller has already decided which clips, in which
        order, trimmed where -- this method's job is to check that decision
        against the media it names and store it as a plan like any other.

        Everything downstream is unchanged: the same validator, the same plan
        row, the same render route. That is the point of the boundary. The
        timeline in the browser is a *proposal*; it becomes an edit only once
        the server has agreed it is renderable, and the render worker never sees
        anything but a stored, validated plan.
        """
        if not cuts:
            raise ValidationError(
                "the timeline is empty",
                hint="Add at least one clip before saving the edit.",
            )

        records = await self._media.records_with_analysis(project_id)
        facts = _media_facts(project_id, records)

        plan = plan_from_cuts(
            project_id=project_id,
            cuts=cuts,
            output=output,
            # A hand-placed bed goes through the same validator as a planned
            # one: the gate does not care who chose the numbers.
            music=music,
            metadata={
                "source": "manual",
                # Which automatic plan this was cut from, when it was cut from
                # one. A hand-edited highlight reel stays traceable to the plan
                # that proposed it rather than appearing out of nowhere.
                "derived_from_edit_plan_id": str(derived_from) if derived_from else None,
            },
        )

        violations = validate_plan(plan, facts)
        if violations:
            # Unlike a planner violation, this one is the user's: they trimmed
            # past the end of a source, or left a clip too short to render. It
            # is reported as a rejected request, not logged as a bug.
            raise PlanInvalidError(violations)

        row = await self._plans.create(
            project_id=project_id,
            plan=plan,
            # A manual edit has no selection to explain -- the user is the
            # selector. Stored with the same keys as an automatic plan so every
            # reader of this column sees one shape.
            selection={"selected": [], "rejected": [], "duplicate_groups": []},
        )
        await self._events.record(
            kind="editplan.created",
            actor="dev-user",
            project_id=project_id,
            payload={
                "edit_plan_id": str(row.id),
                "planner": "manual",
                "segments": len(plan.segments),
                "duration_ms": plan.total_duration_ms,
                "fallback": False,
            },
        )
        await self._session.commit()
        return row

    # ---------------------------------------------------------------- render
    async def create_render(self, *, project_id: ProjectId, edit_plan_id: UUID) -> Any:
        """Create a pending render row for a plan.

        The row exists before the job is dispatched, for the same reason a media
        row exists before its bytes arrive: the caller gets an id to follow, and
        a render that was requested but never queued is visible rather than lost.
        """
        plan_row = await self._plans.get_in_project(edit_plan_id, project_id)
        if plan_row is None:
            raise NotFoundError("edit plan not found in this project")

        render = await self._plans.create_render(project_id=project_id, edit_plan_id=edit_plan_id)
        await self._session.commit()
        return render


def _media_facts(project_id: ProjectId, records: Sequence[MediaRecord]) -> dict[MediaId, MediaFact]:
    """What the plan validator is allowed to know about this project's media.

    One definition, used by both the automatic and the manual path, so a plan
    cut by hand is checked against exactly the same view of the world as one a
    planner produced.
    """
    return {
        record.media_id: MediaFact.from_media(
            media_id=record.media_id,
            project_id=project_id,
            kind=record.kind,
            status=record.status,
            duration_ms=record.duration_ms,
            width=record.width,
            height=record.height,
        )
        for record in records
    }


def _run_record(planner: Planner) -> LlmRunRecord | None:
    """The LLM run behind a plan, when there was one.

    ``isinstance`` rather than duck typing on a ``last_run`` attribute: the
    record's shape is what gets persisted, and accepting anything that happens
    to have that attribute name would let a future planner write a differently
    shaped row into the same table.
    """
    return planner.last_run if isinstance(planner, FallbackPlanner) else None


__all__ = [
    "AnalysisStatus",
    "EditService",
    "MediaRecord",
    "build_candidate",
    "selection_payload",
]
