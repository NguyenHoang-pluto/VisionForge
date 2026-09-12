"""Planning and rendering use cases.

Builds selection candidates from stored analysis, runs a planner, validates what
it produced, persists the plan, and dispatches renders.

The planner is injected as a ``Planner``, never constructed here. That is what
makes swapping the rules engine for an LLM in Phase 5 a wiring change: this
service already cannot tell the difference, because all it ever receives back is
an ``EditPlan``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from visionforge.domain.analysis import AnalysisStatus, AnalyzerName
from visionforge.domain.editplan import MediaFact, PlanInvalidError, validate_plan
from visionforge.domain.errors import ConflictError, NotFoundError, ValidationError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.planner import (
    NoUsableMediaError,
    Planner,
    PlanOutcome,
    PlanRequest,
)
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


def build_candidate(record: MediaRecord) -> Candidate:
    """Flatten one asset's analysis into the signals selection uses.

    Missing analysis produces ``None`` fields rather than defaults. A zero would
    be indistinguishable from a genuinely black frame, and the selector needs to
    reject "never analysed" for a different reason than "analysed and bad".
    """
    quality = record.analysis.get(AnalyzerName.QUALITY, {})
    phash = record.analysis.get(AnalyzerName.PHASH, {})
    scenes = record.analysis.get(AnalyzerName.SCENES, {})

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
    ) -> None:
        self._session = session
        self._media = media_repo
        self._plans = plan_repo
        self._events = events
        self._planner = planner

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

        candidates = [build_candidate(record) for record in records]

        try:
            outcome = self._planner.plan(request, candidates)
        except NoUsableMediaError as exc:
            raise ConflictError(
                str(exc),
                hint="Add more footage, or lower the clip requirement.",
            ) from exc

        facts = {
            record.media_id: MediaFact(
                media_id=record.media_id,
                project_id=project_id,
                is_renderable=(
                    record.status is MediaStatus.READY and record.kind is MediaKind.VIDEO
                ),
                duration_ms=record.duration_ms,
                width=record.width,
                height=record.height,
            )
            for record in records
        }

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

        row = await self._plans.create(
            project_id=project_id,
            plan=outcome.plan,
            selection=selection_payload(outcome.selection),
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
            },
        )
        await self._session.commit()
        return row, outcome

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


__all__ = [
    "AnalysisStatus",
    "EditService",
    "MediaRecord",
    "build_candidate",
    "selection_payload",
]
