"""The AI co-editor use case: propose, validate, apply, and move through history.

    current version -> plan
                    -> shape            (no ids, no names, no paths)
                    -> rules or model   -> EditDelta
                    -> apply_delta      -> new plan, or violations and no change
                    -> new plan row + new version row, in one transaction

Four properties this layer is responsible for, none of which the domain can
enforce on its own because they are about persistence:

**The model never writes.** It returns operations. This service parses them,
applies them deterministically, validates the result against the real media
rows, and only then writes -- and what it writes is a plan document, never a
command. There is no path from a completion to the database that does not pass
through ``apply_delta``.

**Nothing is half-applied.** A rejected patch writes no plan row, no version row
and no event. The previous version stays current and the previous plan stays
byte-identical, which is what makes a failed AI request safe rather than
destructive.

**The current version is the one that gets edited.** Apply takes the version it
was previewed against and refuses if the head has moved -- otherwise a change
previewed against version 3 could land on version 5 and silently mean something
else. Manual edits and AI edits share one head, which is what lets the two
interleave.

**Applying does not render.** A version is a plan; rendering it is a separate,
explicit request through the existing route. A model that could queue FFmpeg
jobs by being asked nicely would be a cost bug at best.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import anyio.to_thread

from visionforge.application.edit_service import MediaRecord, _media_facts
from visionforge.domain.coeditor import (
    COEDIT_PROMPT_VERSION,
    CoEditCommand,
    CoEditFailure,
    CoEditOutcome,
    CoEditSource,
    PlanShape,
    propose,
    recognise_command,
    shape_of,
)
from visionforge.domain.editdelta import (
    DeltaViolation,
    EditDelta,
    parse_operations,
)
from visionforge.domain.editplan import EditPlan, plan_from_payload
from visionforge.domain.errors import ConflictError, NotFoundError, ValidationError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.llm import LlmProvider
from visionforge.domain.patch import PatchContext, PatchOutcome, PlanDiff, apply_delta
from visionforge.domain.render import RenderStatus
from visionforge.domain.style import FPS_PRESETS, PRESET_DIMENSIONS
from visionforge.domain.versions import (
    VersionNode,
    VersionOrigin,
    VersionSummary,
    current_of,
    digest_of,
    redo_target,
    undo_target,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CoEditPreview:
    """What a change would do, before anything is written.

    Carries the concrete operations so the client can hand them straight back to
    ``apply`` -- which re-parses and re-applies them from scratch rather than
    trusting the round trip. That is what stops Apply from re-running a model
    the user has already paid for, without making the client's copy of the
    operations authoritative.
    """

    ok: bool
    outcome: CoEditOutcome
    patch: PatchOutcome | None
    base_version_id: UUID | None
    base_version: int | None

    @property
    def diff(self) -> PlanDiff:
        return self.patch.diff if self.patch else PlanDiff()

    def as_payload(self) -> dict[str, Any]:
        patch = self.patch
        return {
            "ok": self.ok,
            "base_version_id": str(self.base_version_id) if self.base_version_id else None,
            "base_version": self.base_version,
            "operations": (
                [operation.as_payload() for operation in self.outcome.delta.operations]
                if self.outcome.delta
                else []
            ),
            "rationale": self.outcome.delta.rationale if self.outcome.delta else "",
            "diff": self.diff.as_payload(),
            "failure": self.outcome.failure.value if self.outcome.failure else None,
            "detail": self.outcome.detail,
            "source": self.outcome.source.value,
            "provider": self.outcome.provider,
            "model": self.outcome.model,
            "latency_ms": round(self.outcome.latency_ms, 1),
            "violations": [
                violation.as_payload()
                for violation in (
                    patch.violations if patch and not patch.ok else self.outcome.violations
                )
            ],
            # A preview that only needs the rules is safe to apply without a
            # confirmation step; one a model proposed is shown first.
            "needs_confirmation": self.outcome.source is not CoEditSource.RULES,
        }


@dataclass(frozen=True, slots=True)
class AppliedChange:
    """A committed change: the new version, its plan, and what it did."""

    version: Any
    plan_row: Any
    diff: PlanDiff
    outcome: CoEditOutcome


class CoEditService:
    """Co-editing an existing plan, and moving through the versions it produced."""

    def __init__(
        self,
        session: Any,
        media_repo: Any,
        plan_repo: Any,
        versions_repo: Any,
        events: Any,
    ) -> None:
        self._session = session
        self._media = media_repo
        self._plans = plan_repo
        self._versions = versions_repo
        self._events = events

    # --------------------------------------------------------------- history
    async def history(self, project_id: ProjectId, *, limit: int = 50) -> list[VersionSummary]:
        """Every version, newest first, with what it changed and what it rendered.

        The render status travels with the version because "which of these have
        I actually made a file from" is the question a history list is for.
        """
        rows = await self._versions.list_for_project(project_id, limit=limit)
        if not rows:
            return []

        plans: dict[UUID, Any] = {}
        for plan_id in dict.fromkeys(row.edit_plan_id for row in rows):
            # One read per distinct plan, not per version: an undo and the
            # version it restored point at the same plan.
            plans[plan_id] = await self._plans.get_in_project(plan_id, project_id)

        renders = await self._plans.list_renders(project_id, limit=200)
        newest_render: dict[UUID, Any] = {}
        for render in renders:
            # Rows arrive newest first, so the first one seen per plan is the
            # newest -- and a plan re-rendered after an edit shows that render.
            newest_render.setdefault(render.edit_plan_id, render)

        summaries: list[VersionSummary] = []
        for row in rows:
            plan_row = plans.get(row.edit_plan_id)
            render = newest_render.get(row.edit_plan_id)
            summaries.append(
                VersionSummary(
                    id=row.id,
                    version=row.version,
                    parent_id=row.parent_version_id,
                    edit_plan_id=row.edit_plan_id,
                    origin=_origin_of(row.origin),
                    is_current=row.is_current,
                    applied=tuple((row.summary or {}).get("applied", [])),
                    operation_count=row.operation_count,
                    source=row.source or "",
                    provider=row.provider,
                    model=row.model,
                    latency_ms=row.latency_ms,
                    request_digest=row.request_digest,
                    total_duration_ms=plan_row.total_duration_ms if plan_row else 0,
                    segment_count=plan_row.segment_count if plan_row else 0,
                    created_at=row.created_at.isoformat(),
                    render_id=render.id if render else None,
                    render_status=(RenderStatus(render.status).value if render else None),
                )
            )
        return summaries

    async def current_version(self, project_id: ProjectId) -> Any:
        return await self._versions.current(project_id)

    async def record_plan(
        self,
        *,
        project_id: ProjectId,
        edit_plan_id: UUID,
        origin: VersionOrigin,
        request_text: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> Any:
        """Record a plan that a planner or the manual editor produced.

        Called from the plan-creating routes so that generating an edit and
        hand-cutting one both land in the same history the co-editor moves
        through. Without this the co-editor would have no root to patch from and
        an undo could step past the beginning of the story.
        """
        parent = await self._versions.current(project_id)
        return await self._versions.append(
            project_id=project_id,
            edit_plan_id=edit_plan_id,
            origin=origin.value,
            parent_version_id=parent.id if parent else None,
            provider=provider,
            model=model,
            request_digest=digest_of(request_text),
            request_chars=len(request_text or ""),
        )

    # --------------------------------------------------------------- preview
    async def preview(
        self,
        *,
        project_id: ProjectId,
        request_text: str | None,
        provider: LlmProvider | None,
        operations: list[dict[str, Any]] | None = None,
        timeout_s: float = 30.0,
        max_output_tokens: int = 1_500,
    ) -> CoEditPreview:
        """Work out what a change would do. Writes nothing, ever.

        The whole pipeline runs -- resolve, parse, patch, validate against the
        real media rows -- and the result is thrown away except for the diff.
        That is deliberate: the preview a user approves is produced by exactly
        the code that will produce the committed version, so there is no way for
        the two to disagree.
        """
        version, plan = await self._current_plan(project_id)
        context, _ = await self._context(project_id)
        shape = shape_of(plan)

        outcome = await self._resolve(
            shape=shape,
            request_text=request_text,
            operations=operations,
            provider=provider,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
        )
        if not outcome.ok or outcome.delta is None:
            return CoEditPreview(
                ok=False,
                outcome=outcome,
                patch=None,
                base_version_id=version.id,
                base_version=version.version,
            )

        patch = apply_delta(plan, outcome.delta, context)
        return CoEditPreview(
            ok=patch.ok,
            outcome=outcome,
            patch=patch,
            base_version_id=version.id,
            base_version=version.version,
        )

    # ----------------------------------------------------------------- apply
    async def apply(
        self,
        *,
        project_id: ProjectId,
        request_text: str | None,
        provider: LlmProvider | None,
        operations: list[dict[str, Any]] | None = None,
        base_version_id: UUID | None = None,
        timeout_s: float = 30.0,
        max_output_tokens: int = 1_500,
    ) -> AppliedChange:
        """Apply a change and commit it as a new version.

        Refuses if the head has moved since the preview: a change the user
        approved against version 3 must not land on version 5, where the clip
        they meant may be a different clip. Re-previewing is cheap and being
        wrong about which edit was changed is not.
        """
        version, plan = await self._current_plan(project_id)
        if base_version_id is not None and base_version_id != version.id:
            raise ConflictError(
                "the edit has changed since this change was previewed",
                hint="Review the current edit and ask again.",
            )

        context, _ = await self._context(project_id)
        outcome = await self._resolve(
            shape=shape_of(plan),
            request_text=request_text,
            operations=operations,
            provider=provider,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
        )

        if not outcome.ok or outcome.delta is None:
            # Nothing is written. The existing plan and the existing head are
            # exactly as they were, which is the behaviour that makes a provider
            # outage survivable rather than destructive.
            await self._record_failure(project_id, version, outcome, request_text)
            raise CoEditRejected(outcome, ())

        patch = apply_delta(plan, outcome.delta, context)
        if not patch.ok:
            await self._record_failure(project_id, version, outcome, request_text)
            raise CoEditRejected(outcome, patch.violations)

        row = await self._plans.create(
            project_id=project_id,
            plan=patch.plan,
            # A patched plan explains itself through its operations and its
            # parent; the selection that produced the original is not re-run and
            # claiming it would be a copy of somebody else's reasoning.
            selection={"selected": [], "rejected": [], "duplicate_groups": []},
        )
        new_version = await self._versions.append(
            project_id=project_id,
            edit_plan_id=row.id,
            origin=VersionOrigin.CO_EDIT.value,
            parent_version_id=version.id,
            operations=[operation.as_payload() for operation in outcome.delta.operations],
            summary=patch.diff.as_payload(),
            source=outcome.source.value,
            provider=outcome.provider or None,
            model=outcome.model or None,
            prompt_version=COEDIT_PROMPT_VERSION,
            latency_ms=outcome.latency_ms,
            request_digest=digest_of(request_text),
            request_chars=len(request_text or ""),
        )

        await self._events.record(
            kind="editplan.coedited",
            actor="dev-user",
            project_id=project_id,
            payload={
                "edit_plan_id": str(row.id),
                "version": new_version.version,
                "parent_version": version.version,
                "operations": outcome.delta.summary,
                "source": outcome.source.value,
                # The digest travels; the text it fingerprints does not.
                "request_digest": digest_of(request_text),
            },
        )
        await self._session.commit()
        return AppliedChange(version=new_version, plan_row=row, diff=patch.diff, outcome=outcome)

    # ---------------------------------------------------------- undo / redo
    async def undo(self, project_id: ProjectId) -> Any:
        """Move the head back one version. Writes no plan.

        The version being restored already exists and already points at a stored
        plan, so what comes back is the exact edit that was there -- not a
        recomputation of it, and not an inverse delta that might not compose.
        """
        return await self._move(project_id, undo_target, "there is nothing to undo")

    async def redo(self, project_id: ProjectId) -> Any:
        return await self._move(project_id, redo_target, "there is nothing to redo")

    async def _move(
        self,
        project_id: ProjectId,
        pick: Any,
        empty_message: str,
    ) -> Any:
        rows = await self._versions.list_for_project(project_id)
        nodes = _nodes(rows)
        if not nodes:
            raise NotFoundError("this project has no edit history yet")

        target = pick(nodes)
        if target is None:
            raise ConflictError(empty_message)

        await self._versions.set_current(project_id, target.id)
        await self._events.record(
            kind="editplan.version_moved",
            actor="dev-user",
            project_id=project_id,
            payload={"version": target.version, "edit_plan_id": str(target.edit_plan_id)},
        )
        await self._session.commit()

        moved = await self._versions.get_in_project(target.id, project_id)
        return moved

    async def restore(self, project_id: ProjectId, version_id: UUID) -> Any:
        """Jump to any version in the history.

        The generalisation of undo and redo, and the same mechanism: the head
        moves, nothing is written, and the plan that becomes current is the one
        that was stored at the time.
        """
        target = await self._versions.get_in_project(version_id, project_id)
        if target is None:
            raise NotFoundError("version not found in this project")

        await self._versions.set_current(project_id, version_id)
        await self._events.record(
            kind="editplan.version_moved",
            actor="dev-user",
            project_id=project_id,
            payload={"version": target.version, "edit_plan_id": str(target.edit_plan_id)},
        )
        await self._session.commit()
        return await self._versions.get_in_project(version_id, project_id)

    # ------------------------------------------------------------- internals
    async def _resolve(
        self,
        *,
        shape: PlanShape,
        request_text: str | None,
        operations: list[dict[str, Any]] | None,
        provider: LlmProvider | None,
        timeout_s: float,
        max_output_tokens: int,
    ) -> CoEditOutcome:
        """Operations from the client, from the rules, or from a model.

        Client-supplied operations are re-parsed here rather than trusted. They
        came from a preview this server produced, but they arrived over HTTP,
        and "we sent it a moment ago" is not a reason to skip the gate that
        exists precisely because input is untrusted.
        """
        if operations is not None:
            parsed, violations = await anyio.to_thread.run_sync(parse_operations, operations)
            if violations or not parsed:
                return CoEditOutcome(
                    ok=False,
                    failure=CoEditFailure.NO_USABLE_OPERATIONS,
                    detail="; ".join(violation.message for violation in violations)[:300],
                    source=CoEditSource.CLIENT,
                    violations=tuple(violations),
                )
            return CoEditOutcome(
                ok=True,
                delta=EditDelta(
                    operations=parsed,
                    rationale="",
                    source=CoEditSource.CLIENT.value,
                ),
                source=CoEditSource.CLIENT,
            )

        # On a worker thread: the rules are microseconds of regex, but a model
        # is a network round trip, and one slow proposal must not stall every
        # other request sharing this event loop.
        return await anyio.to_thread.run_sync(
            lambda: propose(
                provider,
                shape,
                request_text=request_text,
                timeout_s=timeout_s,
                max_output_tokens=max_output_tokens,
            )
        )

    async def _current_plan(self, project_id: ProjectId) -> tuple[Any, EditPlan]:
        version = await self._versions.current(project_id)
        if version is None:
            raise NotFoundError(
                "this project has no edit to change yet",
            )

        row = await self._plans.get_in_project(version.edit_plan_id, project_id)
        if row is None:
            raise NotFoundError("the current version's plan is missing")

        try:
            return version, plan_from_payload(row.plan)
        except (KeyError, ValueError) as exc:
            # A stored plan this build cannot read is a permanent failure, not a
            # reason to patch something half-understood.
            raise ValidationError(
                "the stored plan cannot be read by this version of the server",
                hint=str(exc),
            ) from exc

    async def _context(self, project_id: ProjectId) -> tuple[PatchContext, Sequence[MediaRecord]]:
        """The server-side facts a patch is checked against.

        Read from the database on every call. A trim is bounded by how long the
        source really is, not by anything the request said -- which is the same
        rule the planner's music facts follow, and for the same reason.
        """
        records = await self._media.records_with_analysis(project_id)
        durations: dict[MediaId, int] = {
            record.media_id: record.duration_ms or 0
            for record in records
            if record.duration_ms is not None
        }
        return (
            PatchContext(
                source_durations=durations,
                dimensions=PRESET_DIMENSIONS,
                fps_presets=FPS_PRESETS,
                media_facts=_media_facts(project_id, records),
            ),
            records,
        )

    async def _record_failure(
        self,
        project_id: ProjectId,
        version: Any,
        outcome: CoEditOutcome,
        request_text: str | None,
    ) -> None:
        """Note that a change was refused, without writing a version for it.

        A refusal is worth seeing -- it is how "the provider is down" or "the
        model keeps proposing invalid trims" becomes visible -- but it is not a
        version: nothing changed, and a history full of non-events is a history
        nobody reads.
        """
        await self._events.record(
            kind="editplan.coedit_rejected",
            actor="dev-user",
            project_id=project_id,
            payload={
                "version": version.version,
                "failure": outcome.failure.value if outcome.failure else "invalid_patch",
                "source": outcome.source.value,
                "request_digest": digest_of(request_text),
            },
        )
        await self._session.commit()


class CoEditRejected(Exception):
    """A change could not be made, and nothing was written.

    Carries both halves of the story: how far the request got (was it read at
    all?) and which validations the resulting plan failed. The route turns it
    into a 422 the panel can explain.
    """

    def __init__(self, outcome: CoEditOutcome, violations: Sequence[DeltaViolation]) -> None:
        self.outcome = outcome
        self.violations = tuple(violations)
        detail = "; ".join(violation.message for violation in violations) or outcome.detail
        super().__init__(detail or "the change could not be applied")


def _nodes(rows: Sequence[Any]) -> list[VersionNode]:
    return [
        VersionNode(
            id=row.id,
            version=row.version,
            parent_id=row.parent_version_id,
            edit_plan_id=row.edit_plan_id,
            is_current=row.is_current,
        )
        for row in rows
    ]


def _origin_of(value: str) -> VersionOrigin:
    """A stored origin, or ``CO_EDIT`` for one this build does not know.

    Version rows are JSON-adjacent history that outlives the code that wrote
    them; an unrecognised origin is a row from another version rather than an
    error worth failing a history listing over.
    """
    try:
        return VersionOrigin(value)
    except ValueError:
        return VersionOrigin.CO_EDIT


__all__ = [
    "AppliedChange",
    "CoEditCommand",
    "CoEditPreview",
    "CoEditRejected",
    "CoEditService",
    "VersionOrigin",
    "current_of",
    "recognise_command",
]
