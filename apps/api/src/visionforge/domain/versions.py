"""Edit versions: the history an undo moves through.

    v1  generated          the plan the planner produced
     └─ v2  co_edit        "lower the music to 40%"
         ├─ v3  co_edit    "remove the third clip"      <- undone
         └─ v4  co_edit    "use bold subtitles"         <- current

A version is a *pointer to a plan plus the story of how it got there*. The plan
rows themselves stay append-only exactly as Phase 4 made them: patching writes a
new ``edit_plans`` row and a version that points at it, so a render can always
be traced to the immutable plan it was built from.

**Undo does not write a plan.** It moves which version is current, back to the
parent -- so undoing is instant, costs nothing, and restores the exact bytes that
were there before rather than a recomputed approximation of them. Redo moves
forward to the most recently created child.

**Branching is allowed and is not a merge problem.** Making a change while two
versions back does not erase the abandoned branch; it adds a new child, and redo
then follows the newer one. That is what every editor does, and the alternative
-- refusing to edit until the user redoes their way to the tip -- is worse.

The functions here are pure and take a list of nodes, so "what does redo do
after an undo and then a fresh edit?" is a unit test with no database. The
repository supplies the nodes; nothing in this module knows what a table is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID


class VersionOrigin(StrEnum):
    """How a version came to exist. Stored, shown, never guessed at."""

    #: A planner produced the plan -- the rules engine or the LLM planner.
    GENERATED = "generated"
    #: The user assembled or edited the timeline by hand.
    MANUAL = "manual"
    #: A validated delta was applied to the previous version.
    CO_EDIT = "co_edit"
    #: The head moved back to a version that already existed.
    UNDO = "undo"
    #: The head moved forward again.
    REDO = "redo"


@dataclass(frozen=True, slots=True)
class VersionNode:
    """One version, reduced to what the lineage rules need.

    Deliberately not the ORM row. Undo and redo are questions about a tree, and
    answering them from ``(id, parent, number)`` keeps the rules testable and
    keeps this module from importing a session.
    """

    id: UUID
    version: int
    parent_id: UUID | None
    edit_plan_id: UUID
    is_current: bool


def current_of(nodes: Sequence[VersionNode]) -> VersionNode | None:
    """The version the project is sitting on.

    Falls back to the highest-numbered version when nothing is marked current,
    which is what a history written before this column existed would look like.
    A project with versions but no head is a bug, not a reason to refuse to
    open the editor.
    """
    for node in nodes:
        if node.is_current:
            return node
    return max(nodes, key=lambda node: node.version) if nodes else None


def undo_target(nodes: Sequence[VersionNode]) -> VersionNode | None:
    """The version an undo would move to, or ``None`` when there is none.

    ``None`` is the first version of a project: there is nothing behind it, and
    saying so lets the UI disable the button rather than offer an action that
    fails.
    """
    current = current_of(nodes)
    if current is None or current.parent_id is None:
        return None
    return next((node for node in nodes if node.id == current.parent_id), None)


def redo_target(nodes: Sequence[VersionNode]) -> VersionNode | None:
    """The version a redo would move to.

    The most recently created child of the current version. "Most recent" is the
    highest version number, which is monotonic per project -- so after undoing
    twice and branching, redo follows the branch the user just made rather than
    the one they abandoned.
    """
    current = current_of(nodes)
    if current is None:
        return None
    children = [node for node in nodes if node.parent_id == current.id]
    return max(children, key=lambda node: node.version) if children else None


def lineage(nodes: Sequence[VersionNode]) -> tuple[VersionNode, ...]:
    """The path from the first version to the current one, in order.

    What a history panel draws: the versions that actually led here, not every
    version ever made. An abandoned branch is still in ``nodes`` and is still
    reachable by redo; it is simply not part of how the current edit came to be.
    """
    by_id = {node.id: node for node in nodes}
    current = current_of(nodes)

    chain: list[VersionNode] = []
    seen: set[UUID] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        chain.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None
    return tuple(reversed(chain))


@dataclass(frozen=True, slots=True)
class VersionSummary:
    """What a version says about itself, for the history list.

    Note what is absent: the user's request text. Only a digest of it is kept,
    the same trade Phase 5 made for planning -- enough to tell two requests
    apart or correlate a repeat, not enough to reconstruct what somebody typed.
    """

    id: UUID
    version: int
    parent_id: UUID | None
    edit_plan_id: UUID
    origin: VersionOrigin
    is_current: bool
    #: One line per operation, as the operations described themselves.
    applied: tuple[str, ...]
    operation_count: int
    #: ``rules``, ``llm`` or ``client`` for a co-edit; empty otherwise.
    source: str
    provider: str | None
    model: str | None
    latency_ms: float | None
    request_digest: str | None
    total_duration_ms: int
    segment_count: int
    created_at: str
    #: The newest render of this version's plan, when there is one.
    render_id: UUID | None = None
    render_status: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "version": self.version,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "edit_plan_id": str(self.edit_plan_id),
            "origin": self.origin.value,
            "is_current": self.is_current,
            "applied": list(self.applied),
            "operation_count": self.operation_count,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "request_digest": self.request_digest,
            "total_duration_ms": self.total_duration_ms,
            "segment_count": self.segment_count,
            "created_at": self.created_at,
            "render_id": str(self.render_id) if self.render_id else None,
            "render_status": self.render_status,
        }


def digest_of(text: str | None) -> str | None:
    """A short, stable fingerprint of a request.

    Stored instead of the text. Enough to notice the same request twice or to
    correlate a support report with a row; useless for recovering what the user
    wrote, which is the intent. The same function the planner uses, kept here so
    both halves of the feature fingerprint identically.
    """
    if not text:
        return None
    import hashlib

    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "VersionNode",
    "VersionOrigin",
    "VersionSummary",
    "current_of",
    "digest_of",
    "lineage",
    "redo_target",
    "undo_target",
]
