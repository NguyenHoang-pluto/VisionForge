"""Render lifecycle.

Separate from ``JobStatus`` on purpose. A job is a unit of work that may be
retried; a render is an artefact that either exists or does not. They diverge:
a job can be RETRY_WAIT while its render is simply still RENDERING, and a render
can be READY long after its job row has aged out.
"""

from __future__ import annotations

from enum import StrEnum


class RenderStatus(StrEnum):
    PENDING = "pending"
    RENDERING = "rendering"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_RENDER_STATUSES: frozenset[RenderStatus] = frozenset(
    {RenderStatus.READY, RenderStatus.FAILED, RenderStatus.CANCELLED}
)


def render_key(project_id: str, render_id: str) -> str:
    """Object key for a finished render.

    Derived from identifiers only, like every other key in the system: there is
    no parameter through which a client string could become a path component.
    """
    return f"projects/{project_id}/renders/{render_id}/output.mp4"
