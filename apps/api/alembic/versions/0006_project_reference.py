"""projects.reference_media_id -- the reference video a project is styled after

One nullable pointer, not a table. A project has at most one active reference at
a time, and the profile derived from it is recomputed from analysis rows on
read rather than stored -- so there is nothing here to keep in step with the
analyzers.

``ON DELETE SET NULL`` rather than a JSON field in ``projects.settings``, which
was the cheaper option: deleting the media a project points at must clear the
pointer, and a JSON field would have left a dangling id that every reader would
have to defend against.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-15 15:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("reference_media_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_projects_reference_media_id",
        "projects",
        "media_assets",
        ["reference_media_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_projects_reference_media_id", "projects", type_="foreignkey")
    op.drop_column("projects", "reference_media_id")
