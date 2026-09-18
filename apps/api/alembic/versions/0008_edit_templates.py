"""edit_templates -- templates a user measured from a video

A template is an edit's structure with the footage taken out: slot lengths,
energies, roles, joins. The library templates ship in code; this table holds
the ones a user measured from a video of their own, so they can be reused in
any project.

Owned by the user rather than a project, which is the whole point: a template
tied to the project it was measured in could not be used anywhere else. The
video it came from is referenced with ``ON DELETE SET NULL``, so deleting that
video -- or its whole project -- leaves the template standing.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-18 16:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edit_templates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=60), nullable=False),
        sa.Column("source_media_id", sa.Uuid(), nullable=True),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("extraction", JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_media_id"], ["media_assets.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_edit_templates_user_created", "edit_templates", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_edit_templates_user_created", table_name="edit_templates")
    op.drop_table("edit_templates")
