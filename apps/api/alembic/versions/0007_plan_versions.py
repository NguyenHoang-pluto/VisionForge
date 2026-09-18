"""edit_plan_versions -- the history an undo moves through

A version is a pointer to a plan plus the story of how it got there. The plan
rows stay append-only: patching writes a new ``edit_plans`` row and a version
that points at it, so a render is always traceable to the immutable plan it was
built from, and undo restores the exact bytes rather than a recomputation.

``parent_version_id`` is a self-reference, which makes the history a tree rather
than a list: editing while two versions back adds a branch instead of erasing
one. The partial unique index is what keeps exactly one head per project --
enforced by the database rather than by whoever remembers to clear the old one.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-16 09:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edit_plan_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("edit_plan_id", sa.Uuid(), nullable=False),
        sa.Column("parent_version_id", sa.Uuid(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("operations", JSONB(), nullable=True),
        sa.Column("operation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary", JSONB(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=16), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("request_digest", sa.String(length=32), nullable=True),
        sa.Column("request_chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["edit_plan_id"], ["edit_plans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["parent_version_id"], ["edit_plan_versions.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint("project_id", "version", name="uq_version_project_version"),
    )
    op.create_index("ix_edit_plan_versions_project_id", "edit_plan_versions", ["project_id"])
    op.create_index("ix_edit_plan_versions_edit_plan_id", "edit_plan_versions", ["edit_plan_id"])
    op.create_index(
        "ix_versions_project_created", "edit_plan_versions", ["project_id", "created_at"]
    )
    # One head per project. Partial, so the many not-current rows do not collide.
    op.create_index(
        "uq_version_project_current",
        "edit_plan_versions",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index("uq_version_project_current", table_name="edit_plan_versions")
    op.drop_index("ix_versions_project_created", table_name="edit_plan_versions")
    op.drop_index("ix_edit_plan_versions_edit_plan_id", table_name="edit_plan_versions")
    op.drop_index("ix_edit_plan_versions_project_id", table_name="edit_plan_versions")
    op.drop_table("edit_plan_versions")
