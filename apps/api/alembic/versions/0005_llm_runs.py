"""llm_runs -- observability for LLM planning

One row per planning call, successful or not. The rows worth having are the
failures: a provider timeout, a rejected directive, a fallback to the rules
engine. A column on edit_plans could not record those, because a failed run
produces no plan.

Stores a digest of the user's request, never the request itself, and never the
prompt or the completion.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-12 17:39:49.455774
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("edit_plan_id", sa.Uuid(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("request_digest", sa.String(length=32), nullable=True),
        sa.Column("request_chars", sa.Integer(), nullable=False),
        sa.Column("fallback_reason", sa.String(length=48), nullable=True),
        sa.Column("fallback_detail", sa.Text(), nullable=True),
        sa.Column("violations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["edit_plan_id"], ["edit_plans.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_llm_runs_project_created", "llm_runs", ["project_id", "created_at"], unique=False
    )
    op.create_index(op.f("ix_llm_runs_project_id"), "llm_runs", ["project_id"], unique=False)
    op.create_index(op.f("ix_llm_runs_status"), "llm_runs", ["status"], unique=False)
    op.create_index(
        "ix_llm_runs_status_created", "llm_runs", ["status", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_llm_runs_status_created", table_name="llm_runs")
    op.drop_index(op.f("ix_llm_runs_status"), table_name="llm_runs")
    op.drop_index(op.f("ix_llm_runs_project_id"), table_name="llm_runs")
    op.drop_index("ix_llm_runs_project_created", table_name="llm_runs")
    op.drop_table("llm_runs")
