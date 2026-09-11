"""Enable the pgvector extension.

The first migration installs the extension rather than creating tables. Phase 0
decided embeddings arrive in week 5 for dedupe and selection; installing the
extension now means that work is a plain column addition rather than a migration
that has to run as a superuser at an awkward moment.

Revision ID: 0001
Revises:
Create Date: 2026-09-11
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
