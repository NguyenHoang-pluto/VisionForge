"""Declarative base for all ORM models.

Lives apart from the engine so that Alembic can import metadata without opening
a connection pool.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for VisionForge ORM models.

    Phase 1 defines no tables: the first migration only installs the ``vector``
    extension. Domain tables arrive in Phase 2 with the media and job modules.
    """
