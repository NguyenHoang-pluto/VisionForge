"""Synchronous session factory for Celery workers.

Workers are synchronous processes doing blocking work (subprocesses, file I/O).
Driving an async engine from them would mean running a loop per task for no
benefit. ``postgresql+psycopg`` (psycopg3) backs both engines, so this is the same
driver the API uses, not a second dependency.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from visionforge.core.config import get_settings

_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def get_sync_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_size=2,
            max_overflow=2,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_sync_sessionmaker() -> sessionmaker[Session]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(bind=get_sync_engine(), expire_on_commit=False)
    return _sessionmaker


def dispose_sync_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _sessionmaker = None
