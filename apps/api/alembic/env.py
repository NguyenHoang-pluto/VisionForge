"""Alembic environment.

The database URL comes from ``Settings`` rather than alembic.ini so that
migrations and the application can never disagree about which database they mean.

Migrations run through a *synchronous* engine. Alembic is a one-shot CLI
operation with no concurrency to exploit, and a sync engine avoids an event-loop
dependency the migration step does not need -- which on Windows also sidesteps
the psycopg/ProactorEventLoop incompatibility entirely.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from visionforge.core.config import get_settings
from visionforge.infra.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _database_url() -> str:
    """The application's database URL, escaped for alembic's configparser.

    No driver rewrite is needed: ``postgresql+psycopg`` is psycopg3, whose single
    dialect backs both the sync engine used here and the async engine used by the
    application. That dual-mode support is why psycopg3 was chosen over psycopg2.
    """
    return get_settings().database_url.replace("%", "%%")


config.set_main_option("sqlalchemy.url", _database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
