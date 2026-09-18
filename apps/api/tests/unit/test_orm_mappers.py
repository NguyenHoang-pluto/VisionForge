"""The ORM mappers configure.

SQLAlchemy resolves relationships lazily, at first use. A mapping that cannot be
configured therefore raises nothing at import and nothing at startup -- it
raises on the first query, which in an API means a 500 on every route at once.

Phase 8 hit exactly that. Adding ``projects.reference_media_id`` created a second
foreign-key path between ``projects`` and ``media_assets``, and SQLAlchemy could
no longer tell which one ``Project.media`` meant. Every unit test passed, mypy
passed, ruff passed, the migration applied, and the application was completely
broken -- discovered by the acceptance script on its first HTTP call.

One assertion, no database, a fraction of a second.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import configure_mappers

from visionforge.infra.db.base import Base
from visionforge.infra.db.models import MediaAsset, Project


class TestMapperConfiguration:
    def test_every_mapping_resolves(self) -> None:
        """The whole registry, not one model: a broken relationship on any of
        them fails every query, not just queries that touch it."""
        configure_mappers()

    def test_a_project_still_means_its_own_media(self) -> None:
        """The ambiguity had two readings and only one is right.

        ``Project.media`` is "assets belonging to this project", which is the
        ``media_assets.project_id`` path -- not "the asset this project
        references", which is the new column pointing the other way.
        """
        configure_mappers()
        relationship = sa.inspect(Project).relationships["media"]
        assert {column.name for column in relationship.local_columns} == {"id"}
        assert relationship.mapper.class_ is MediaAsset

    def test_the_reference_column_points_at_media(self) -> None:
        column = sa.inspect(Project).columns["reference_media_id"]
        assert column.nullable is True
        targets = {fk.column.table.name for fk in column.foreign_keys}
        assert targets == {"media_assets"}

    def test_deleting_the_referenced_media_clears_the_pointer(self) -> None:
        """``ON DELETE SET NULL``. A dangling id would be a reference every
        reader downstream has to defend against."""
        column = sa.inspect(Project).columns["reference_media_id"]
        assert {fk.ondelete for fk in column.foreign_keys} == {"SET NULL"}

    def test_the_metadata_has_no_unreachable_tables(self) -> None:
        """A sanity check on the registry as a whole: every mapped class is in
        the same metadata the migrations build."""
        mapped = {
            mapper.class_.__tablename__
            for mapper in Base.registry.mappers
            if hasattr(mapper.class_, "__tablename__")
        }
        assert mapped <= set(Base.metadata.tables)
