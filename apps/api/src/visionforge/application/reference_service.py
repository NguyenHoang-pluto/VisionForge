"""Nominating a reference video, and reading what it measures as.

Thin by design. The measuring is the analyzers' job and the interpreting is the
domain's; what is left for this layer is the part that touches the database and
the part that enforces ownership -- and those are the two parts that must not
live anywhere else.

Two rules are enforced here and nowhere else, because here is where the request
first meets the database:

**A reference is a clip in this project.** The media id arrives from the client
and is resolved through ``get_in_project`` before anything is written, so a user
cannot point their project at someone else's footage and read its measurements
back out through the profile endpoint.

**A reference is a video.** An audio file has no shot length and an image has no
cuts; nominating either would produce a profile with nothing in it and a user
with no idea why.
"""

from __future__ import annotations

import logging

from visionforge.domain.analysis import AnalyzerName
from visionforge.domain.errors import NotFoundError, ValidationError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.reference import ReferenceProfile, profile_from
from visionforge.infra.db.models import Project
from visionforge.infra.db.repositories import MediaRepository, ProjectRepository

logger = logging.getLogger(__name__)

#: The analyzers a profile is built from. Every one is optional: a reference
#: with only scene detection still yields pacing, and the profile says which
#: features it could not measure rather than inventing them.
PROFILE_ANALYZERS: tuple[AnalyzerName, ...] = (
    AnalyzerName.SCENES,
    AnalyzerName.QUALITY,
    AnalyzerName.DYNAMICS,
    AnalyzerName.BEATS,
)


class ReferenceService:
    def __init__(self, projects: ProjectRepository, media: MediaRepository) -> None:
        self._projects = projects
        self._media = media

    async def set_reference(self, project: Project, media_id: MediaId) -> MediaId:
        """Nominate a clip. Raises rather than storing something unusable."""
        asset = await self._media.get_in_project(media_id, project.id)
        if asset is None:
            # Deliberately the same error a genuinely missing id produces. A
            # distinct "not yours" would confirm the id exists somewhere, which
            # is exactly what an id-probing caller is trying to learn.
            raise NotFoundError("media not found in this project")

        kind = MediaKind(asset.kind)
        if kind is not MediaKind.VIDEO:
            raise ValidationError(
                f"a reference must be a video, not {kind.value}",
                hint="Pick a clip whose cutting and colour can be measured.",
            )
        if MediaStatus(asset.status) is not MediaStatus.READY:
            raise ValidationError(
                "this clip is still being processed",
                hint="Wait for ingest to finish, then nominate it again.",
            )

        await self._projects.set_reference(project, media_id)
        return media_id

    async def clear_reference(self, project: Project) -> None:
        await self._projects.set_reference(project, None)

    async def profile_for_project(self, project: Project) -> ReferenceProfile | None:
        """The current reference's measured style, or ``None`` if there is none.

        Recomputed from analysis rows every time rather than stored. The
        derivation is deterministic, so this costs a little arithmetic and buys
        a profile that can never be stale against a re-analysis -- and one
        version bump in the domain re-derives every project's profile with no
        migration.
        """
        if project.reference_media_id is None:
            return None

        media_id = MediaId(project.reference_media_id)
        asset = await self._media.get_in_project(media_id, project.id)
        if asset is None:
            # The FK is ON DELETE SET NULL, so this is only reachable if the row
            # moved projects, which nothing does. Treated as "no reference"
            # rather than as an error: the user can nominate another.
            logger.warning(
                "reference media missing for project",
                extra={"project_id": str(project.id), "media_id": str(media_id)},
            )
            return None

        payloads = await self._media.analysis_payloads(media_id, PROFILE_ANALYZERS)
        return profile_from(
            media_id=media_id,
            duration_ms=asset.duration_ms or 0,
            scenes=payloads.get(AnalyzerName.SCENES),
            quality=payloads.get(AnalyzerName.QUALITY),
            dynamics=payloads.get(AnalyzerName.DYNAMICS),
            beats=payloads.get(AnalyzerName.BEATS),
        )

    @staticmethod
    def reference_id(project: Project) -> MediaId | None:
        return MediaId(project.reference_media_id) if project.reference_media_id else None


def project_id_of(project: Project) -> ProjectId:
    return ProjectId(project.id)
