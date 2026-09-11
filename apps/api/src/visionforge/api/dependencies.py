"""FastAPI dependency providers.

The probe list is assembled here, in one place, so a test can override it with
fakes and exercise the readiness logic without any infrastructure running.
"""

from __future__ import annotations

from visionforge.application.health_service import HealthService
from visionforge.core.config import get_settings
from visionforge.domain.health import HealthProbe
from visionforge.infra.db import DatabaseProbe
from visionforge.infra.redis import RedisProbe
from visionforge.infra.storage import StorageProbe


def get_health_probes() -> list[HealthProbe]:
    return [DatabaseProbe(), RedisProbe(), StorageProbe()]


def get_health_service() -> HealthService:
    return HealthService(
        probes=get_health_probes(),
        timeout_s=get_settings().readiness_timeout_s,
    )
