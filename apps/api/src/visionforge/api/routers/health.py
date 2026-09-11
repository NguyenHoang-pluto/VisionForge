"""Liveness, readiness and version endpoints.

Liveness and readiness are separate on purpose: liveness answers "is this process
alive" (restart me if not), readiness answers "can this process serve traffic"
(stop routing to me if not). Conflating them causes a container to be killed when
its database is merely slow.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from visionforge import __version__
from visionforge.api.dependencies import get_health_service
from visionforge.api.schemas.health import (
    ComponentHealthResponse,
    HealthResponse,
    ReadinessResponse,
    VersionResponse,
)
from visionforge.application.health_service import HealthService
from visionforge.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Shallow check. Does not touch any dependency."""
    return HealthResponse(status="ok")


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """The process is running and the event loop is responsive."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(
    response: Response,
    service: HealthService = Depends(get_health_service),
) -> ReadinessResponse:
    """Deep check. Returns 503 when a required dependency is unreachable."""
    report = await service.readiness()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ok" if report.ready else "not_ready",
        components=[
            ComponentHealthResponse(
                name=component.name,
                status=component.status.value,
                latency_ms=component.latency_ms,
                detail=component.detail,
            )
            for component in report.components
        ],
    )


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    settings = get_settings()
    return VersionResponse(
        name=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
