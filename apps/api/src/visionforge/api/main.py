"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visionforge import __version__
from visionforge.api.middleware import RequestContextMiddleware
from visionforge.api.routers import health
from visionforge.core.config import get_settings
from visionforge.core.logging import configure_logging
from visionforge.infra.db import dispose_engine
from visionforge.infra.redis import close_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down.

    Deliberately does not connect to anything on boot: the process must start
    even when Postgres is down, and report that through ``/health/ready``. A
    service that refuses to start cannot tell you why it is unhealthy.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "starting",
        extra={"version": __version__, "environment": settings.environment},
    )
    try:
        yield
    finally:
        await dispose_engine()
        await close_redis()
        logger.info("stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="VisionForge API",
        version=__version__,
        description="AI-assisted media creation platform.",
        lifespan=lifespan,
        docs_url="/docs" if settings.is_local else None,
        redoc_url=None,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.include_router(health.router)
    return app


app = create_app()
