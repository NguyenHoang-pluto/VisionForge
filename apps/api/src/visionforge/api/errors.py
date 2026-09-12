"""Domain errors -> HTTP responses.

One handler, registered once, so no route has to remember to translate. The
response shape is stable across every endpoint: ``{code, message, hint}``. Stack
traces go to the log, never to the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from visionforge.core.context import get_request_id
from visionforge.domain.errors import VisionForgeError

logger = logging.getLogger(__name__)


async def visionforge_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, VisionForgeError)
    if exc.http_status >= 500:
        logger.exception("unhandled domain error", extra={"path": request.url.path})
    else:
        logger.info(
            "request rejected",
            extra={"path": request.url.path, "code": exc.code, "reason": exc.message},
        )

    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "hint": exc.hint,
                "request_id": get_request_id(),
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(VisionForgeError, visionforge_error_handler)
