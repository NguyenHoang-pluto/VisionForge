"""Development entrypoint: ``python -m visionforge``.

Exists so that the Windows event-loop policy is installed *before* uvicorn
creates its loop. Running ``uvicorn visionforge.api.main:app`` directly works on
Linux but fails on Windows, because by the time uvicorn imports the app module
the loop already exists. In production the API is started by a process manager
on Linux, where this wrapper is unnecessary but harmless.
"""

from __future__ import annotations

from visionforge.core.runtime import configure_event_loop_policy


def main() -> None:
    configure_event_loop_policy()

    import uvicorn

    from visionforge.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "visionforge.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_local and settings.api_reload,
        log_config=None,  # visionforge.core.logging owns log configuration
    )


if __name__ == "__main__":
    main()
