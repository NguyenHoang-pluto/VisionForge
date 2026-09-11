"""Platform runtime fixes applied before the event loop is created.

Windows defaults to ``ProactorEventLoop``, which psycopg's async driver cannot
use. Any process that opens an async database connection on Windows must install
the selector policy *before* the loop exists -- doing it after the loop is
running has no effect.

This is a host-platform workaround, not an architecture change: on Linux (CI and
production) the function is a no-op.
"""

from __future__ import annotations

import asyncio
import sys


def configure_event_loop_policy() -> None:
    """Install an event loop policy compatible with psycopg's async driver."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
