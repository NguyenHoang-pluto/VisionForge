"""Request-scoped correlation ID.

A single ``ContextVar`` so that a log line emitted deep in the call stack can be
tied back to the HTTP request that caused it. This is the minimum groundwork for
the distributed tracing planned in week 15 -- it is deliberately not more.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()
