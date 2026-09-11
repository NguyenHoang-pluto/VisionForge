"""Log lines are valid JSON and carry the required fields."""

from __future__ import annotations

import json
import logging

from visionforge.core.context import set_request_id
from visionforge.core.logging import JsonFormatter


def _record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="visionforge.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_format_emits_required_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert set(payload) >= {"timestamp", "level", "logger", "message"}
    assert payload["level"] == "INFO"
    assert payload["logger"] == "visionforge.test"
    assert payload["message"] == "hello world"


def test_extra_fields_and_request_id_ride_along() -> None:
    set_request_id("abc123")
    payload = json.loads(JsonFormatter().format(_record(job_id="j-1")))

    assert payload["request_id"] == "abc123"
    assert payload["job_id"] == "j-1"
