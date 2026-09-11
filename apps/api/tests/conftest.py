"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from visionforge.core.runtime import configure_event_loop_policy

os.environ.setdefault("ENVIRONMENT", "ci")
configure_event_loop_policy()


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Settings are cached per process; clear around every test."""
    from visionforge.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
