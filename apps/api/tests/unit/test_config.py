"""Configuration loads from the environment and exposes no secrets by accident."""

from __future__ import annotations

import pytest

from visionforge.core.config import Settings, get_settings


def test_defaults_are_usable_without_any_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ENVIRONMENT", "LOG_LEVEL", "DATABASE_URL", "REDIS_URL"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.environment == "local"
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.redis_url.startswith("redis://")
    assert len(settings.buckets) == 3


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.is_local is False


def test_secrets_are_not_leaked_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_SECRET_KEY", "super-secret-value")
    get_settings.cache_clear()

    settings = get_settings()
    assert "super-secret-value" not in repr(settings)
    assert settings.s3_secret_key.get_secret_value() == "super-secret-value"


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()
