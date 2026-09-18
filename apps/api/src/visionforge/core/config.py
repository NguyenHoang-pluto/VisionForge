"""Application configuration.

All configuration is read once, here, through Pydantic Settings. Phase 0 rule:
no ``os.environ`` lookups at call sites.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    environment: Environment = "local"
    log_level: str = "INFO"
    app_name: str = "visionforge-api"

    # --- Database ---
    database_url: str = "postgresql+psycopg://visionforge:visionforge@localhost:5442/visionforge"
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_s: int = 10

    # --- Redis (broker + cache; never the source of truth for job state) ---
    redis_url: str = "redis://localhost:6389/0"

    # --- Object storage (MinIO locally, S3 later) ---
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key: SecretStr = SecretStr("visionforge")
    s3_secret_key: SecretStr = SecretStr("visionforge-dev-secret")
    s3_bucket_media: str = "visionforge-media"
    s3_bucket_derivatives: str = "visionforge-derivatives"
    s3_bucket_renders: str = "visionforge-renders"

    # --- HTTP ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True

    # --- Rendering ---
    #: Whether the render worker's machine has an NVIDIA GPU with NVENC. Off by
    #: default, because a render asked of an encoder that is not there fails;
    #: set ``RENDER_GPU=true`` where there is one. 8K is offered only then.
    render_gpu: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Readiness ---
    readiness_timeout_s: float = 2.0

    # --- LLM planning (Phase 5) ---
    #: Off by default. Every deployment without a key plans deterministically
    #: and works completely; AI is an addition, never a dependency.
    llm_enabled: bool = False
    #: One of: anthropic, openai, stub. The stub is a deterministic local
    #: provider for tests and is refused in production.
    llm_provider: str = "anthropic"
    llm_model: str = ""
    #: Overridable so a compatible gateway or a locally-hosted model can be
    #: pointed at without new code. Empty means the provider's own default.
    llm_base_url: str = ""
    #: **Server-side only.** Read here, used by the provider adapters, and never
    #: serialised into a response, a log line, a plan or a database row.
    llm_api_key: SecretStr | None = None
    #: Hard ceiling on one planning call. Chosen to be shorter than a user will
    #: wait: past this the rules engine is a better answer than a slower one.
    llm_timeout_s: float = 30.0
    llm_max_output_tokens: int = 2_000

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    @property
    def llm_configured(self) -> bool:
        """Whether AI planning could run. Not whether it will succeed.

        The stub needs no key; every real provider does. This answers the
        question the UI asks -- "should the AI option be offered?" -- without
        touching the key itself.
        """
        if not self.llm_enabled:
            return False
        if self.llm_provider.strip().lower() == "stub":
            return self.environment != "production"
        return bool(self.llm_api_key and self.llm_api_key.get_secret_value().strip())

    @property
    def buckets(self) -> tuple[str, str, str]:
        """The three buckets the storage layer bootstraps and probes."""
        return (self.s3_bucket_media, self.s3_bucket_derivatives, self.s3_bucket_renders)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is parsed once. Tests clear the cache via
    ``get_settings.cache_clear()``.
    """
    return Settings()
