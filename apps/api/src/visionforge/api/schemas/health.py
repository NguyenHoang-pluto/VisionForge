"""Health endpoint response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ComponentHealthResponse(BaseModel):
    name: str
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str = Field(examples=["ok", "not_ready"])
    components: list[ComponentHealthResponse]


class VersionResponse(BaseModel):
    name: str
    version: str
    environment: str
