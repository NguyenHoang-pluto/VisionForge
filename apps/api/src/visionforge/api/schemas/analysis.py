"""Request and response models for the analysis endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from visionforge.api.schemas.media import JobResponse


class AnalyzeRequest(BaseModel):
    """Queue analysis for one asset, or for the whole project when omitted."""

    media_id: UUID | None = None
    #: Which lanes to run. Both by default. Useful for re-running only the CPU
    #: signals after a threshold change without paying for GPU inference again.
    lanes: list[str] | None = Field(default=None, examples=[["cpu", "gpu"]])


class AnalyzeResponse(BaseModel):
    queued: int
    media_count: int
    lanes: list[str]
    jobs: list[JobResponse]


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    media_id: UUID
    analyzer: str
    analyzer_version: str
    status: str
    payload: dict[str, Any]
    metrics: dict[str, Any] | None = None
    #: Whether an embedding was stored. The 512 floats themselves are not
    #: serialised: they are large, and no client has a use for the raw vector
    #: that the similarity endpoint does not serve better.
    has_embedding: bool = False
    created_at: datetime


class AnalysisListResponse(BaseModel):
    items: list[AnalysisResponse]
    total: int


class SimilarityHit(BaseModel):
    media_id: UUID
    #: Cosine distance: 0 is identical, 2 is opposite.
    distance: float
    #: 1 - distance, for callers that prefer "higher is closer".
    similarity: float


class SimilarMediaResponse(BaseModel):
    query_media_id: UUID
    results: list[SimilarityHit]
