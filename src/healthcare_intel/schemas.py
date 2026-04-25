from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=4)
    latitude: float | None = None
    longitude: float | None = None
    top_k: int = 20


class QueryResult(BaseModel):
    name: str
    address_city: str | None = None
    address_stateOrRegion: str | None = None
    address_zipOrPostcode: str | None = None
    trust_score: float
    trust_band: str
    distance_km: float | None = None
    matched_capabilities: list[str]
    citations: list[str]


class TrustScore(BaseModel):
    trust_score: float
    trust_band: str
    confidence_low: float
    confidence_high: float
    contradiction_flags: list[str]
    missing_critical_fields: list[str]


class TraceStep(BaseModel):
    step: str
    details: dict[str, Any]
