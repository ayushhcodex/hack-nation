from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=4)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    top_k: int = 20


class QueryResult(BaseModel):
    name: str
    address_city: Optional[str] = None
    address_stateOrRegion: Optional[str] = None
    address_zipOrPostcode: Optional[str] = None
    trust_score: float
    trust_band: str
    distance_km: Optional[float] = None
    matched_capabilities: List[str]
    citations: List[str]


class TrustScore(BaseModel):
    trust_score: float
    trust_band: str
    confidence_low: float
    confidence_high: float
    contradiction_flags: List[str]
    missing_critical_fields: List[str]


class TraceStep(BaseModel):
    step: str
    details: dict
