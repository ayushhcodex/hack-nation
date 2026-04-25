from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from healthcare_intel.reasoning.query_engine import run_query
from healthcare_intel.schemas import QueryRequest

app = FastAPI(title="Agentic Healthcare Intelligence API", version="0.1.0")

DATA_PATH = Path("outputs/facilities_enriched.parquet")
DESERT_PATH = Path("outputs/specialized_deserts.parquet")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query_facilities(payload: QueryRequest) -> dict:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Run pipeline first to generate outputs")

    facilities = pd.read_parquet(DATA_PATH)
    result = run_query(
        facilities,
        payload.query,
        latitude=payload.latitude,
        longitude=payload.longitude,
        top_k=payload.top_k,
    )
    return {"count": len(result), "results": result.to_dict(orient="records")}


@app.get("/deserts")
def get_deserts(limit: int = 100) -> dict:
    if not DESERT_PATH.exists():
        raise HTTPException(status_code=404, detail="Run pipeline first to generate outputs")
    deserts = pd.read_parquet(DESERT_PATH).head(limit)
    return {"count": len(deserts), "results": deserts.to_dict(orient="records")}
