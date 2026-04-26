from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from healthcare_intel.reasoning.query_engine import run_query
from healthcare_intel.schemas import QueryRequest

app = FastAPI(title="HealthIntel India — Agentic Healthcare Intelligence", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path("outputs/facilities_enriched.parquet")
DESERT_PATH = Path("outputs/specialized_deserts.parquet")
CRITICAL_DESERT_PATH = Path("outputs/critical_deserts.parquet")
RECOMMENDATIONS_PATH = Path("outputs/deployment_recommendations.json")
TRACE_PATH = Path("outputs/pipeline_trace.jsonl")
FRONTEND_DIR = Path("frontend")

# Cache loaded data
_cache: dict = {}


def _load_facilities() -> pd.DataFrame:
    if "facilities" not in _cache:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Run pipeline first to generate outputs")
        _cache["facilities"] = pd.read_parquet(DATA_PATH)
    return _cache["facilities"]


def _load_deserts() -> pd.DataFrame:
    if "deserts" not in _cache:
        if not DESERT_PATH.exists():
            raise HTTPException(status_code=404, detail="Run pipeline first to generate outputs")
        _cache["deserts"] = pd.read_parquet(DESERT_PATH)
    return _cache["deserts"]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "HealthIntel India"}


@app.post("/api/query")
def query_facilities(payload: QueryRequest) -> dict:
    facilities = _load_facilities()
    result = run_query(
        facilities,
        payload.query,
        latitude=payload.latitude,
        longitude=payload.longitude,
        top_k=payload.top_k,
    )
    
    # Build chain of thought trace
    chain_of_thought = _build_query_trace(facilities, payload, result)
    
    formatted_results = []
    for _, row in result.iterrows():
        try:
            evidence = json.loads(row.get("extraction_evidence", "[]"))
        except (json.JSONDecodeError, TypeError):
            evidence = []
            
        try:
            uncertainty_flags = json.loads(row.get("uncertainty_flags", "[]"))
        except (json.JSONDecodeError, TypeError):
            uncertainty_flags = []
            
        formatted_results.append({
            "name": str(row.get("name", "Unknown")),
            "hospital": str(row.get("name", "Unknown")),
            "facility_id": str(row.get("facility_id", "")),
            "address_city": str(row.get("address_city", "Unknown")),
            "address_stateOrRegion": str(row.get("address_stateOrRegion", "Unknown")),
            "trust_score": float(row.get("trust_score", 0.0)),
            "trust_band": str(row.get("trust_band", "low")),
            "matched_capabilities": str(row.get("matched_capabilities", "")),
            "confidence": float(row.get("confidence", 0.0)),
            "uncertainty_flags": uncertainty_flags,
            "evidence": evidence,
            "contradiction_flags": json.dumps(uncertainty_flags)
        })
    
    return {
        "count": len(formatted_results),
        "results": formatted_results,
        "chain_of_thought": chain_of_thought,
    }


def _build_query_trace(facilities: pd.DataFrame, payload: QueryRequest, result: pd.DataFrame) -> list[dict]:
    """Generate Chain of Thought steps for query transparency."""
    steps = result.attrs.get("trace_steps", [])
    
    # Extract the matched capabilities string we saved in the dataframe
    matched_caps = result.iloc[0]["matched_capabilities"] if not result.empty and "matched_capabilities" in result.columns else "none"
    
    if matched_caps != "none" and matched_caps != "":
        steps.append({
            "agent": "Filter",
            "action": f"Applied filters",
            "detail": f"Filtered to {len(result)} matching facilities based on constraints",
        })
    
    steps.append({
        "agent": "Verifier",
        "action": "Cross-checked trust scores and contradictions",
        "detail": f"{sum(1 for _, r in result.iterrows() if r.get('contradiction_count', r.get('contradiction_flags', '[]')) not in (0, '[]'))} facilities have trust issues" if not result.empty else "No results to verify",
    })
    
    steps.append({
        "agent": "Ranker",
        "action": "Ranked by trust (75%) + proximity (25%)",
        "detail": f"Top result: {result.iloc[0]['name']} (trust: {result.iloc[0].get('trust_score', 'N/A')})" if not result.empty else "No results found",
    })
    
    return steps


@app.get("/api/facilities")
def get_facilities(
    state: Optional[str] = None,
    trust_band: Optional[str] = None,
    facility_type: Optional[str] = None,
    capability: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
) -> dict:
    facilities = _load_facilities()
    
    if state:
        facilities = facilities[facilities["address_stateOrRegion"].str.lower() == state.lower()]
    if trust_band:
        facilities = facilities[facilities["trust_band"] == trust_band]
    if facility_type:
        facilities = facilities[facilities["facilityTypeId"] == facility_type.lower()]
    if capability and capability in facilities.columns:
        facilities = facilities[facilities[capability] == True]
    
    total = len(facilities)
    page = facilities.iloc[offset:offset + limit]
    
    cols = [
        "facility_id", "name", "address_city", "address_stateOrRegion",
        "address_zipOrPostcode", "facilityTypeId", "operatorTypeId",
        "trust_score", "trust_band", "confidence_low", "confidence_high",
        "contradiction_count", "contradiction_flags", "capabilities_found",
        "avg_confidence", "latitude", "longitude", "data_completeness",
    ]
    existing = [c for c in cols if c in page.columns]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": json.loads(page[existing].to_json(orient="records", default_handler=str)),
    }


@app.get("/api/facilities/{facility_id}")
def get_facility_detail(facility_id: str) -> dict:
    facilities = _load_facilities()
    match = facilities[facilities["facility_id"] == facility_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Facility not found")
    
    row = match.iloc[0]
    return json.loads(row.to_json(default_handler=str))


@app.get("/api/stats")
def get_stats() -> dict:
    facilities = _load_facilities()
    deserts = _load_deserts()
    
    # Capability counts
    cap_cols = [c for c in facilities.columns if c.startswith("has_") and facilities[c].dtype == bool]
    capability_counts = {c: int(facilities[c].sum()) for c in sorted(cap_cols)}
    
    # State-level summary
    state_summary = (
        facilities.groupby("address_stateOrRegion")
        .agg(
            count=("name", "size"),
            avg_trust=("trust_score", "mean"),
        )
        .round(4)
        .sort_values("count", ascending=False)
        .head(30)
    )
    
    # Trust distribution
    trust_dist = facilities["trust_band"].value_counts().to_dict()
    
    # Facility type distribution
    type_dist = facilities["facilityTypeId"].value_counts().to_dict()
    
    # Desert summary
    risk_dist = deserts["risk_tier"].value_counts().to_dict() if "risk_tier" in deserts.columns else {}
    
    return {
        "total_facilities": len(facilities),
        "total_desert_regions": len(deserts),
        "avg_trust_score": round(float(facilities["trust_score"].mean()), 4),
        "trust_distribution": trust_dist,
        "facility_type_distribution": type_dist,
        "capability_counts": capability_counts,
        "state_summary": json.loads(state_summary.to_json(orient="index")),
        "desert_risk_distribution": risk_dist,
        "total_contradictions": int(facilities.get("contradiction_count", pd.Series([0])).sum()),
    }


@app.get("/api/deserts")
def get_deserts(
    risk_tier: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = Query(default=100, le=500),
) -> dict:
    deserts = _load_deserts()
    
    if risk_tier:
        deserts = deserts[deserts["risk_tier"] == risk_tier]
    if state:
        deserts = deserts[deserts["address_stateOrRegion"].str.lower() == state.lower()]
    
    return {
        "count": len(deserts.head(limit)),
        "results": json.loads(deserts.head(limit).to_json(orient="records", default_handler=str)),
    }


@app.get("/api/deserts/geojson")
def get_deserts_geojson() -> dict:
    """Return desert data as GeoJSON-like structure for map overlay."""
    deserts = _load_deserts()
    features = []
    
    for _, row in deserts.iterrows():
        lat = row.get("region_lat")
        lon = row.get("region_lon")
        if pd.notna(lat) and pd.notna(lon):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "state": row.get("address_stateOrRegion", ""),
                    "pin_code": str(row.get("address_zipOrPostcode", "")),
                    "desert_score": float(row.get("desert_score", 0)),
                    "risk_tier": row.get("risk_tier", "unknown"),
                    "facilities_in_region": int(row.get("facilities_in_region", 0)),
                    "hospitals_in_region": int(row.get("hospitals_in_region", 0)),
                    "missing_high_acuity": row.get("missing_high_acuity", ""),
                    "avg_trust_score": float(row.get("avg_trust_score", 0)),
                },
            })
    
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/map/facilities")
def get_facilities_for_map() -> dict:
    """Return minimal facility data for map markers."""
    facilities = _load_facilities()
    cols = ["facility_id", "name", "latitude", "longitude", "trust_score", "trust_band",
            "facilityTypeId", "address_city", "address_stateOrRegion", "capabilities_found"]
    existing = [c for c in cols if c in facilities.columns]
    valid = facilities[existing].dropna(subset=["latitude", "longitude"])
    return {
        "count": len(valid),
        "facilities": json.loads(valid.to_json(orient="records", default_handler=str)),
    }


@app.get("/api/recommendations")
def get_recommendations() -> dict:
    if not RECOMMENDATIONS_PATH.exists():
        return {"count": 0, "recommendations": []}
    with open(RECOMMENDATIONS_PATH) as f:
        recs = json.load(f)
    return {"count": len(recs), "recommendations": recs}


@app.get("/api/trace")
def get_pipeline_trace() -> dict:
    if not TRACE_PATH.exists():
        return {"steps": []}
    steps = []
    with open(TRACE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return {"steps": steps}


# Serve frontend
if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
    
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
