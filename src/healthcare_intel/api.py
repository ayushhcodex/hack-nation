from __future__ import annotations

import ast
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_listish(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text or text.lower() in {"[]", "none", "null", "nan"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [text]


def _to_citations(raw: object) -> list[str]:
    return _parse_listish(raw)


def _to_flags(raw: object) -> list[str]:
    return _parse_listish(raw)


def _parse_json_object(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _compute_query_evaluation(
    payload: QueryRequest,
    facilities: pd.DataFrame,
    deserts: pd.DataFrame,
    result: pd.DataFrame,
    formatted_results: list[dict],
    chain_of_thought: list[dict],
    exact_match: bool,
) -> dict:
    count = len(formatted_results)
    avg_trust = 0.0
    if count:
        avg_trust = float(sum(float(r.get("trust_score", 0.0)) for r in formatted_results) / count)
    avg_trust = _clip01(avg_trust)

    evidence_coverage = 0.0
    contradiction_rate = 0.0
    if count:
        evidence_coverage = sum(1 for r in formatted_results if r.get("evidence")) / count
        contradiction_rate = sum(1 for r in formatted_results if r.get("uncertainty_flags")) / count

    retrieval_coverage = _clip01(count / max(1, payload.top_k))
    consistency = _clip01(1.0 - contradiction_rate)

    discovery_verification = _clip01(
        (0.35 * avg_trust)
        + (0.25 * evidence_coverage)
        + (0.20 * consistency)
        + (0.20 * retrieval_coverage)
    )

    ids = [str(r.get("facility_id", "")) for r in formatted_results if r.get("facility_id")]
    subset = facilities[facilities["facility_id"].astype(str).isin(ids)] if ids and "facility_id" in facilities.columns else facilities.head(0)

    avg_confidence = 0.0
    if not subset.empty and "avg_confidence" in subset.columns:
        avg_confidence = _clip01(float(pd.to_numeric(subset["avg_confidence"], errors="coerce").fillna(0.0).mean()))

    text_density = 0.0
    if not subset.empty and "full_text" in subset.columns:
        mean_chars = float(subset["full_text"].fillna("").astype(str).str.len().mean())
        text_density = _clip01(mean_chars / 800.0)

    # IDP quality: how well extracted evidence is actually grounded in messy free-form notes.
    grounded_hits = 0
    evidence_snippets = 0
    corroboration_total = 0.0
    corroboration_rows = 0
    if not subset.empty:
        for _, row in subset.iterrows():
            full_text = str(row.get("full_text", "") or "")
            full_text_lower = full_text.lower()
            evidence_obj = _parse_json_object(row.get("extraction_evidence", "{}"))
            for info in evidence_obj.values():
                if isinstance(info, dict):
                    snippets = info.get("evidence", [])
                    if isinstance(snippets, list):
                        for s in snippets:
                            txt = str(s).strip()
                            if not txt:
                                continue
                            evidence_snippets += 1
                            if txt.lower() in full_text_lower:
                                grounded_hits += 1

            trace_obj = _parse_json_object(row.get("extraction_trace", "{}"))
            merged = float(trace_obj.get("pass3_merged_count", 0) or 0)
            cross = float(trace_obj.get("cross_validated_count", 0) or 0)
            if merged > 0:
                corroboration_total += _clip01(cross / merged)
                corroboration_rows += 1

    grounding_score = _clip01((grounded_hits / evidence_snippets) if evidence_snippets else 0.0)
    corroboration_score = _clip01((corroboration_total / corroboration_rows) if corroboration_rows else 0.0)

    idp_innovation = _clip01(
        (0.35 * avg_confidence)
        + (0.35 * grounding_score)
        + (0.20 * corroboration_score)
        + (0.10 * text_density)
    )

    risk_tier = deserts.get("risk_tier", pd.Series(dtype=str)).astype(str).str.lower()
    high_or_critical = (risk_tier.isin(["critical", "high"]).sum() / max(1, len(deserts))) if len(deserts) else 0.0
    actionability = _clip01(count / 10.0)
    social_impact = _clip01((0.60 * high_or_critical) + (0.40 * actionability))

    agents = {str(step.get("agent", "")) for step in chain_of_thought}
    cot_completeness = _clip01(
        (
            (1 if ("LLMQueryPlanner" in agents or "QueryParser" in agents) else 0)
            + (1 if "Verifier" in agents else 0)
            + (1 if "Ranker" in agents else 0)
        ) / 3.0
    )
    trace_depth = _clip01(len(chain_of_thought) / 4.0)
    ux_transparency = _clip01((0.50 * cot_completeness) + (0.30 * trace_depth) + (0.20 * evidence_coverage))

    overall = _clip01(
        (0.35 * discovery_verification)
        + (0.30 * idp_innovation)
        + (0.25 * social_impact)
        + (0.10 * ux_transparency)
    )

    return {
        "overall_score": round(overall * 100, 1),
        "confidence_label": "high" if overall >= 0.75 else "medium" if overall >= 0.5 else "low",
        "rubric": {
            "discovery_verification": round(discovery_verification * 100, 1),
            "idp_innovation": round(idp_innovation * 100, 1),
            "social_impact_utility": round(social_impact * 100, 1),
            "ux_transparency": round(ux_transparency * 100, 1),
        },
        "diagnostics": {
            "exact_match": exact_match,
            "result_count": count,
            "retrieval_coverage": round(retrieval_coverage * 100, 1),
            "avg_trust_score": round(avg_trust * 100, 1),
            "evidence_coverage": round(evidence_coverage * 100, 1),
            "consistency": round(consistency * 100, 1),
            "grounded_evidence": round(grounding_score * 100, 1),
            "cross_pass_corroboration": round(corroboration_score * 100, 1),
            "query": payload.query,
        },
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "HealthIntel India"}


@app.post("/api/query")
def query_facilities(payload: QueryRequest) -> dict:
    facilities = _load_facilities()
    deserts = _load_deserts()
    result = run_query(
        facilities,
        payload.query,
        latitude=payload.latitude,
        longitude=payload.longitude,
        top_k=payload.top_k,
    )
    
    # Build chain of thought trace
    chain_of_thought = _build_query_trace(facilities, payload, result)
    used_relaxed_fallback = any(step.get("agent") == "Fallback" for step in chain_of_thought)
    
    formatted_results = []
    for _, row in result.iterrows():
        evidence = _to_citations(row.get("citations", []))
        uncertainty_flags = _to_flags(row.get("contradiction_flags", "[]"))
            
        formatted_results.append({
            "name": str(row.get("name", "Unknown")),
            "hospital": str(row.get("name", "Unknown")),
            "facility_id": str(row.get("facility_id", "")),
            "address_city": str(row.get("address_city", "Unknown")),
            "address_stateOrRegion": str(row.get("address_stateOrRegion", "Unknown")),
            "capacity": int(row.get("capacity", 0)) if pd.notna(row.get("capacity")) else None,
            "trust_score": float(row.get("trust_score", 0.0)),
            "trust_band": str(row.get("trust_band", "low")),
            "matched_capabilities": str(row.get("matched_capabilities", "")),
            "confidence": float(row.get("confidence", 0.0)),
            "uncertainty_flags": uncertainty_flags,
            "citations": evidence,
            "evidence": evidence,
            "contradiction_flags": json.dumps(uncertainty_flags)
        })

    summary: dict = {}
    lower_query = payload.query.lower()
    if "bed" in lower_query and "capacity" in result.columns:
        capacity = pd.to_numeric(result["capacity"], errors="coerce")
        valid_capacity = capacity.dropna()
        summary = {
            "total_beds": int(valid_capacity.sum()) if not valid_capacity.empty else 0,
            "facilities_with_bed_data": int(valid_capacity.count()),
        }

    evaluation = _compute_query_evaluation(
        payload=payload,
        facilities=facilities,
        deserts=deserts,
        result=result,
        formatted_results=formatted_results,
        chain_of_thought=chain_of_thought,
        exact_match=not used_relaxed_fallback,
    )
    
    return {
        "count": len(formatted_results),
        "exact_match": not used_relaxed_fallback,
        "summary": summary,
        "evaluation": evaluation,
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
