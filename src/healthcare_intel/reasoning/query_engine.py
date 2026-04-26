from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass

import pandas as pd

from healthcare_intel.observability import ObservabilityTracker, estimate_trace_cost
from healthcare_intel.utils import haversine_km
from healthcare_intel.config import settings
from healthcare_intel.reasoning.vector_search import semantic_search

QUERY_TO_CAPABILITY = {
    "icu": "has_icu",
    "functional icu": "has_functional_icu",
    "oxygen": "has_oxygen",
    "neonatal": "has_neonatal_bed",
    "oncology": "has_oncology",
    "dialysis": "has_dialysis",
    "trauma": "has_trauma",
    "emergency surgery": "has_emergency_surgery",
    "appendectomy": "has_appendectomy",
    "anesthesiologist": "has_anesthesiologist",
    "parttime": "uses_parttime_doctors",
    "part-time": "uses_parttime_doctors",
}


@dataclass
class ParsedQuery:
    raw_query: str
    required_capabilities: list[str]
    required_state: str | None


def _parse_query_fallback(query: str, known_states: list[str]) -> ParsedQuery:
    q = query.lower()
    required_capabilities: list[str] = []
    for phrase, col in QUERY_TO_CAPABILITY.items():
        if phrase in q and col not in required_capabilities:
            required_capabilities.append(col)

    required_state = None
    for state in known_states:
        if state and state.lower() in q:
            required_state = state
            break

    return ParsedQuery(
        raw_query=query,
        required_capabilities=required_capabilities,
        required_state=required_state,
    )


def _llm_parse_query(query: str, known_states: list[str]) -> tuple[ParsedQuery | None, str]:
    if "api.databricks.com" in settings.databricks_host or "云" in settings.databricks_host or not settings.databricks_token or settings.databricks_host == "https://dbc-xxxxxxx.cloud.databricks.com":
        return None, "Databricks LLM credentials missing. Provide DATABRICKS_HOST."
        
    url = f"{settings.databricks_host.rstrip('/')}/serving-endpoints/{settings.llm_endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json"
    }

    schema_cols = [
        "has_icu", "has_functional_icu", "has_oxygen", "has_neonatal", "has_oncology", "has_dialysis", 
        "has_trauma", "has_emergency_surgery", "has_appendectomy", "has_anesthesiologist", "uses_parttime_doctors",
        "has_general_surgery", "has_maternity", "has_eye_care", "has_dental", "has_dermatology", "has_xray", 
        "has_lab", "has_family_medicine", "has_internal_medicine", "has_urology", "has_gastro", "has_pulmonology",
        "has_psychiatry", "has_rehab", "has_blood_bank", "has_ventilator", "has_ambulance", "has_ct_scan", "has_mri",
        "has_opd", "has_cashless", "has_pediatric", "has_24x7", "has_ot", "has_surgical_capability"
    ]

    prompt = f"""
You are an autonomous AI Query Planner mapping user requests to relational database filters.
Extract the user's intent into exactly the following JSON structure. 
Available Capability Columns: {schema_cols}
Known States: {known_states[:50]} # Truncated to 50 for prompt size...

User Query: "{query}"

Output Format:
{{
  "required_capabilities": ["list_of_exact_column_names_needed_to_satisfy_intent"],
  "required_state": "state_name_if_mentioned_or_null",
  "reasoning": "brief explanation"
}}

Rules:
- Do NOT hallucinate columns. ONLY pick from the Available Capability Columns list.
- Use explicit semantic reasoning based on medical knowledge (e.g. 'dental' => 'has_dental', 'heart' => 'has_cardiology').
- If they ask for part-time, pick 'uses_parttime_doctors'.
- Return strictly valid JSON.
"""
    data = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.1}
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=10) as res:
            resp_data = json.loads(res.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            req_caps = [c for c in result.get("required_capabilities", []) if c in schema_cols]
            
            state = result.get("required_state")
            if state:
                state_match = next((s for s in known_states if s.lower() == state.lower()), None)
                state = state_match
                
            return ParsedQuery(
                raw_query=query,
                required_capabilities=req_caps,
                required_state=state,
            ), result.get("reasoning", "")
    except Exception as e:
        return None, str(e)


def parse_query(query: str, known_states: list[str], trace_steps: list) -> ParsedQuery:
    parsed, reason = _llm_parse_query(query, known_states)
    if parsed:
        trace_steps.append({
            "agent": "LLMQueryPlanner",
            "action": "Parsed query logic",
            "detail": f"Constraints: {parsed.required_capabilities}. Reasoning: {reason[:100]}",
            "state_filter": parsed.required_state or "none",
        })
        return parsed
        
    trace_steps.append({
        "agent": "QueryParser",
        "action": "LLM failed/unconfigured. Used keyword fallback.",
        "detail": str(reason),
    })
    fallback_res = _parse_query_fallback(query, known_states)
    trace_steps.append({
        "agent": "QueryParser",
        "action": "Keyword regex fallback matching",
        "detail": f"Extracted capabilities: {fallback_res.required_capabilities or 'none'}",
        "state_filter": fallback_res.required_state or "none",
    })
    return fallback_res


def run_query(
    facilities: pd.DataFrame,
    query: str,
    latitude: float | None = None,
    longitude: float | None = None,
    top_k: int = 20,
    tracker: ObservabilityTracker | None = None,
) -> pd.DataFrame:
    active_tracker = tracker or ObservabilityTracker(run_name="query_engine", enabled=False)

    with active_tracker.span("reasoning_query", query=query, top_k=top_k):
        trace_steps: list[dict[str, str]] = []
        known_states = (
            facilities["address_stateOrRegion"].fillna("").astype(str).str.strip().drop_duplicates().tolist()
        )
        parsed = parse_query(query, known_states, trace_steps)
        working = facilities.copy()
        
        # 1. Semantic Vector Search (Optional)
        semantic_results = []
        if settings.use_vector_search:
            # Build Metadata Filters explicitly for the vector engine
            query_filters = {}
            if parsed.required_state:
                query_filters["address_stateOrRegion"] = parsed.required_state
            for cap in parsed.required_capabilities:
                query_filters[cap] = True

            trace_steps.append({
                "agent": "VectorSearch",
                "action": "Semantic search with Metadata Pre-Filtering",
                "detail": f"Executed against Mosaic AI. Database filters: {query_filters}"
            })
            # Fetch using native Databricks Vector Search filters
            vec_res = semantic_search(
                query_text=query, 
                filters=query_filters if query_filters else None,
                num_results=200
            )
            if vec_res:
                semantic_results = [r.get("facility_id") for r in vec_res if "facility_id" in r]
                res_count = len(semantic_results)
                
                # Boost matched items or filter by them. If we have hits, we only keep hits.
                # If the vector search returns nothing, we fall back gracefully.
                if semantic_results:
                    trace_steps[-1]["detail"] = f"Retrieved {res_count} semantically related facilities"
                    working = working[working["facility_id"].isin(semantic_results)].copy()
                else:
                    trace_steps[-1]["detail"] = "Query returned no semantic results, falling back to keywords"
            else:
                trace_steps[-1]["detail"] = "Vector search unavailable or failed, falling back to keywords"

        # 2. Basic text fallback (if Vector Search is off and no explicit capability matched)
        if not semantic_results and not parsed.required_capabilities:
            tokens = [t.strip() for t in query.lower().replace(",", " ").split() if len(t) > 3 and t not in ["nearest", "find", "show", "hospital", "clinic", "facility", "with", "that", "can"]]
            if tokens and "full_text" in working.columns:
                trace_steps.append({
                    "agent": "QueryParser",
                    "action": f"Basic text matching fallback for {tokens}",
                    "detail": "Semantic search inactive. Used basic text matching."
                })
                mask = pd.Series([False]*len(working), index=working.index)
                for t in tokens:
                    mask = mask | working["full_text"].str.lower().str.contains(t, na=False, regex=False)
                    mask = mask | working["name"].str.lower().str.contains(t, na=False, regex=False)
                if mask.any():
                    working = working[mask]
                    
        if parsed.required_state:
            working = working[
                working["address_stateOrRegion"].str.lower() == parsed.required_state.lower()
            ]

        for cap in parsed.required_capabilities:
            if cap in working.columns:
                working = working[working[cap] == True]  # noqa: E712

        if latitude is not None and longitude is not None:
            working = working.copy()
            working["distance_km"] = working.apply(
                lambda r: haversine_km(
                    latitude,
                    longitude,
                    float(r["latitude"]),
                    float(r["longitude"]),
                )
                if pd.notna(r["latitude"]) and pd.notna(r["longitude"])
                else float("inf"),
                axis=1,
            )
        else:
            working = working.copy()
            working["distance_km"] = float("inf")

        # Blend trust and proximity for practical triage ranking.
        working["ranking_score"] = (
            (working["trust_score"].fillna(0.0) * 0.75)
            + ((1 / (1 + working["distance_km"].replace(float("inf"), 5000))) * 0.25)
        )

        working = working.sort_values(by=["ranking_score", "trust_score"], ascending=False).head(top_k)

        def _citations(row: pd.Series) -> list[str]:
            raw = row.get("extraction_evidence", "{}")
            evidence = json.loads(raw) if isinstance(raw, str) else {}
            snippets = []
            for v in evidence.values():
                if isinstance(v, list):
                    snippets.extend(v)
            return snippets[:5]

        working["matched_capabilities"] = ", ".join(parsed.required_capabilities)
        working["citations"] = working.apply(_citations, axis=1)

        active_tracker.log_metrics(
            estimate_trace_cost(
                num_rows=len(working),
                text_chars=len(query),
                query_count=1,
            )
        )

        cols = [
            "name",
            "address_city",
            "address_stateOrRegion",
            "address_zipOrPostcode",
            "trust_score",
            "trust_band",
            "distance_km",
            "matched_capabilities",
            "citations",
            "contradiction_flags",
            "facility_id",
        ]
        existing = [c for c in cols if c in working.columns]
        final_df = working[existing].reset_index(drop=True)
        # Attach the trace to the dataframe properties so the API can use it
        final_df.attrs["trace_steps"] = trace_steps
        return final_df
