from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from healthcare_intel.observability import ObservabilityTracker, estimate_trace_cost
from healthcare_intel.utils import haversine_km

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


@dataclass(slots=True)
class ParsedQuery:
    raw_query: str
    required_capabilities: list[str]
    required_state: str | None


def parse_query(query: str, known_states: list[str]) -> ParsedQuery:
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
        known_states = (
            facilities["address_stateOrRegion"].fillna("").astype(str).str.strip().drop_duplicates().tolist()
        )
        parsed = parse_query(query, known_states)

        working = facilities.copy()

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
        ]
        existing = [c for c in cols if c in working.columns]
        return working[existing].reset_index(drop=True)
