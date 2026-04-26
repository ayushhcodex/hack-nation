"""Enhanced Medical Desert Detection with distance analysis and deployment recommendations."""
from __future__ import annotations

import math

import pandas as pd

from healthbricks_india.utils import haversine_km

HIGH_ACUITY_CAPS = [
    "has_oncology",
    "has_dialysis",
    "has_trauma",
    "has_emergency_surgery",
    "has_functional_icu",
]

MODERATE_ACUITY_CAPS = [
    "has_neonatal",
    "has_cardiology",
    "has_neurology",
    "has_blood_bank",
    "has_surgical_capability",
]

BASIC_CAPS_FOR_DESERT = [
    "has_xray",
    "has_lab",
    "has_maternity",
    "has_ambulance",
    "has_24x7",
]


def _classify_facility_tier(ftype: str) -> str:
    """Classify facility into tiers."""
    ftype = str(ftype).lower().strip()
    if ftype == "hospital":
        return "tier_1"
    elif ftype == "clinic":
        return "tier_2"
    else:
        return "tier_3"


def identify_specialized_deserts(df: pd.DataFrame) -> pd.DataFrame:
    """Identify medical deserts with enhanced multi-layer scoring."""
    region_cols = ["address_stateOrRegion", "address_zipOrPostcode"]
    working = df.copy()

    # Ensure all capability columns exist
    for cap in HIGH_ACUITY_CAPS + MODERATE_ACUITY_CAPS + BASIC_CAPS_FOR_DESERT:
        if cap not in working.columns:
            working[cap] = False

    # Classify facility tiers
    working["facility_tier"] = working.get("facilityTypeId", "unknown").map(_classify_facility_tier)

    # Group by region
    grouped = (
        working.groupby(region_cols, dropna=False)[HIGH_ACUITY_CAPS + MODERATE_ACUITY_CAPS]
        .sum()
        .reset_index()
    )

    # Facility counts per region
    region_counts = (
        working.groupby(region_cols, dropna=False)
        .agg(
            facilities_in_region=("name", "size"),
            hospitals_in_region=("facility_tier", lambda x: (x == "tier_1").sum()),
            clinics_in_region=("facility_tier", lambda x: (x == "tier_2").sum()),
        )
        .reset_index()
    )

    grouped = grouped.merge(region_counts, on=region_cols, how="left")

    # Average trust in region
    trust_avg = (
        working.groupby(region_cols, dropna=False)["trust_score"]
        .mean()
        .reset_index()
        .rename(columns={"trust_score": "avg_trust_score"})
    )
    grouped = grouped.merge(trust_avg, on=region_cols, how="left")
    grouped["avg_trust_score"] = grouped["avg_trust_score"].round(4)

    # Get representative lat/lon for each region
    coords = (
        working.groupby(region_cols, dropna=False)[["latitude", "longitude"]]
        .mean()
        .reset_index()
        .rename(columns={"latitude": "region_lat", "longitude": "region_lon"})
    )
    grouped = grouped.merge(coords, on=region_cols, how="left")

    # High acuity coverage score
    grouped["high_acuity_coverage"] = grouped[HIGH_ACUITY_CAPS].sum(axis=1)
    grouped["moderate_acuity_coverage"] = grouped[MODERATE_ACUITY_CAPS].sum(axis=1)
    grouped["total_capability_count"] = len(HIGH_ACUITY_CAPS) + len(MODERATE_ACUITY_CAPS)

    # Desert score (weighted: high acuity matters more)
    max_high = len(HIGH_ACUITY_CAPS)
    max_mod = len(MODERATE_ACUITY_CAPS)
    
    high_ratio = (grouped["high_acuity_coverage"] / max_high).clip(0, 1)
    mod_ratio = (grouped["moderate_acuity_coverage"] / max_mod).clip(0, 1)
    
    # Weighted desert score (high acuity = 70%, moderate = 30%)
    coverage_ratio = (high_ratio * 0.7 + mod_ratio * 0.3).round(4)
    grouped["coverage_ratio"] = coverage_ratio
    grouped["desert_score"] = (1 - coverage_ratio).round(4)

    # Risk tier
    grouped["risk_tier"] = grouped["desert_score"].map(
        lambda x: "critical" if x >= 0.8 else "high" if x >= 0.6 else "moderate" if x >= 0.3 else "low"
    )

    # Missing capabilities list
    def _missing_caps(row):
        missing = []
        for cap in HIGH_ACUITY_CAPS:
            if row.get(cap, 0) == 0:
                label = cap.replace("has_", "").replace("_", " ").title()
                missing.append(label)
        return missing

    grouped["missing_high_acuity"] = grouped.apply(
        lambda r: ", ".join(_missing_caps(r)) if _missing_caps(r) else "None", axis=1
    )

    # Sort by desert score (worst first)
    grouped = grouped.sort_values(
        by=["desert_score", "facilities_in_region"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return grouped


def compute_nearest_facility_distances(
    deserts: pd.DataFrame,
    facilities: pd.DataFrame,
) -> pd.DataFrame:
    """For each desert region, find distance to nearest facility with each high-acuity capability."""
    results = []
    
    for _, desert_row in deserts.iterrows():
        if pd.isna(desert_row.get("region_lat")) or pd.isna(desert_row.get("region_lon")):
            continue
        
        lat = float(desert_row["region_lat"])
        lon = float(desert_row["region_lon"])
        
        distances = {}
        for cap in HIGH_ACUITY_CAPS:
            cap_facilities = facilities[facilities.get(cap, False) == True]
            if cap_facilities.empty:
                distances[f"nearest_{cap}_km"] = -1
                distances[f"nearest_{cap}_name"] = "None in dataset"
                continue
            
            valid = cap_facilities.dropna(subset=["latitude", "longitude"])
            if valid.empty:
                distances[f"nearest_{cap}_km"] = -1
                distances[f"nearest_{cap}_name"] = "No coordinates"
                continue
            
            dists = valid.apply(
                lambda r: haversine_km(lat, lon, float(r["latitude"]), float(r["longitude"])),
                axis=1,
            )
            min_idx = dists.idxmin()
            distances[f"nearest_{cap}_km"] = round(dists[min_idx], 1)
            distances[f"nearest_{cap}_name"] = valid.loc[min_idx, "name"]
        
        results.append({
            "address_stateOrRegion": desert_row.get("address_stateOrRegion", ""),
            "address_zipOrPostcode": desert_row.get("address_zipOrPostcode", ""),
            **distances,
        })
    
    if not results:
        return pd.DataFrame()
    
    dist_df = pd.DataFrame(results)
    return deserts.merge(dist_df, on=["address_stateOrRegion", "address_zipOrPostcode"], how="left")


def generate_deployment_recommendations(deserts: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Generate actionable deployment recommendations from desert analysis."""
    critical = deserts[deserts["risk_tier"].isin(["critical", "high"])].head(top_n)
    
    recommendations = []
    for _, row in critical.iterrows():
        state = row.get("address_stateOrRegion", "Unknown")
        pin = row.get("address_zipOrPostcode", "Unknown")
        score = row.get("desert_score", 0)
        missing = row.get("missing_high_acuity", "")
        facilities = row.get("facilities_in_region", 0)
        hospitals = row.get("hospitals_in_region", 0)
        
        # Determine priority deployment
        priority = []
        for cap in HIGH_ACUITY_CAPS:
            if row.get(cap, 0) == 0:
                nearest_km = row.get(f"nearest_{cap}_km", -1)
                cap_label = cap.replace("has_", "").replace("_", " ").title()
                if nearest_km > 100:
                    priority.append({
                        "capability": cap_label,
                        "nearest_km": nearest_km,
                        "urgency": "critical",
                    })
                elif nearest_km > 50:
                    priority.append({
                        "capability": cap_label,
                        "nearest_km": nearest_km,
                        "urgency": "high",
                    })
                elif nearest_km > 0:
                    priority.append({
                        "capability": cap_label,
                        "nearest_km": nearest_km,
                        "urgency": "moderate",
                    })
        
        recommendations.append({
            "state": state,
            "pin_code": str(pin),
            "desert_score": score,
            "risk_tier": row.get("risk_tier", "unknown"),
            "existing_facilities": int(facilities),
            "existing_hospitals": int(hospitals),
            "missing_capabilities": missing,
            "priority_deployments": priority,
            "lat": row.get("region_lat"),
            "lon": row.get("region_lon"),
        })
    
    return recommendations
