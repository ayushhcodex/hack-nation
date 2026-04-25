from __future__ import annotations

import pandas as pd

HIGH_ACUITY_CAPS = [
    "has_oncology",
    "has_dialysis",
    "has_trauma",
    "has_emergency_surgery",
    "has_functional_icu",
]


def identify_specialized_deserts(df: pd.DataFrame) -> pd.DataFrame:
    region_cols = ["address_stateOrRegion", "address_zipOrPostcode"]
    working = df.copy()

    for cap in HIGH_ACUITY_CAPS:
        if cap not in working.columns:
            working[cap] = False

    grouped = (
        working.groupby(region_cols, dropna=False)[HIGH_ACUITY_CAPS]
        .sum()
        .reset_index()
    )

    grouped["facilities_in_region"] = (
        working.groupby(region_cols, dropna=False)
        .size()
        .reset_index(name="n")["n"]
    )

    grouped["acuity_coverage_count"] = grouped[HIGH_ACUITY_CAPS].sum(axis=1)
    grouped["required_capability_count"] = len(HIGH_ACUITY_CAPS)
    grouped["coverage_ratio"] = (
        grouped["acuity_coverage_count"] / grouped["required_capability_count"]
    ).round(4)
    grouped["desert_score"] = (1 - grouped["coverage_ratio"]).round(4)

    grouped["risk_tier"] = grouped["desert_score"].map(
        lambda x: "critical" if x >= 0.8 else "high" if x >= 0.6 else "moderate" if x >= 0.3 else "low"
    )

    return grouped.sort_values(by=["desert_score", "facilities_in_region"], ascending=[False, True]).reset_index(drop=True)
