from __future__ import annotations

from pathlib import Path

import pandas as pd

from healthcare_intel.utils import safe_text

TEXT_COLUMNS = [
    "description",
    "specialties",
    "procedure",
    "equipment",
    "capability",
    "address_line1",
    "address_line2",
    "address_line3",
    "address_city",
    "address_stateOrRegion",
    "address_zipOrPostcode",
]

NUMERIC_COLUMNS = ["numberDoctors", "capacity", "latitude", "longitude"]


def load_facility_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(safe_text)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["name"] = df.get("name", "").map(safe_text)
    df["address_country"] = df.get("address_country", "").map(safe_text)

    text_parts = ["description", "specialties", "procedure", "equipment", "capability"]
    df["full_text"] = df[text_parts].fillna("").agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True)

    df["address_zipOrPostcode"] = df["address_zipOrPostcode"].astype(str).str.replace(".0", "", regex=False)

    return df
