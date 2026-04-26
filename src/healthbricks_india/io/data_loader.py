from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from healthbricks_india.utils import safe_text

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

JSON_ARRAY_COLUMNS = ["specialties", "procedure", "equipment", "capability"]


def _parse_json_field(value: object) -> list[str]:
    """Parse a JSON array string into a list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "[]"}:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    except (json.JSONDecodeError, ValueError):
        return [text]


def _flatten_json_to_text(value: object) -> str:
    """Parse JSON array and join into a single text string."""
    items = _parse_json_field(value)
    return ". ".join(items) if items else ""


def load_facility_data(path: Path) -> pd.DataFrame:
    """Load facility data from CSV or XLSX, with smart JSON parsing."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    # Parse JSON array columns into flat text for NLP processing
    for col in JSON_ARRAY_COLUMNS:
        if col in df.columns:
            # Store original JSON for structured extraction
            df[f"{col}_raw"] = df[col].copy()
            df[f"{col}_parsed"] = df[col].map(_parse_json_field)
            # Flatten to text for unstructured extraction
            df[col] = df[col].map(_flatten_json_to_text)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        if col not in JSON_ARRAY_COLUMNS:
            df[col] = df[col].map(safe_text)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["name"] = df.get("name", "").map(safe_text)
    df["address_country"] = df.get("address_country", "").map(safe_text)

    # Build full text corpus from all text fields
    text_parts = ["description", "specialties", "procedure", "equipment", "capability"]
    df["full_text"] = (
        df[text_parts]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
    )

    df["address_zipOrPostcode"] = (
        df["address_zipOrPostcode"].astype(str).str.replace(".0", "", regex=False)
    )

    # Facility type normalization
    if "facilityTypeId" in df.columns:
        df["facilityTypeId"] = df["facilityTypeId"].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df["facilityTypeId"] = "unknown"

    # Operator type normalization 
    if "operatorTypeId" in df.columns:
        df["operatorTypeId"] = df["operatorTypeId"].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df["operatorTypeId"] = "unknown"

    # Data completeness score (% of non-null fields)
    total_cols = len(df.columns)
    df["data_completeness"] = (
        df.notna().sum(axis=1) / total_cols
    ).round(4)

    # Specialty count (from parsed arrays)
    if "specialties_parsed" in df.columns:
        df["specialty_count"] = df["specialties_parsed"].map(len)
    else:
        df["specialty_count"] = 0

    # Web presence score
    web_cols = [
        "officialWebsite", "facebookLink", "twitterLink",
        "linkedinLink", "instagramLink",
    ]
    existing_web_cols = [c for c in web_cols if c in df.columns]
    if existing_web_cols:
        df["web_presence_score"] = df[existing_web_cols].notna().sum(axis=1) / len(web_cols)
    else:
        df["web_presence_score"] = 0.0

    # Assign stable facility_id
    df["facility_id"] = [f"F{str(i).zfill(5)}" for i in range(len(df))]

    return df


# Backward compatibility alias
def load_facility_csv(path: Path) -> pd.DataFrame:
    """Legacy alias — now supports both CSV and XLSX."""
    return load_facility_data(path)
