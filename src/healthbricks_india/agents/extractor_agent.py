from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from healthbricks_india.utils import contains_any, split_sentences

CAPABILITY_TERMS = {
    "has_icu": ["icu", "intensive care"],
    "has_oxygen": ["oxygen", "o2", "oxygen plant", "oxygen cylinder"],
    "has_neonatal_bed": ["nicu", "neonatal", "newborn intensive", "neonatology"],
    "has_oncology": ["oncology", "cancer care", "radiotherapy", "chemotherapy"],
    "has_dialysis": ["dialysis", "hemodialysis", "nephrology dialysis"],
    "has_trauma": ["trauma", "accident and emergency", "er", "emergency medicine"],
    "has_emergency_surgery": ["emergency surgery", "urgent surgery", "general surgery"],
    "has_appendectomy": ["appendectomy", "appendicectomy"],
    "has_anesthesiologist": ["anesthesiologist", "anaesthesiologist", "anesthesia specialist"],
    "has_24x7": ["24/7", "24x7", "round the clock", "day and night"],
    "uses_parttime_doctors": ["part-time", "part time", "visiting consultant", "locum"],
}


def _collect_evidence(text: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    for sentence in split_sentences(text):
        if contains_any(sentence, terms):
            hits.append(sentence)
    return hits[:3]


def extract_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        text = " ".join(
            [
                str(row.get("description", "")),
                str(row.get("specialties", "")),
                str(row.get("procedure", "")),
                str(row.get("equipment", "")),
                str(row.get("capability", "")),
            ]
        )
        text = re.sub(r"\s+", " ", text).strip()

        out: dict[str, Any] = {}
        evidence: dict[str, list[str]] = {}

        for flag, terms in CAPABILITY_TERMS.items():
            ev = _collect_evidence(text, terms)
            out[flag] = bool(ev)
            evidence[flag] = ev

        out["has_functional_icu"] = bool(out["has_icu"] and out["has_oxygen"])

        rows.append(
            {
                **out,
                "extraction_evidence": json.dumps(evidence, ensure_ascii=False),
                "extraction_text_length": len(text),
            }
        )

    extracted = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), extracted], axis=1)
