from __future__ import annotations

import json
import math

import pandas as pd

CRITICAL_FIELDS = [
    "name",
    "address_stateOrRegion",
    "address_zipOrPostcode",
    "description",
    "equipment",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _confidence_interval(prob: float, n_eff: float) -> tuple[float, float]:
    if n_eff <= 1:
        return 0.0, 1.0
    margin = 1.96 * math.sqrt((prob * (1 - prob)) / n_eff)
    return _clamp(prob - margin), _clamp(prob + margin)


def score_trust(df: pd.DataFrame) -> pd.DataFrame:
    outputs = []

    for _, row in df.iterrows():
        contradiction_flags: list[str] = []
        missing_critical_fields = [f for f in CRITICAL_FIELDS if not str(row.get(f, "")).strip()]

        positive_signals = 0
        negative_signals = 0

        if row.get("has_functional_icu", False):
            positive_signals += 2
        if row.get("has_anesthesiologist", False):
            positive_signals += 2
        if row.get("has_emergency_surgery", False):
            positive_signals += 2
        if row.get("has_24x7", False):
            positive_signals += 1
        if row.get("uses_parttime_doctors", False):
            positive_signals += 1

        if row.get("has_icu", False) and not row.get("has_oxygen", False):
            contradiction_flags.append("ICU claimed but oxygen support not evidenced")
            negative_signals += 3

        if (
            row.get("has_emergency_surgery", False)
            or row.get("has_appendectomy", False)
        ) and not row.get("has_anesthesiologist", False):
            contradiction_flags.append(
                "Emergency surgery/appendectomy claimed without anesthesiologist evidence"
            )
            negative_signals += 3

        if row.get("has_24x7", False) and (pd.isna(row.get("numberDoctors")) or row.get("numberDoctors", 0) <= 0):
            contradiction_flags.append("24/7 claim with no doctor count available")
            negative_signals += 2

        if len(missing_critical_fields) >= 3:
            contradiction_flags.append("Too many critical fields are missing")
            negative_signals += 2

        evidence = json.loads(row.get("extraction_evidence", "{}"))
        evidence_count = sum(len(v) for v in evidence.values() if isinstance(v, list))
        if evidence_count >= 6:
            positive_signals += 2
        elif evidence_count <= 1:
            negative_signals += 2

        alpha = 1 + positive_signals
        beta = 1 + negative_signals + (len(missing_critical_fields) / 2)
        trust_score = alpha / (alpha + beta)
        conf_low, conf_high = _confidence_interval(trust_score, alpha + beta)

        trust_band = "high" if trust_score >= 0.75 else "medium" if trust_score >= 0.5 else "low"

        outputs.append(
            {
                "trust_score": round(trust_score, 4),
                "trust_band": trust_band,
                "confidence_low": round(conf_low, 4),
                "confidence_high": round(conf_high, 4),
                "contradiction_flags": json.dumps(contradiction_flags, ensure_ascii=False),
                "missing_critical_fields": json.dumps(missing_critical_fields),
            }
        )

    trust_df = pd.DataFrame(outputs)
    return pd.concat([df.reset_index(drop=True), trust_df], axis=1)
