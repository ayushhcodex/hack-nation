"""Enhanced Trust Scorer with 8 contradiction rules, peer comparison, and Bayesian confidence."""
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

CONTRADICTION_RULES = [
    {
        "id": "surgery_without_anesthesia",
        "when": ["has_emergency_surgery", "has_appendectomy", "has_general_surgery"],
        "requires": ["has_anesthesiologist"],
        "mode": "any_when_all_req",
        "severity": "high",
        "penalty": 3,
        "message": "Claims surgical capability but no anesthesiologist evidence",
    },
    {
        "id": "icu_without_oxygen",
        "when": ["has_icu"],
        "requires": ["has_oxygen"],
        "mode": "any_when_all_req",
        "severity": "high",
        "penalty": 3,
        "message": "ICU claimed but oxygen support not evidenced",
    },
    {
        "id": "always_open_without_doctors",
        "when": ["has_24x7"],
        "requires": [],
        "mode": "check_doctors",
        "severity": "medium",
        "penalty": 2,
        "message": "24/7 claim with no doctor count available",
    },
    {
        "id": "neonatal_without_pediatrics",
        "when": ["has_neonatal"],
        "requires": ["has_pediatric"],
        "mode": "any_when_all_req",
        "severity": "high",
        "penalty": 3,
        "message": "Neonatal care but no pediatric specialist evidenced",
    },
    {
        "id": "emergency_without_support",
        "when": ["has_emergency_surgery"],
        "requires": ["has_ambulance", "has_trauma"],
        "mode": "any_when_any_req",
        "severity": "medium",
        "penalty": 2,
        "message": "Emergency surgery claimed without ambulance or trauma setup",
    },
    {
        "id": "hospital_no_equipment",
        "when": [],
        "requires": [],
        "mode": "check_hospital_equipment",
        "severity": "medium",
        "penalty": 2,
        "message": "Hospital type but no equipment or capability details listed",
    },
    {
        "id": "specialist_in_pharmacy",
        "when": ["has_oncology", "has_dialysis", "has_icu", "has_neonatal"],
        "requires": [],
        "mode": "check_facility_type",
        "severity": "high",
        "penalty": 3,
        "message": "High-acuity capability claimed by pharmacy/dentist facility",
    },
    {
        "id": "blood_bank_without_lab",
        "when": ["has_blood_bank"],
        "requires": ["has_lab"],
        "mode": "any_when_all_req",
        "severity": "medium",
        "penalty": 2,
        "message": "Blood bank without laboratory support",
    },
]

HIGH_VALUE_CAPS = [
    "has_functional_icu", "has_surgical_capability", "has_anesthesiologist",
    "has_emergency_surgery", "has_oncology", "has_dialysis", "has_trauma",
    "has_neonatal", "has_cardiology", "has_neurology",
]

MODERATE_VALUE_CAPS = [
    "has_24x7", "has_ot", "has_xray", "has_ct_scan", "has_mri",
    "has_lab", "has_blood_bank", "has_ventilator", "has_ambulance",
    "has_maternity", "has_orthopedic",
]

BASIC_CAPS = [
    "has_opd", "has_cashless", "has_dental", "has_eye_care",
    "has_pediatric", "has_family_medicine", "has_internal_medicine",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _confidence_interval(prob: float, n_eff: float) -> tuple[float, float]:
    if n_eff <= 1:
        return 0.0, 1.0
    margin = 1.96 * math.sqrt((prob * (1 - prob)) / n_eff)
    return _clamp(prob - margin), _clamp(prob + margin)


def _check_contradiction(row: pd.Series, rule: dict) -> str | None:
    """Check a single contradiction rule against a facility row."""
    mode = rule["mode"]
    
    if mode == "any_when_all_req":
        # If any "when" capability is true, ALL "requires" must be true
        triggered = any(row.get(w, False) for w in rule["when"])
        if triggered:
            missing = [r for r in rule["requires"] if not row.get(r, False)]
            if missing:
                return rule["message"]
    
    elif mode == "any_when_any_req":
        # If any "when" is true, at least ONE "requires" must be true
        triggered = any(row.get(w, False) for w in rule["when"])
        if triggered:
            has_any = any(row.get(r, False) for r in rule["requires"])
            if not has_any:
                return rule["message"]
    
    elif mode == "check_doctors":
        if row.get("has_24x7", False):
            n_docs = row.get("numberDoctors")
            if pd.isna(n_docs) or n_docs <= 0:
                return rule["message"]
    
    elif mode == "check_hospital_equipment":
        ftype = str(row.get("facilityTypeId", "")).lower()
        if ftype == "hospital":
            equip = str(row.get("equipment", ""))
            cap = str(row.get("capability", ""))
            if len(equip.strip()) < 5 and len(cap.strip()) < 10:
                return rule["message"]
    
    elif mode == "check_facility_type":
        ftype = str(row.get("facilityTypeId", "")).lower()
        if ftype in ("pharmacy", "farmacy", "dentist"):
            triggered = any(row.get(w, False) for w in rule["when"])
            if triggered:
                return rule["message"]
    
    return None


def compute_confidence(extracted_record: dict, contradictions: list[str], evidence: dict) -> dict:
    uncertainty_flags = []
    confidence_reason = []

    evidence_count = 0
    for key, ev_list in evidence.items():
        if isinstance(ev_list, list):
            evidence_count += len(ev_list)
        elif isinstance(ev_list, dict) and "evidence" in ev_list:
            evidence_count += len(ev_list["evidence"])
            
    evidence_score = _clamp(evidence_count / 5.0)
    if evidence_count <= 1:
        uncertainty_flags.append("Sparse supporting evidence")
    confidence_reason.append(f"Evidence score: {evidence_score:.2f} ({evidence_count} sentences found)")

    contradiction_count = len(contradictions)
    contradiction_penalty = _clamp(contradiction_count / 3.0)
    for c in contradictions:
        uncertainty_flags.append(f"Conflicting capability: {c}")
    confidence_reason.append(f"Contradiction penalty: {contradiction_penalty:.2f} ({contradiction_count} contradictions)")

    critical_fields = ["has_icu", "has_ventilator", "has_anesthesiologist", "has_emergency_surgery"]
    fields_present = 0
    for field in critical_fields:
        is_true = bool(extracted_record.get(field, False))
        has_ev = len(evidence.get(field, [])) > 0 if isinstance(evidence, dict) else False
        
        if is_true or has_ev:
            fields_present += 1
        else:
            if field == "has_anesthesiologist":
                uncertainty_flags.append("Missing anesthesiologist data")
            elif field == "has_icu":
                uncertainty_flags.append("Weak ICU evidence")
            elif field == "has_emergency_surgery":
                uncertainty_flags.append("Missing surgery capability data")

    completeness_score = fields_present / len(critical_fields) if critical_fields else 0.0
    confidence_reason.append(f"Completeness score: {completeness_score:.2f}")

    confidence = round((0.4 * evidence_score) + (0.3 * (1.0 - contradiction_penalty)) + (0.3 * completeness_score), 4)

    return {
        "confidence": confidence,
        "uncertainty_flags": uncertainty_flags,
        "confidence_reason": confidence_reason
    }


def score_trust(df: pd.DataFrame) -> pd.DataFrame:
    """Score trust for each facility using enhanced multi-signal approach."""
    outputs = []

    # Pre-compute state-level averages for peer comparison
    cap_cols = [c for c in df.columns if c.startswith("has_") and df[c].dtype == bool]
    state_avgs = {}
    if cap_cols and "address_stateOrRegion" in df.columns:
        for state, group in df.groupby("address_stateOrRegion"):
            state_avgs[state] = group[cap_cols].mean().to_dict()

    for _, row in df.iterrows():
        contradiction_flags: list[str] = []
        missing_critical_fields = [f for f in CRITICAL_FIELDS if not str(row.get(f, "")).strip()]

        positive_signals = 0.0
        negative_signals = 0.0

        # Positive signals from high-value capabilities with confidence weighting
        for cap in HIGH_VALUE_CAPS:
            if row.get(cap, False):
                conf = row.get(f"{cap}_confidence", 0.5)
                positive_signals += 2 * conf

        for cap in MODERATE_VALUE_CAPS:
            if row.get(cap, False):
                conf = row.get(f"{cap}_confidence", 0.5)
                positive_signals += 1 * conf

        for cap in BASIC_CAPS:
            if row.get(cap, False):
                positive_signals += 0.5

        # Run contradiction checks
        for rule in CONTRADICTION_RULES:
            issue = _check_contradiction(row, rule)
            if issue:
                contradiction_flags.append(issue)
                negative_signals += rule["penalty"]

        # Missing fields penalty
        if len(missing_critical_fields) >= 3:
            contradiction_flags.append("Too many critical fields are missing")
            negative_signals += 2

        # Evidence richness signal
        evidence = {}
        try:
            evidence = json.loads(row.get("extraction_evidence", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        
        evidence_count = sum(
            len(v.get("evidence", [])) if isinstance(v, dict) else (len(v) if isinstance(v, list) else 0)
            for v in evidence.values()
        )
        if evidence_count >= 8:
            positive_signals += 3
        elif evidence_count >= 4:
            positive_signals += 1.5
        elif evidence_count <= 1:
            negative_signals += 2

        # Data completeness bonus
        completeness = row.get("data_completeness", 0.5)
        positive_signals += completeness * 2

        # Web presence bonus
        web_score = row.get("web_presence_score", 0)
        positive_signals += web_score

        # Peer comparison
        peer_flag = ""
        state = row.get("address_stateOrRegion", "")
        if state in state_avgs:
            caps_found = row.get("capabilities_found", 0)
            state_avg_caps = sum(state_avgs[state].values())
            if caps_found > state_avg_caps * 3 and caps_found > 10:
                peer_flag = f"Claims {caps_found} capabilities vs state avg {state_avg_caps:.1f} — potentially over-reported"
                negative_signals += 1
            elif caps_found < state_avg_caps * 0.2 and caps_found <= 1:
                peer_flag = f"Only {caps_found} capabilities vs state avg {state_avg_caps:.1f} — potentially under-reported"

        # Bayesian trust score
        alpha = 1 + positive_signals
        beta = 1 + negative_signals + (len(missing_critical_fields) / 2)
        trust_score = alpha / (alpha + beta)
        conf_low, conf_high = _confidence_interval(trust_score, alpha + beta)

        # Compute uncertainty confidence layer
        extracted_record = row.to_dict()
        conf_result = compute_confidence(extracted_record, contradiction_flags, evidence)

        trust_band = (
            "high" if trust_score >= 0.75
            else "medium" if trust_score >= 0.5
            else "low"
        )

        outputs.append({
            "trust_score": round(trust_score, 4),
            "trust_band": trust_band,
            "confidence": conf_result["confidence"],
            "uncertainty_flags": json.dumps(conf_result["uncertainty_flags"], ensure_ascii=False),
            "confidence_reason": json.dumps(conf_result["confidence_reason"], ensure_ascii=False),
            "confidence_low": round(conf_low, 4),
            "confidence_high": round(conf_high, 4),
            "contradiction_flags": json.dumps(contradiction_flags, ensure_ascii=False),
            "contradiction_count": len(contradiction_flags),
            "missing_critical_fields": json.dumps(missing_critical_fields),
            "peer_comparison_flag": peer_flag,
            "positive_signals": round(positive_signals, 2),
            "negative_signals": round(negative_signals, 2),
            "trust_reasoning": json.dumps({
                "positive_signals": round(positive_signals, 2),
                "negative_signals": round(negative_signals, 2),
                "contradictions": contradiction_flags,
                "missing_fields": missing_critical_fields,
                "evidence_count": evidence_count,
                "data_completeness": round(completeness, 3),
                "peer_flag": peer_flag,
                "formula": f"alpha={round(alpha,2)} / (alpha + beta={round(alpha+beta,2)}) = {round(trust_score,4)}",
            }, ensure_ascii=False),
        })

    trust_df = pd.DataFrame(outputs)
    return pd.concat([df.reset_index(drop=True), trust_df], axis=1)
