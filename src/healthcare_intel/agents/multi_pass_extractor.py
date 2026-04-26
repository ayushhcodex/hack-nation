"""Multi-Pass Capability Extractor with Confidence Scoring.

Three extraction passes:
  Pass 1 (Structured): Extract from parsed JSON arrays in specialties
  Pass 2 (Unstructured): Scan free-text fields for keyword matches with evidence
  Pass 3 (Cross-Validation): Cross-reference fields to adjust confidence up/down
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from healthcare_intel.utils import contains_any, split_sentences
from healthcare_intel.config import settings

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "indian_medical_terms.yaml"

def _load_terms() -> dict[str, list[str]]:
    """Load capability terms from YAML config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config.get("capability_terms", {})
    # Fallback minimal terms
    return {
        "has_icu": ["icu", "intensive care"],
        "has_oxygen": ["oxygen", "o2"],
        "has_emergency_surgery": ["emergency surgery", "general surgery"],
        "has_anesthesiologist": ["anesthesiologist", "anaesthesiologist"],
        "has_24x7": ["24/7", "24x7", "round the clock", "24 hours"],
    }

CAPABILITY_TERMS = _load_terms()

# Map specialties JSON values to capability flags
SPECIALTY_TO_CAPABILITY = {
    "oncology": "has_oncology",
    "cardiology": "has_cardiology",
    "neurology": "has_neurology",
    "neurosurgery": "has_neurology",
    "orthopedicSurgery": "has_orthopedic",
    "pediatrics": "has_pediatric",
    "neonatology": "has_neonatal",
    "nephrology": "has_dialysis",
    "generalSurgery": "has_general_surgery",
    "gynecologyAndObstetrics": "has_maternity",
    "ophthalmology": "has_eye_care",
    "dentistry": "has_dental",
    "endodontics": "has_dental",
    "orthodontics": "has_dental",
    "dermatology": "has_dermatology",
    "radiology": "has_xray",
    "clinicalPathology": "has_lab",
    "hematology": "has_lab",
    "anatomicPathology": "has_lab",
    "familyMedicine": "has_family_medicine",
    "internalMedicine": "has_internal_medicine",
    "emergencyMedicine": "has_trauma",
    "anesthesiology": "has_anesthesiologist",
    "urology": "has_urology",
    "gastroenterology": "has_gastro",
    "pulmonology": "has_pulmonology",
    "psychiatry": "has_psychiatry",
    "physicalMedicineAndRehabilitation": "has_rehab",
}


def _collect_evidence(text: str, terms: list[str]) -> list[str]:
    """Find sentences containing any of the search terms."""
    hits: list[str] = []
    for sentence in split_sentences(text):
        if contains_any(sentence, terms):
            hits.append(sentence.strip())
    return hits[:3]


def _pass1_structured(row: pd.Series) -> dict[str, dict]:
    """Pass 1: Extract capabilities from structured specialties JSON arrays."""
    results: dict[str, dict] = {}
    
    specialties_parsed = row.get("specialties_parsed", [])
    if not isinstance(specialties_parsed, list):
        specialties_parsed = []
    
    for specialty in specialties_parsed:
        s = str(specialty).strip()
        cap = SPECIALTY_TO_CAPABILITY.get(s)
        if cap and cap not in results:
            results[cap] = {
                "confidence": 0.7,
                "source": "structured_specialty",
                "evidence": [f"Listed specialty: {s}"],
                "pass": 1,
            }
    
    return results


def _llm_extract_capabilities(text: str) -> dict:
    if not text.strip():
        raise ValueError("Empty text")
        
    url = f"{settings.databricks_host.rstrip('/')}/serving-endpoints/{settings.llm_endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
You are a medical intelligence extractor. Read the following hospital notes and extract exactly the requested fields into JSON. Do not include markdown blocks or any other text.
Input Text: {text}

Output JSON Schema:
{{
  "icu_available": bool,
  "has_ventilator": bool,
  "has_anesthesiologist": bool,
  "surgery_capable": bool,
  "availability_24_7": bool,
  "staffing_pattern": "part_time | full_time | unknown",
  "confidence": float,
  "evidence": {{
    "icu_available": ["exact substring from text proving ICU exists"],
    "has_ventilator": ["...", ...],
    "has_anesthesiologist": ["..."],
    "surgery_capable": ["..."],
    "availability_24_7": ["..."],
    "staffing_pattern": ["..."]
  }}
}}

Rules:
- "evidence" MUST ONLY contain EXACT substrings copied verbatim from the "Input Text". Do not summarize.
- Missing info -> set bools to false or null, and evidence arrays to empty [].
- Infer implicit meaning (e.g. "visiting doctor" -> part_time).
"""
    data = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000, "temperature": 0.1}
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=12) as res:
            resp_data = json.loads(res.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            # Anti-hallucination strict check
            for k, ev_list in result.get("evidence", {}).items():
                if not isinstance(ev_list, list): continue
                for ev in ev_list:
                    if str(ev).strip() and str(ev) not in text:
                        raise ValueError(f"Hallucinated evidence detected: {ev}")
                        
            return result
    except Exception as e:
        raise ValueError(f"LLM extraction failed: {e}")


def _pass2_regex_fallback(row: pd.Series) -> dict[str, dict]:
    """Fallback: Scan free-text fields for keyword matches."""
    results: dict[str, dict] = {}
    
    text_fields = {
        "description": str(row.get("description", "")),
        "capability": str(row.get("capability", "")),
        "procedure": str(row.get("procedure", "")),
        "equipment": str(row.get("equipment", "")),
    }
    
    combined_text = " ".join(text_fields.values())
    combined_text = re.sub(r"\s+", " ", combined_text).strip()
    
    for flag, terms in CAPABILITY_TERMS.items():
        evidence = _collect_evidence(combined_text, terms)
        if evidence:
            # Track which fields the evidence came from
            field_sources = []
            for field_name, field_text in text_fields.items():
                if any(contains_any(field_text, [t]) for t in terms):
                    field_sources.append(field_name)
            
            multi_field = len(field_sources) > 1
            conf = 0.8 if multi_field else 0.6
            
            results[flag] = {
                "confidence": conf,
                "source": "unstructured_text",
                "evidence": evidence,
                "fields_found_in": field_sources,
                "multi_field_corroboration": multi_field,
                "pass": 2,
            }
    
    return results


def _pass2_unstructured(row: pd.Series) -> dict[str, dict]:
    """Pass 2: LLM reasoning with regex fallback."""
    text_fields = {
        "description": str(row.get("description", "")),
        "capability": str(row.get("capability", "")),
        "procedure": str(row.get("procedure", "")),
        "equipment": str(row.get("equipment", "")),
    }
    
    combined_text = " ".join(text_fields.values())
    combined_text = re.sub(r"\s+", " ", combined_text).strip()
    
    llm_ready = (
        bool(settings.databricks_token)
        and settings.databricks_host
        and settings.databricks_host != "https://dbc-xxxxxxx.cloud.databricks.com"
    )

    if len(combined_text) > 10 and llm_ready:
        try:
            llm_res = _llm_extract_capabilities(combined_text)
            results = {}
            conf = min(max(llm_res.get("confidence", 0.7), 0.1), 1.0)
            
            mapping = {
                "has_icu": "icu_available",
                "has_ventilator": "has_ventilator",
                "has_anesthesiologist": "has_anesthesiologist",
                "has_emergency_surgery": "surgery_capable",
                "has_24x7": "availability_24_7",
            }
            
            for cap_flag, llm_key in mapping.items():
                if llm_res.get(llm_key) is True:
                    evidence = llm_res.get("evidence", {}).get(llm_key, [])
                    results[cap_flag] = {
                        "confidence": conf,
                        "source": "llm_agent",
                        "evidence": evidence,
                        "fields_found_in": ["unstructured_combined"],
                        "multi_field_corroboration": False,
                        "pass": 2,
                    }
                    
            if llm_res.get("staffing_pattern") == "part_time":
                evidence = llm_res.get("evidence", {}).get("staffing_pattern", [])
                results["uses_parttime_doctors"] = {
                    "confidence": conf,
                    "source": "llm_agent",
                    "evidence": evidence,
                    "fields_found_in": ["unstructured_combined"],
                    "multi_field_corroboration": False,
                    "pass": 2,
                }
            return results
        except Exception:
            # Silently fallback to old regex approach if LLM fails or hallucinates
            pass
            
    return _pass2_regex_fallback(row)


def _pass3_cross_validate(
    pass1: dict[str, dict],
    pass2: dict[str, dict],
    row: pd.Series,
) -> dict[str, dict]:
    """Pass 3: Cross-reference passes to adjust confidence."""
    merged: dict[str, dict] = {}
    all_keys = set(pass1.keys()) | set(pass2.keys())
    
    for key in all_keys:
        p1 = pass1.get(key)
        p2 = pass2.get(key)
        
        if p1 and p2:
            # Both passes found it → high confidence
            merged[key] = {
                "confidence": min(0.95, max(p1["confidence"], p2["confidence"]) + 0.15),
                "source": "cross_validated",
                "evidence": list(dict.fromkeys(p1.get("evidence", []) + p2.get("evidence", []))),
                "passes_found": [1, 2],
                "reasoning": "Corroborated by both structured specialty and unstructured text",
            }
        elif p1:
            # Only structured → moderate confidence
            merged[key] = {
                **p1,
                "reasoning": "Found in structured specialties only; no text corroboration",
            }
        elif p2:
            # Only unstructured → depends on multi-field
            reasoning = (
                "Found in multiple text fields (corroborated)"
                if p2.get("multi_field_corroboration")
                else "Found in text only (single field)"
            )
            merged[key] = {**p2, "reasoning": reasoning}
    
    # Derived capabilities
    icu_conf = merged.get("has_icu", {}).get("confidence", 0)
    oxy_conf = merged.get("has_oxygen", {}).get("confidence", 0)
    if icu_conf > 0 and oxy_conf > 0:
        merged["has_functional_icu"] = {
            "confidence": round(min(icu_conf, oxy_conf), 4),
            "source": "derived",
            "evidence": (
                merged.get("has_icu", {}).get("evidence", [])
                + merged.get("has_oxygen", {}).get("evidence", [])
            ),
            "reasoning": "Derived: ICU + Oxygen both confirmed",
        }
    
    ot_conf = merged.get("has_ot", {}).get("confidence", 0)
    anes_conf = merged.get("has_anesthesiologist", {}).get("confidence", 0)
    if ot_conf > 0 and anes_conf > 0:
        merged["has_surgical_capability"] = {
            "confidence": round(min(ot_conf, anes_conf), 4),
            "source": "derived",
            "evidence": (
                merged.get("has_ot", {}).get("evidence", [])
                + merged.get("has_anesthesiologist", {}).get("evidence", [])
            ),
            "reasoning": "Derived: OT + Anesthesiologist both confirmed",
        }
    
    # Facility type sanity check
    facility_type = str(row.get("facilityTypeId", "")).lower()
    if facility_type in ("pharmacy", "farmacy", "dentist"):
        for high_acuity in ["has_icu", "has_oncology", "has_dialysis", "has_neonatal", "has_emergency_surgery"]:
            if high_acuity in merged:
                merged[high_acuity]["confidence"] = max(0.1, merged[high_acuity]["confidence"] - 0.4)
                merged[high_acuity]["reasoning"] += f" [DOWNGRADED: unlikely for {facility_type}]"
    
    return merged


def extract_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Run 3-pass extraction on all facility rows."""
    rows: list[dict[str, Any]] = []
    
    for _, row in df.iterrows():
        pass1 = _pass1_structured(row)
        pass2 = _pass2_unstructured(row)
        merged = _pass3_cross_validate(pass1, pass2, row)
        
        out: dict[str, Any] = {}
        
        # Set boolean flags and confidence scores
        all_caps = set(CAPABILITY_TERMS.keys()) | set(SPECIALTY_TO_CAPABILITY.values()) | {
            "has_functional_icu", "has_surgical_capability",
            "has_family_medicine", "has_internal_medicine", "has_dermatology",
            "has_urology", "has_gastro", "has_pulmonology", "has_psychiatry", "has_rehab",
            "has_ventilator", "uses_parttime_doctors"
        }
        
        for cap in sorted(all_caps):
            info = merged.get(cap)
            if info:
                out[cap] = info["confidence"] >= 0.3
                out[f"{cap}_confidence"] = round(info["confidence"], 4)
            else:
                out[cap] = False
                out[f"{cap}_confidence"] = 0.0
        
        # Evidence and extraction metadata
        evidence_detail = {
            k: {
                "evidence": v.get("evidence", []),
                "confidence": v.get("confidence", 0),
                "source": v.get("source", ""),
                "reasoning": v.get("reasoning", ""),
            }
            for k, v in merged.items()
        }
        
        out["extraction_evidence"] = json.dumps(evidence_detail, ensure_ascii=False)
        out["extraction_text_length"] = len(str(row.get("full_text", "")))
        out["capabilities_found"] = sum(1 for c in all_caps if out.get(c, False))
        out["avg_confidence"] = round(
            sum(out.get(f"{c}_confidence", 0) for c in all_caps if out.get(c, False))
            / max(1, out["capabilities_found"]),
            4,
        )
        
        # Trace for chain of thought
        out["extraction_trace"] = json.dumps({
            "pass1_structured_count": len(pass1),
            "pass2_unstructured_count": len(pass2),
            "pass3_merged_count": len(merged),
            "cross_validated_count": sum(
                1 for v in merged.values() if v.get("source") == "cross_validated"
            ),
            "derived_count": sum(
                1 for v in merged.values() if v.get("source") == "derived"
            ),
        }, ensure_ascii=False)
        
        rows.append(out)
    
    extracted = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), extracted], axis=1)
