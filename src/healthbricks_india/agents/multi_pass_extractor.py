"""Dual-Pass Capability Extractor with Probabilistic Arbitration.

Architecture:
  Pass 1 (Structured): Extract from parsed JSON arrays in specialties
  Pass 2A (Keyword): ALWAYS runs — scans free-text fields for all 30+ capabilities
  Pass 2B (LLM Semantic): Runs when Databricks credentials available — deep semantic extraction
  Pass 3 (Arbitration): Merges both Pass 2 results via evidence-backed confidence scoring

Key improvement over the original:
  - Keyword and LLM passes ALWAYS both run (no fallback/skipping)
  - Every facility gets full 30+ capability coverage from keywords
  - LLM adds semantic depth and implicit inference on top
  - Probabilistic arbitration resolves disagreements with evidence weighting
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from healthbricks_india.utils import contains_any, split_sentences
import mlflow
from healthbricks_india.config import settings

logger = logging.getLogger(__name__)

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

# LLM output key → internal capability flag mapping
LLM_KEY_TO_CAPABILITY = {
    "icu_available": "has_icu",
    "has_ventilator": "has_ventilator",
    "has_anesthesiologist": "has_anesthesiologist",
    "surgery_capable": "has_emergency_surgery",
    "availability_24_7": "has_24x7",
}


def _collect_evidence(text: str, terms: list[str]) -> list[str]:
    """Find sentences containing any of the search terms."""
    hits: list[str] = []
    for sentence in split_sentences(text):
        if contains_any(sentence, terms):
            hits.append(sentence.strip())
    return hits[:3]


# ---------------------------------------------------------------------------
# Pass 1: Structured specialty mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pass 2A: Keyword-based extraction (ALWAYS runs for every row)
# ---------------------------------------------------------------------------

def _pass2a_keyword(row: pd.Series) -> dict[str, dict]:
    """Pass 2A: Scan free-text fields for keyword matches across ALL capabilities.

    This pass ALWAYS runs, providing the full 30+ capability baseline.
    It does not depend on LLM availability.
    """
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
                "source": "keyword",
                "evidence": evidence,
                "fields_found_in": field_sources,
                "multi_field_corroboration": multi_field,
                "pass": "2a",
            }

    return results


# ---------------------------------------------------------------------------
# Pass 2B: LLM semantic extraction (runs when credentials available)
# ---------------------------------------------------------------------------

@mlflow.trace(name="llm_extract", span_type="LLM")
def _llm_extract_capabilities(text: str) -> dict:
    """Call Databricks Llama 3.3 for deep semantic extraction.

    Returns raw LLM response dict or raises ValueError on failure.
    """
    if not text.strip():
        raise ValueError("Empty text")

    url = f"{settings.databricks_host.rstrip('/')}/serving-endpoints/{settings.llm_endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json",
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
                if not isinstance(ev_list, list):
                    continue
                for ev in ev_list:
                    if str(ev).strip() and str(ev) not in text:
                        raise ValueError(f"Hallucinated evidence detected: {ev}")

            return result
    except Exception as e:
        raise ValueError(f"LLM extraction failed: {e}")


def _pass2b_llm(row: pd.Series) -> dict[str, dict]:
    """Pass 2B: LLM semantic extraction. Returns empty dict on failure (never blocks).

    This pass runs independently of Pass 2A. If it fails, keyword results
    stand alone — no capability is ever lost.
    """
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

    if len(combined_text) <= 10 or not llm_ready:
        return {}

    try:
        llm_res = _llm_extract_capabilities(combined_text)
        results: dict[str, dict] = {}
        conf = min(max(llm_res.get("confidence", 0.7), 0.1), 1.0)

        for llm_key, cap_flag in LLM_KEY_TO_CAPABILITY.items():
            if llm_res.get(llm_key) is True:
                evidence = llm_res.get("evidence", {}).get(llm_key, [])
                results[cap_flag] = {
                    "confidence": conf,
                    "source": "llm_agent",
                    "evidence": evidence,
                    "fields_found_in": ["unstructured_combined"],
                    "multi_field_corroboration": False,
                    "pass": "2b",
                }

        if llm_res.get("staffing_pattern") == "part_time":
            evidence = llm_res.get("evidence", {}).get("staffing_pattern", [])
            results["uses_parttime_doctors"] = {
                "confidence": conf,
                "source": "llm_agent",
                "evidence": evidence,
                "fields_found_in": ["unstructured_combined"],
                "multi_field_corroboration": False,
                "pass": "2b",
            }

        return results
    except Exception:
        # LLM failed — keyword results will stand alone. No capability is lost.
        return {}


# ---------------------------------------------------------------------------
# Pass 3: Probabilistic Arbitration (merges structured + keyword + LLM)
# ---------------------------------------------------------------------------

def _pass3_arbitrate(
    pass1: dict[str, dict],
    pass2a_kw: dict[str, dict],
    pass2b_llm: dict[str, dict],
    row: pd.Series,
) -> dict[str, dict]:
    """Pass 3: Merge all extraction results via evidence-backed arbitration.

    Arbitration rules:
      - All three agree  → highest confidence + 0.2 (capped at 0.95)
      - Two sources agree → max confidence + 0.15 (cross-validated)
      - LLM only (keyword disagrees) → LLM confidence × 0.8 (slight discount)
      - Keyword only (LLM explicitly returned false) → keyword confidence × 0.7
      - Keyword only (LLM unavailable/failed) → keyword confidence (stands alone)
      - Structured only → 0.7 (moderate)
    """
    merged: dict[str, dict] = {}
    all_keys = set(pass1.keys()) | set(pass2a_kw.keys()) | set(pass2b_llm.keys())
    llm_ran = len(pass2b_llm) > 0 or (
        bool(settings.databricks_token)
        and settings.databricks_host
        and settings.databricks_host != "https://dbc-xxxxxxx.cloud.databricks.com"
    )

    for key in all_keys:
        p1 = pass1.get(key)
        kw = pass2a_kw.get(key)
        llm = pass2b_llm.get(key)

        sources_found = []
        all_evidence: list[str] = []
        confidences: list[float] = []

        if p1:
            sources_found.append("structured")
            all_evidence.extend(p1.get("evidence", []))
            confidences.append(p1["confidence"])
        if kw:
            sources_found.append("keyword")
            all_evidence.extend(kw.get("evidence", []))
            confidences.append(kw["confidence"])
        if llm:
            sources_found.append("llm")
            all_evidence.extend(llm.get("evidence", []))
            confidences.append(llm["confidence"])

        # Deduplicate evidence while preserving order
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            if e not in seen:
                seen.add(e)
                unique_evidence.append(e)

        # --- Arbitration logic ---
        num_sources = len(sources_found)
        base_conf = max(confidences) if confidences else 0.0

        if num_sources >= 3:
            # All three sources agree
            final_conf = min(0.95, base_conf + 0.2)
            source_tag = "triple_validated"
            reasoning = "Confirmed by structured specialty, keyword scan, AND LLM semantic extraction"
        elif num_sources == 2:
            if "llm" in sources_found and "keyword" in sources_found:
                # Both unstructured methods agree — strong signal
                final_conf = min(0.95, base_conf + 0.15)
                source_tag = "cross_validated"
                reasoning = "Corroborated by both keyword scan and LLM semantic extraction"
            elif "structured" in sources_found:
                # Structured + one unstructured
                final_conf = min(0.95, base_conf + 0.15)
                source_tag = "cross_validated"
                other = [s for s in sources_found if s != "structured"][0]
                reasoning = f"Corroborated by structured specialty and {other} extraction"
            else:
                final_conf = min(0.95, base_conf + 0.15)
                source_tag = "cross_validated"
                reasoning = f"Corroborated by {' and '.join(sources_found)}"
        elif num_sources == 1:
            source = sources_found[0]
            if source == "llm":
                # LLM found it but keyword didn't — LLM inferred implicitly
                final_conf = base_conf * 0.85
                source_tag = "llm_only"
                reasoning = "Detected by LLM semantic inference only (no keyword match)"
            elif source == "keyword":
                if llm_ran and key in LLM_KEY_TO_CAPABILITY.values():
                    # LLM ran and explicitly did NOT find this capability
                    final_conf = base_conf * 0.7
                    source_tag = "keyword_only_llm_disagrees"
                    reasoning = "Keyword matched but LLM did not confirm — possible false positive"
                else:
                    # LLM didn't run, or this cap isn't in LLM's scope
                    final_conf = base_conf
                    source_tag = "keyword_only"
                    reasoning = "Detected by keyword scan (LLM did not cover this capability)"
            else:
                # Structured only
                final_conf = base_conf
                source_tag = "structured_only"
                reasoning = "Found in structured specialties only; no text corroboration"
        else:
            continue  # Shouldn't happen but safety guard

        merged[key] = {
            "confidence": round(min(0.95, max(0.05, final_conf)), 4),
            "source": source_tag,
            "evidence": unique_evidence,
            "sources_found": sources_found,
            "reasoning": reasoning,
        }

    # --- Derived capabilities ---
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
            "sources_found": ["derived"],
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
            "sources_found": ["derived"],
            "reasoning": "Derived: OT + Anesthesiologist both confirmed",
        }

    # --- Facility type sanity check ---
    facility_type = str(row.get("facilityTypeId", "")).lower()
    if facility_type in ("pharmacy", "farmacy", "dentist"):
        for high_acuity in ["has_icu", "has_oncology", "has_dialysis", "has_neonatal", "has_emergency_surgery"]:
            if high_acuity in merged:
                merged[high_acuity]["confidence"] = max(0.05, merged[high_acuity]["confidence"] - 0.4)
                merged[high_acuity]["reasoning"] += f" [DOWNGRADED: unlikely for {facility_type}]"

    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@mlflow.trace(name="dual_pass_extractor", span_type="AGENT")
def extract_capabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Run dual-pass extraction on all facility rows.

    For every row:
      1. Pass 1: Structured specialty mapping (instant)
      2. Pass 2A: Keyword scan for ALL 30+ capabilities (instant)
      3. Pass 2B: LLM semantic extraction (when available, adds depth)
      4. Pass 3: Probabilistic arbitration merging all results
    """
    rows: list[dict[str, Any]] = []
    llm_success_count = 0
    llm_fail_count = 0
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        # Pass 1: Structured (instant)
        pass1 = _pass1_structured(row)

        # Pass 2A: Keyword — ALWAYS runs (instant, full 30+ capabilities)
        pass2a_kw = _pass2a_keyword(row)

        # Pass 2B: LLM — runs when available (adds semantic depth)
        pass2b_llm = _pass2b_llm(row)
        if pass2b_llm:
            llm_success_count += 1
        elif (
            bool(settings.databricks_token)
            and settings.databricks_host
            and settings.databricks_host != "https://dbc-xxxxxxx.cloud.databricks.com"
        ):
            llm_fail_count += 1

        # Pass 3: Probabilistic arbitration
        merged = _pass3_arbitrate(pass1, pass2a_kw, pass2b_llm, row)

        out: dict[str, Any] = {}

        # Set boolean flags and confidence scores
        all_caps = set(CAPABILITY_TERMS.keys()) | set(SPECIALTY_TO_CAPABILITY.values()) | {
            "has_functional_icu", "has_surgical_capability",
            "has_family_medicine", "has_internal_medicine", "has_dermatology",
            "has_urology", "has_gastro", "has_pulmonology", "has_psychiatry", "has_rehab",
            "has_ventilator", "uses_parttime_doctors",
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
                "sources_found": v.get("sources_found", []),
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

        # Extraction trace for chain of thought
        out["extraction_trace"] = json.dumps({
            "pass1_structured_count": len(pass1),
            "pass2a_keyword_count": len(pass2a_kw),
            "pass2b_llm_count": len(pass2b_llm),
            "pass3_merged_count": len(merged),
            "cross_validated_count": sum(
                1 for v in merged.values() if "cross_validated" in v.get("source", "") or v.get("source") == "triple_validated"
            ),
            "llm_only_count": sum(
                1 for v in merged.values() if v.get("source") == "llm_only"
            ),
            "keyword_only_count": sum(
                1 for v in merged.values() if "keyword_only" in v.get("source", "")
            ),
            "derived_count": sum(
                1 for v in merged.values() if v.get("source") == "derived"
            ),
        }, ensure_ascii=False)

        rows.append(out)

        if (idx + 1) % 500 == 0:
            logger.info(f"[Extractor] Processed {idx + 1}/{total} facilities (LLM: {llm_success_count} ok, {llm_fail_count} failed)")

    logger.info(
        f"[Extractor] Complete: {total} facilities. "
        f"LLM enriched: {llm_success_count}/{total}. "
        f"LLM failures: {llm_fail_count}. "
        f"Keyword baseline: {total}/{total} (always runs)."
    )

    extracted = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), extracted], axis=1)
