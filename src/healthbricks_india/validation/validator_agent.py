from __future__ import annotations

import json
import re
import urllib.request
import pandas as pd

import mlflow
from healthbricks_india.config import settings

MEDICAL_STANDARDS = {
    "has_emergency_surgery": ["has_anesthesiologist"],
    "has_appendectomy": ["has_anesthesiologist"],
    "has_icu": ["has_oxygen"],
}


@mlflow.trace(name="llm_validate", span_type="LLM")
def _llm_validate_capabilities(capabilities: dict, metadata: dict) -> list[str]:
    if (
        not settings.databricks_host
        or settings.databricks_host == "https://dbc-xxxxxxx.cloud.databricks.com"
        or "YOUR-DATABRICKS" in settings.databricks_host
        or not settings.databricks_token
    ):
        raise ValueError("Databricks LLM credentials not configured")

    url = f"{settings.databricks_host.rstrip('/')}/serving-endpoints/{settings.llm_endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json",
    }
    
    prompt = f"""
You are a Medical Consistency Auditor. Review the facility capabilities and metadata below.
Identify any logical medical contradictions (e.g. 'Clinic claims advanced surgery but is just a Dentist').
If there are no contradictions, return an empty array.
If there are contradictions, return an array of short strings detailing the exact issue.

Metadata context:
{json.dumps(metadata, indent=2)}

Capabilities claimed:
{json.dumps(capabilities, indent=2)}

Respond with STRICT JSON format:
{{
  "contradictions": [
    "string detail 1"
  ]
}}
"""
    data = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0.1}
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
            result = json.loads(re.sub(r"```json|```", "", content).strip())
            return result.get("contradictions", [])
    except Exception as e:
        raise ValueError(f"LLM validation failed: {e}")


@mlflow.trace(name="validator_agent", span_type="AGENT")
def validate_against_standards(df: pd.DataFrame) -> pd.DataFrame:
    issues = []

    for _, row in df.iterrows():
        record_issues = []
        
        metadata = {
            "facilityType": row.get("facilityTypeId", "Unknown"),
            "operatorType": row.get("operatorTypeId", "Unknown"),
        }
        
        caps = {
            k: bool(row[k]) for k in [
                "has_icu", "has_ventilator", "has_anesthesiologist", 
                "has_emergency_surgery", "has_appendectomy", "has_oxygen", "has_neonatal"
            ] if k in row and row[k]
        }
        
        if caps:
            try:
                llm_issues = _llm_validate_capabilities(caps, metadata)
                record_issues.extend(llm_issues)
            except Exception:
                for capability, deps in MEDICAL_STANDARDS.items():
                    if row.get(capability, False):
                        for dep in deps:
                            if not row.get(dep, False):
                                record_issues.append(f"{capability} is true but dependency {dep} is not evidenced")

        issues.append(
            {
                "validator_issues": json.dumps(record_issues, ensure_ascii=False),
                "validator_issue_count": len(record_issues),
            }
        )

    issues_df = pd.DataFrame(issues)
    return pd.concat([df.reset_index(drop=True), issues_df], axis=1)
