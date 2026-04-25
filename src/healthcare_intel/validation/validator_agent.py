from __future__ import annotations

import json

import pandas as pd

MEDICAL_STANDARDS = {
    "has_emergency_surgery": ["has_anesthesiologist"],
    "has_appendectomy": ["has_anesthesiologist"],
    "has_icu": ["has_oxygen"],
}


def validate_against_standards(df: pd.DataFrame) -> pd.DataFrame:
    issues = []

    for _, row in df.iterrows():
        record_issues = []
        for capability, deps in MEDICAL_STANDARDS.items():
            if row.get(capability, False):
                for dep in deps:
                    if not row.get(dep, False):
                        record_issues.append(
                            f"{capability} is true but dependency {dep} is not evidenced"
                        )

        issues.append(
            {
                "validator_issues": json.dumps(record_issues, ensure_ascii=False),
                "validator_issue_count": len(record_issues),
            }
        )

    issues_df = pd.DataFrame(issues)
    return pd.concat([df.reset_index(drop=True), issues_df], axis=1)
