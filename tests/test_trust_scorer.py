import json

import pandas as pd

from healthcare_intel.validation.trust_scorer import score_trust


def test_trust_scorer_flags_missing_anesthesiologist() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "Demo Hospital",
                "address_stateOrRegion": "Bihar",
                "address_zipOrPostcode": "800001",
                "description": "Emergency surgery available 24/7",
                "equipment": "ICU, oxygen",
                "has_icu": True,
                "has_oxygen": True,
                "has_emergency_surgery": True,
                "has_appendectomy": True,
                "has_anesthesiologist": False,
                "has_24x7": True,
                "uses_parttime_doctors": True,
                "numberDoctors": 5,
                "extraction_evidence": json.dumps({"has_emergency_surgery": ["Emergency surgery available"]}),
            }
        ]
    )

    out = score_trust(df)
    assert out.loc[0, "trust_score"] < 0.75
    assert "anesthesiologist" in out.loc[0, "contradiction_flags"].lower()
