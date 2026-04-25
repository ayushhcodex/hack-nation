import pandas as pd

from healthcare_intel.reasoning.query_engine import run_query


def test_query_engine_filters_by_capability_and_state() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "A",
                "address_city": "Patna",
                "address_stateOrRegion": "Bihar",
                "address_zipOrPostcode": "800001",
                "trust_score": 0.9,
                "trust_band": "high",
                "has_appendectomy": True,
                "uses_parttime_doctors": True,
                "extraction_evidence": "{}",
            },
            {
                "name": "B",
                "address_city": "Lucknow",
                "address_stateOrRegion": "Uttar Pradesh",
                "address_zipOrPostcode": "226001",
                "trust_score": 0.95,
                "trust_band": "high",
                "has_appendectomy": True,
                "uses_parttime_doctors": True,
                "extraction_evidence": "{}",
            },
        ]
    )

    result = run_query(
        df,
        "Find in Bihar for appendectomy and parttime doctors",
        top_k=10,
    )
    assert len(result) == 1
    assert result.loc[0, "name"] == "A"
