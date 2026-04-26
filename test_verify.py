import urllib.request
import json
from unittest.mock import patch, MagicMock
from io import BytesIO

# Import the actual implementations
from healthbricks_india.agents.multi_pass_extractor import _llm_extract_capabilities, _pass2_unstructured
from healthbricks_india.reasoning.query_engine import _llm_parse_query
import pandas as pd

# Fake successful Databricks LLM response payload for TASK 1 (Extraction)
fake_extraction_response = {
    "choices": [{
        "message": {
            "content": json.dumps({
                "icu_available": True,
                "staffing_pattern": "part_time",
                "confidence": 0.9,
                "has_anesthesiologist": False,
                "evidence": {
                    "icu_available": ["Intensive Care Unit fully operational since 2012"],
                    "staffing_pattern": ["visiting specialist doctors"]
                }
            })
        }
    }]
}

# Fake successful Databricks LLM response payload for TASK 2 (Query Parsing)
fake_query_response = {
    "choices": [{
        "message": {
            "content": json.dumps({
                "required_capabilities": ["has_icu", "uses_parttime_doctors"],
                "required_state": "Delhi",
                "reasoning": "Extracted constraints based on standard semantics."
            })
        }
    }]
}


def test_task1():
    print("\n--- Testing Task 1 (Extraction) ---")
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(fake_extraction_response).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    # Simulated unstructured free-text note
    text = "We have an Intensive Care Unit fully operational since 2012 along with visiting specialist doctors."
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        # 1. Test raw LLM extraction
        res = _llm_extract_capabilities(text)
        print("[Pass] Successfully parsed JSON from LLM.")
        print(f"       Extracted Data: icu={res.get('icu_available')}, staff={res.get('staffing_pattern')}")

        # 2. Test full pipeline mapping in `_pass2_unstructured`
        row = pd.Series({
            "description": text,
            "capability": "",
            "procedure": "",
            "equipment": ""
        })
        normalized_results = _pass2_unstructured(row)
        print("[Pass] Successfully normalized into existing capability schema:")
        for cap, meta in normalized_results.items():
            print(f"       => {cap}: confidence {meta['confidence']}, source: {meta['source']}")


def test_task2():
    print("\n--- Testing Task 2 (Query Planner) ---")
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(fake_query_response).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        import healthbricks_india.config
        # Override host temporarily to pass the missing host check
        healthbricks_india.config.settings.databricks_host = "https://dbc-realworkspace.cloud.databricks.com"
        
        parsed, reason = _llm_parse_query("query about delhi icu part time", ["Delhi", "Maharashtra"])
        print("[Pass] Successfully invoked LLM to parse generic text to constraints:")
        print(f"       Capabilities Matched: {parsed.required_capabilities}")
        print(f"       State Matched: {parsed.required_state}")
        print(f"       Reasoning: {reason}")


if __name__ == "__main__":
    test_task1()
    test_task2()
