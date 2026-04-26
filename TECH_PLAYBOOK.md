# HealthBricks India: Technical Playbook & Demo Cheat Sheet

This document contains the complete architectural breakdown of the Agentic System. Use this as your master cheat sheet for answering Judge Q&A during your pitch.

---

## 1. ARCHITECTURE
**Pipeline Flow (Step-by-Step)**
1. **Ingestion**: Raw India healthcare CSV data is loaded via `data_loader.py`.
2. **First-Pass Regex (Speed)**: `multi_pass_extractor.py` runs a lightning-fast keyword scan for basic capabilities to structure easy data.
3. **Second-Pass LLM (Reasoning)**: Unstructured, messy text is dynamically passed to the Databricks Llama 3.3 Agent to intelligently extract hidden attributes.
4. **Validation Layer**: `validator_agent.py` audits extracted data against facility metadata using an LLM to flag medical paradoxes.
5. **Trust Scoring**: `trust_scorer.py` aggregates evidence and contradictions to assign a definitive 0-100% Trust Score.
6. **Persistence**: Clean, structured dataframe is exported to `outputs/facilities_enriched.parquet`.
7. **Semantic Indexing**: Pipeline attempts to push the cleaned data to Mosaic AI Vector Search.

---

## 2. FILE STRUCTURE
**Key Files & Purpose**
- `src/healthbricks_india/agents/multi_pass_extractor.py`: The core LLM extraction engine converting messy raw text into Strict JSON.
- `src/healthbricks_india/agents/genie_orchestrator.py`: The LangChain-style Autonomous Router mapping natural language to backend Tool Functions.
- `src/healthbricks_india/reasoning/query_engine.py`: The AI Query Planner that parses user queries (e.g. "rural clinics") into search parameters.
- `src/healthbricks_india/reasoning/vector_search.py`: The interface layer for Databricks Mosaic AI Hybrid Search.
- `src/healthbricks_india/validation/validator_agent.py`: The Medical Auditor LLM that spots logic contradictions.
- `src/healthbricks_india/validation/trust_scorer.py`: The mathematical engine grading facility reliability.
- `frontend/app.js` & `styles.css`: The map clustering engine and glassmorphism interface.

---

## 3. EXTRACTION (`multi_pass_extractor.py`)
**Current Logic**
Bypasses basic regex for complex notes. Connects to `databricks-meta-llama-3-3-70b-instruct` to extract capabilities. It enforces **Verbatim Anchoring**, rejecting answers if the LLM hallucinates evidence not found in the original string.

**Sample Input → Output**
*Input:* `"The site has basic checkups. Dr. Singh brings ventilator machinery occasionally for severe trauma."*
*Output JSON:* 
```json
{
  "has_ventilator": true,
  "surgery_capable": false,
  "staffing_pattern": "part_time",
  "evidence": {
    "has_ventilator": ["Dr. Singh brings ventilator machinery occasionally"]
  }
}
```

---

## 4. QUERY ENGINE
**How query is parsed**
User types a query. Instead of regex, it hits `LLMQueryPlanner`. The LLM identifies the geographic target (`required_state`) and the medical targets (`required_capabilities`).

**Example Query → Execution Path**
*User Input:* `"Find a facility in Bihar for emergency appendectomy with part-time doctors"`
*Execution Path:*
1. LLM breaks it down into structured constraints: `state="Bihar"`, `caps=["has_appendectomy", "uses_parttime"]`.
2. The UI receives a `chain_of_thought` trace showing this breakdown.
3. The engine filters the dataframe or queries Vector Search for these exact metadata attributes.
4. Results are sorted by `trust_score` (Descending).

---

## 5. TRUST SCORER
**Rules & Scoring Formula**
The `trust_scorer.py` evaluates every row utilizing a weighted equation:
- **Base Score (40%)**: Derived from the LLM's confidence rating of the extracted text.
- **Evidence Weight (40%)**: Boosts score if explicit verbatim evidence citations are mapped to the capabilities.
- **Penalties (20%)**: Drastically reduces the score if `validator_agent.py` found contradictions.
- *Scaling:* The final float is scaled from 0-1, mapped into a "High/Medium/Low" trust band, and color-coordinated in the UI.

---

## 6. VALIDATOR
**What contradictions are checked**
The logic operates dynamically via Llama 3.3. It feeds the LLM the extracted capabilities alongside the `facilityType`.
*Examples of what it catches:*
- Anesthesiologist missing, but "Emergency Surgery" is flagged.
- Oxygen missing, but "ICU" is flagged.
- A "Primary Care Clinic" facility metadata tag claiming to have "Advanced Neurosurgery".

---

## 7. VECTOR SEARCH
**On/Off functionality**
- Configured by `.env` via `USE_VECTOR_SEARCH = true/false`.
- **Where used:** In `query_engine.py`, if true, the system skips basic Pandas DataFrame filtering and offloads the hybrid semantic query to the Databricks Mosaic API, finding results that are contextually similar even if exact keywords don't match.

---

## 8. API OUTPUT
**Sample `/api/query` Response JSON**
```json
{
  "results": [
    {
      "name": "Apollo Clinic",
      "address_stateOrRegion": "Bihar",
      "trust_score": 0.89,
      "trust_band": "high",
      "matched_capabilities": "has_appendectomy",
      "citations": "[\"Performs basic appendectomies\"]"
    }
  ],
  "chain_of_thought": [
    {
      "agent": "LLMQueryPlanner",
      "action": "Parsed query logic",
      "detail": "Constraints: ['has_appendectomy']. Reasoning: Target constraint."
    },
    {
      "agent": "Verifier",
      "action": "Cross-checked trust scores",
      "detail": "No results to verify"
    }
  ]
}
```

---

## 9. DEMO FLOW
**What to show the judges (Live Pitch Strategy)**
1. **The Discover Tab:** Type a complex query (e.g. *"Appendectomy in Rural Bihar under low trust"*).
2. **Show the Chain of Thought:** Scroll down and point explicitly to the **🧠 Agent Chain of Thought** trace box to prove it's transparent, not a black box.
3. **The Modal:** Click a facility result. Show the popup highlighting Verbatim Extraction Evidence and any Contradiction Alerts (Neuro-Symbolic AI).
4. **The Desert Map:** Switch to the map tab. Let the massive 10,000-node Leaflet MarkerClusters animate inward. Click a red cluster to show a "High-Risk Medical Desert" flagged by your logic.
