# Pitch Context: HealthBricks India

## 1. Project Identity
*   **Name**: HealthBricks India
*   **Mission**: Restructuring India's messy healthcare data into a high-trust, actionable intelligence network for crisis response and infrastructure planning.
*   **Target**: 10,000+ facilities across India (unstructured clinical notes, specialties, and supply data).

## 2. The Problem: "The Data Fog"
*   Healthcare data in India is messy, unstructured, and often untrustworthy.
*   Critical resource gaps (ICUs, Oxygen, Anesthesia) are hidden inside lengthy text descriptions.
*   Decision-makers can't distinguish between a "reliable provider" and a "claimed capability" without a manual audit.
*   "Medical Deserts" (regions with zero critical care access) are invisible without specialized analytics.

## 3. The Innovation: Agentic Neuro-Symbolic Intelligence
Our system doesn't just "search" data; it **reconstructs** it with three distinct layers of intelligence:

### Layer 1: Scalable Dual-Pass Extraction
*   **Parallel Execution**: Runs LLM Semantic Extraction (Llama 3.3) and Vectorized Keyword Scanning simultaneously.
*   **Probabilistic Arbitration**: If the LLM and the keyword scanner disagree, a Bayesian arbitration layer decides the final confidence score based on evidence richness.
*   **Verbatim Anchoring**: Every single capability found (e.g., "Has ICU") is backed by an exact, highlighted quote from the original source text to prevent AI hallucinations.

### Layer 2: Bayesian Trust Scoring Engine
*   **Vectorized Logic**: Processes 10,000 facilities in seconds (vs. hours with standard loops).
*   **High-Acuity Penalties**: Claims are cross-checked against reality. (e.g., A "Pharmacy" claiming "Emergency Surgery" is automatically flagged for a 40% trust penalty).
*   **Medical Standard Validation**: Identifies clinical paradoxes like "Surgery capability claimed without an Anesthesiologist."

### Layer 3: Crisis Mapping & Desert Detection
*   **Medical Desert Detection**: Identifies regions (PIN-code level) with systemic gaps in Lifesaving care (Dialysis, Trauma, Neonatal care).
*   **Actionable Logistics**: Calculates the exact distance (km) to the nearest functional provider for deployment planning.

## 4. Technical Stack
*   **Intelligence**: Databricks Mosaic AI (Llama 3.3 70B), Vector Search.
*   **Observability**: MLflow Trace (Chain-of-Thought transparency for every query).
*   **Backend**: Python (Pandas/Vectorized Ops), FastAPI.
*   **Frontend**: Interactive Crisis Dashboard with Leaflet.js MarkerClustering and Bayesian Audit Gauges.

## 5. Winning Differentiators (Why We Win)
1.  **Trust-First RAG**: We don't just retrieve results; we rank them by **Trust Score**. We prioritize verified, medically consistent facilities over ones with high-marketing keywords.
2.  **Total Transparency**: MLflow traces allow judges to see the "Agentic Reasoning" behind the Query Planner (how it converts natural language into structured medical logic).
3.  **Scalability**: Our vectorized engine can handle 1,000,000 facilities with the same latency as 10,000.
4.  **Social Impact**: Beyond search—we provide a roadmap for where the government needs to build the next "HealthBrick."
