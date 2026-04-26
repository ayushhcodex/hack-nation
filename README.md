<div align="center">

# HealthIntel India 🏥 
### Agentic Healthcare Intelligence mapped for 1.4 Billion Lives

*A Databricks for Good Hackathon Project*

</div>

## 📖 Overview
HealthIntel India is a fully autonomous, Multi-Agent AI System designed to parse, synthesize, and map the medical capabilities of over 10,000 Indian healthcare facilities. Built for the *"Serving a Nation"* problem statement, this system abandons rigid logic and simple SQL in favor of **Agentic Engineering**. It intelligently identifies critical Medical Deserts and produces verifiable Chain-of-Thought trust metrics for NGO health planners.

## 🚀 The Tech Stack
- **Core Intelligence Engine:** Databricks Foundation Model APIs (Specifically `Meta Llama 3.3 70B Instruct`).
- **Semantic Retrieval:** Mosaic AI Vector Search.
- **Backend Server:** Python, FastAPI, Uvicorn, Pandas.
- **Frontend Dashboard:** Vanilla HTML5, CSS3 (Glassmorphism UI), JavaScript (ES6), Leaflet.js (MarkerCluster).

---

## 🛠️ The Architecture & Contributions

Over the course of development, this application was rapidly evolved from a static deterministic parser into a living **Agentic Neuro-Symbolic System** across 4 major milestones:

### 1. Unstructured Data Extraction Agent (`multi_pass_extractor.py`)
Real-world hospital data is terribly messy. Initially, this application used a simple Regex scanner (e.g., looking for the word "ICU"). 
- **The Upgrade:** The extraction core was replaced with a Databricks Llama 3.3 Agent. It dynamically reads free-form clinical notes and mathematically infers missing capabilities. 
- **Verbatim Anchoring:** To prevent medical hallucinations, the Agent forces every single extracted capability to be anchored to an exact sub-string citation.

### 2. Multi-Attribute LLM Query Planner (`query_engine.py`)
Standard systems use rigid dictionary filters to search for facilities. 
- **The Upgrade:** A Natural Language Query Planner. When a user asks *"Find rural clinics near Bihar with part-time dentists"*, the LLM planner digests the constraints, formats them into a strict search schema, and pipes the semantic logic into the Databricks Vector Search DB.

### 3. Autonomous Genie Orchestrator (`genie_orchestrator.py`)
Instead of developers writing manual Python files to execute code, the Pipeline operates completely autonomously.
- **Detailed Orchestration Engine:** The `chat_and_execute(prompt: str)` function acts as a LangChain-style router. The system feeds the Databricks model live schemas for internal Python tools (like `sync_vector_index` or `build_knowledge_base`). The LLM understands the prompt intent, selects the exact mathematical sequence of python functions to call, and executes the data pipeline dynamically. 
- **Self-Healing:** If an API step fails in the multi-tool chain, the orchestrator interrupts the data loop and reports exactly which payload failed.

### 4. LLM Validator Agent & Dynamic Mapping  (`validator_agent.py` & `app.js`)
Medical data is full of errors, so we instituted a stringent "Self-Correction Loop".
- **Metadata Context Injection:** An LLM Medical Auditor reads the claims a hospital makes alongside its **Metadata** (e.g., checking if a "Primary Care Clinic" is claiming to have "Advanced Neurosurgery"). If it spots a logical paradox, it flags the facility with a Trust Score penalty.
- **Dynamic Crisis Mapping UI:** The frontend map was overhauled with `Leaflet.markercluster`. Thousands of Databricks facility nodes are dynamically clustered into glowing heatmap nodes that scale intelligently across the map of India at 60 FPS without crashing the browser.

---

## 💻 Running the Application Locally

The application acts securely as a local bridge to the Databricks Cloud. You will gracefully bypass CORS limits by having your Python Backend trigger the AI.

1. **Configure Environment:** Ensure your `.env` contains your Databricks API keys.
```env
DATABRICKS_HOST="https://dbc-YOUR-WORKSPACE.cloud.databricks.com/"
DATABRICKS_TOKEN="dapi..."
LLM_ENDPOINT_NAME="databricks-meta-llama-3-3-70b-instruct"
```

2. **Start the API & Web Server:**
```bash
export PYTHONPATH=src
python3 -m uvicorn healthcare_intel.api:app --reload --port 8000
```

3. **Explore HealthIntel:**
Navigate to `http://127.0.0.1:8000/` to test the Discovery search engine and visualize the Medical Deserts Map in real-time.
