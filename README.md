# Agentic Healthcare Intelligence System (India)

This project builds an end-to-end reasoning layer for healthcare discovery across 10k+ Indian facility records with null-heavy, unstructured text.

## What You Get

- Massive unstructured extraction from free text fields (`description`, `specialties`, `procedure`, `equipment`, `capability`)
- Multi-attribute reasoning query engine (not just keyword matching)
- Trust Scorer with contradiction checks and confidence intervals
- Validator agent against baseline medical logic rules
- Medical desert detection by state and PIN code
- Row-level citation snippets for explainability
- Trace logs for each pipeline stage (MLflow-ready integration path)
- MLflow-backed observability spans and run metrics with estimated trace cost
- Mosaic AI Vector Search sync and semantic retrieval module
- Genie-style autonomous multi-step task orchestrator
- FastAPI service for query and desert endpoints
- Streamlit dashboard for planner/NGO usage
- Databricks notebook for Free Edition execution

## Repository Layout

- `src/healthcare_intel/io/data_loader.py`: CSV ingestion and null-safe normalization
- `src/healthcare_intel/agents/extractor_agent.py`: capability extraction from unstructured text
- `src/healthcare_intel/validation/trust_scorer.py`: trust score + contradictions + confidence bounds
- `src/healthcare_intel/validation/validator_agent.py`: standards cross-check
- `src/healthcare_intel/reasoning/query_engine.py`: multi-attribute query parser and ranking
- `src/healthcare_intel/reasoning/vector_search.py`: Mosaic AI Vector Search sync and retrieval
- `src/healthcare_intel/analytics/desert_detection.py`: specialized desert detection
- `src/healthcare_intel/pipeline.py`: full orchestration and artifact output
- `src/healthcare_intel/observability.py`: MLflow tracing + cost estimate metrics
- `src/healthcare_intel/agents/genie_orchestrator.py`: autonomous multi-step execution engine
- `src/healthcare_intel/api.py`: API endpoints
- `src/healthcare_intel/dashboard_app.py`: dashboard UI
- `notebooks/01_databricks_end_to_end.ipynb`: Databricks-ready notebook

## Dataset

Put your CSV in `data/` (or any path) with your provided columns, including null values.

Example filename:

- `data/VF_Hackathon_Dataset_India_Large.csv`

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) Run pipeline

```bash
python scripts/run_pipeline.py --dataset data/VF_Hackathon_Dataset_India_Large.csv --output outputs
```

To disable MLflow instrumentation:

```bash
python scripts/run_pipeline.py --dataset data/VF_Hackathon_Dataset_India_Large.csv --output outputs --disable-mlflow
```

To sync to Mosaic AI Vector Search while running:

```bash
python scripts/run_pipeline.py --dataset data/VF_Hackathon_Dataset_India_Large.csv --output outputs --vector-endpoint <endpoint_name> --vector-index <catalog.schema.index_name>
```

### 3) Launch API

```bash
uvicorn healthcare_intel.api:app --reload
```

### 4) Launch dashboard

```bash
streamlit run src/healthcare_intel/dashboard_app.py
```

### 5) Run Genie-style autonomous task

```bash
python scripts/run_genie_task.py --action build_knowledge_base --params "{\"dataset_path\": \"data/VF_Hackathon_Dataset_India_Large.csv\", \"output_dir\": \"outputs\", \"enable_mlflow\": true}"
```

## Output Artifacts

- `outputs/facilities_enriched.parquet`
- `outputs/facilities_enriched.csv`
- `outputs/specialized_deserts.parquet`
- `outputs/specialized_deserts.csv`
- `outputs/pipeline_trace.jsonl`

## Trust Scoring Summary

Score blends:

- positive evidence support,
- contradiction penalties,
- missing critical fields,
- confidence interval around estimated reliability.

Example contradiction:

- Emergency surgery claim without anesthesiologist evidence.

## Sample Query

"Find the nearest facility in rural Bihar that can perform an emergency appendectomy and typically leverages parttime doctors"

Use API `/query` with optional `latitude` and `longitude` to rank by trust + distance.

## Databricks Notes

- Use serverless compute in Databricks Free Edition.
- Upload CSV to DBFS/Unity Catalog Volume.
- Open notebook `notebooks/01_databricks_end_to_end.ipynb` and execute cells.
- Use `src/healthcare_intel/reasoning/vector_search.py` or CLI flags to sync outputs to Mosaic AI Vector Search.

## Stretch Extension Hooks

- Add LLM extraction with Databricks Model Serving endpoint in extractor stage.
- Log step spans to MLflow Tracing for every recommendation path.
- Add self-correction loops by re-scoring low-confidence rows with validator prompts.
- Build PIN-level India choropleth from `specialized_deserts.csv`.
